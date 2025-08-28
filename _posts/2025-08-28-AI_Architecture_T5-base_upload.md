---
layout: single
title:  "AI 아키텍쳐 5. T5-base"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

요즘, 처음부터 글을 직접 쓰는 사람은 거의 없을 것이다 AI가 워낙 글을 잘 써주기도 하고 말이다.  
이 글은 GPT5 기반으로 만든 글이다 성능 좋네  

## T5 (Text-to-Text Transfer Transformer)

BERT가 ‘이해’를, GPT-2가 ‘생성’을 상징했다면, **T5(Text-to-Text Transfer Transformer)**는 그 둘을 하나의 관점으로 묶어 **“모든 NLP를 텍스트 → 텍스트 변환 문제로 통합”**한 모델이다.  
구글이 제시한 이 관점은 번역, 요약, 질의응답, 분류, 자연어 추론까지 모두 **하나의 인터페이스**로 다룬다.  
즉, 입력 앞에 **태스크 프리픽스(task prefix)**를 붙여 모델이 해야 할 일을 자연어로 지정하고, 출력은 항상 텍스트로 받는다.  

예시:  
- 번역: `translate English to Korean: How are you?` → `잘 지내?`  
- 요약: `summarize: This article explains ...` → `이 글은 ...을 설명한다.`  
- 분류: `sst2 sentence: The movie was great` → `positive`  

이 글에서는 **T5-base**를 중심으로, 설계 철학, 아키텍처, 학습 목표(Span Corruption), 상대적 위치 바이어스, 최적화(Adafactor), 디코딩 전략, 파인튜닝 레시피까지 **레이어 단위**로 깊이 있게 파헤친다.  

---

## 스펙 개요 (T5 계열)  

| 모델 | 파라미터 수 | 인코더 레이어 | 디코더 레이어 | d_model | n_head | d_ff | Vocab |
|---|---:|---:|---:|---:|---:|---:|---:|
| T5-small | ~60M | 6 | 6 | 512 | 8 | 2048 | 32k |
| **T5-base** | **~220M** | **12** | **12** | **768** | **12** | **3072** | **32k** |
| T5-large | ~770M | 24 | 24 | 1024 | 16 | 4096 | 32k |
| T5-3B | ~3B | 24 | 24 | 1024 | 32 | 16384 | 32k |
| T5-11B | ~11B | 24 | 24 | 1024 | 128 | 65536 | 32k |

- 토크나이저: **SentencePiece(Unigram)**, vocab 약 32k, **센티넬 토큰** `<extra_id_0> ... <extra_id_99>`.  
- 임베딩 공유: **입력 임베딩**과 **출력 소프트맥스 가중치** **공유(weight tying)**.  
- 정규화: 표준 LayerNorm이 아닌 **RMSNorm** 변형(**T5LayerNorm**).  
- 위치: 절대 위치 임베딩이 아니라 **상대적 위치 바이어스(Relative Position Bias)**.  
- 활성함수: 기본 **ReLU** (T5.1.1 변형은 **Gated GELU** 등을 사용하지만, 본 글은 원 T5-base 기준).  
- 최적화: **Adafactor**, inverse square-root 계열 스케줄, label smoothing.  

---

## 철학: 단일 목표로 통합되는 확장성

T5의 핵심은 **단일 학습 목표**로 거의 모든 다운스트림 태스크를 커버하는 것이다.  
우리는 조건부 생성 확률을 최대화한다:  

$$
\max_{\theta}\;\log p_\theta(\mathbf{y}\mid \mathbf{x})
= \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, \mathbf{x})
$$

여기서 입력 $\mathbf{x}$는 태스크 프리픽스를 포함한 토큰 시퀀스, 출력 $\mathbf{y}$는 해당 태스크의 정답 시퀀스다.  
**하나의 목적함수**로 통일했기에, 사전학습과 파인튜닝, 제로샷/퓨샷 사용까지 동일한 뼈대를 공유한다.  

---

## 입력 표현과 토크나이저

### SentencePiece(Unigram)
문장을 서브워드 단위로 분해한다. Unigram 모델은 후보 어휘 집합에서 가능도 기반으로 최적의 서브워드 분해를 고른다.  

- 장점: 언어 불변성, OOV 최소화, 작은 vocab으로 광범위한 텍스트 커버.
- 구현: 공백 포함 처리, 숫자/기호의 안정적 처리.  

### 센티넬 토큰
스팬 마스킹을 위해 **예약된 특수 토큰** `<extra_id_n>`을 사용한다.  
입력 텍스트에서 제거된 연속 스팬마다 하나의 센티넬이 들어가고, **타깃 시퀀스**는 해당 스팬들을 **센티넬로 구분하여 나열**한다.  

예:  
- 입력(마스킹 후): `The <extra_id_0> over the <extra_id_1>.`  
- 타깃: `<extra_id_0> quick brown fox <extra_id_1> lazy dog <extra_id_2>`  

---

## 임베딩과 파라미터 공유  

T5는 **임베딩 공유(weight tying)**를 채택한다.  
즉, 입력 임베딩 행렬 $E \in \mathbb{R}^{V \times d}$와 출력 로짓의 선형 변환 가중치 $W \in \mathbb{R}^{d \times V}$가 다음과 같이 연결된다:  

$$
W = E^\top
$$

출력 로짓은
$$
\mathbf{z}_t = \mathbf{h}_t W = \mathbf{h}_t E^\top
$$
이며, 소프트맥스로 확률을 만든다:
$$
p_\theta(y_t = v \mid \cdot) = \frac{\exp(z_{t,v})}{\sum_{v'} \exp(z_{t,v'})}.
$$

**효과**: 파라미터 수 절감, 입력/출력 공간의 일관성, 학습 안정성 향상.  

---

## 정규화: RMSNorm(T5LayerNorm)

표준 LayerNorm은 평균을 빼고 분산으로 나누지만, T5는 **RMSNorm**을 사용한다:  

$$
\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{g}
$$

- 평균을 빼지 않고 **RMS로만 정규화**.  
- $\mathbf{g}$는 학습 가능한 게인 벡터.  
- **Pre-Norm** 구조: 각 서브레이어(어텐션/FFN) **입력에** RMSNorm을 적용한 후 잔차를 더한다.    
  즉,  
  $$
  \mathbf{y} = \mathbf{x} + \mathrm{Dropout}\big(f(\mathrm{RMSNorm}(\mathbf{x}))\big).
  $$

**이점**: 깊은 네트워크에서 학습 안정성 증대, 그래디언트 흐름 개선.  

---

## 상대적 위치 바이어스(Relative Position Bias)  

T5는 절대 위치 임베딩을 쓰지 않는다. 대신 **어텐션 로짓**에 **상대적 거리 기반 바이어스**를 더한다.  
헤드 $h$에 대한 어텐션 로짓은  

$$
\mathbf{A}^{(h)} = \frac{Q^{(h)} {K^{(h)}}^\top}{\sqrt{d_k}} + \mathbf{B}^{(h)}
$$

여기서 $\mathbf{B}^{(h)} \in \mathbb{R}^{L\times L}$는 토큰 $i$와 $j$의 상대적 거리 $\Delta = j-i$에 대해 **버킷화(bucketing)**된 임베딩에서 조회한 값:  

$$
B^{(h)}_{ij} = b^{(h)}\big(\mathrm{bucket}(\Delta)\big).
$$

버킷 함수는 작은 거리일수록 **고해상도**, 먼 거리는 **로그 스케일**로 **저해상도** 버킷에 매핑한다.  
이 방식은 긴 컨텍스트에서도 일반화가 잘 된다.  

**소프트맥스**로 가중치를 만들고 값 집계:  
$$
\mathrm{Attn}^{(h)}(\mathbf{X}) = \mathrm{softmax}\left(\mathbf{A}^{(h)}\right)V^{(h)}.
$$

멀티헤드는  
$$
\mathrm{MHA}(\mathbf{X})
= \mathrm{Concat}\big(\mathrm{Attn}^{(1)}(\mathbf{X}),\dots,\mathrm{Attn}^{(H)}(\mathbf{X})\big)W^O.
$$

---

## 인코더-디코더 스택 구조  

T5는 **Encoder-Decoder** 구조를 가진다.  

### 인코더 레이어(×12, T5-base)
각 레이어는  
1) **Self-Attention (RMSNorm → MHA → Dropout)**  
2) **FFN (RMSNorm → Dense(ReLU) → Dense → Dropout)**  
3) **Residual Add**  
로 구성된다.  

포지션-와이즈 FFN:  
$$
\mathrm{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2,
$$
여기서 $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, T5-base는 $d=768$, $d_{ff}=3072$.

### 디코더 레이어(×12, T5-base)
디코더는 두 종류의 어텐션을 가진다:  
1) **Masked Self-Attention**: 미래 토큰을 보지 않도록 **look-ahead mask** 적용.  
   $$
   \tilde{\mathbf{A}}^{(h)} = \frac{Q^{(h)} {K^{(h)}}^\top}{\sqrt{d_k}} + \mathbf{B}^{(h)} + \mathbf{M},
   $$
   $\mathbf{M}$은 상삼각(미래) 위치에 $-\infty$를 넣는 마스크 행렬.  

2) **Encoder-Decoder Cross-Attention**: 디코더 쿼리가 인코더 키/밸류에 주의.  
   $$
   \mathbf{A}_{\text{cross}}^{(h)} = \frac{Q_{\text{dec}}^{(h)} {K_{\text{enc}}^{(h)}}^\top}{\sqrt{d_k}} + \mathbf{B}_{\text{cross}}^{(h)}.
   $$
   (교차에서도 상대적 바이어스를 둘 수 있으나, 구현은 인코더-디코더 길이/정렬에 맞춤)  

3) **FFN**, 그리고 각 서브레이어 앞에 **RMSNorm**, 뒤에 **Residual Add**.  

---

## 사전학습 목표: Span Corruption(Denoising)  

BERT의 MLM은 개별 토큰을 가리는 반면, T5는 **연속된 스팬**을 가린다.  
**노이즈 밀도** $\rho$ (예: 0.15)와 **평균 스팬 길이** $\lambda$ (예: 3)를 정하고, 전체 토큰 중 약 $\rho$ 비율만큼을 **길이 ~ Poisson/Geometric 분포**에서 샘플한 스팬들로 덮는다.  

입력 시퀀스 $\mathbf{x} = (x_1, \dots, x_n)$에서 마스킹된 스팬들을 $\{\mathcal{S}_k\}$라 하면,  
- **입력**은 각 스팬을 **단일 센티넬**로 대체:  
  $$
  \tilde{\mathbf{x}} = \mathrm{mask\_spans}(\mathbf{x}; \{\mathcal{S}_k\}) \in \mathcal{V}^*,
  $$
- **타깃**은 제거된 스팬들을 **센티넬로 구분하여 연결**:  
  $$
  \mathbf{y}^\star = \big\langle \langle\text{extra\_id}_0\rangle, \mathbf{x}_{\mathcal{S}_1}, \langle\text{extra\_id}_1\rangle, \mathbf{x}_{\mathcal{S}_2}, \dots, \langle\text{extra\_id}_K\rangle \big\rangle.
  $$

학습 손실(teacher forcing, label smoothing $\alpha$):  
$$
\mathcal{L}(\theta) = - \sum_{t=1}^{T} \sum_{v=1}^{V}
\tilde{y}_{t,v}\;\log p_\theta(y_t=v \mid y_{<t}, \tilde{\mathbf{x}}),
$$
$$
\tilde{y}_{t,v} =
\begin{cases}
1-\alpha & \text{if } v = y_t\\
\frac{\alpha}{V-1} & \text{otherwise}
\end{cases}.
$$

**왜 스팬인가? (필요성)**  
문장 구조는 종종 **연속된 구문 단위**로 조직된다.  
스팬 단위 복원은 **문맥 이해 + 생성 능력**을 동시에 훈련하며, 디코더를 **적극적으로** 사용한다.  

---

## 최적화: Adafactor  

거대한 모델과 시퀀스 길이를 다루기 위해 T5는 **Adafactor**를 사용한다(Adam의 메모리 비용을 줄이는 변형).  
핵심 아이디어는 2차 모멘트(분산) 추정을 **행/열 팩터**로 근사하는 것:  

- 파라미터 행렬 $W \in \mathbb{R}^{m\times n}$에 대해
  $$
  \mathbb{E}[g\odot g] \approx \mathbf{r}\mathbf{c}^\top
  $$
  형태로 근사(여기서 $\mathbf{r} \in \mathbb{R}^m$, $\mathbf{c} \in \mathbb{R}^n$).

업데이트 스케치:  
$$
\mathbf{v}_r \leftarrow \beta_2 \mathbf{v}_r + (1-\beta_2)\; \mathrm{mean}_{\text{cols}}(g\odot g),\quad
\mathbf{v}_c \leftarrow \beta_2 \mathbf{v}_c + (1-\beta_2)\; \mathrm{mean}_{\text{rows}}(g\odot g),
$$

$$
\hat{\mathbf{V}} \approx \frac{\mathbf{v}_r \mathbf{v}_c^\top}{\mathrm{mean}(\mathbf{v}_r)}
$$

$$
W \leftarrow W - \eta \frac{g}{\sqrt{\hat{\mathbf{V}}} + \epsilon}.
$$

러닝레이트는 보통 **inverse square-root 스케줄**:  
$$
\eta_t = \eta_0 \cdot \min\left(t^{-1/2}, \frac{t}{t_{\text{warmup}}^{3/2}}\right).
$$

---

## 컴퓨팅 복잡도와 메모리  

- 어텐션 복잡도: $O(L^2 d)$ (길이 $L$의 시퀀스, 모델 차원 $d$).  
- 멀티헤드 $H$에서 각 헤드는 $d_k = d/H$.  
- 인코더-디코더의 총 비용은 **인코더 self-attn + 디코더 masked self-attn + cross-attn**의 합.  

메모리 팁:  
- **Gradient Checkpointing**, **Mixed Precision**, **Adafactor**로 메모리 사용 최적화.  
- 파인튜닝에서는 **LoRA/IA³** 같은 어댑터 기법으로 파라미터 효율을 높일 수 있다(원 논문 외 응용).  

---

## 디코딩: 빔 서치와 샘플링  

### 빔 서치(Beam Search)  
길이 보정(length penalty) $\alpha$를 쓰는 점수:  
$$
s(\mathbf{y}) = \frac{1}{(5+|\mathbf{y}|)^\alpha/(5+1)^\alpha} \sum_{t} \log p(y_t \mid y_{<t}, \mathbf{x}).
$$

- 장점: 결정적이고 높은 점수의 시퀀스 선호.  
- 단점: **다양성** 부족, 반복적 표현 문제.  

### 확률적 샘플링  
- **Top-k**: 상위 $k$개만 후보.  
- **Nucleus (Top-p)**: 누적확률 $p$를 넘을 때까지 후보 포함.  
- **Temperature $\tau$**:  
  $$
  p_\tau(v) \propto \exp\big(z_v/\tau\big).
  $$

실무 팁:  
- 요약: **beam 4–8 + length penalty**가 흔히 안정적.  
- 생성적 글쓰기: **top-p 0.9, temperature 0.7** 같은 세팅이 다양성↑.  

---

## 레이어-바이-레이어 해부  

### 1) 인코더 Self-Attention  
양방향 주의:  
$$
\mathbf{A}^{(h)}_{\text{enc}} = \frac{Q^{(h)}_{\text{enc}} {K^{(h)}_{\text{enc}}}^\top}{\sqrt{d_k}} + \mathbf{B}^{(h)}_{\text{enc}},
\quad
\mathrm{softmax}(\mathbf{A}^{(h)}_{\text{enc}})V^{(h)}_{\text{enc}}.
$$
- 전체 문맥을 글로벌하게 요약.  
- 문법/의미/장거리 의존성을 캡처.  

### 2) 디코더 Masked Self-Attention  
미래를 차단하는 마스크 $\mathbf{M}$:  
$$
\mathrm{softmax}\Big(\frac{Q_{\text{dec}}K_{\text{dec}}^\top}{\sqrt{d_k}} + \mathbf{B}_{\text{dec}} + \mathbf{M}\Big)V_{\text{dec}}.
$$
- 자연스러운 **자기회귀 생성**을 학습.  

### 3) Cross-Attention  
인코더 표현을 조건으로 출력 생성:  
$$
\mathrm{softmax}\Big(\frac{Q_{\text{dec}}K_{\text{enc}}^\top}{\sqrt{d_k}} + \mathbf{B}_{\text{cross}}\Big)V_{\text{enc}}.
$$
- 입력의 관련 부분을 정밀하게 참조(번역, 질의응답, 추출 요약 등에서 핵심).  

### 4) FFN(포지션-와이즈)  
각 토큰을 독립적으로 고차 특징으로 변환:  
$$
\mathbf{h}' = \max(0, \mathbf{h}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2.
$$

---

## 학습 데이터: C4(Colossal Clean Crawled Corpus)  

웹 크롤링 텍스트를 **강하게 정제(cleaning)**한 데이터셋.  
- 품질 낮은 페이지, 중복, boilerplate 제거.  
- 영어 중심(원 T5), 다만 멀티링구얼 변형도 존재.  
- 시퀀스 길이(프리트레인): 보통 512 토큰 단위 조각을 많이 사용(작업/리소스에 따라 변형 가능).  

---

## 왜 T5는 통합이 중요한가? (설계 의도)  

- **단일 인터페이스**: 프롬프트(태스크 프리픽스)를 수정하는 것만으로 태스크 전환.  
- **스케일 친화적**: 같은 목적함수로 데이터/모델 규모를 키우면 성능이 확장.  
- **사전학습 목표가 생성 친화적**: 스팬 복원은 생성 디코더 훈련에 직접적.  

BERT와 비교:  
- BERT: 인코더만, MLM/NSP 중심, 출력은 주로 분류 헤드.  
- T5: 인코더-디코더, 스팬 복원으로 **생성/이해** 모두 강화, 출력은 항상 텍스트.  

GPT-2와 비교:  
- GPT-2: 디코더만, 자기회귀 LM(조건부 입력은 프롬프트로 연결).  
- T5: 입력과 출력 사이 **크로스 어텐션**으로 양방향 컨텍스트-조건화에 유리.  

BART와 비교:  
- BART: 다양한 노이즈(토큰삭제/치환/순서교란 등)로 디노이징.  
- T5: **스팬 복원** 하나에 집중, **상대적 위치 바이어스 + RMSNorm** 등 구현 디테일 차별화.  

---

## 훈련 트릭과 안정화  

- **Label Smoothing**: $\alpha \approx 0.1$ 추천(태스크/데이터에 따라 조절).  
- **Dropout**: 0.1 근처(대규모 사전학습에서는 낮추거나 0).  
- **Warmup**: 수천~수만 스텝 워밍업 후, inverse sqrt.  
- **Grad Clipping**: 1.0 근처 권장.  
- **Mixed Precision**: 속도/메모리 효율↑.  
- **Batching**: 동적 패딩으로 낭비 최소화.  

---

## 프롬프트 설계(태스크 프리픽스)  

T5는 **자연어 태스크 명시**를 선호한다. 예:  

- 요약: `summarize: <문서>`  
- 번역: `translate English to Korean: <영문>`  
- 질의응답(추출): `question: <질문> context: <문맥>`  
- 분류: `sst2 sentence: <문장>` → `positive/negative`

프리픽스 설계는 **성능에 민감**하다. 명료하고 일관된 프리픽스를 유지하라.  

---

## 추론 파이프라인(의사코드)

```python

# Encoder

X\_in = tokenize(prefix + input\_text)           # SentencePiece IDs
X\_emb = embed(X\_in)                            # shared embedding
H\_enc = EncoderStack(X\_emb, rel\_pos\_bias, rmsnorm\_pre)

# Decoder (autoregressive)

y = \[BOS or sentinel start]                    # task-dependent
for t in range(max\_len):
Y\_emb = embed(y)
H\_dec = DecoderStack(Y\_emb, H\_enc, rel\_pos\_bias, mask, rmsnorm\_pre)
logits = H\_dec\[-1] @ E^T
p = softmax(logits\[-1])
y.append(sample(p) or beam\_step(p))
if y\[-1] == EOS: break

return detokenize(y)

```

---

## 파인튜닝  

- **데이터 포맷**: 항상 “텍스트 → 텍스트”. 라벨도 텍스트화(예: `entailment`, `contradiction`).  
- **학습률**: 1e-4 ~ 3e-4 (베이스 기준)에서 시작, 스케줄러로 감쇠.  
- **배치**: 64~2048 토큰/스텝(자원에 맞춤).  
- **시퀀스 길이**: 작업에 맞게 256/512/1024.  
- **정규화**: label smoothing, dropout 조절.  
- **체크포인트 전략**: dev set 로스/메트릭 기준 early stopping.  

**하드 태스크 팁**  
- 추출 요약/QA: 프리픽스에 `context:`/`question:` 명시적으로 붙이기.  
- 장문 요약: 길이 보정 있는 빔 서치 + 반복 억제(ngram blocking).  
- 멀티태스크: 서로 다른 태스크를 균형 있게 샘플링.  

---

## 오류 모드와 디버깅

- **반복적 생성(repetition)**: temperature/Top-p 조절, n-gram 반복 금지, 길이 보정 조정.  
- **할루시네이션**: 더 강한 조건화(명확한 프리픽스/컨텍스트), 길이 제한, 수치/엔티티 평가 도입.  
- **느린 디코딩**: 빔 축소, 캐싱 활용, INT8/FP8 디코딩 고려.  

---

## 수식으로 보는 T5의 학습/추론 핵심  

### 1) 조건부 생성  
$$
p_\theta(\mathbf{y}\mid \mathbf{x}) = \prod_{t=1}^{T} p_\theta(y_t \mid \mathbf{y}_{<t}, \mathbf{x})
= \prod_{t=1}^{T} \mathrm{softmax}\big(\mathbf{h}_t E^\top\big)_{y_t}.
$$

### 2) 상대적 위치 바이어스 결합 어텐션  
$$
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{B}\right) \mathbf{V}.
$$

### 3) RMSNorm 기반 Pre-Norm 잔차  
$$
\mathbf{y} = \mathbf{x} + f\big(\mathrm{RMSNorm}(\mathbf{x})\big),\quad
\mathbf{z} = \mathbf{y} + g\big(\mathrm{RMSNorm}(\mathbf{y})\big).
$$

### 4) 스팬 노이징 타깃 구성  
$$
\mathbf{y}^\star = \big[\langle e_0\rangle, \mathbf{x}_{\mathcal{S}_1}, \langle e_1\rangle, \mathbf{x}_{\mathcal{S}_2}, \dots, \langle e_K\rangle \big].
$$

### 5) 라벨 스무딩 크로스 엔트로피  
$$
\mathcal{L} = -\sum_{t}\sum_{v}\tilde{y}_{t,v}\log p_\theta(y_t=v\mid\cdot),\quad
\tilde{y}_{t,v} =
\begin{cases}
1-\alpha & v=y_t\\
\frac{\alpha}{V-1} & \text{else}
\end{cases}.
$$

---

## BERT / GPT-2 / BART와의 구조적 대비 (요약 표)

| 축 | BERT | GPT-2 | BART | **T5** |
|---|---|---|---|---|
| 구조 | Encoder | Decoder | Enc-Dec | **Enc-Dec** |
| 사전학습 | MLM(+NSP) | LM | 디노이징(다양) | **Span Corruption** |
| 위치 | 절대(learned) | 절대(learned) | 절대 | **상대 바이어스** |
| 정규화 | LayerNorm | LayerNorm | LayerNorm | **RMSNorm** |
| 임베딩 공유 | 경우에 따라 | 경우에 따라 | 경우에 따라 | **입력-출력 공유** |
| 인터페이스 | 주로 분류 헤드 | 프롬프트 생성 | 인코더-디코더 | **Text→Text 통합** |

---

## 구현 디테일(주의점)

- **Pad/Mask 처리**: 인코더 패딩 마스크, 디코더 **look-ahead mask** 동시 처리 필요.  
- **상대 바이어스 버킷**: 길이 증가 시 버킷 범위/분할이 어텐션 품질에 영향.  
- **공유 임베딩**: 학습률/정규화가 임베딩과 로짓 양쪽에 동시에 영향.  
- **RMSNorm epsilon**: 너무 작으면 수치 불안정, 보통 $10^{-6}$ 수준.

---

## Hugging Face 사용 예  

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-base"
tok = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

task_input = "summarize: This is a long article about T5..."
x = tok(task_input, return_tensors="pt")

# Beam search + length penalty
y = model.generate(
    **x,
    num_beams=4,
    length_penalty=1.0,
    max_new_tokens=128,
    no_repeat_ngram_size=3
)
print(tok.decode(y[0], skip_special_tokens=True))
```

파인튜닝 팁:  

* 입력은 항상 `"<task>: <input>"` 형태 유지.  
* 라벨 텍스트는 가능한 **짧고 일관**되게(분류 라벨도 텍스트).  

---

## 성능/일반화에 기여하는 요소 정리

1. **스팬 단위 디노이징**: 문장 구조/패턴 복원 능력 강화 → 다양한 다운스트림에 전이.
2. **Enc-Dec + Cross-Attn**: 입력과 출력의 정밀한 정렬/조건화 → 번역/요약에 최적.
3. **상대적 위치 바이어스**: 긴 문맥 일반화 및 위치 불변성.
4. **RMSNorm + Pre-Norm**: 깊은 스택에서의 안정적 학습.
5. **임베딩 공유**: 파라미터 절감과 표현 일관성.
6. **Adafactor**: 대규모 학습에서의 메모리/속도 현실성.

---

## T5-base를 어디에 써야 할까?

* **요약**(뉴스, 과학 논문 초록, 회의록): 안정적이고 정보 보존률이 좋은 편.
* **번역**: 소규모 도메인 적응 시 강력.
* **질의응답**: `question:`, `context:` 프리픽스로 명확한 조건화.
* **분류/추론**: 라벨을 텍스트로 정하면 자연스러운 적응 가능.
* **데이터 증강**: 설명 생성, 패러프레이즈, 템플릿 채우기.

---

## 한 걸음 더: 변형/후속 작업

* **FLAN-T5**: 지시어(Instruction) 튜닝으로 제로샷/퓨샷 향상.
* **UL2**: 다양한 학습 목표 혼합(마스킹/자기회귀/Prefix-LM).
* **T5.1.1**: 학습 안정화, 활성함수 변경(GeGLU), 드롭아웃 조정 등.
* **Parameter-Efficient Tuning**: LoRA, Prefix/Prompt Tuning, BitFit 등.

---

## 결론

**T5-base**는 “텍스트→텍스트”라는 단일 패러다임으로 NLP를 통합했다.
스팬 디노이징, 인코더-디코더, 상대적 위치 바이어스, RMSNorm, 임베딩 공유, Adafactor 최적화까지—디테일 하나하나가 **확장성과 범용성**을 위해 설계되어 있다.  

BERT가 이해를, GPT-2가 생성을 열었다면, \*\*T5는 ‘지시 가능한 범용 변환기’\*\*로 그 사이를 단단히 메웠다.  
오늘날의 인스트럭션 튜닝(FLAN-T5), 태스크 통합적 파인튜닝, 멀티태스크 학습의 토대에는 T5의 철학이 흐른다.  

바로 사용해보고 싶다면:  

* 모델: `t5-base`  
* 인터페이스: **자연어 프리픽스 + 텍스트 라벨**  
* 디코딩: 과제에 맞춰 **beam / top-p / temperature**를 조합  

마지막으로, 링크:  

* Hugging Face: [https://huggingface.co/t5-base](https://huggingface.co/t5-base)

캬 GPT5 성능 좋네 내가 읽어도 이건 잘 쓴 글 같다.  