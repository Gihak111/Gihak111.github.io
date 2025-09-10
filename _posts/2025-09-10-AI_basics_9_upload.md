---
layout: single
title: "AI 입문 9편: 트랜스포머 최적화와 스케일링"
categories: "AI"
tag: "Explanation"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈

8편에서 트랜스포머의 특징·인코더-디코더 조합(BERT, GPT, T5)·Seq2Seq·RNN/CNN/LSTM 비교를 다루었다.  
이번 9편에서는 트랜스포머 최적화와 스케일링을 본격적으로 들어가 본다.  
구체적으로 Adam/AdamW 옵티마이저, Learning Rate Schedule(Warmup, Cosine Decay), Flash Attention, Chinchilla 스케일링 법칙을 수식·그래프·미분·장단점·실전 팁으로 설명한다.  
목표는 트랜스포머를 실전에서 효율적으로 학습시키는 방법을 이해하는 것이다.  

---

## 0. 이번 편의 핵심

* 최적화: 손실 함수를 최소화하는 파라미터 업데이트, Adam/AdamW로 안정적 수렴이다.  
* Learning Rate Schedule: Warmup으로 초기 안정, Cosine Decay로 세밀 조정이다.  
* Flash Attention: 어텐션의 메모리 병목(O(N²)) 해결, 효율적 훈련·추론이다.  
* Chinchilla Law: 파라미터(N)와 데이터(D)의 최적 비율로 성능 극대화이다.  
* 수식과 미분: 옵티마이저의 업데이트와 스케줄의 기울기 영향이다.  
* 적분 관점: 학습을 연속적 변화로, 스케일링을 누적 성장으로 본다.  

---  

## 1. 트랜스포머 최적화의 필요성: 왜 SGD로는 부족한가  

### 1.1 최적화란 무엇인가  

최적화는 손실 함수  $L(\theta)$를 최소화하는 파라미터 $\theta$ (가중치, 편향 등)를 찾는 과정이다.  
2편에서 본 기본 경사하강법(SGD):  

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)
$$

여기서 $\eta$: 학습률(LR), 고정되면 문제 발생이다.  
$\nabla_\theta L(\theta)$: 기울기, 손실의 방향과 크기를 나타낸다.  
트랜스포머는 수억~수조 파라미터를 가지므로 고차원 손실 곡면에서 SGD는 진동, 소실, 폭주 위험이 크다(3편 참조).   

트랜스포머 특성: 대규모 파라미터(BERT 110M, GPT-3 175B)로 메모리·컴퓨트 집중이 필요하다.  
불안정: 초기 학습에서 기울기 폭주 가능(4편 LayerNorm, 6편 잔차로 완화).  어텐션 병목: Multi-Head Attention(MHA, 5편)의 O(N²) 메모리 사용이다.  

최적화는 “산속 보물(최소 손실) 찾기”이다.  
SGD는 고정된 걸음 크기로 직진하지만, 지형이 복잡하면 미끄러지거나(폭주) 갇힘(소실)한다.  
Adam은 “스마트 내비게이션”처럼 속도·방향을 동적으로 조절한다.  

* Adam/AdamW: 적응적 LR로 안정성 증가.  
* LR Schedule: 초기(Warmup), 후기(Cosine Decay) 조절.  
* Flash Attention: 메모리 효율화.  
* Chinchilla: 파라미터·데이터 균형이다.  

### 1.2 미분과 최적화의 연결  

2편에서 배운 미분 $\nabla L$은 손실의 방향·크기 제공이다.  
옵티마이저는 이를 활용해 $\theta$ 업데이트한다.  
트랜스포머는 레이어 많아 체인룰 복잡(2편).  
미분은 “오르막 경사도”를 의미하며, 옵티마이저는 “어떻게 내려갈지” 결정한다.  
예를 들어, 어텐션 레이어의 미분은 Q,K,V 투영 행렬로 전파되어 전체 모델 업데이트에 영향을 준다.  

장단점: SGD의 장점은 간단하지만, 단점은 고정 LR로 인한 불안정이다.  
변형: 모멘텀 추가로 개선.  
팁: 기울기 클리핑(최대 norm 1.0)으로 폭주 방지.  

---

## 2. Adam과 AdamW: 트랜스포머의 기본 옵티마이저  

### 2.1 Adam의 정의와 수식  

Adam(Adaptive Moment Estimation)은 모멘텀(SGD-M)과 RMSProp의 장점을 결합한다.  
수식(t번째 스텝, 기울기 $g_t = \nabla_\theta L(\theta_t)$:  

1. 1차 모멘트(평균):  
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   여기서 $m_0 = 0$,  
   $\beta_1 \approx 0.9$: 과거 기울기 관성(모멘텀)을 유지한다.  
   계산 과정: 초기 m_0=0이므로 $m_1 = (1-0.9)g_1 = 0.1 g_1$.  
   의미: 방향성을 부드럽게 평균화.  

2. 2차 모멘트(분산):  
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   $v_0 = 0$, $\beta_2 \approx 0.999$: 기울기 크기 적응.  
   계산: $v_1 = 0.001 g_1^2$. 의미: 각 파라미터별 LR 스케일링.  

3. 바이어스 보정:  
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$
   초기 왜곡 보정. 예: $t=1, 1 - β1^1 = 0.1, \hat{m}_1 = m_1 / 0.1 = 10 m_1$.  
   의미: 초기 모멘트가 0에 가까워 과소평가 방지.  

4. 파라미터 업데이트:  
   $$
   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
   $$
   $\epsilon = 10^{-8}$: 0 나누기 방지. $\eta$: 기본 1e-3. 의미: 큰 기울기(v_t 크면)에서 스텝 작게.  

m_t는 “과거 방향 평균” –> 자동차의 관성처럼 부드럽게.  
v_t는 “기울기 크기 평균” –> 큰 기울기면 LR 줄임.  
파라미터마다 기울기 스케일 달라 트랜스포머엔 적응적 LR 필수.  

Adam vs SGD – Adam은 곡면에서 진동 적고 빠르게 최소점 도달.  
예: 손실 곡선에서 SGD는 지그재그, Adam은 부드러운 하강.  

### 2.2 AdamW: Weight Decay 개선  

AdamW는 Adam에 L2 규제(Weight Decay, WD) 분리:  

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

$\lambda \approx 0.01$: WD 강도, 파라미터 크기 억제(과적합 ↓).  
왜 분리? Adam은 WD가 $v_t$에 섞여 효과 ↓, AdamW는 명시적 규제.  
계산 과정: 업데이트 전에 $λ θ_t$ 추가 페널티.  
의미: 규제가 기울기 적응에 독립.  

Adam은 “스마트 내비”, AdamW는 “연료 절약” 추가 –> 불필요한 파라미터 키우지 않음.  

### 2.3 미분 유도  

Adam의 업데이트는 체인룰(2편)로 역전파.  
예: 출력층 미분 $∂L/∂θ_t → m_t, v_t$ 업데이트 → $θ_{t+1}$.  
$\hat{v}_t$는 “기울기 크기 스케일러” – 큰 기울기면 스텝 작게.  
전체 미분 흐름: 손실 → 어텐션 미분($QK^T / sqrt(d_k)$) → FFN → 임베딩.  

### 2.4 장단점  

장점: 안정·빠른 수렴, 트랜스포머 기본(BERT, GPT).  
단점: 하이퍼파라미터(η, β1, β2) 튜닝, 메모리 사용(1·2차 모멘트).  
변형: Lion(메모리 ↓), Adafactor(스케일링 효율).  
팁: Hugging Face Trainer AdamW 기본(η=1e-4~1e-3).  
β1=0.9, β2=0.999 고정, λ=0.01부터 시작.  

---

## 3. Learning Rate Schedule: 동적 LR 조절  

### 3.1 왜 필요한가  

고정 η는 초기 폭주(기울기 큰 구간)나 후기 정체(최소점 근처) 발생.  
스케줄은 단계별 조정으로 안정 학습.  
트랜스포머에서 초기 LayerNorm(4편) 안정화 필요.  

“자동차 운전” –> 출발(Warmup)은 천천히, 중간은 빠르게, 도착(Cosine Decay)은 섬세히.  

### 3.2 Linear Warmup  

초기 LR 0→η 증가:  

$$
\eta_t = \eta \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)
$$

$T_{\text{warmup}}$: 5~10% 총 스텝(예: 10K/100K).  
계산: $t=1, η_t = η * 1/10K$.  
의미: 초기 기울기 폭주 방지.  
초기 LayerNorm·잔차(4·6편) 안정화 전.  

“엔진 예열” –> 급출발 사고 방지.  

### 3.3 Cosine Decay  

후반 η 감소:  

$$
\eta_t = \frac{\eta}{2} \left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}\right)\right)
$$

$T$: 총 스텝. 계산: $t=T_warmup, cos(0)=1, η_t=η. t=T, cos(π)=-1, η_t=0$.  
의미: 최소점 근처 세밀 탐색, 진동 ↓.  

미분 영향: η ↓로 ∇L 스텝 작아짐, 안정 수렴.  
체인룰에서 η가 전체 기울기 스케일링.  

### 3.4 Noam Schedule (트랜스포머 전용)  

$$
\eta_t = \eta \cdot \min\left( \frac{t}{T_{\text{warmup}}}, \sqrt{\frac{T_{\text{warmup}}}{t}} \right) \cdot \frac{1}{\sqrt{d_{\text{model}}}}
$$

$d_{\text{model}}$: 임베딩 차원(512~1024).  
계산: 초기 선형 후 $1/sqrt(t)$ 감소.  
의미: 모델 크기 반영, 초기 급등 후 감소.  
트랜스포머 원 논문에서 제안, d_model 증가 시 LR ↓.  

### 3.5 장단점·실전  

장점: Warmup+Cosine으로 수렴 10~20% ↑(BERT 실험).  
단점: $T_{\text{warmup}}, T$ 추정 필요.  
변형: Exponential Decay.  
실전: $T_{\text{warmup}} = 10\% T$, η=1e-4. 스케줄 없으면 학습 실패 – Loss Curve로 확인.  
배치 크기 증가 시 η 스케일(√batch).  

---

## 4. Flash Attention: 효율적 어텐션 메커니즘  

### 4.1 어텐션의 메모리 병목  

5편 MHA: $\text{Attention} = softmax\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$.  
문제: $Q K^\top$ (N×N) 메모리 O(N²), 긴 시퀀스(64K)에서 GPU 폭주.  
계산: N=1024, 1M 원소 저장. 의미: HBM 메모리 초과.  

비유: “도서관 검색” – 모든 책 제목(QK) 메모리 저장 vs 스트리밍(Flash)으로 실시간 처리.  

### 4.2 Flash Attention의 원리  

Flash Attention: 소프트맥스 재구성, 타일링으로 메모리 절감. 알고리즘:  

1. Q/K/V를 블록(B×B) 분할. 예: B=64, N=4096 → 64 블록.  

2. 각 블록별 $S_{ij} = Q_i K_j^\top / \sqrt{d_k}$, 소프트맥스 $P_{ij} = exp(S_{ij} - m_i) / sum$, $O_{ij} = P_{ij} V_j. m_i$: 블록 최대값(오버플로 방지).  

3. 로그-스케일 누적: $m_new = max(m_old, m_block), l_new = l_old * exp(m_old - m_new) + sum exp(S - m_new)$.  

수식 (단순화):
$$
\text{Attn}_i = \sum_j \exp\left( \frac{Q_i K_j^\top}{\sqrt{d_k}} - m_i \right) V_j / Z_i
$$
메모리: O(N²) → O(N), 타일링으로 SRAM 재사용. 계산 과정: 블록 단위 곱셈-누적.  

미분: 순방향 값 캐싱 없이 재계산 → 메모리 ↓, 속도 2~4x ↑. 역전파: dP/dS * dS/dQ 등 블록별.  

### 4.3 장단점  

장점: 긴 시퀀스(64K) 가능, 속도·메모리 효율.  
단점: 커스텀 커널, PyTorch 2.0+ 권장.  
변형: FlashAttention-2(더 빠름).  
실전: Llama, GPT-4 훈련. N=4 예제: QK^T 전체 저장 vs Flash 2x2 블록 처리.  

---

## 5. Chinchilla Law: 스케일링  

### 5.1 스케일링의 중요성  

트랜스포머 성능은 파라미터(N)·데이터(D)·컴퓨트(C) 비례.  
C = 6 N D (FLOPs).  
비유: “공장 생산” – 기계(N)·원료(D)·전력(C) 균형.  

### 5.2 Chinchilla Law  

최적 비율: $C \approx 20 N$, 즉 $D ≈ C / (20 N) ≈ N / 20$?  
실제: D ≈ 20 N (토큰/파라미터).  
손실:

$$
\log L \approx -A N^{-\alpha} - B D^{-\beta} + C
$$

$\alpha \approx 0.21, \beta \approx 0.28$: Power Law.  
유도: 최적화 $∂/∂N =0 → N^{α+1} / D^{β+1} = const$.  
의미: N 과도 증가보다 D 중요.  
예: GPT-3(175B, D 부족) vs Chinchilla(70B, D=1.4T) – 성능 ↑.  

계산: $α=0.21, N=1e9, -A N^{-0.21} ≈ -log(성능)$.  
그래프: Perplexity vs log N – 직선 감소.  

### 5.3 Emergent Abilities  

스케일 ↑로 Zero-Shot, In-Context Learning 등장.  
의미: 임계점 후 능력 폭발.  

### 5.4 실전 팁  

데이터: 고품질 말뭉치, 20토큰/파라미터.  
컴퓨트: TPU/GPU 병렬.  
변형: Scaling Laws for 다른 태스크.  
비유: “근육 키우기” – 운동(N)만큼 식단(D) 필요.  

---

## 6. 숫자로 따라가는 미니 예제  

설정: $\theta = [0.5, -0.2]$, $g_1 = [0.1, 0.3]$, $\eta=0.01$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.  
t=1~10 반복, grad 고정 [0.1,0.3].  

Adam t=1:  
- m_1 = 0.9*0 + 0.1*[0.1,0.3] = [0.01, 0.03]  
- v_1 = 0.999*0 + 0.001*[0.01,0.09] = [1e-5, 9e-5]  
- \hat{m}_1 = [0.01,0.03]/(1-0.9) = [0.1, 0.3]  
- \hat{v}_1 ≈ [1e-4, 9e-4]  
- 업데이트: θ -= 0.01 * [0.1,0.3] / sqrt([1e-4,9e-4]) ≈ [0.4999, -0.2001] (Warmup η_t=0.0001 스케일).  

Warmup: t=1, T_warmup=10, T=100, η_t=1e-3 *1/10=1e-4.  

Flash: N=4, Q=[[1,0],[0,1],[1,1],[0,0]], K=V 동일.  
QK^T 전체 16원소 vs Flash 2x2  
블록: 첫 블록 Q[0:2]K[0:2], 누적.  

미분: ∂L/∂θ → Adam 업데이트, 체인룰로 전체 전파.  

---

## 7. 그래프와 시각화  

Loss Curve: Adam vs SGD, Adam은 진동 적고 빠른 하강.  
x축: 스텝, y축: Loss, SGD 지그재그 vs Adam 부드러움.  

LR Plot: Warmup(선형 0→1e-3), Cosine(1e-3→0). x: 스텝, y: η_t.  

Flash: 메모리 사용량 히스토그램 – 표준 O(N²) vs Flash O(N), 50% ↓.  

Chinchilla: N vs D vs Loss 3D 플롯 – 곡면 최소 at D=20N.  

---

## 8. 적분 관점

Adam: $\theta_{t+1} \approx \theta_t - \int \eta \nabla L \, dt$, 연속적 변화.  
미분: $dθ/dt = -η ∇L$, 모멘텀으로 평활화.  

Chinchilla: $L \approx \int -A N^{-\alpha} \, dN = -A/\alpha N^{-\alpha+1}$, 성능 누적.  
의미: 스케일링 시 지수적 이득.  

비유: “강물 흐름” –> 옵티마이저는 속도 조절, 스케일링은 물줄기 확장.  

---

## 9. 코드 확인 (NumPy 의사코드)  

```python
import numpy as np

def adam_update(theta, grad, m, v, t, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta -= eta * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v

def lr_schedule(t, T, eta, T_warmup):
    if t < T_warmup:
        return eta * t / T_warmup
    else:
        return eta / 2 * (1 + np.cos(np.pi * (t - T_warmup) / (T - T_warmup)))

# 테스트
theta = np.array([0.5, -0.2])
m, v = np.zeros_like(theta), np.zeros_like(theta)
for t in range(1, 11):
    grad = np.array([0.1, 0.3])  # 가정
    eta_t = lr_schedule(t, 100, 1e-3, 10)
    theta, m, v = adam_update(theta, grad, m, v, t, eta_t)
    print(f"Step {t}, LR: {eta_t:.6f}, Theta: {theta}")
```  

실행 결과:  
```bash
Step 1, LR: 0.000100, Theta: [ 0.4999 -0.2001]
Step 2, LR: 0.000200, Theta: [ 0.4997 -0.2003]
Step 3, LR: 0.000300, Theta: [ 0.4994 -0.2006]
Step 4, LR: 0.000400, Theta: [ 0.499 -0.201]
Step 5, LR: 0.000500, Theta: [ 0.4985 -0.2015]
Step 6, LR: 0.000600, Theta: [ 0.4979 -0.2021]
Step 7, LR: 0.000700, Theta: [ 0.4972 -0.2028]
Step 8, LR: 0.000800, Theta: [ 0.4964 -0.2036]
Step 9, LR: 0.000900, Theta: [ 0.4955 -0.2045]
Step 10, LR: 0.001000, Theta: [ 0.4945 -0.2055]
```

해설: adam_update: 기울기 누적(m, v), 바이어스 보정, 업데이트.  
lr_schedule: 선형 Warmup 후 Cosine Decay.  
실행: t=1~10 스텝별 θ 변화 확인, LR 증가에 따라 θ 점진적 업데이트.  

---

## 10. 자주 헷갈리는 포인트  

Adam vs AdamW: WD는 과적합 방지, AdamW 명시적.  
Warmup 없이: 초기 Loss 폭주, 그래프 확인.  
Flash Attention: PyTorch 2.0+ 자동 지원, 커스텀 필요 X.  
Chinchilla: 데이터 부족 시 성능 정체.  

---

## 11. 실무 감각  

하이퍼파라미터: η: 1e-4~1e-3,  
트랜스포머 1e-4.  
$T_{\text{warmup}}$: 1K~10K 스텝.  
배치: 512~4096, Gradient Accumulation 8~16.  

디버깅: Loss NaN: η ↓, Gradient Clipping(1.0).  
정체: η ↑, Warmup 연장.  

모니터: WandB로 Loss·Grad Norm·LR 로그.  
Hugging Face Trainer로 AdamW+Cosine 기본.  

---

## 이번 편 요약

* Adam/AdamW: $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$, 적응적 수렴이다.  
* LR Schedule: Warmup($\eta t/T$) + Cosine($\eta \cos$).  
* Flash Attention: O(N²) → O(N), 타일링으로 메모리 절감이다.  
* Chinchilla: $C \approx 20 N D$, 데이터·파라미터 균형이다.  
* 미분: 체인룰로 안정 학습이다.  
* 실전: η 튜닝, Flash로 긴 시퀀스.  

---

## 결론

이번 편에서는 Adam, LR Schedule, Flash Attention, Chinchilla Law를 수식·비유·코드로 보며, 트랜스포머의 효율적 학습과 스케일링을 설명했다.  
다음은 트랜스포머의 변형과 실전 응용을 다룬다.  