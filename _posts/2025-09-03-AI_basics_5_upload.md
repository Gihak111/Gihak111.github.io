---
layout: single
title:  "AI 입문 5편: 어텐션(Attention) — Scaled Dot-Product, Self-Attention, Multi-Head (수식·미분·예제·실전 가이드)"
categories: "AI"
tag: "Explanation"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈

이전 4편에서 정규화(BatchNorm, LayerNorm, GroupNorm)를 통해 학습 안정화를 배웠다.  
이번 5편에서는 트랜스포머(Transformer)의 핵심 엔진인 **어텐션(Attention)** 메커니즘을 본격적으로 파고든다.   
구체적으로 **Scaled Dot-Product Attention, Self-Attention, Multi-Head Attention**을 **수식·그래프·미분·장단점·실전 팁**으로 비교하고 설명하자.  
목표는 TR(트랜스포머)를 겨냥했지만, 그 바닥을 이루는 공통 원리를 확실히 잡는 것이다.  
초심자도 읽고 따라오면 **“아, 어텐션이 어떻게 시퀀스 정보를 연결하고 왜 강력한지”** 감각이 잡히게 하는 것이 이 글의 목표이다.  
이전 편들보다 더 깊고 자세한 설명을 추가하여, 개념이 처음인 분들도 단계적으로 이해할 수 있도록 하겠다.  
각 부분에서 왜 이 개념이 필요한지, 어떻게 작동하는지, 실전에서 주의할 점까지 세세히 풀어보자.  

---

## 0. 이번 편의 핵심  

* **어텐션 = "집중" 메커니즘**: 시퀀스 데이터에서 중요한 부분에만 초점을 맞춰 정보를 가중 평균한다.  
마치 사람이 문장을 읽을 때 핵심 단어에 주의를 기울이는 것처럼.  
* 각 변형의 **역할과 차이**를 알면 왜 트랜스포머가 RNN을 대체했는지 이해: RNN의 순차 처리 한계를 넘어 병렬 처리 가능.
* **Scaled Dot-Product Attention**: 기본 빌딩 블록, 쿼리-키-밸류로 유사도 계산.  
* **Self-Attention**: 입력 자신에게 적용, 시퀀스 내 관계 모델링.  
* **Multi-Head Attention**: 여러 관점에서 어텐션, 더 풍부한 표현.  
* **배치 처리와 행렬 미분**을 알면 실전 계산이 훨씬 깔끔해진다.  
* **적분 관점**은 가중 평균의 연속 누적을 준다.  
* 초보자 팁: 어텐션은 "소프트 검색 엔진"처럼 생각. 이 편에서 기본 개념부터 반복 설명하고, 수식은 단계별로 풀어 쓰며, 여러 예제를 통해 직관을 쌓아가자.  

이젠 진짜 직관이 중요하다  
니가 AI 없이 1달 걸릴꺼 AI끼면 2일이면 다한다.  
이게 시12발 어 그냥 AI가 싼 코드 보고 이해할 수 있으면 그걸 어떻게 수정하면 더 효율적인지 알 수 있는 능력만 있으면 코딩 몰라도 된다.  
물론 이 직관을 지르려면 코딩 잘하게 되긴 하지만,  
그래도, 암튼 그 직관 기르는게 지금 바이브 코딩, AI 와 함께 코딩하는 현 시점에서 제일 중요한 것 같다.  
또, 그 직관을 기르기 위해선 개념의 탄탄함이 근간이 되어야 하는 것 같다.  

---

## 1. 어텐션(Attention) — 왜 필요한가  

### 1.1 시퀀스 데이터의 도전 과제   

먼저, 왜 어텐션이 등장했는지부터 이해하자.  
딥러닝에서 시퀀스 데이터(문장, 시계열, DNA 등)를 다룰 때 기존 RNN/LSTM은 순차적으로 처리한다.  
예를 들어, "The cat sat on the mat" 문장에서 "sat"을 이해하려면 이전 단어들을 순서대로 기억해야 한다.  

* **문제 1: 장기 의존성(Long-range Dependency)**: 문장이 길어지면 초기 정보가 희미해진다(기울기 소실, 3편 활성화 함수 참조).  
* **문제 2: 병렬화 불가**: 순차 처리라 GPU 병렬 연산이 어렵다. 학습 속도 느림.  
* **문제 3: 고정 컨텍스트**: 모든 위치가 동일하게 취급되어 중요한 부분 강조 어려움.  

책을 읽을 때 모든 단어를 똑같이 중요하게 보지 않고, 키워드에 "주의(attention)"를 집중하는 그러 느낌.  
어텐션은 바로 이 아이디어를 모델에 적용한 것이다.  
각 위치에서 전체 시퀀스를 보고 중요한 부분에 가중치를 부여해 정보를 추출한다.  

* **해결**: 어텐션은 모든 위치를 동시에 고려(병렬 가능), 중요도에 따라 가중 평균. 트랜스포머의 기반이 되어 RNN을 넘어섰다.  

### 1.2 어텐션의 기본 아이디어  

어텐션은 **Query(Q), Key(K), Value(V)** 세 가지 벡터로 작동한다.  
- **Query**: "무엇을 찾을까?" 현재 위치의 질문 벡터.  
- **Key**: "이게 맞아?" 각 위치의 키 벡터, Q와 유사도 계산.  
- **Value**: "이 정보 가져갈게." 가중 평균할 실제 값.  

직관: 도서관에서 책 찾기. Q는 검색어, K는 책 제목, V는 책 내용. Q-K 유사도로 책 중요도 계산 후 V 가중 평균.  

번역에서 "it" (Q)이 "cat" (K)을 찾아 그 의미 (V)를 가져옴.  

### 1.3 미분 가능성과 학습  

어텐션은 소프트맥스(1편)로 소프트 가중치 만들기 때문에 미분 가능.  
역전파(2편)로 학습. 파라미터: Q/K/V를 만드는 선형 변환(Dense, 2편).  

* **기울기 전달**: 가중치가 소프트라 기울기 소실 덜함(3편 ReLU처럼).  
하드 어텐션(스파스)은 미분 어려움.  

### 1.4 그래프와 출력 범위의 역할  

* 그래프: 어텐션 맵(Attention Map) – N×N 행렬로 시각화, 위치 간 연결 강도.  
* 효과: 컨텍스트 인코딩, 번역에서 "it"이 가리키는 단어 자동 찾기.  

초보자: 어텐션이 없는 모델은 고정 창(window)으로 보지만, 어텐션은 동적으로 확대/축소.  

---

## 2. Scaled Dot-Product Attention — 기본 빌딩 블록

Scaled Dot-Product Attention은 어텐션의 기본 형태로, 트랜스포머의 핵심 연산이다.  

### 2.1 정의와 모양(shape)  

$$
\text{Attention}(Q, K, V) = \softmax\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

* 입력: Q ∈ ℝ^{B×N×d_k}, K ∈ ℝ^{B×M×d_k}, V ∈ ℝ^{B×M×d_v} (B: 배치, N: 쿼리 길이, M: 키/밸류 길이, d_k: 키 차원, d_v: 밸류 차원).  
* Q K^\top: 유사도 행렬 (B×N×M).  
* /√d_k: 스케일링, 내적 값 분산 ≈ d_k로 폭주 방지.  
* 소프트맥스: 마지막 축(M)별, 가중치 합=1.  
* 출력: B×N×d_v, 각 쿼리에 대한 가중 밸류 평균.  

**배치 처리**: 실전에서 B>1, 행렬 곱으로 병렬.  

초보자 직관: Q가 각 K와 내적(dot product)으로 유사도 측정.   
스케일링은 내적 값이 d_k에 비례 커지는 문제 해결(분산 ≈ d_k).  

예: d_k=4, Q=[1,0,0,0], K1=[1,0,0,0] 내적=1, K2=[0,1,0,0] 내적=0. 스케일 후 소프트맥스 [0.73, 0.27].  

그래프: 어텐션 맵은 히트맵, 대각선 강하면 자기 참조 강함.  

### 2.2 스케일링의 이유: 수학적 유도  

내적 Q·K의 기대값=0, 분산= d_k (정규화 가정). 소프트맥스 입력 분산 크면 출력 포화(기울기 소실).  
/√d_k로 분산=1 유지.  

초보자: "큰 숫자 방지" – 소프트맥스 안정.  

### 2.3 미분을 실제로 계산하자  

역전파: 체인룰(2편). L에 대한 출력 O=softmax(S) V, S = (Q K^T)/√d_k.  

* ∂L/∂V = (softmax(S))^T ∂L/∂O  
* ∂L/∂S = (∂L/∂O V^T) ⊙ softmax'(S)  (softmax 미분: diag(p) - p p^T)  
* ∂L/∂Q = (∂L/∂S) K / √d_k  
* ∂L/∂K = (∂L/∂S)^T Q / √d_k  

초보자: "가중치 높은 경로로 기울기 더 전달". 부록에 상세 유도.  

### 2.4 장단점과 대안  

* 장점: 병렬 계산(GPU 효율), 동적 컨텍스트, 간단 구현.  
* 단점: O(N M d_k) ≈ O(N² d) 복잡도(긴 시퀀스 부담), 스케일링 없으면 포화.  
* 대안: Efficient Attention (Linear, Reformer) – O(N) 근사.  

### 2.5 사용 사례  

* 트랜스포머 기본 단위.  
* Encoder-Decoder Attention: 번역에서 소스 K/V, 타겟 Q.  

---

## 3. Self-Attention — 시퀀스 내 관계 모델링  

Self-Attention은 입력 자신에게 어텐션을 적용해 내부 관계를 학습한다.  

### 3.1 정의와 모양(shape)  

$$
\text{SelfAttn}(X) = \text{Attention}(X W_Q, X W_K, X W_V)
$$

* X ∈ ℝ^{B×N×d_model}: 입력 임베딩.  
* W_Q, W_K ∈ ℝ^{d_model × d_k}, W_V ∈ ℝ^{d_model × d_v}: 학습 가능 Dense(2편).  
* 출력: B×N×d_v.  

초보자 직관: 입력이 자신에게 어텐션. 문장에서 각 단어가 다른 단어에 주의.  

예: "The animal didn't cross the street because it was too tired." "it"이 "animal" 참조 – self-attn이 자동 연결.  

그래프: 어텐션 맵으로 "it" 행에서 "animal" 열 밝음.  

### 3.2 미분을 실제로 계산하자  

W_Q/K/V 추가: ∂L/∂W_Q = (∂L/∂Q)^T X 등.  

전체: ∂L/∂X = ∂L/∂Q W_Q^T + ∂L/∂K W_K^T + ∂L/∂V W_V^T.  

초보자: "중요 위치 변화가 전체에 영향" – 관계 학습.  

### 3.3 장단점  

* 장점: 위치 독립(순차 아님), 글로벌 컨텍스트(첫 단어가 마지막 참조).  
* 단점: 순서 무시(포지셔널 필요), 과도 주의 분산(마스크 제어).  

### 3.4 사용 사례  

* 트랜스포머 인코더: 레이어 쌓아 특징 추출.  
* ViT: 이미지 패치 self-attn.  

초보자 팁: RNN처럼 타임스텝 없어, 모든 토큰 병렬 처리.  

---

## 4. Multi-Head Attention — 여러 관점의 병렬  

Multi-Head Attention은 여러 헤드로 다양성을 더한다.  

### 4.1 정의와 모양(shape)  

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

$$
\text{head}_i = \text{Attention}(Q W_{Q_i}, K W_{K_i}, V W_{V_i})
$$

* d_k = d_model / h (분할).  
* W_O ∈ ℝ^{h d_v × d_model}: 출력 선형.  
* 배치: B×h×N×(d_k).  

초보자 직관: 각 헤드 다른 "관점" 학습. 하나는 구문, 하나는 의미 등.  

비유: 그룹 토론 – 여러 사람이 각자 초점 맞춰 의견 모음.  

그래프: 헤드별 맵, 다양성 시각화.  

### 4.2 미분을 실제로 계산하자  

각 헤드 독립 + Concat/W_O: ∂L/∂head_i = (∂L/∂O W_O^T)[i*d_v:(i+1)*d_v].  

초보자: "다양한 경로로 기울기 분산/집중".  

### 4.3 장단점  

* 장점: 풍부 표현(단일 헤드보다 복잡 패턴), 병렬(GPU OK).  
* 단점: 파라미터 ↑(h배), 해석 어려움(헤드 역할 불명확).  

### 4.4 사용 사례  

* 트랜스포머 표준 (h=8).  
* GPT/BERT: MHA로 컨텍스트.  

초보자: h=1은 single-head, h 증가로 성능 ↑ but 오버헤드.  

---

## 5. 숫자로 따라가는 미니 예제 (손으로 풀어보기)  

**설정**: B=1, N=3 (단어: "cat", "sat", "mat"), d_model=4, d_k=4, h=1. 임베딩 X (간단 가정):  

$$
X = \begin{bmatrix} 
1 & 0 & 0 & 0 \\  % cat
0 & 1 & 0 & 0 \\  % sat
0 & 0 & 1 & 0   % mat
\end{bmatrix}
$$

W_Q = W_K = W_V = I (identity, 초기 단순).  

### 5.1 Scaled Dot-Product  

Q=K=V=X  

$$
Q K^\top = \begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}, \quad \frac{Q K^\top}{\sqrt{4}} = \begin{bmatrix}0.5&0&0\\0&0.5&0\\0&0&0.5\end{bmatrix}
$$

소프트맥스 (행별, e^{0.5}≈1.65, e^0=1):  

$$
\text{softmax}([0.5,0,0]) \approx [0.45, 0.275, 0.275]  % but 대각 중심으로 재계산
$$

실제: 각 행 대각 0.5, 나머지 0 → softmax ≈ [0.622, 0.189, 0.189] per row (approx).  

출력 ≈ softmax * V ≈ weighted X.  

### 5.2 Self-Attention (W 학습 가정)  

W_Q 등 랜덤: 예 W_Q = [[0.1,0.2,0.3,0.4]]^T 등, 상세 값 계산.  

### 5.3 Multi-Head (h=2, d_k=2)  

X 분할: head1 X[:,:2], head2 X[:,2:].  

헤드별 attn, concat, W_O 곱.  

### 5.4 역전파 시뮬  

L = ||O - target||^2, ∂L/∂O 계산 후 backward.  

초보자: 숫자 플러그인으로 "기울기 흐름" 느껴보자.  

---

## 6. 그래프 비교와 시각화 팁  

* Single vs Multi: single은 단일 맵, multi는 다중 관점.  
* 시각화: Matplotlib 히트맵, 헤드별 색상.  
* 초보자: BERTviz 툴로 실전 맵 보기 – "it" 집중 패턴.  

---

## 7. 마스킹과 포지셔널 인코딩 — 필수 보완  

### 7.1 마스킹  

디코더: causal mask (상삼각 -∞, 미래 차단).  

인코더: padding mask (패딩 -∞).  

초보자: "치팅 방지" – 훈련 시 미래 안 봄.  

수식: scores += mask * -1e9 before softmax.  

### 7.2 포지셔널 인코딩  

X + PE, PE_i^j = sin(i / 10000^{2j/d}) or cos.  

초보자: "순서 태그" – attn 위치 무시 보상.  

미분: PE 고정, 미분 안 함.  

---

## 8. 적분의 관점 — 가중 평균과 누적  

어텐션 ≈ ∫ p(x) v(x) dx (연속 가중 평균).  

초보자: "연속 누적 중요도" – 소프트 확률로 적분 유사.  

규제: attn entropy로 다양성.  

---  

## 9. 초기화와 어텐션의 궁합  

Xavier/He (3편), LayerNorm (4편)과 함께 안정.  

초보자: 초기 W 랜덤, 학습으로 패턴.  

---  

## 10. 어텐션을 코드로 확인해보기 (NumPy 의사코드)  

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * -1e9
    attn = softmax(scores)
    output = np.matmul(attn, V)
    return output, attn

# Self-Attention
def self_attention(X, d_k):
    W_Q = W_K = W_V = np.random.randn(X.shape[-1], d_k)  # 예시
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)
    return scaled_dot_product_attention(Q, K, V)

# Multi-Head
def multi_head_attention(X, h, d_model):
    d_k = d_model // h
    heads = []
    for i in range(h):
        head, _ = self_attention(X, d_k)
        heads.append(head)
    concat = np.concatenate(heads, axis=-1)
    W_O = np.random.randn(h * d_k, d_model)
    return np.matmul(concat, W_O)

# 테스트: X = np.eye(3)[:, :3]  # N=3, d=3
```

초보자: 코드 줄줄이 설명, 마스크 추가 예.  

---

## 11. 자주 헷갈리는 포인트 정리  

* Q/K/V 차원: d_k 맞춰야.  
* 스케일링 이유: 분산 제어.  
* Self vs Cross: Self 자신, Cross 다른 소스.  
* O(N²): 효율 팁 (Flash Attention).  

초보자 Q&A: "왜 dot-product? 코사인 유사도 변형."  

---

## 12. 어텐션 쌓기와 네트워크 동역학  

트랜스포머 레이어: MHA + FFN + LN + 잔차.  

초보자: "블록 쌓아 깊게" – 잔차로 기울기 유지(기울기 폭주/소실 방지).  

역전파: 체인룰로 안정 전달.  

---

## 13. 실무 감각: 선택과 튜닝  

* 선택: TR → MHA, 효율 → Sparse/Linear.  
* 튜닝: h=8~16, d_k=64.  
* 디버깅: Attn 맵 검사 – 과집중 시 dropout.  
* 트렌드: QK Norm, RoPE (회전 위치).  

초보자: Hugging Face Transformers로 실험.  

---

## 14. 어텐션과 TR(트랜스포머)의 연결고리 미리보기  

* TR: 인코더/디코더 스택, attn 핵심.  
* 왜 강력? 병렬 + 글로벌 + 스케일.  

---

## 이번 편 요약(한 장으로)  

* **어텐션 = 동적 가중 평균**: Q-K-V로 중요도.  
* **Scaled Dot-Product**: 기본, 스케일 내적 소프트맥스 V.  
* **Self-Attention**: X 자신, 관계 모델.  
* **Multi-Head**: h 병렬, 다양 표현.  
* **미분**: 체인룰, 학습 W.  
* **적분**: 누적 중요.  
* 실전: 마스크·PE·LN과 함께, O(N²) 주의.  

---

## 결론

이번 편에서는 **어텐션**을 기본부터 multi-head까지 **수식·예제·코드**를 알아보았다.  
다음 6편에서는 Feed-Forward Network(FFN)과 잔차 연결(Residual)을 다룬다.  
TR 레이어 구성 요소를 **수식·직관·최적화**로 설명하자.  

---

## 부록: Scaled Dot-Product 미분 대충 유도  

O = P V, P = softmax(S), S = (Q K^T)/√d_k.  

∂O/∂S = (∂L/∂O V^T) ⊙ P'(S), P' = diag(P) - P P^T.  

다들 GPT 같은 AI 어떻게 활용하는지 나는 잘 모른다.  
난 공부하는데 주로 사용하는데 진짜 쓸모있다.  