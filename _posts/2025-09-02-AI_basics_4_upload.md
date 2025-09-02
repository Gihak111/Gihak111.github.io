---
layout: single
title: "AI 입문 4편 확장 통합판: 정규화(Normalization) – BatchNorm, LayerNorm, GroupNorm"
categories: "AI"
tag: "Normalization, BatchNorm, LayerNorm, GroupNorm"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈

이전 편에서 활성화 함수(Sigmoid/Tanh/ReLU/GELU)를 통해 비선형성을 정리하였다.  
이번 4편에서는 신경망 학습을 안정화하는 정규화(Normalization)를 본격적으로 파고든다.  
구체적으로 BatchNorm, LayerNorm, GroupNorm을 비교한다.  

---

# 서문

- 각 정규화 수식의 항 하나하나가 왜 존재하는지 이해한다.  
- BatchNorm / LayerNorm / GroupNorm를 언제·왜·어떻게 쓰는지 결정할 수 있다.  
- 손으로 따라할 수 있는 수치 예제와 간단한 NumPy / PyTorch 코드(순전파 포함)를 통해 직접 확인할 수 있다.  
- 수식 유도·역전파 도출·디버깅/실험 설계까지 실제 모델링에 바로 쓸 수 있는 단계적 가이드를 제공한다.  

---

# 이번 편의 핵심

정규화는 입력 분포 안정화이다.  
각 방법의 대상 축·학습 가능성을 알면 왜 특정 모델에서 특정 정규화가 선택되는지 이해한다.  

- 정규화 = 입력 분포 안정화: 층마다 데이터 스케일/분포가 변하는 문제를 해결한다.  
- BatchNorm: CNN의 고전, 배치 통계 사용 but 배치 크기 의존.  
- LayerNorm: Transformer의 기본, 시퀀스/토큰 독립.  
- GroupNorm: 중간 지점, 배치 크기 무관.  
- 선택 기준: 아키텍처·배치 크기·태스크에 따라 다르다. 실험과 직관으로 익힌다.  
- 적분 관점에서 평균·분산 누적을 연결한다.  

정규화의 핵심 질문은 "어떤 축에서 평균과 분산을 구할 것인가?"이다.  
이 한 선택이 모델 성능과 안정성에 큰 영향을 준다.  
간단한 규칙: CNN + large B → BN, CNN + small B → GN, Transformer → LN. 항상 eps, momentum, gamma/beta 초기화와 추론 모드 설정을 신경 쓴다.  

최근, 수학공부 열심히 하냐는 말에 좀 많이 긁혀서 수식 증명까지 다 때려박다보니까 글이 엄청나게 길어졌다.  
이번편만 이럴 예정이니 눈 감고 넘어가 주면 좋겠다.  
잠고로 gpt 쓴 글이다  
공부할때 많이 도움된다  

---

# 목차

1. 정규화의 목표와 공통 형태  
2. 텐서 축과 어떤 축으로 정규화할지 결정하는 법  
3. Batch Normalization (BN)  
   - 순전파 수식  
   - running mean/var 업데이트 수식  
   - 역전파 상세 유도  
   - 수치 예제(단계별 계산)  
   - 소배치 문제와 대안(SyncBN, BatchRenorm 등)  
4. Layer Normalization (LN)  
   - 수식·역전파  
   - Transformer에서의 Pre-LN vs Post-LN   
   - RMSNorm, ScaleNorm 등 변형  
5. Group Normalization (GN)  
   - 동작 원리와 수식  
   - 그룹 수 G 선택 가이드  
6. 정규화 축(axes)별 통계적 성질 비교    
   - 통계량의 분산(표본 평균의 분산) 분석  
   - 왜 BN은 소배치에 약한가(수식으로 설명)  
7. 수치 예제 비교 (BN vs LN vs GN)  
8. 손으로 해보는 미니 예제(정확 계산)  
9. 실전 팁·하이퍼파라미터·디버깅 체크리스트  
10. 실험 설계(비교 실험 의사코드)  
11. PyTorch 예제 — 훈련/추론 차이까지 확인  
12. 간단한 NumPy 확인 코드(포워드만)  
13. 자주 하는 실수 & Q&A  
14. 정규화 쌓기와 네트워크 동역학  
15. 실무 감각: 선택과 튜닝  
16. 확장: 변형들(간단히)  
17. 고급 주제: BatchRenorm, Weight Standardization, EvoNorm  
18. 그래프 비교와 시각화 팁  
19. 기울기 문제와 해결 연계  
20. 초기화와 정규화의 궁합  
21. 정규화와 Transformer의 연결고리 미리보기  
22. 결론 및 추천  
23. 부록: BN 역전파 유도 자세한 전개(수학적으로)  
24. 부록: 빠른 참고 표  

---

# 1. 정규화의 목표와 공통 형태 (단계별 사고)  

정규화의 목표는 학습 중 각 층으로 들어오는 입력의 분포(평균·분산)를 안정화시켜 학습을 빠르고 안정적으로 만드는 것이다. 이는 다음 두 가지 효과를 야기한다.  

1. 학습률을 더 크게 잡을 수 있다 — 입력 스케일의 변화에 둔감해져서 옵티마 탐색이 빨라진다.  
2. 훈련 안정성/일반화에 도움 — 일부 경우에는 규제 효과(regularization)처럼 동작한다.  

내부 공변량 이동(Internal Covariate Shift): 신경망 학습 중 가중치 업데이트로 각 층 입력 분포가 변한다(ICS).  

- 문제: 이전 층 출력이 다음 층 입력 → 분포 변화로 학습 불안정, 기울기 소실/폭주 유발.  
- 예: 활성화 입력이 크게 변하면 Sigmoid 소실 심화.  
- 해결: 각 층 입력을 평균 0, 분산 1로 정규화 → 안정적 학습.  

정규화는 미분 가능해야 역전파 OK이다. 일부(γ, β)는 학습되어 표현력 유지한다.  

- 기울기 전달: 정규화 후 스케일/시프트로 원래 분포 일부 복원.  
- 효과: 학습 속도 ↑, 초기화 덜 민감, 규제 효과(노이즈 추가).  

공통 형태(모든 정규화의 본질):  

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad y = \gamma\hat{x} + \beta
$$

- $\mu,\sigma^2$는 어떤 축(axes)에서 계산하느냐에 따라 달라진다.  
- $\gamma,\beta$는 학습 가능한 스케일·바이어스(affine) 파라미터로, 정규화된 표현의 표현력을 회복한다.  

항목별 단계적 의미:  

1. $x$ — 정규화 대상인 입력(스칼라 또는 텐서 원소).  
2. $\mu$ — 평균(어느 축에서 계산하는지가 핵심). 배치/샘플/그룹 등 선택 가능.  
3. $\sigma^2$ — 분산(같은 축에서 계산). 흩어짐을 측정.  
4. $\varepsilon$ — 작은 상수(예: $10^{-5}\sim10^{-6}$). 수치 안정성 확보(분모 0 방지).  
5. $\hat{x}$ — 표준화된 값(평균 0, 분산 1에 가깝게 조정).  
6. $\gamma,\ \beta$ — 학습 가능한 스케일(확대)과 시프트(이동). 정규화 이후에도 모델이 필요한 분포로 복원할 수 있게 해준다.  

정리: 정규화 설계는 어떤 축에서 평균·분산을 구할 것인가의 문제이다.  

---

# 2. 텐서 형상과 정규화 축 결정  

대표 텐서 형상:  

- CNN: $x\in\mathbb{R}^{B\times C\times H\times W}$ (Batch, Channel, Height, Width)  
- Transformer / MLP: $x\in\mathbb{R}^{B\times T\times d}$ (Batch, Time/Token, Feature)  

각 정규화는 대체로 다음 축에서 평균/분산을 계산한다:  

- BatchNorm (2D): 채널별로 배치+공간 축 $B,H,W$에서 평균/분산을 계산 → 채널마다 하나의 정규화 스칼라 생성 (CNN에 적합)  
- LayerNorm: 각 샘플(토큰) 내의 feature 축에서 계산(예: d) → 배치 크기 무관(시퀀스에 적합)  
- GroupNorm: 채널을 G개 그룹으로 나누고 그룹별로 계산 → 배치 무관, 소배치 CNN에서 적절  

결정 기준(빠른 체크리스트):  

1. 입력이 시퀀스(Transformer)? → LayerNorm  
2. CNN이고 충분한 배치 크기 가능? → BatchNorm  
3. CNN이고 소배치/온라인 환경? → GroupNorm  

---

# 3. Batch Normalization (BN) — 심층 분석  

## 3.1 목적 한 문장  

배치 단위로 통계를 계산해 채널(feature)별로 입력 스케일을 맞추어 학습을 가속하고 안정화한다. 특히 CNN에서 강력한 효과를 보인다.  

## 3.2 순전파(Forward)  

채널 $c$에 대해, 입력 $x_{b,c,h,w}$에 대한 채널별 배치 통계:  

$$
\mu_{B,c} = \frac{1}{m} \sum_{b,h,w} x_{b,c,h,w}, \qquad
\sigma_{B,c}^2 = \frac{1}{m} \sum_{b,h,w} (x_{b,c,h,w} - \mu_{B,c})^2
$$

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_{B,c}}{\sqrt{\sigma_{B,c}^2 + \varepsilon}},\qquad
y_{b,c,h,w} = \gamma_c \hat{x}_{b,c,h,w} + \beta_c
$$

- m = B × H × W: 정규화에 참여한 원소 수(배치와 공간 차원 포함).  
- 채널별(c)로 통계를 내므로 같은 채널의 다른 샘플/공간을 섞어 평균을 구한다.  

수식 해석(단계):  

1. 동일 채널의 여러 샘플(b)과 모든 공간(h,w)을 모아 평균을 계산 → 그 채널의 전반적 레벨(스케일)을 추정.  
2. 분산은 값들의 흩어짐(스케일 불확실성)을 측정.  
3. 표준편차(s)를 사용(분산의 제곱근)하여 스케일이 동일 단위로 맞춰짐.  
4. 작은 $\varepsilon$로 수치 안정성 확보.  
5. $\gamma_c,\beta_c$는 채널별 보정 파라미터—정규화로 잃은 표현력을 복원.  

## 3.3 Running statistics (훈련 ↔ 추론 연결)  

훈련 시: 배치 통계를 사용하고, 동시에 지수이동평균(EMA)으로 running mean/var를 업데이트한다:  

$$
\text{running\_mean}_c \leftarrow (1-\alpha) \cdot \text{running\_mean}_c + \alpha \cdot \mu_{B,c}
$$

$$
\text{running\_var}_c \leftarrow (1-\alpha) \cdot \text{running\_var}_c + \alpha \cdot \sigma_{B,c}^2
$$

(프레임워크마다 momentum 또는 alpha 표기 차이 있음. 예: PyTorch의 기본 momentum=0.1는 위 식의 $\alpha$와 유사한 개념.)  

추론 시: 배치 통계가 없으므로 running mean/var를 사용해서 정규화한다.  

해석: 훈련 시의 배치 노이즈는 규제 효과를 제공하지만, 추론 시에는 안정적인 통계가 필요하므로 running stat을 사용한다. 이것이 훈련/추론 모드가 다른 이유이다.  

## 3.4 역전파(Backward) — 단계별 유도  

목표: 손실 $L$에 대해 $\frac{\partial L}{\partial x_{i}}$를 구하는 것(인덱스 $i$는 정규화에 포함된 모든 scalar 원소를 1차원으로 펼친 인덱스라고 생각).  

간단화: 1차원 배열 $x = (x_1,\dots,x_N)$에 대해($N=M$과 동일),  

정규화 변수:  

$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i,\quad
\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2,\quad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},\quad y_i = \gamma\hat{x}_i + \beta
$$

이제 체인룰로 미분한다.  

### (A) $\frac{\partial L}{\partial \gamma},\ \frac{\partial L}{\partial \beta}$

$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^N \frac{\partial L}{\partial y_i} \cdot \hat{x}_i,\qquad
\frac{\partial L}{\partial \beta} = \sum_{i=1}^N \frac{\partial L}{\partial y_i}
$$

(이건 직관적으로 명확하다 — $y_i$가 $\gamma,\beta$에 선형이므로.)  

### (B) $\frac{\partial L}{\partial x_i}$ 유도 (핵심)  

체인룰을 세 번 연쇄해서 적용한다:  

$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^N \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial x_i}
$$

$\frac{\partial y_j}{\partial \hat{x}_j} = \gamma$ 이므로,  

$$
\frac{\partial L}{\partial x_i} = \gamma \sum_{j=1}^N \frac{\partial L}{\partial y_j} \cdot \frac{\partial \hat{x}_j}{\partial x_i}
$$

다음으로 $\hat{x}_j = (x_j - \mu)/s$ (여기서 $s = \sqrt{\sigma^2 + \varepsilon}$).

$$
\frac{\partial \hat{x}_j}{\partial x_i} = \frac{1}{s}(\delta_{ij} - \frac{1}{N}) - \frac{x_j - \mu}{s^3} \cdot \frac{1}{N}(x_i - \mu)
$$

이 식은 체인룰과 $\mu,\sigma^2$가 $x$에 의존하는 것을 고려해서 유도된 표준 식이다.  

정리하면(좀 더 보기 쉬운 형태):  

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{s} \Bigg( \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{N} \sum_{j} \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \cdot \frac{1}{N} \sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \Bigg)
$$

여기서 $\frac{\partial L}{\partial \hat{x}_j} = \frac{\partial L}{\partial y_j} \cdot \gamma$ 라고 두고 시작하면 최종식은 다음과 같이 자주 쓰인다.  

BN의 역전파 요지: 입력 각각에 대한 그래디언트는 그 입력 자신의 기여와 정규화된 평균/분산에 의한 보정항(모든 샘플에 걸친 합으로 표현됨)을 포함한다.  

### (C) 유도(중간 단계 상세)  

1. $s = \sqrt{\sigma^2 + \varepsilon}$ 이므로 $\partial s / \partial x_i = \frac{1}{2s} \cdot 2\cdot \frac{1}{N} (x_i - \mu) - $ (아래의 $\mu$ 의존성을 포함)  

2. $\mu$에 대한 기울기: $\partial\mu/\partial x_i = 1/N$.  

3. $\sigma^2$에 대한 기울기: $\partial\sigma^2/\partial x_i = \frac{2}{N}(x_i - \mu) - \frac{2}{N} \sum_k (x_k - \mu)\cdot \partial\mu/\partial x_i = \frac{2}{N}(x_i - \mu)$ (두 항의 상쇄로 인해 최종적으로 이렇게 간단해지는 표기를 쓴다).  

(실제 유도는 조금 장황하지만, 프레임워크가 정확히 구현하므로 결과 공식만 외워도 무방하다.)  

정규화는 평균·분산을 계산하는 과정 때문에 역전파에서 추가 항(평균/분산에 대한 미분)이 들어간다. 체인룰을 따라 아래 항들이 포함된다:  

- dL/dx는 직접 항(표준화된 항에 의한 것)과 평균/분산의 변화가 x에 미치는 영향(모든 요소에 걸쳐 전파되는 상호작용)을 합쳐 계산한다.  
- 실제 유도는 다소 장대하지만, 프레임워크(PyTorch, TensorFlow)는 이를 자동으로 처리하므로 실제 사용자는 forward/모듈 사용에 집중하면 된다.  

## 3.5 수치 예제: BN 역전파 계산을 단계별로 해보자  

설정: 1차원(채널=1), $N=2$ (작은 배치). 입력 $x = [1.0, 3.0]$. $\gamma=2.0,\ \beta=0$. eps 아주 작다고 가정($\varepsilon=0$으로 근사). 손실 함수: $L = \frac{1}{2}(y_1^2 + y_2^2)$ (MSE with zero target).  

단계 1: 평균·분산 계산  

- $\mu = (1+3)/2 = 2.0$.
- $\sigma^2 = ((1-2)^2 + (3-2)^2)/2 = (1 + 1)/2 = 1.0$.
- $s = \sqrt{\sigma^2 + \varepsilon} = 1.0$.

단계 2: 정규화, 출력 계산  

- $\hat{x} = [(1-2)/1, (3-2)/1] = [-1, +1]$.
- $y = \gamma \hat{x} + \beta = 2 \cdot [-1, +1] = [-2, +2]$.

단계 3: 손실과 $\partial L/\partial y$

- $L = 0.5((-2)^2 + 2^2) = 0.5(4+4) = 4.0$
- $\frac{\partial L}{\partial y} = [y_1, y_2] = [-2, +2]$ (왜냐하면 $d(0.5 y^2)/dy = y$).

단계 4: $\partial L/\partial \gamma, \partial L/\partial \beta$

- $\partial L/\partial \gamma = \sum_i (\partial L/\partial y_i) \hat{x_i} = (-2)(-1) + (2)(1) = 2 + 2 = 4.$
- $\partial L/\partial \beta = \sum_i (\partial L/\partial y_i) = -2 + 2 = 0.$

단계 5: $\partial L/\partial x$ — 공식 사용:  

공식:  

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{s} \Big( \frac{\partial L}{\partial y_i} - \frac{1}{N} \sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \cdot \frac{1}{N} \sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \Big)
$$

- $\sum_j \frac{\partial L}{\partial y_j} = -2 + 2 = 0$.
- $\sum_j \frac{\partial L}{\partial y_j} \hat{x}_j = (-2)(-1) + 2(1) = 2 + 2 = 4$.

따라서  

- $\frac{\partial L}{\partial x_1} = \frac{2}{1} \Big( -2 - 0 - (-1) \cdot \frac{4}{2} \Big) = 2 \Big( -2 + 2 \Big) = 0.$
- $\frac{\partial L}{\partial x_2} = \frac{2}{1} \Big( 2 - 0 - (1) \cdot \frac{4}{2} \Big) = 2 \Big( 2 - 2 \Big) = 0.$

결과: $\partial L / \partial x = [0,0]$.

해석: 이 간단한 예에서는 대칭성 때문에 입력에 대한 그래디언트가 0이 된다(모든 변화가 $\gamma$ 쪽으로 흘러감). 작은 예제로 역전파 공식을 직접 계산해 보면 수식이 틀리지 않았음을 확인할 수 있다.  

이 예제는 역전파 공식을 수치적으로 검증하는 데 좋은 연습이다. 실제 네트워크에서는 그렇게 자주 0이 되지는 않는다.  

## 3.6 장단점 요약  

- 장점: 강력한 규제 효과(배치 노이즈), 학습 가속, 초기화 민감도 감소.  
- 단점: 배치 크기 의존 (작은 배치에서는 통계 추정 불안정), 추론 시 running stat 필요로 인한 훈련/추론 불일치 문제, 시퀀스 모델에 부적합.  

BN 특이: 훈련/추론 모드. running avg로 추론 안정.  

## 3.7 실전 팁  

- CNN + 배치 크기 충분(B ≥ 32 권장) → BN 우선 고려.  
- 소배치 환경이면 GN(또는 LN)로 대체 권장.  
- SyncBN: multi-GPU 환경에서 배치 통계를 동기화하여 안정화.  

## 3.8 소배치 문제와 대안(SyncBN, BatchRenorm 등)  

배치 통계 의존 → 소배치에 취약. 배치 크기 작으면 통계 불안정.  

- SyncBN: 멀티 GPU 환경에서 각 GPU의 배치 통계를 동기화하여 effective batch size를 키운다. 대규모 분산 학습에서 BN의 효용을 살리려면 유용.  

## 3.9 사용 사례  

- CNN(ResNet 등) 히든 레이어.  
- 초기 딥러닝 표준.  

---

# 4. Layer Normalization (LN) — 샘플 내부 정규화 (Step-by-step)  

## 4.1 목적 한 문장  

각 샘플의 모든 피처(feature) 차원에서 평균·분산을 계산해, 배치 크기와 상관없이 입력을 안정화한다. Transformer에서 표준이다.  

## 4.2 수식과 항 해설  

한 샘플의 d차원 벡터에 대해:  

$$
\mu_L = \frac{1}{d} \sum_{j=1}^d x_j, \qquad \sigma_L^2 = \frac{1}{d} \sum_{j=1}^d (x_j - \mu_L)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu_L}{\sqrt{\sigma_L^2 + \varepsilon}}, \qquad y_j = \gamma_j \hat{x}_j + \beta_j
$$

- 특징: 샘플 단위 통계만 사용하므로 배치 크기와 무관.  

## 4.3 역전파 요약  

LN의 역전파는 BN의 1차원 버전과 매우 유사하다.  
평균과 분산을 feature 축에서 취한다는 점만 다르다.  
결과 공식은 BN에서 축만 바뀐 것과 형태상 동일하다.  

## 4.4 장단점 요약

- 장점: 배치 무관 → 소규모 배치/온라인 학습에서도 안정적. Transformer 등 시퀀스 모델에 적합.  
- 단점: BN이 제공하는 규제(배치 노이즈) 효과가 부족할 수 있음. 피처 차원이 클 경우 계산 비용 고려.  

## 4.5 Transformer에서의 Pre-LN vs Post-LN

- Post-LN(초기형): 서브레이어(예: Self-Attention) 결과에 LN을 적용하고 잔차 연결 → 초기 모델(Transformer)에서 사용.  
- Pre-LN(현대형 권장): 서브레이어 입력에 LN을 적용 → 잔차 연결로 들어가는 경로가 더 '직선'이 되어 깊이 증가 시에도 기울기 소실 문제가 완화.  

간단 규칙: 딥 트랜스포머엔 Pre-LN를 권장.  

## 4.6 RMSNorm, ScaleNorm 등 변형

- RMSNorm: 평균 중심 연산을 생략하고 제곱평균(RMS)만 사용.  
- 이론적 장점: 계산량/메모리 절감, 일부 경우에서 성능 비슷.  
- ScaleNorm: Transformer 계열에서 계산비/안정성 개선을 노리는 변형.  

## 4.7 언제 사용할까

- Transformer (Self-Attention 후), RNN에서 배치 독립성이 필요할 때.  
- 소배치/온라인 환경.  

## 4.8 사용 사례

- Transformer (어텐션/MLP 후).  
- NLP/시퀀스 모델 표준.  
- 현대: ViT 등 이미지에도.  

---

# 5. Group Normalization (GN) — 그룹 단위 정규화 (Step-by-step)  

## 5.1 목적 한 문장  

채널 축을 G개의 그룹으로 나눈 뒤, 그룹별로 평균·분산을 계산해 배치 비의존적이면서 채널 간 상관관계를 일부 유지한다. 소배치 CNN에 적합하다.

## 5.2 수식과 항 해설 (한 샘플 내 그룹 g에 대해)

$$
\mu_g = \frac{1}{m} \sum_{i\in g} x_i, \qquad \sigma_g^2 = \frac{1}{m} \sum_{i\in g} (x_i - \mu_g)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_g}{\sqrt{\sigma_g^2 + \varepsilon}}, \qquad y_i = \gamma_i \hat{x}_i + \beta_i
$$

- m = (C/G) × H × W: 그룹 내 원소 수.  
- G: 그룹 수(하이퍼파라미터). 경험적으로 G=32가 자주 추천되지만, C < 32인 경우 G=C로 설정.  

## 5.3 직관  

- G=1이면 전체 채널을 하나의 그룹으로 묶어 LayerNorm과 동일하게 동작(샘플 내 전체 채널 정규화).  
- G=C이면 채널별 정규화(인스턴스·채널 단위).  

## 5.4 미분

그룹 내 체인룰.  

## 5.5 장단점

- 장점: 배치 무관, BN의 장점 일부(채널 결합 통계) 보존, 소배치에서 안정.  
- 단점: G 값 튜닝 필요, 그룹 경계에서 정보 손실 가능성.  

## 5.6 왜 소배치에서 GN이 좋은가?

GN은 배치 차원 B를 통계 계산에서 제외한다.  
따라서 배치 크기 변화에 의해 통계의 분산이 커지는 문제가 사라진다.  
단점: BN이 주는 배치레벨의 암묵적 규제 이득은 얻기 어렵다.  

## 5.7 하이퍼파라미터 G 선택 가이드  

- 권장 경험칙: G=32 (단, C가 32보다 작다면 G=C로 설정하여 채널별 정규화처럼 동작).  
- 그룹이 너무 작으면(예: G=C) → 채널별 정규화(LN/instance-norm과 비슷), 너무 크면(G=1) → 채널 전체를 묶는 LN과 비슷.  

## 5.8 사용 사례  

- 소규모 배치 CNN.  
- GN as BN 대체.  

---

# 6. 통계적 관점: 표본 평균의 분산 분석(왜 BN에 배치크기가 중요한가)  

정규화에서 가장 핵심적인 통계는 표본 평균 $\hat{\mu}$와 표본 분산 $\hat{\sigma}^2$의 추정 분산(variance of estimator)이다.  

단순화: 독립 동일 분포(각 원소 분산 $\sigma_x^2$) 가정 아래, 표본 평균의 분산은:  

$$
\mathrm{Var}(\hat{\mu}) = \frac{\sigma_x^2}{N}
$$

여기서 N은 평균을 계산하는 원소 수(예: BN에서는 N=BHW)이다. 따라서 N이 작으면 평균 추정의 분산이 커진다 → 정규화된 값의 노이즈가 커져 학습이 불안정해짐.  

결론: BN은 평균/분산을 배치+공간에서 추정하므로 N이 충분히 클 때 안정적이다. 소배치(또는 작은 H,W)에서는 N이 작아지고 통계 노이즈가 커진다.  

정규화 축(axes)별 통계적 성질 비교:  

- 통계량의 분산(표본 평균의 분산) 분석.  
- 왜 BN은 소배치에 약한가(수식으로 설명).  

---

# 7. 수치 예제 비교 (BN vs LN vs GN) — 같은 입력으로 따라가기  

입력 예시 (단순화): B=2, C=2, H=1, W=1, 값 배열:  

$$
X = \begin{bmatrix}
[1,\ 2] \\
[3,\ 4]
\end{bmatrix}
$$

(의미: 두 샘플, 각 샘플에 채널 2개)  

## 7.1 BatchNorm (채널 기준)  

- 채널0: 값 [1,3] → μ=2, σ^2=((1-2)^2+(3-2)^2)/2 = (1+1)/2 = 1 → σ=1 → x̂=[-1, +1]  
- 채널1: 값 [2,4] → μ=3, σ^2=1 → x̂=[-1, +1]  

결과(샘플별):  

$$
[[-1, -1], [ +1, +1]]
$$

μ_B = [2,3], σ_B^2 ≈ [2,2], ε=0.

$\hat x ≈ [[[-0.707, -0.707]], [[0.707, 0.707]]] (γ=1,β=0 가정)$

BN은 배치 혼합.  

## 7.2 LayerNorm (샘플 기준)  

- 샘플1 [1,2]: μ=1.5, σ^2=((1-1.5)^2+(2-1.5)^2)/2 = (0.25+0.25)/2 = 0.25 → σ=0.5 → x̂=[-1, +1]  
- 샘플2 [3,4]: μ=3.5, σ^2=0.25 → σ=0.5 → x̂=[-1, +1]  

결과(샘플별):  

$$
[[-1, +1], [-1, +1]]
$$

샘플1: μ=1.5, σ^2=0.5 → \hat x ≈ [[-0.707, 0.707]]  

샘플2: μ=3.5, σ^2=0.5 → \hat x ≈ [[-0.707, 0.707]]  

LN은 샘플 내 균형.  

## 7.3 GroupNorm  

- G=1이면 샘플 내 전체 채널을 하나 그룹으로 묶으므로 LN과 동일: 결과는 LN과 같음.  
- G=2이면 채널별 독립 정규화(각 채널에 대해 단독 평균/분산 계산) → BN과 유사하지만 배치 차원은 포함하지 않음.  
- G=2: 각 채널 독립 정규화.  

결론: 같은 원소라도 "어떤 축에서 평균·분산을 취하느냐"에 따라 결과가 달라진다.  

## 7.4 역전파 시뮬 (간단 dL/dy=1)  

dL/dx = 체인룰로 정규화 역산(평균/분산 영향).  

직관: BN은 배치 분포 그래프 변동, LN은 샘플별 히스토그램 안정, GN은 그룹 크기 따라.  

시각화: 분포 플롯 전/후.  

직관: 깊은 층에서 분포 유지 → 안정.  

---

# 8. 손으로 해보는 예제  

### 예제 A — BatchNorm 간단 계산  

설정: B=2, C=1, H=W=1, x = [1.0, 3.0], γ=1, β=0, ε=0  

1. μ = (1+3)/2 = 2.0  
2. σ^2 = ((1-2)^2 + (3-2)^2)/2 = (1 + 1)/2 = 1.0  
3. σ = 1.0  
4. x̂ = [(1-2)/1, (3-2)/1] = [-1, +1]  
5. y = γ·x̂ + β = [-1, +1]  

### 예제 B — LayerNorm 간단 계산  

설정: x = [1, 3, 5], d=3, γ=1, β=0  

1. μ = (1+3+5)/3 = 3.0  
2. σ^2 = ((1-3)^2 + (3-3)^2 + (5-3)^2)/3 = (4+0+4)/3 = 8/3 ≈ 2.6666667  
3. σ = √(8/3) ≈ 1.6329931619  
4. x̂ ≈ [ (1-3)/1.63299, 0, (5-3)/1.63299 ] ≈ [-1.224744871, 0, +1.224744871]  

(숫자 하나하나 직접 계산해보면 정규화가 어떻게 작동하는지 직관이 생긴다.)  

---

# 9. 실전 팁·하이퍼파라미터·디버깅 체크리스트  

1. 아키텍처 확인: CNN → BN 우선, Transformer → LN.  
2. 배치 크기 점검: BN 사용 시 B 충분한가? 아니면 GN으로 전환.  
3. 추론 모드 확인: PyTorch 등에서 model.eval() 호출 여부 확인(BN 관련 문제 대다수는 여기에 있음).  
4. eps와 momentum 기본값: eps=1e-5 ~ 1e-3, momentum(=alpha)=0.1(프레임워크마다 표기 차이 있음).  
5. gamma/beta 초기화: γ=1, β=0. 필요에 따라 γ를 0으로 초기화해 잔차 블록 안정화 기법 사용 가능.  

- eps: 일반적으로 $1\times10^{-5}$ ~ $1\times10^{-3}$. 너무 작으면 숫자 불안정(NaN), 너무 크면 정규화가 덜 효과적.  
- momentum(running stats): 보통 0.1~0.01. 데이터가 매우 비정상적으로 바뀌는 경우 더 크게.  
- gamma 초기화: 1.0 권장. 일부 경우(잔차 블록 끝 등) 0으로 초기화하면 잔차 경로 가교 역할을 하여 안정화(예: FixUp 대안).  
- weight decay와 gamma: gamma(스케일 파라미터)에 L2 규제를 적용하면 모델 성능에 악영향을 줄 수 있음. 보통 optimizer 설정에서 gamma/beta를 weight decay에서 제외하는 것이 권장된다.  
- affine=False: 드물게 정규화 자체만 원할 때 사용.  

디버깅 체크리스트(실전 문제 해결 순서):  

1. model.eval()를 잊지 않았나? — 추론 시 BN이 train 모드면 성능 급락.  
2. 러닝 통계가 수렴했나? — 작은 training run으로 running_mean/std의 값이 안정적인지 확인.  
3. 배치 크기 너무 작지 않나? — BN 사용시 1~2 정도의 작은 배치는 매우 불안정.  
4. GN/LN로 바꿔보자 — 소배치 또는 온라인 환경이면 GN 또는 LN을 시도.  
5. 의심 레이어의 출력 평균/표준편차 로깅 — forward hook을 걸어 확인.  

간단한 PyTorch forward hook 예제:  

```python
# 요지: 각 미들 레이어 출력의 mean/std를 찍어본다.
hooks = []
for name, module in model.named_modules():
    if any(k in name.lower() for k in ['bn','ln','norm','group']):
        def _hook(m, inp, outp, n=name):
            print(f"{n}: mean={outp.mean().item():.4f}, std={outp.std().item():.4f}")
        hooks.append(module.register_forward_hook(_hook))
```

---

# 10. 실험 설계(비교 실험 의사코드)  

목표: 같은 CNN 아키텍처에 BN/GN/LN을 교체해가며 학습 곡선을 비교  

의사코드:

```python
for norm in ['batch','group','layer']:
    model = build_cnn(norm_type=norm)
    train(model, epochs=50)
    plot_train_val_accuracy()
```

비교 포인트: 수렴 속도, 최종 검증 정확도, 배치 크기 변화에 따른 안정성(B=128 → B=32 → B=8)  

실험 목표: CIFAR-10에서 같은 아키텍처에 BN vs GN vs LN를 적용하여 수렴 속도·최종 정확도·소배치 강건성을 비교.  

의사코드:  

```python
# 모델 아키텍처는 간단한 ResNet-ish
def build_model(norm_type='batch'):
    if norm_type == 'batch': Norm = nn.BatchNorm2d
    elif norm_type == 'group': Norm = lambda C: nn.GroupNorm(32, C)
    elif norm_type == 'layer': Norm = lambda C: nn.LayerNorm([C, H, W])
    # ... build conv blocks with Norm

# 학습 루프 동일하게 맞추기
for norm in ['batch','group','layer']:
    model = build_model(norm)
    train(model)
    evaluate(model)
```

측정지표: 학습 곡선(훈련/검증 loss), 최종 검증 정확도, 배치 크기 변화에 따른 성능(예: B=128,32,8)  

---

# 11. PyTorch 예제 — 훈련/추론 차이까지 확인  

```python
import torch
import torch.nn as nn

# -------------------------------
# 1. Batch Normalization 예제
# -------------------------------
x = torch.randn(4, 3, 8, 8)  # (B, C, H, W)
bn = nn.BatchNorm2d(3)       # 채널=3

# 훈련 모드
bn.train()
train_out = bn(x)

# 추론 모드
bn.eval()
eval_out = bn(x)

print("BN - train mean (per channel):", train_out.mean([0,2,3]))
print("BN - eval mean (per channel):", eval_out.mean([0,2,3]))

# -------------------------------
# 2. Layer Normalization 예제
# -------------------------------
x = torch.randn(2, 5)  # (B, d)
ln = nn.LayerNorm(5)

ln.train()
train_out = ln(x)

ln.eval()
eval_out = ln(x)

print("LN - train mean (per sample):", train_out.mean(-1))
print("LN - eval mean (per sample):", eval_out.mean(-1))

# -------------------------------
# 3. Group Normalization 예제
# -------------------------------
x = torch.randn(2, 6, 4, 4)  # (B, C, H, W)
gn = nn.GroupNorm(3, 6)      # 그룹=3, 채널=6

train_out = gn(x)
eval_out = gn(x)   # GN은 train/eval 모드 차이 없음

print("GN - mean per group (approx):", train_out.mean([0,2,3]))
```

### 해설  

- BN: train() 모드에서는 배치 통계, eval() 모드에서는 running mean/var 사용 → 출력 분포 달라짐.  
- LN: 샘플 단위 통계만 사용하므로 train/eval 차이 없음(단, γ/β의 학습 상태는 영향을 줌).  
- GN: 그룹 단위 통계만 사용, train/eval 차이 없음.  

---

# 12. 간단한 NumPy 확인 코드(포워드만) — 손으로 확인할 때 유용  

```python
import numpy as np

def batch_norm_forward(x, eps=1e-5):
    # x: (B, C, H, W)
    mu = np.mean(x, axis=(0,2,3), keepdims=True)
    var = np.var(x, axis=(0,2,3), keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat

def layer_norm_forward(x, eps=1e-5):
    # x: (B, C) or (B, T, d) 사용 시 마지막 축에 적용
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat

def group_norm_forward(x, G, eps=1e-5):
    # x: (B, C, H, W)
    N, C, H, W = x.shape
    x_reshaped = x.reshape(N, G, C//G, H, W)
    mu = np.mean(x_reshaped, axis=(2,3,4), keepdims=True)
    var = np.var(x_reshaped, axis=(2,3,4), keepdims=True)
    x_hat = (x_reshaped - mu) / np.sqrt(var + eps)
    return x_hat.reshape(N, C, H, W)

# 테스트 (CNN 스타일 x: (B,C,H,W))
x = np.array([[[[1]], [[2]]], [[[3]], [[4]]]])  # (2,2,1,1)
print("BN:", batch_norm_forward(x))
print("LN:", layer_norm_forward(x.reshape(2,2)))
print("GN (G=1):", group_norm_forward(x, G=1))
```

이 코드로 각 정규화 출력 확인한다. 실전은 torch.nn 등.  

---

# 13. 자주 하는 실수 & Q&A  

Q1: BN을 쓰는데 배포에서 성능이 크게 떨어져요. 왜?  

A1: 추론 시 model.eval()을 호출하지 않아 running mean/var가 아닌 배치 통계를 사용했을 가능성이 큽니다.  

Q2: 소배치인데 BN을 꼭 써야 할까요?  

A2: 소배치에서는 BN 통계가 불안정합니다. GN이나 LN으로 바꾸거나 SyncBN(다중 GPU 동기화)을 고려하세요.  

Q3: γ, β에 weight decay를 적용해도 될까요?  

A3: 보통 γ,β는 weight decay 대상에서 제외합니다(성능 손실 우려).  

Q4: Norm을 활성화 함수 앞에 두나요, 뒤에 두나요?  

A4: 관습적으로 ResNet 계열은 Conv → BN → Activation 순을 사용하지만, 실험적 차이가 있으므로 작은 모델로 실험해보는 것이 좋습니다.  

자주 헷갈리는 포인트 정리:  

- 축 차이: BN 배치, LN 피처.  
- γ/β 위치: 항상 마지막.  
- 추론 모드: BN만 running stats.  
- 변형: InstanceNorm (이미지 스타일).  

---

# 14. 정규화 쌓기와 네트워크 동역학  

일반 패턴: Conv/Dense → Norm → Activation 또는 Conv/Dense → Activation → Norm 둘 다 사용 사례 존재.  

다층에서의 정규화는 각 층의 입력 분포를 제어하여 깊은 네트워크의 학습을 안정화.  

역전파 관점에서는 정규화가 기울기 폭발/소실을 완화하는 역할을 한다.  

다층: Dense + Act + Norm.  

역전파: Norm 미분 포함(안정 전달).  

---

# 15. 실무 감각: 선택과 튜닝  

- 선택 기준: CNN → BN, Transformer → LN, 소배치 CNN → GN.  
- 튜닝 포인트: G의 선택(예: G=32 권장)을 실험으로 검증. eps와 momentum(=alpha)도 실험적으로 조정.  
- 디버깅: 레이어별 출력 분포(평균/분산) 모니터링 → 값이 NaN 또는 비정상적으로 클 때 정규화 추가.  
- 실험: Norm 위치(Act 전/후) 비교.  
- 디버깅: 분포 모니터 → 불안정 시 Norm 추가.  
- 트렌드: RMSNorm (LN 변형, TR 효율).  

---

# 16. 확장: 변형들(간단히)  

- InstanceNorm: 스타일 전이 등에서 사용(그램 차원의 정규화).  
- LayerScale / RMSNorm: Transformer 계열에서 계산비/안정성 개선을 노리는 변형.  
- SyncBN / GhostBN: 분산 학습에서 BN 한계를 보완.  

---

# 17. 고급 주제: BatchRenorm, Weight Standardization, EvoNorm  

## 17.1 BatchRenorm  

BN의 훈련/추론 모드 불일치를 보정하기 위한 방법으로, 훈련 시에도 running statistics를 더욱 적극 반영한다. 핵심은 보정 계수:  

$$
r = \frac{\sigma_{B}}{\sigma_{running}},\qquad d = \frac{\mu_B - \mu_{running}}{\sigma_{running}}
$$

이를 이용해 정규화 전후를 조정한다. 구현 복잡성은 약간 증가하지만 소배치일 때 안정성이 개선된다.  

## 17.2 Weight Standardization + GN

Weight Standardization(WS)은 Conv 필터에 대해 가중치 정규화를 수행하여 스케일을 안정화하는 기법이다.  
GN과 함께 쓰이면 BN 없이도 학습이 잘 되는 현대 CNN 설계(예: ConvNeXt)에서 성능 향상을 보인다.  

## 17.3 EvoNorm  

정규화와 활성화 함수를 통합한 최신 연구 계열. 실제 적용 시 좋은 성능을 내지만 하이퍼파라미터가 늘어난다.  

---

# 18. 그래프 비교와 시각화 팁  

- BN: 배치 분포 그래프 변동.  
- LN: 샘플별 히스토그램 안정.  
- GN: 그룹 크기 따라.  
- 시각화: 분포 플롯 전/후.  

직관: 깊은 층에서 분포 유지 → 안정.  

---

# 19. 기울기 문제와 해결 연계  

ICS 완화: 정규화로 활성화 입력 안정 → 소실/폭주 ↓.  

ICS 극복.  

---

# 20. 초기화와 정규화의 궁합

정규화 있으면 초기화 덜 민감(분포 재조정).  

---

# 21. 정규화와 Transformer의 연결고리 미리보기

- TR: LN으로 시퀀스 안정, 잔차 연결과 함께.  
- 왜 LN? 배치 무관, 토큰 독립 처리.  

---

# 22. 결론 및 추천 (실무 요약)

정규화의 본질은 "어떤 축에서 평균과 분산을 취할 것인가"이다. 이 한 선택이 모델 성능과 안정성에 큰 영향을 준다.  

1. Transformer·시퀀스 → LayerNorm(Pre-LN 권장)  
2. CNN이고 충분한 배치 → BatchNorm  
3. CNN이고 소배치/온라인/분산불리환경 → GroupNorm(+Weight Standardization)  
4. gamma/beta에 weight decay를 적용하지 말 것(대부분 권장)  
5. 디버깅: forward hook으로 각 레이어 출력 분포 확인 → model.eval() 누락/러닝 통계 문제 우선 의심  

이번 편에서는 정규화를 BatchNorm부터 GroupNorm까지 수식·예제·코드로 비교하며 왜 각자가 언제 유리한지 해부하였다.  
이 감각이 생기면, 대형 모델 학습이 훨씬 수월해진다.

정규화 = 분포 안정: ICS 극복.  

- BatchNorm: 배치 평균/분산, 규제 강 but 배치 의존.  
- LayerNorm: 피처 평균/분산, 배치 무관 TR 표준.  
- GroupNorm: 그룹 평균/분산, 중간 균형.  
- 미분: 체인룰 전달, 학습 γ/β.  
- 적분: 분산 누적 직관.  

다음 5편에서는 어텐션(Attention)을 다룬다.  
Self-Attention/Multi-Head를 수식·직관·TR 핵심으로 설명하고, 왜 TR의 엔진인지 밝힌다.  

부록: 더 보고 싶다면  

- 역전파(full derivation) 문서(요청 시 재첨부 가능).  
- PyTorch custom layer 구현 예제(훈련/추론 동작 포함) 추가 가능.  

---

# 23. 부록: BN 역전파 유도 자세한 전개(수학적으로)  

(이 절은 위 요약의 수식들을 더 엄밀히 유도한 것이다. 수식을 차근차근 따라가며 풀면 BN의 역전파 식이 왜 그렇게 생기는지 이해할 수 있다.)  

설정: $x_1,...,x_N$에 대해  

$$
\mu = \frac{1}{N}\sum_i x_i,\quad
\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2,\quad
s=\sqrt{\sigma^2+\varepsilon}
$$

$\hat{x}_i = (x_i - \mu)/s$, $y_i = \gamma \hat{x}_i + \beta$.

$L$를 쓸 때,  

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \gamma \frac{\partial \hat{x}_j}{\partial x_i}.
$$

따라서 $\partial \hat{x}_j / \partial x_i$를 계산하면 된다. $\hat{x}_j = (x_j - \mu) s^{-1}$ 이므로,  

$$
\frac{\partial \hat{x}_j}{\partial x_i} = s^{-1}(\delta_{ij} - \frac{1}{N}) + (x_j - \mu) \cdot \frac{\partial s^{-1}}{\partial x_i}.
$$

그리고 $s^{-1} = (\sigma^2 + \varepsilon)^{-1/2}$,  

$$
\frac{\partial s^{-1}}{\partial x_i} = -\frac{1}{2}(\sigma^2+\varepsilon)^{-3/2} \cdot \frac{\partial\sigma^2}{\partial x_i}.
$$

$\partial \sigma^2 / \partial x_i = \frac{2}{N}(x_i - \mu) - \frac{2}{N}\sum_k (x_k - \mu) \cdot \frac{\partial \mu}{\partial x_i} = \frac{2}{N}(x_i - \mu)$ (두 번째 항이 0이 됨을 보일 수 있다). 결합하면 위의 최종식으로 도달한다.  

(수식 전개는 이 문서의 목적상 여기까지 적는다. 더 엄밀한 변형을 원하면 알려달라.)  

---

# 24. 부록: 빠른 참고 표

| 상황                         | 추천 정규화             | 이유/비고               |
| -------------------------- | ------------------ | ------------------- |
| Transformer, NLP           | LayerNorm (Pre-LN) | 배치 불필요, 시퀀스 안정성     |
| CNN, large batch           | BatchNorm          | 학습 가속/규제 효과         |
| CNN, small batch or online | GroupNorm (+WS)    | 배치 무관, 안정적          |
| Multi-GPU distributed      | SyncBN             | 배치 통계 동기화로 BN 효과 복원 |

---