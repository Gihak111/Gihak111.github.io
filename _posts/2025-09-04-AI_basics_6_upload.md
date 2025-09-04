---
layout: single
title:  "AI 입문 6편: FFN과 잔차 연결"
categories: "AI"
tag: "Explanation"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈

5편에서는 어텐션(Scaled Dot-Product, Self-Attention, Multi-Head)을 상세히 풀며 트랜스포머의 핵심 엔진을 이해했다.  
이번 6편에서는 트랜스포머 레이어의 나머지 주요 구성 요소인 **Feed-Forward Network(FFN)**과 **잔차 연결(Residual Connection)**을 본격적으로 파고든다.  
구체적으로 **FFN의 구조와 미분, 잔차 연결의 역할과 수학적 이점**을 **수식·그래프·미분·장단점·실전 팁**으로 비교하며 알아보자.  

---

## 0. 이번 편의 핵심

* **FFN = 위치별 독립 변환 모듈**: 5편의 어텐션이 시퀀스 전체의 관계를 모델링하는 "글로벌" 모듈이라면, FFN은 각 토큰(또는 위치)에서 독립적으로 특징을 변환하는 "로컬" 모듈이다. 이는 비선형성을 더해 모델의 표현력을 높여준다.  
* **잔차 연결 = 정보 보존 지름길**: 깊은 네트워크에서 층이 쌓일수록 정보가 왜곡되거나 기울기가 사라지는 문제를 해결한다. 입력을 출력에 직접 더함으로써 "변화량"만 학습하게 해준다.  
* 각 요소의 **수식과 미분**을 알면 트랜스포머 레이어가 어떻게 쌓이는지 이해할 수 있다: 어텐션(5편) + FFN + 잔차 + 정규화(4편) = 하나의 완전한 블록.  
* **선택 기준**: 트랜스포머에서 FFN은 보통 GELU(3편) 활성화와 Dense Layer(2편) 두 개를 사용하며, 잔차 연결은 거의 모든 현대 딥러닝 아키텍처에서 필수이다.  
* 적분 관점에서 FFN의 층별 누적 변환과 잔차 연결의 다중 경로를 "연속적인 함수 근사"로 연결해 보자. 이 편에서 초보자를 위해 수식을 하나하나 풀어가며, 왜 이게 학습에 도움이 되는지 반복해서 설명하겠다.  

---

## 1. Feed-Forward Network(FFN) — 위치별 비선형 변환의 핵심  

### 1.1 FFN이 왜 필요한가: 배경과 동기  

먼저, 트랜스포머 모델에서 FFN이 왜 등장하는지부터 알아보자.  
5편에서 배운 어텐션 메커니즘은 시퀀스 데이터의 각 위치(토큰) 간 관계를 모델링한다.  
예를 들어, 문장 "The cat sat on the mat"에서 "cat"과 "sat"의 관계를 계산한다.  
하지만 어텐션의 출력은 기본적으로 선형 연산(내적과 가중 평균)의 결과이다.  
선형 연산만으로는 복잡한 비선형 패턴을 학습하기 어렵다(3편에서 설명한 선형성의 한계 참조).  

더 쉽게 설명하자면, 어텐션이 "친구들과 대화하며 정보를 교환"하는 과정이라면, FFN은 "각자가 집에 가서 그 정보를 스스로 소화하고 재해석"하는 과정이다.  
각 토큰이 독립적으로 자신의 임베딩 벡터를 변환하여 더 풍부한 특징을 추출한다.  
이는 RNN이나 LSTM의 히든 상태 업데이트와 유사하지만, FFN은 위치별로 병렬 처리되므로 GPU에서 훨씬 빠르다.  

* **문제 1: 표현력 부족**: 어텐션만으로는 모델이 단순한 선형 변환에 머무를 수 있습니다. FFN이 비선형 활성화(3편)를 추가해 복잡한 함수 근사를 가능하게 한다.  
* **문제 2: 로컬 vs 글로벌**: 어텐션은 글로벌(전체 시퀀스)하지만, FFN은 로컬(각 위치 독립)로 균형을 맞춘다.  
* **트랜스포머에서의 역할**: 각 레이어에서 Self-Attention(5편) 후 FFN이 적용되어 특징을 재정제한다.  

FFN을 빼고 트랜스포머를 학습하면 성능이 급격히 떨어진다.  
왜냐하면 비선형성이 없으면 모델이 얕은 학습만 하기 때문이다.  

### 1.2 FFN의 기본 수식과 구조: 단계별 분해  

트랜스포머의 FFN은 보통 두 개의 Dense Layer(2편)와 하나의 활성화 함수(3편)로 구성된다.  
수식을 자세히 보자:  

$$
\text{FFN}(\mathbf{x}) = W_2 \left( f(W_1 \mathbf{x} + \mathbf{b}_1) \right) + \mathbf{b}_2
$$

여기서:

$$ 
\mathbf{x} ∈ \mathbb{R}^{d_{\text{model}}}
$$
: 입력 벡터 (하나의 토큰 임베딩, d_model은 임베딩 차원, 보통 512~1024).  
$$
W_1 ∈ \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}
$$
: 첫 번째 가중치 행렬 (d_ff는 중간 차원, 보통 4 × d_model로 확장).  
$$
\mathbf{b}_1 ∈ \mathbb{R}^{d_{\text{ff}}}
$$
: 첫 번째 편향.  

$$
f
$$
: 활성화 함수, 트랜스포머 원본은 ReLU지만 현대 모델(BERT, GPT)은 GELU(3편).  
$$
W_2 ∈ \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}
$$
: 두 번째 가중치 행렬 (차원 복귀).  
$$
\mathbf{b}_2 ∈ \mathbb{R}^{d_{\text{model}}}
$$
: 두 번째 편향.  

첫 번째 Dense($W_1$)는 입력을 더 큰 공간($d_ff$)으로 "확장"하여 세밀한 특징을 추출한다.  
비유로, 작은 방에서 큰 방으로 옮겨 물건을 자세히 분류하는 것과 같다.   
활성화 f는 비선형성을 추가해 복잡한 패턴을 만든다.  
두 번째 Dense($W_2$)는 다시 원래 차원으로 "압축"하여 다음 레이어와 호환되게 합니다.  

단계별 계산 과정 (초보자 위해 수학 유도):  
1. 첫 번째 선형 변환:  
   $$
   \mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1
   $$
   - $\mathbf{b}_1$ 편향은 이동(translation)을 허용해 더 유연한 매핑(2편 참조). 없이면 원점 통과 강제.
   - 중요성: $d_ff > d_model$이므로 "과잉 표현"으로 정보 손실 없이 세부 처리.

2. 활성화 적용:  
   $$
   \mathbf{a} = f(\mathbf{z}_1)
   $$
   - $f = GELU(z) = z \Phi(z), \Phi(z)$는 정규 누적 분포(3편).
   - 왜 GELU? ReLU보다 매끄러워 기울기 소실 적고, 음수 약간 허용(죽은 뉴런 방지).
   - 유도: GELU는 확률적 관점에서 z를 "게이팅" – 적분 관점(2편)에서 누적 확률로 스무딩.

3. 두 번째 선형 변환:
   $$
   \mathbf{z}_2 = W_2 \mathbf{a} + \mathbf{b}_2
   $$
   - 왜 $W_2$? 압축하며 중요한 특징 선택. 큰 방에서 정리 후 작은 상자로 포장.  

전체 시퀀스 입력 $X ∈ \mathbb{R}^{N \times d_{\text{model}}}$ (N: 토큰 수)에 대한 배치 버전:
$$
Z_1 = X W_1^\top + \mathbf{1} \mathbf{b}_1^\top \quad (\mathbb{R}^{N \times d_{\text{ff}}})
$$
- $\mathbf{1} ∈ \mathbb{R}^{N \times 1}: ones$ 벡터, 편향 브로드캐스트(2편).
- 왜 $^\top$? 코드에서 행렬 곱 순서(초보자: NumPy matmul처럼).

$$
A = f(Z_1) \quad (요소별, 병렬)
$$

$$
\text{FFN}(X) = A W_2^\top + \mathbf{1} \mathbf{b}_2^\top \quad (\mathbb{R}^{N \times d_{\text{model}}})
$$

각 행(토큰)이 독립이니 $O(N d_ff d_model)$ 복잡도지만 GPU 빠름.  
수식에서 X가 행렬이니 한 번 계산으로 전체 처리 – 병렬성의 핵심.  

### 1.3 FFN의 미분과 역전파: 체인룰 유도와 설명  

학습은 역전파(2편)로, FFN의 미분을 자세히 유도하겠다.  
손실 L에 대한 그라디언트:  

먼저, 출력에 대한 그라디언트 $∂L/∂FFN$ 주어짐 (상위 레이어에서).

1. 두 번째 Dense 미분:
   $$
   \frac{\partial L}{\partial \mathbf{z}_2} = \frac{\partial L}{\partial \text{FFN}}
   $$
   - 초보자: 이는 출력 변화가 손실에 미치는 직접 영향.

   $$
   \frac{\partial L}{\partial W_2} = A^\top \frac{\partial L}{\partial \mathbf{z}_2}
   $$
   - 유도: $L = g(\mathbf{z}_2), \mathbf{z}_2 = W_2 \mathbf{a}$, 체인룰 $\frac{\partial L}{\partial W_2} = \frac{\partial \mathbf{z}_2}{\partial W_2}^\top \frac{\partial L}{\partial \mathbf{z}_2} = \mathbf{a}^\top \frac{\partial L}{\partial \mathbf{z}_2}$ (행렬 미분 규칙, 2편).
   -  $W_2$ 업데이트 방향 -> A가 큰 특징 강조하기 때문에 중요하다.  

   $$
   \frac{\partial L}{\partial \mathbf{b}_2} = \mathbf{1}^\top \frac{\partial L}{\partial \mathbf{z}_2}
   $$
   - 배치 합: 편향은 모든 토큰 공유.  

   $$
   \frac{\partial L}{\partial \mathbf{a}} = \frac{\partial L}{\partial \mathbf{z}_2} W_2^\top
   $$
   - 전파: 다음으로 기울기 전달.

2. 활성화 미분:
   $$
   \frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{a}} \odot f'(\mathbf{z}_1)
   $$
   - $⊙$: 요소별 곱(3편). $f' = GELU' = \Phi(\mathbf{z}_1) + \mathbf{z}_1 \phi(\mathbf{z}_1), \phi$는 정규 밀도.
   - 유도: $a = f(z_1)$, 체인룰 직접. 요소별로 해야 활성화 독립.  
   - 초보자: $f'$가 0이면 기울기 소실되기 때문에 GELU가 ReLU보다 더 좋다.  

3. 첫 번째 Dense 미분:  
   $$
   \frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial \mathbf{z}_1}
   $$
   - 유사 $W_2$.

   $$
   \frac{\partial L}{\partial \mathbf{b}_1} = \mathbf{1}^\top \frac{\partial L}{\partial \mathbf{z}_1}
   $$

   $$
   \frac{\partial L}{\partial X} = \frac{\partial L}{\partial \mathbf{z}_1} W_1^\top
   $$
   - 이전 레이어(어텐션 출력)로 전달.  

이 미분 과정이 왜 학습을 가능하게 하냐면, 체인룰로 뒤에서 앞으로 기울기가 흐르지만, 활성화 f'가 곱해지며 약해질 수 있다.  
여기서 잔차 연결(다음 섹션)이 도와준다.  
수학적으로, FFN 미분은 Dense(2편)와 활성화(3편)의 조합으로 이전 지식 연결된다.  

### 1.4 FFN의 장단점: 상세 분석  

* 장점:  
  - **병렬 처리 가능**: 위치 독립, 어텐션과 달리 $O(N d_ff d_model)$ but $N<<d_ff$.
  - **표현력 향상**: $d_ff$ 확장으로 모델 용량 ↑. 예: $d_model=512, d_ff=2048$ 시 파라미터 $512*2048*2 ≈ 2M$.
  - **간단 구현**: Dense 두 개 – 코드 쉬움.  
  - **비선형성 추가**: 활성화로 복잡 패턴 학습(보편 근사 정리).  

* 단점:  
  - **파라미터 오버헤드**: 대형 모델에서 FFN이 대부분 파라미터 차지 (GPT-3 175B 중 FFN 비중 높음).  
  - **과적합 위험**: d_ff 많을 시 훈련 데이터 memorization – 규제(2편 L2) 필요.  
  - **계산 비용**: 확장/축소로 FLOPs 증가, but GPU 최적이라 알빠노.  

실험으로 d_ff 변동 – 작으면 underfit, 크면 overfit. 균형 맞추면 된다 이거 중요함.  

### 1.5 FFN의 사용 사례와 변형  

* 표준: Transformer MLP (FFN = GELU 활성화).  
* 변형: Swish(z * sigmoid(z)) or ReLU – GELU가 성능 우위(3편).  
* 다른 모델: ViT(이미지 패치 FFN), GPT 시리즈.  

Hugging Face 코드에서 FFN 확인해 봐라 torch.nn.Linear 두 개 있고 그럴 꺼다.  

---

## 2. 잔차 연결(Residual Connection) — 깊은 네트워크의 기울기 보존자  

### 2.1 잔차 연결이 왜 필요한가: 문제 배경 상세  

깊은 네트워크(많은 레이어)에서 주요 문제는 "기울기 소실(vanishing gradient)"과 "기울기 폭주(exploding gradient)"이다(3편).  
층이 쌓일수록 미분의 곱(체인룰)이 0이나 ∞로 가기 쉽다.  
예: 100층 네트워크에서 각 층 미분 0.9라면 총 $0.9^100 ≈ 2.656e-5$ –> 거의 0.  

긴 파이프라인에서 물(기울기)이 끝까지 도달하지 못한다.  
잔차 연결은 "지름길 파이프"를 추가해 직접 흐르게 한다.  
ResNet(2015)에서 처음 소개되었고, 트랜스포머가 채택해 깊은 모델(12~24층)학습이 가능해 졌다.  

* **문제 1: 정보 왜곡**: 층 통과 시 입력 정보 손실.  
* **문제 2: 학습 어려움**: 초기 층 기울기 약해 업데이트 느림.  
* **트랜스포머에서의 필요**: 어텐션 + FFN 쌓기 위해 필수 – 없이면 훈련 불안정.  


### 2.2 잔차 연결의 기본 수식과 구조: 유도와 해설

기본 아이디어:
$$
\mathbf{y} = \mathbf{x} + F(\mathbf{x})
$$

- $\mathbf{x}$: 입력.  
- $F$: 변환 함수 (FFN or 어텐션).  
- $\mathbf{y}$: 출력.  

$y = x (identity 경로) + F(x) (잔차 경로)$. 
학습 목표: $F(x)$를 "최적 변화량"으로 한다.  

identity 함수 학습 정확히 1이어야해서 어렵다.  
잔차의 경우 $F(x)=0$ 시 identity 자동.  

트랜스포머 레이어 적용:  
1. 어텐션 부분:  
   $$
   \mathbf{x}' = \text{LayerNorm}(\mathbf{x}) \quad (4편, 분포 안정)
   $$
   - 왜 LN 먼저나면, 그래야 입력 정규화로 변환 안정되기 때문이다.  

   $$
   \mathbf{y} = \mathbf{x} + \text{MultiHeadAttention}(\mathbf{x}', \mathbf{x}', \mathbf{x}') \quad (5편 MHA)
   $$
   - + $\mathbf{x}$: 원 입력 보존.  

2. FFN 부분:
   $$
   \mathbf{y}' = \text{LayerNorm}(\mathbf{y})
   $$

   $$
   \mathbf{z} = \mathbf{y} + \text{FFN}(\mathbf{y}')
   $$
   - 전체: x → z (하나의 레이어 출력).  

시퀀스 버전: X 행렬 동일, +X는 요소별.  

수식에서 $+x$가 왜 중요하냐면,  
$F$가 "잘못" 학습해도 $x$가 보존되어 정보 손실 적어진다.  
깊이 L층 시 총 경로 $2^L$ –> 기울기 다각화.  

변형 수식 (pre-norm vs post-norm):  
- Original Transformer: LN after + .  
- Modern (GPT): LN before F, + after.  

$$
\mathbf{y} = \text{LN}(\mathbf{x} + F(\mathbf{x}))
$$

vs

$$
\mathbf{y} = \mathbf{x} + F(\text{LN}(\mathbf{x}))
$$

- pre-norm 더 안정적이라 차이가 생기는 거다.  

### 2.3 잔차 연결의 미분과 수학적 이점: 상세 유도  

역전파에서 잔차의 마법:  
기본:  
$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left( 1 + \frac{\partial F}{\partial \mathbf{x}} \right)
$$

- 1 없으면 : $\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial F}{\partial \mathbf{x}}$
- +1 있으면: 직접 경로, 미분 곱 피할 수 있다.  

유도 단계 :  
1. $y = x + F(x)$
2. $∂y/∂x = 1 + ∂F/∂x$ (야코비안).
3. 체인룰: $∂L/∂x = (∂y/∂x)^⊤ ∂L/∂y = (1 + ∂F/∂x)^⊤ ∂L/∂y$

깊은 네트워크 유도: L층, $y_l = y_{l-1} + F_l(y_{l-1})$

총 미분:
$$
\frac{\partial L}{\partial y_0} = \frac{\partial L}{\partial y_L} \prod_{l=1}^L \left( 1 + \frac{\partial F_l}{\partial y_{l-1}} \right)
$$
- 곱 대신 합 경로 확장: binomial theorem처럼 $(1 + a)^L = sum binom$.  
곱은 소실 되기 쉽다.($0.9^L →0$), 합은 유지된다(최소 1).  
적분 관점: 잔차 $≈ Euler method for ODE dy/dz = F(y)$, z=깊이 –> 연속 모델(Neural ODE).

### 2.4 잔차 연결의 장단점: 깊이 분석  

* 장점:  
  - **깊이 확장**: ResNet-152처럼 100+층 가능 –> 트랜스포머 48층(BERT-large).  
  - **수렴 속도 ↑**: 초기 F≈0으로 identity, 빠른 초기 학습.  
  - **정보 보존**: degradation 문제 해결(깊어질수록 성능 ↓ 방지).  
  - **기울기 안정**: +1 경로로 소실/폭주 완화.  

* 단점:
  - **추가 연산**: + 연산 미미 but 메모리(gradient 저장).  
  - **설계 주의**: F 입력/출력 차원 맞춰야 (projection if not).  
  - **해석 어려움**: 경로 많아 블랙박스.  

잔차 빼면 훈련 실패한다.  
꼬우면 너가 실험해 봐라  

### 2.5 잔차 연결의 사용 사례와 변형  

* 표준: Transformer, ResNet.  
* 변형: DenseNet (dense 연결), Highway Net (게이트).  
* 현대: Vision Transformer, GPT – pre-norm 잔차.  

---

## 3. FFN과 잔차의 결합: 트랜스포머 레이어 전체 수식과 흐름  

전체 레이어 수식:  
$$
\text{AttnOut} = X + \text{MHA}(\text{LN}(X)) \quad (잔차1)
$$

$$
\text{FFNOut} = \text{AttnOut} + \text{FFN}(\text{LN}(\text{AttnOut})) \quad (잔차2)
$$

LN으로 입력 안정 → F(어텐션 or FFN) 변환 → + 원본 보존.  
미분 전체: 잔차 덕에 $∂L/∂X = ∂L/∂FFNOut * (1 + ∂FFN/∂LN + ... )$ –> 안정된다.  

레이어 쌓기: L회 반복, 각 블록 독립 but 누적.  
레고 블록 –> 각 블록(어텐션+FFN+잔차)이 쌓여 탑이 되는 거랑 같은 느낌.  

### 3.1 레이어 미분 유도 예  

어텐션 + 잔차 미분:  
$∂L/∂X = ∂L/∂y (1 + ∂MHA/∂x' * ∂LN/∂X) + ...$  

LN 미분 복잡하지만, 체인이다.  
잔차가 안전빵 해준다.  

---

## 4. 숫자로 따라가는 미니 예제 (손으로 풀어보기)  
N=2 (토큰), d_model=2, d_ff=4.  

$X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$

$W_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \\ 0.7 & 0.8 \end{bmatrix}^⊤ (2x4)$

$b_1 = [0,0,0,0]$

$Z_1 = X W_1 = 계산 단계: 첫 행 1*0.1+2*0.3 =0.7, etc.$

$A = ReLU(Z_1) = max(0, Z_1)$

W_2 = 랜덤, FFN 출력.

잔차: FFN + X.

미분: $∂L/∂FFN = [[1,1],[1,1]] $가정, backward 단계별 계산.

---

## 5. 그래프 비교와 시각화 팁   

* FFN: 입력/출력 벡터 플롯 – 확대/활성화/축소.  
* 잔차: 네트워크 다이어그램, 지름길 화살표.  
* 시각화: TensorBoard로 기울기 히스토그램 – 잔차 있/없 비교.  

PyTorch hook으로 중간 값 로그만 해도 탈 없긴 하다.  

---

## 6. 적분의 관점 — FFN과 잔차의 연속 해석

잔차: $y = x + F(x) ≈ x + ε F(x/ε), ε→0 시 dy/dz = F(y)$, z=층.

FFN: 누적 변환 as 적분 $∫ F dz.$

연속 흐름 –> 이산 층을 continuous로 봄, 안정성 올라간다.  

---

## 7. 초기화와 FFN/잔차의 궁합  

He 초기화(3편 ReLU용) for FFN – 분산 $2/fan_in$.

LN(4편) + 잔차: 초기화 덜 민감.  

---

## 8. FFN과 잔차를 코드로 확인해보기 (NumPy 의사코드)  

```python
import numpy as np

def gelu(z):
    return 0.5 * z * (1 + np.erf(z / np.sqrt(2)))

def ffn(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1  # (N, d_ff)
    A = gelu(Z1)
    return A @ W2.T + b2  # (N, d_model)

def layer_norm(X, gamma=1, beta=0, eps=1e-5):
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

def transformer_layer(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2):  # 단순 MHA 생략
    # 어텐션 부분 생략, FFN + 잔차 fokus
    attn_out = X  # placeholder
    norm1 = layer_norm(attn_out)
    ffn_out = ffn(norm1, W1, b1, W2, b2)
    return attn_out + ffn_out  # 잔차

# 테스트 데이터
X = np.array([[1,2], [3,4]])
# 랜덤 W 등 초기화
# 출력 계산
```

---

## 9. 자주 헷갈리는 포인트 정리  

* LN 위치: pre vs post – pre 더 안정.  
* 잔차 없이: 훈련 실패 – 왜? 기울기 0.  
* d_ff 선택: 4*d_model 표준, but 튜닝.  
* FFN vs Dense: FFN은 두 Dense + act.  

잔차는 보존을 위해 +x이다.

---

## 10. FFN/잔차 쌓기와 네트워크 동역학  

L 레이어: for l in 1 to L: X = transformer_layer(X)  
동역학: 잔차로 선형 동역학, FFN 비선형.  
초보자: "안정적 진화" – 층별 작은 변화 누적.  

---

## 11. 실무 감각: 선택과 튜닝  

* FFN: d_ff=2048 (BERT-base), GELU.  
* 잔차: 항상 on, off 시 baseline.  
* 튜닝: lr schedule with 잔차 – warm up.  
* 디버깅: gradient norm 모니터 – 폭주 시 clip.  
* 트렌드: FFN 효율화 (MoE, sparse).  

---

## 12. FFN/잔차와 TR(트랜스포머)의 연결고리 미리보기

TR 인코더: 6~12 레이어, 각 어텐션 + FFN + 잔차 + LN.  
디코더: 마스크 attn 추가.  
왜 완성? 이 블록으로 encoder-decoder (번역 등).  

---

## 이번 편 요약(한 장으로)

* **FFN = 두 Dense + act**: $\text{FFN}(x) = W2 f(W1 x + b1) + b2$, 위치 독립 변환.  
* **잔차 = x + F(x)**: 기울기 보존, $∂L/∂x = ∂L/∂y (1 + ∂F/∂x)$.  
* **결합**: LN + F + 잔차 블록.  
* **미분**: 체인룰 유도, 소실 방지.  
* **적분**: ODE-like 연속.  
* 실전: d_ff 튜닝, GELU, pre-norm.  

---

## 결론

이번 편에서는 **FFN과 잔차 연결**을 수식 중심으로 알아보았다.  
각 수식의 의미와 중요성을 반복 설명해 감각이 잡힐 것이다.  
이 바탕으로 트랜스포머 전체가 보일 것이다.  

다음 7편에서는 **포지셔널 인코딩(Positional Encoding)**을 다룬다.  
위치 정보를 추가하는 방법을 **수식·직관·변형**으로 설명하고, 왜 어텐션에 필수인지 밝히자.  