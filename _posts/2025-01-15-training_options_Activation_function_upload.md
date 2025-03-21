---
layout: single
title:  "활성화 함수와 다른 이것저것"
categories: "Deep_Learning"
tag: "model-optimization-deployment"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


다양한 딥러닝에서 사용되는 활성화 함수(Activation Functions)와 그 외 의 다양한 함수들을 알아보고 그 특징을 아래에 정리하겟다.  
활성화 함수는 뉴런의 출력을 결정하며, 모델의 학습 성능과 표현력을 크게 좌우한다.  
전에 한번, 활성화 함수에 대해 정리했던 적이 있다.  
이 글도 참고하자.  


### 1. 전통적 활성화 함수  
#### (1) Sigmoid  
- 정의:  
  \[
  \text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
  \]
- 특징:  
  - 출력 범위: (0, 1).  
  - 주로 확률 계산에 사용.  
  - 단점: Vanishing Gradient 문제 (x가 크거나 작을 때 기울기가 0에 가까워짐).  
  - 활용: 초창기 뉴럴 네트워크, 이진 분류.  

#### (2) Tanh (Hyperbolic Tangent   
- 정의:  
  \[
  \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]
- 특징:  
  - 출력 범위: (-1, 1).  
  - Sigmoid보다 출력 중심이 0에 가까워 학습 수렴이 빠름.  
  - 단점: 여전히 Vanishing Gradient 문제 존재.  
  - 활용: 순환 신경망(RNN) 등.  


### 2. Rectified Linear Unit (ReLU) 및 변형
#### (1) ReLU  
- 정의:  
  \[
  \text{ReLU}(x) = \max(0, x)
  \]
- 특징:  
  - 출력 범위: [0, ∞).  
  - 단순하고 계산 효율성 높음.  
  - 단점: Dead Neuron 문제 (입력이 음수일 때 기울기가 0).  
  - 활용: 딥러닝 모델 대부분에서 기본 활성화 함수로 사용.

#### (2) Leaky ReLU  
- 정의:  
  \[
  \text{Leaky ReLU}(x) = 
  \begin{cases} 
  x & \text{if } x > 0 \\
  \alpha x & \text{if } x \leq 0
  \end{cases}
  \]
  - 여기서 \(\alpha\)는 작은 양수 (일반적으로 0.01).  
- 특징:  
  - 음수 입력에 대해 기울기를 유지.  
  - Dead Neuron 문제 완화.  

#### (3) Parametric ReLU (PReLU)  
- **정의**: Leaky ReLU에서 \(\alpha\)를 학습 가능한 파라미터로 설정.  
- **특징**:  
  - 학습 과정에서 \(\alpha\)를 최적화.  

#### (4) Exponential Linear Unit (ELU)  
- 정의:  
  \[
  \text{ELU}(x) = 
  \begin{cases} 
  x & \text{if } x > 0 \\
  \alpha(e^x - 1) & \text{if } x \leq 0
  \end{cases}
  \]
- 특징:  
  - ReLU와 비슷하지만 음수 입력에 대해 부드러운 변화 제공.  
  - 출력 평균이 0에 가까워 학습 안정화.  

#### (5) GELU (Gaussian Error Linear Unit)  
- 정의:  
  \[
  \text{GELU}(x) = x \cdot \Phi(x)
  \]
  - 여기서 \(\Phi(x)\)는 정규 분포의 누적 분포 함수.  
- 특징:  
  - 입력에 대해 부드럽고 확률적 활성화 제공.  
  - 트랜스포머 모델에서 많이 사용 (예: BERT).  


### 3. Softmax
- 정의:  
  \[
  \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
  \]
- 특징:  
  - 출력 범위: (0, 1).  
  - 출력의 합이 항상 1이 되어 확률 해석 가능.  
  - 활용: 다중 클래스 분류의 출력층.  


### 4. Swish 및 최신 함수  
#### (1) Swish  
- 정의:  
  \[
  \text{Swish}(x) = x \cdot \text{Sigmoid}(x)
  \]
- 특징:  
  - Google이 제안한 활성화 함수.  
  - 학습 안정성과 성능 우수.  
  - 활용: EfficientNet 등 최신 모델.  

#### (2) Mish  
- 정의:  
  \[
  \text{Mish}(x) = x \cdot \tanh(\text{Softplus}(x))
  \]
  - 여기서 \(\text{Softplus}(x) = \ln(1 + e^x)\).  
- 특징:  
  - Swish와 유사하지만 더 부드럽고 학습에 강건.  


### 5. Custom 및 상황별 함수  
#### (1) Maxout  
- 정의: 여러 선형 함수의 최대값을 선택.  
  \[
  \text{Maxout}(x) = \max(x_1, x_2, ..., x_k)
  \]
- 특징:  
  - 다양한 형태의 함수 표현 가능.  
  - 계산 비용 높음.  

#### (2) Hard Sigmoid / Hard Tanh  
- **정의**:  
  - Sigmoid/Tanh를 근사하여 계산 효율성을 높임.  
- 활용: 임베디드 환경이나 경량화 모델.  

#### (3) Sinusoidal Activation  
- 정의:  
  \[
  \text{Sin}(x) = \sin(x)
  \]
- 특징:  
  - 주기적 데이터 처리에 활용 (예: 시계열).  


### 6. 비선형적 변환 함수  
#### (1) Softplus  
- 정의:  
  \[
  \text{Softplus}(x) = \ln(1 + e^x)
  \]
- 특징:  
  - ReLU의 부드러운 버전.  
  - 음수 기울기를 완전히 제거하지 않음.  

#### (2) Thresholded ReLU  
- 정의:  
  \[
  \text{Thresholded ReLU}(x) = 
  \begin{cases} 
  x & \text{if } x > \theta \\
  0 & \text{otherwise}
  \end{cases}
  \]
  - \(\theta\): 임계값.  
- 특징:  
  - 특정 임계값 이상에서만 활성화.  


### 7. 활성화 함수 비교  
| 함수       | 출력 범위     | 주요 특징                       | 단점                         | 활용 분야                |
|------------|---------------|---------------------------------|-----------------------------|--------------------------|
| Sigmoid    | (0, 1)        | 확률 계산                     | Vanishing Gradient 문제     | 이진 분류               |
| Tanh       | (-1, 1)       | 중심이 0                       | Vanishing Gradient 문제     | 순환 신경망(RNN)         |
| ReLU       | [0, ∞)        | 단순, 계산 효율성              | Dead Neuron 문제            | CNN, 딥러닝 전반         |
| Leaky ReLU | (-∞, ∞)       | 음수 기울기 유지               | 최적 \(\alpha\) 찾기 필요  | CNN, 회귀               |
| ELU        | (-α, ∞)       | 음수 활성화 부드러움           | 계산 복잡성 증가            | CNN, 고급 모델          |
| Swish      | (-∞, ∞)       | 부드러운 곡선                 | 계산 비용                   | EfficientNet 등          |
| Softmax    | (0, 1)        | 출력의 합이 1                  | 클래스 간 상호의존성        | 다중 클래스 분류         |


위 함수들은 딥러닝 모델의 목적과 데이터 특성에 따라 적절히 선택해서 사용해야 합니다.  
특히, 최신 모델에서는 Swish, GELU와 같은 부드럽고 강력한 함수들이 점점 더 자주 쓰이고 있다.  

지난번에 올린 활성화 함수들도 있다. 이 글도 참고하자.  

