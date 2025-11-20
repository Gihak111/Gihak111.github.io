---
layout: single
title:  "PINNs"
categories: "AI"
tag: "PINN"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 데이터와 물리 법칙의 결합  
## 1. 서론: 데이터 의존성(Data-Driven)의 한계와 물리학의 통합  
전통적인 딥러닝 모델은 방대한 데이터셋에 의존하여 입력과 출력 사이의 비선형 매핑을 학습하는 '블랙박스' 접근법을 취한다.  
그러나 실제 공학이나 물리 현상(유체 역학, 열전달 등)을 모델링할 때, 데이터가 부족하거나 고비용인 경우가 많다.  
PINN은 이러한 딥러닝 모델에 지배 방정식(Governing Equations)인 편미분방정식(PDE)을 손실 함수(Loss Function)의 형태로 제약 조건화하여 통합한다.  
이는 순수 데이터 기반 접근법과 전통적인 수치 해석법(FEM, FDM)의 하이브리드 모델로 정의할 수 있다.  
[Physics-informed neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)  

## 2. 선형대수학적 메커니즘 1: 신경망을 통한 함수 공간의 투영
PINN의 핵심은 신경망을 미지의 해(Solution) $u(t, x)$를 근사하는 Universal Function Approximator로 간주하는 것이다.  
이를 선형대수학적으로 해석하면, 고차원 파라미터 공간에서 데이터 공간으로의 비선형 변환이다.  

신경망의 $l$번째 레이어에서의 연산은 다음과 같은 아핀 변환(Affine Transformation)과 비선형 활성화 함수(Activation Function) $\sigma$의 합성으로 표현된다.  

$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

여기서 $W^{(l)}$은 가중치 행렬(Weight Matrix), $b^{(l)}$는 편향 벡터(Bias Vector)이다.  
선형대수학적 관점에서 딥러닝 학습은 이 행렬 $W$와 벡터 $b$의 최적 조합을 찾아, 입력 공간의 벡터 $x$를 우리가 원하는 해 공간의 벡터 $u$로 매핑하는 선형 변환의 연속체(composition)를 구성하는 과정이다.  
PINN은 이 과정에서 자동 미분(Automatic Differentiation, AD)을 사용하여 물리 법칙을 강제한다.  

## 3. 자동 미분(Automatic Differentiation)의 선형대수학적 실체: 야코비 행렬의 연쇄
이는 야코비 행렬(Jacobian Matrix)과 벡터의 곱(Jacobian-Vector Product, JVP)의 연속이다.  
입력 $x \in \mathbb{R}^{d_{in}}$가 $L$개의 레이어를 거쳐 출력 $u \in \mathbb{R}^{d_{out}}$이 되는 과정을 생각하자.  
물리 방정식(PDE)의 잔차(Residual)를 계산하려면 입력 좌표 $x$에 대한 출력 $u$의 2계 도함수(Hessian)까지 필요하다.  

$$\frac{\partial u}{\partial x} = J_L \cdot J_{L-1} \cdots J_1$$

여기서 $J_l$은 $l$번째 레이어의 야코비 행렬이다.  
PINN의 학습 과정인 역전파(Backpropagation)는 벡터-야코비 곱(Vector-Jacobian Product, VJP)을 통해 손실 함수의 그라디언트를 구하는 과정이다.  
즉, PINN은 순전파(Forward pass)에서 좌표에 대한 미분값(JVP)을 계산하여 물리 법칙을 구성하고, 역전파(Backward pass)에서 파라미터에 대한 미분값(VJP)을 계산하여 학습한다.  
이 두 가지 서로 다른 방향의 행렬 연산이 동시에 맞물려 돌아가는 것이 PINN의 계산적 실체다.  


## 4. 핵심 구조: 물리 정보 잔차(Physics Residual)의 선형 결합
Raissi 등의 논문에 따르면, PINN의 목적 함수는 관측 데이터에 대한 오차($MSE_{u}$)와 물리 방정식의 위배 정도를 나타내는 잔차($MSE_{f}$)의 선형 결합으로 정의된다.  

$$\mathcal{L}(\theta) = w_{u}MSE_{u} + w_{f}MSE_{f}$$

- 물리 법칙의 행렬 연산화  
일반적인 비선형 편미분방정식을 다음과 같이 정의하자.  
$$f(t, x) := \frac{\partial u}{\partial t} + \mathcal{N}[u] = 0$$
여기서 $\mathcal{N}[u]$는 $u$에 대한 비선형 연산자(Non-linear Operator)이다.  
전통적인 수치해석(FEM 등)에서는 이 방정식을 풀기 위해 도메인을 격자(Mesh)로 나누고, 이를 거대 희소 행렬(Sparse Matrix) 시스템 $Ax=b$ 형태로 변환하여 역행렬을 구하거나 반복법으로 해를 구한다.  
반면, PINN은 행렬 역연산(Inversion)을 수행하지 않는다.  
대신, 신경망 출력 $u(t, x)$를 미분 연산자에 통과시켜 $f(t, x)$를 계산한다.  
이때 자동 미분은 연쇄 법칙(Chain Rule)의 반복 적용이며, 이는 야코비 행렬(Jacobian Matrix)의 곱으로 표현된다.  
즉, 물리 법칙 준수 여부를 확인하는 과정 자체가 거대한 야코비 행렬 연산 과정이다.  

## 5. 최적화 과정의 기하학적 해석: Hessian과 Loss Landscape
PINN의 학습은 파라미터 $\theta$ (모든 $W, b$의 집합) 공간에서 손실 함수 $\mathcal{L}(\theta)$의 전역 최솟값을 찾는 비선형 최적화 문제다.  

$$\theta^* = \arg\min_{\theta} (MSE_{u} + MSE_{f})$$

이 과정에서 선형대수학적으로 중요한 것은 손실 함수의 곡률을 나타내는 헤세 행렬(Hessian Matrix) $H$이다.  
$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$$

PINN의 손실 함수는 매우 복잡하고 비볼록(Non-convex)하며, 특히 PDE 제약 조건으로 인해 Loss Landscape가 일반 딥러닝보다 훨씬 거칠다(ill-conditioned).  
이는 헤세 행렬의 고유값(Eigenvalue) 분포가 넓게 퍼져 있음을 의미하며(큰 Condition Number), 경사 하강법(Gradient Descent)의 수렴을 어렵게 만든다.  
따라서 최근 연구들은 헤세 행렬의 정보를 활용하거나, 고유값의 균형을 맞추기 위한 가중치 조절 알고리즘(예: Neural Tangent Kernel 이론 기반)을 도입한다.  

- **최적화 난제 1: 그라디언트 병리(Gradient Pathology)와 벡터의 직교성**  
PINN의 손실 함수는 데이터 손실 $\mathcal{L}_{u}$와 물리 손실 $\mathcal{L}_{f}$의 합이다.  
$$\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{u} + \lambda \nabla_\theta \mathcal{L}_{f}$$

선형대수학적 관점에서 학습이 성공하려면 두 그라디언트 벡터 $\nabla_\theta \mathcal{L}_{u}$와 $\nabla_\theta \mathcal{L}_{f}$가 파라미터 공간에서 서로 협력적인 방향(Cooperative direction)을 가리켜야 한다.  
그러나 Wang et al.(2021)의 연구에 따르면, 고차원 문제에서 이 두 벡터는 종종 직교(Orthogonal)하거나 상충하는 방향을 가리킨다.  
더 심각한 문제는 두 벡터의 놈(Norm)의 불균형이다.  

$$||\nabla_\theta \mathcal{L}_{f}|| \gg ||\nabla_\theta \mathcal{L}_{u}|| \quad \text{혹은 그 반대}$$

만약 물리 손실의 그라디언트 노름이 압도적으로 크다면, 최적화 궤적은 물리 방정식만 만족시키려 하고 경계 조건(데이터)을 무시하는 방향으로 편향된다.  
이는 행렬의 조건수(Condition Number)가 매우 나쁜 선형 시스템을 푸는 것과 같다.  
이를 해결하기 위해 각 손실 항의 그라디언트 노름을 동적으로 맞춰주는 GradNorm 같은 기법이 필수적이다.  
이는 기하학적으로 두 벡터의 크기를 리스케일링하여 타협점(Pareto Optimality)을 찾는 과정이다.  


- **최적화 난제 2: 스펙트럼 편향(Spectral Bias)과 NTK 이론**  
신경망 학습 역학을 설명하는 Neural Tangent Kernel (NTK) 이론은 PINN의 수렴 특성을 고유값 분해(Eigendecomposition)로 명쾌하게 설명한다.  

학습 시간 $t$에 따른 잔차의 감소는 다음과 같이 지수적으로 표현된다.  
$$\frac{d}{dt} (u - u^*) \approx - \eta \sum_{i} e^{-\lambda_i t} v_i v_i^T (u - u^*)$$
여기서 $\lambda_i$와 $v_i$는 NTK 행렬 $K(x, x')$의 고유값과 고유벡터다.  

- 큰 고유값 ($\lambda_{large}$) : 저주파 성분(Low frequency components). 학습 초기 빠르게 수렴한다.  
- 작은 고유값 ($\lambda_{small}$) : 고주파 성분(High frequency components). 수렴에 매우 오랜 시간이 걸리거나 아예 학습되지 않는다.  

PDE 문제는 종종 급격한 변화나 특이점(Singularity)을 포함하므로 고주파 성분이 중요하다. 하지만 PINN은 태생적으로 저주파수 우선 학습(Low-frequency bias) 경향을 가진다.  
선형대수학적으로 보면, 이는 시스템 행렬이 Stiff(뻣뻣한) 성질을 가지며, 고유값 스펙트럼이 매우 넓게 퍼져 있어(Wide Spectrum) 특정 방향으로만 하강이 일어나고 다른 방향으로는 정체되는 현상이다.  
이를 극복하기 위해 푸리에 특징 매핑(Fourier Feature Mapping) 등을 통해 입력 공간을 고차원 고주파 영역으로 강제 투영하는 기법이 사용된다.  

- **기저 함수(Basis Function)의 관점: 고정 기저 vs 적응형 기저**
전통적인 유한요소법(FEM)과 PINN을 선형 공간의 기저(Basis) 관점에서 비교하면 그 차이가 명확하다.  

- FEM : 공간을 격자(Mesh)로 나누고, 각 격자 위에서 정의된 지역적 다항식(Local Polynomials)을 기저로 사용한다.  
    $$u_h(x) = \sum_{i=1}^{N} c_i \phi_i(x)$$
    여기서 $\phi_i$는 미리 정의된 고정된 형태다(예: Hat function).  
    행렬 $A$는 희소(Sparse)하지만, 차원이 커지면(3D 이상) $N$이 기하급수적으로 증가하는 '차원의 저주'에 직면한다.  

- PINN : 신경망 자체가 전역적이고 적응형인 비선형 기저(Global Adaptive Non-linear Basis)를 구성한다.  
    $$u_{NN}(x) = \sum_{i=1}^{N} w^{(out)}_i \sigma(w^{(in)}_i x + b)$$
    PINN은 학습을 통해 문제 해결에 가장 적합한 기저 함수(활성화 함수의 조합) 자체를 찾아낸다.  
    이는 선형대수학적으로 부분 공간(Subspace)을 데이터와 방정식에 맞춰 최적의 형태로 변형(Deformation)시키는 것과 같다.  
    덕분에 PINN은 격자가 필요 없는(Mesh-free) 접근이 가능하며, 고차원 문제에서도 파라미터 수가 선형적으로만 증가하는 효율성을 가진다.  

## 6. Neural Tangent Kernel (NTK) 관점의 스펙트럼 편향
최근 이론적 분석(Wang et al., 2021 등)은 PINN의 학습 역학을 Neural Tangent Kernel (NTK) 이론으로 설명한다.  
무한 너비의 신경망은 선형화된 모델로 근사할 수 있으며, 이때 학습은 커널 회귀(Kernel Regression)와 유사해진다.  

NTK 행렬 $K$의 고유값 분해(Eigendecomposition)를 통해 분석하면, PINN은 스펙트럼 편향(Spectral Bias) 현상을 보인다.  
- 큰 고유값에 해당하는 저주파 성분(전체적인 해의 형상)은 빠르게 학습된다.  
- 작은 고유값에 해당하는 고주파 성분(해의 세밀한 변화나 급격한 변동)은 학습 속도가 매우 느리다.  

이는 선형대수학적으로 볼 때, PDE의 잔차 항($MSE_{f}$)과 경계 조건 항($MSE_{u}$)의 그라디언트 벡터들이 서로 다른 스케일과 방향을 가지기 때문에 발생하는 그라디언트 병리(Gradient Pathology) 현상으로 해석된다.  
이를 해결하기 위해 그라디언트 벡터들의 내적이나 노름(Norm)을 조정하는 기법들이 필수적으로 요구된다.  

### 7. 결론 및 시사점
PINN은 미분방정식을 '푸는(Solving)' 행위를 행렬 $A$를 전치시키거나 분해하는 수치해석적 관점에서, 최적의 파라미터 $\theta$를 탐색하는 최적화(Optimization) 문제로 패러다임을 전환했다.  
선형대수학적 관점에서 PINN은  
1) 고차원 공간의 아핀 변환을 통한 함수 근사  
2) 야코비 행렬 연산을 통한 물리 법칙의 잔차 계산  
3) 헤세 행렬의 고유값 분포에 따른 최적화 궤적 탐색  
이라는 세 가지 축으로 요약된다.  

PINN을 깊이 이해한다는 것은 단순히 딥러닝을 물리 문제에 쓴다는 것이 아니다.  
그것은 "편미분방정식의 해 공간을 파라미터 공간으로 사영(Projection)시켰을 때 발생하는 비선형 최적화 문제의 기하학적 구조와 스펙트럼 특성을 제어하는 기술"이다.  
우리는 지금 $Ax=b$라는 선형 시스템 풀이의 안락함을 떠나, 헤세 행렬의 안장점(Saddle Point)과 그라디언트 벡터의 충돌이 난무하는 비볼록(Non-convex) 최적화의 거친 바다를 항해하여 물리 법칙을 근사하고 있다.  

