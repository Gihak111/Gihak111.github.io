---
layout: single
title:  "PINN +"
categories: "AI"
tag: "PINN"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 개쩌는 PINN 너무 잼있다.  
## 1. PINN의 정의: 물리학을 규제(Regularization)로 도입한 신경망

PINN(Physics-Informed Neural Networks)은 편미분방정식(PDE)의 잔차(Residual)를 손실 함수(Loss Function)에 포함하여,  
학습 데이터가 존재하지 않는 영역에서도 모델이 물리 법칙을 따르도록 강제하는 딥러닝 프레임워크이다.  

기존의 데이터 주도(Data-Driven) 딥러닝이 "정답 데이터(Label)와의 오차 최소화"에 집중한다면, PINN은 "데이터 오차와 물리 방정식 위배 오차의 동시 최소화"를 목표로 한다.  
이는 물리 법칙을 일종의 규제(Regularization) 항으로 사용하여, 데이터가 희소하거나 없는 영역에서도 해(Solution)의 물리적 타당성을 보장한다.  

## 2. 수식적 원리: 잔차(Residual)의 정의

PINN의 핵심은 미분 방정식의 좌변과 우변의 차이인 잔차(Residual)를 0으로 만드는 것이다.  
일반적인 편미분방정식을 다음과 같이 정의한다.  

$$\mathcal{N}[u(x, t)] = f(x, t)$$

* $u(x, t)$: 구하고자 하는 해 (예: 전위, 온도)  
* $\mathcal{N}[\cdot]$: 미분 연산자 (Differential Operator)  
* $f(x, t)$: 소스 항 (Source term)  

이때 신경망 모델을 $\hat{u}(x, t; \theta)$라고 할 때, 잔차 함수 $r$은 다음과 같이 정의된다.  

$$r(x, t; \theta) := \mathcal{N}[\hat{u}(x, t; \theta)] - f(x, t)$$  

신경망 $\hat{u}$가 물리 법칙을 완벽히 만족한다면,  
잔차 $r$은 모든 영역에서 0이 되어야 한다.  
따라서 PINN의 학습 목표는 이 잔차 $r$을 0에 수렴시키는 파라미터 $\theta$를 찾는 것이다.  

## 3. 손실 함수(Loss Function)의 구성

PINN의 전체 손실 함수 $\mathcal{L}(\theta)$는 데이터 손실과 물리 손실의 가중 합으로 구성된다.  

$$\mathcal{L}(\theta) = w_{data}\mathcal{L}_{data} + w_{physics}\mathcal{L}_{physics}$$

1.  데이터 손실 ($\mathcal{L}_{data}$): 초기 조건 및 경계 조건(Boundary Condition)에 해당한다. 실제 관측값 또는 경계값이 존재하는 지점($N_u$)에서의 예측 오차를 계산한다.  
    $$\mathcal{L}_{data} = \frac{1}{N_u} \sum_{i=1}^{N_u} | \hat{u}(x_u^i, t_u^i) - u^i |^2$$

2.  물리 손실 ($\mathcal{L}_{physics}$): 도메인 내부의 임의의 좌표점($N_f$, Collocation Points)에서 물리 방정식의 위배 정도를 계산한다. 이곳에는 정답 데이터($u$)가 필요 없으며, 오직 잔차($r$)의 크기만을 최소화한다.  
    $$\mathcal{L}_{physics} = \frac{1}{N_f} \sum_{j=1}^{N_f} | r(x_f^j, t_f^j) |^2$$

## 4. 핵심 기술: 자동 미분(Automatic Differentiation)

잔차를 계산하기 위해서는 신경망의 출력 $\hat{u}$를 입력 변수($x, y, t$)로 미분해야 한다. PINN은 수치 미분(Numerical Differentiation)의 근사 오차 문제를 해결하기 위해 **자동 미분(Automatic Differentiation, AD)**을 사용한다.

AD는 딥러닝 프레임워크(PyTorch, TensorFlow 등)의 계산 그래프(Computational Graph)를 통해 연쇄 법칙(Chain Rule)을 적용한다. 이를 통해 입력 좌표에 대한 출력의 미분 계수(예: $\frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}$)를 근사치가 아닌 해석학적으로 정확한 값으로 산출한다.  

즉, 모델이 선정된 함수의 조건을 강제적으로 만족시키는 방향으로 학습되게 된다.  

## 5. 모델 아키텍처 (Architecture)

PINN은 일반적으로 완전 연결 신경망(Fully Connected Network, MLP) 구조를 기반으로 하되, 미분 방정식 해결을 위해 다음과 같은 특성을 갖는다.  

- 입력층 (Input Layer): 시공간 좌표 벡터 $(x, y, t)$를 입력으로 받는다.  
- 활성화 함수 (Activation Function): 2계 미분 이상의 고계 도함수 정보를 보존해야 하므로, ReLU와 같은 조각별 선형 함수(Piecewise Linear Function)는 사용할 수 없다. 대신 Tanh, SiLU (Swish), Sin 등 전 구간에서 미분 가능한(Smooth) 함수를 필수적으로 사용한다. ReLU를 사용할 경우 2계 미분값이 0이 되어 라플라스 방정식 등의 학습이 불가능해진다.  
- 출력층 (Output Layer): 해당 좌표에서의 물리량 예측값 $\hat{u}$를 출력한다.  

## 6. 학습 과정 및 라플라스/푸아송 적용

학습은 다음의 절차를 반복하며 진행된다.  

1.  순전파 (Forward Pass): 임의의 좌표 $(x, y)$를 신경망에 입력하여 예측값 $\hat{u}$를 얻는다.  
2.  미분 계산: 자동 미분을 통해 공간에 대한 2계 도함수($u_{xx}, u_{yy}$)를 계산한다.  
3.  잔차 평가:  
    - 라플라스 방정식 : 소스 항이 없으므로, 잔차는 $u_{xx} + u_{yy}$가 된다. 이 값이 0이 되도록 학습한다.  
    - 푸아송 방정식 : 소스 항 $f(x, y)$가 존재하므로, 잔차는 $(u_{xx} + u_{yy}) - f(x, y)$가 된다. 2계 미분의 합이 소스 항과 일치하도록 학습한다.  
4.  역전파 (Backpropagation) : 계산된 잔차의 제곱합(Loss)을 줄이는 방향으로 가중치를 갱신한다.  

결론적으로 PINN은 물리 방정식을 만족하지 못할 경우 손실값이 증가하는 구조를 통해,  
신경망이 스스로 물리 법칙을 준수하는 함수 형태로 수렴하게 만드는 방법론이다.  

## 결론
이번 정리가 저번에 올린 정리보다 훨신 잘 정리된 것 같다.  
전에껀 한번 읽고 gemini랑 쓴 거고, 이건 4번정도 읽고 gemini랑 쓴 거다.  
하지만 확실히 이해할 수 있어 너무 재미있다.  