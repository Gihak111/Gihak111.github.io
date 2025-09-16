---
layout: single
title:  "Gradient Descent method"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 경사 하강법

머신러닝 모델 학습의 근본적인 목표는 주어진 데이터를 가장 잘 설명하는 최적의 파라미터(Parameter)를 찾는 것이다.  
여기서 '최적'이란 모델의 예측이 실제 정답과 가장 가깝게 일치하는 상태를 의미한다.  

경사 하강법(Gradient Descent)은 이 최적의 파라미터를 찾기 위해 사용되는 핵심적인 1차(first-order) 반복 최적화 알고리즘이다.  
이 알고리즘은 모델의 '성능'을 정량적으로 측정하는 지표를 최소화하는 방향으로 파라미터를 점진적으로 업데이트한다.  

### 2. 비용 함수 J(θ)란 무엇인가?

모델의 성능을 측정하는 지표를 비용 함수(Cost Function)라 하며, 기호로는 주로 $J(\theta)$를 사용한다.  
이는 손실 함수(Loss Function) 또는 목적 함수(Objective Function)라고도 불린다.  

이 함수의 핵심 역할은 현재 모델의 파라미터 $\theta$가 얼마나 나쁜지를 측정하는 것이다.  
즉, 모델의 예측이 실제 정답과 얼마나 차이가 나는지를 하나의 숫자(스칼라 값)로 나타낸다.

* $J(\theta)$의 값이 **크다**: 모델의 예측이 실제 정답과 많이 다르다는 의미이다. (성능이 나쁘다)  
* $J(\theta)$의 값이 **작다**: 모델의 예측이 실제 정답과 거의 일치한다는 의미이다. (성능이 좋다)  

따라서 머신러닝 학습의 목표는 $J(\theta)$의 값을 가능한 0에 가깝게 만드는 최적의 파라미터 $\theta$를 찾는 것이다.  

$J(\theta)$는 해결하고자 하는 문제의 종류에 따라 다른 형태를 사용하며, 대표적인 예는 다음과 같다.  

* 평균 제곱 오차 (Mean Squared Error, MSE): 주로 연속된 값을 예측하는 회귀(Regression) 문제에서 사용된다. 오차의 제곱에 대한 평균값으로, 오차가 클수록 더 큰 불이익을 부여한다.  
    $$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\text{예측값}^{(i)} - \text{실제값}^{(i)})^2$$

* 교차 엔트로피 (Cross-Entropy): 주로 범주를 예측하는 분류(Classification) 문제에서 사용된다. 모델이 정답을 확신을 가지고 틀릴 경우 $J$ 값이 무한대에 가깝게 치솟으며 엄청나게 큰 불이익을 주는 특징이 있다.  
    $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

### 3. 경사 하강법의 선형대수학적 접근

경사 하강법은 비용 함수 $J(\theta)$라는 지형에서 가장 낮은 지점을 찾아가는 과정이며, 이 과정은 선형대수학의 벡터 연산을 통해 명확하게 설명된다.  

#### 3.1. 파라미터 벡터와 그라디언트 벡터

모델의 모든 파라미터(가중치와 편향)는 $n$차원 공간 $\mathbb{R}^n$에 존재하는 하나의 **파라미터 벡터** $\theta$로 표현할 수 있다.  

$$\theta = \begin{pmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{pmatrix} \in \mathbb{R}^n$$

**그라디언트(Gradient)** $\nabla J(\theta)$는 비용 함수 $J$를 각 파라미터 $\theta_i$로 편미분(partial derivative)한 결과를 원소로 가지는 벡터이다.  

$$\nabla J(\theta) = \begin{pmatrix} \frac{\partial J}{\partial \theta_1} \\ \frac{\partial J}{\partial \theta_2} \\ \vdots \\ \frac{\partial J}{\partial \theta_n \end{pmatrix}$$

이 그라디언트 벡터 $\nabla J(\theta)$는 $n$차원 파라미터 공간의 한 점 $\theta$에서 함수 $J$의 값이 **가장 가파르게 증가하는 방향(오르막길)**을 가리킨다.  
이 벡터의 크기(magnitude)는 해당 방향으로의 변화율(기울기의 크기)을 의미한다.  

#### 3.2. 파라미터 업데이트 규칙

경사 하강법은 비용 함수를 최소화하기 위해 그라디언트가 가리키는 방향의 **반대 방향(내리막길)**으로 파라미터 벡터를 이동시킨다.  
이 과정은 다음의 업데이트 규칙을 반복적으로 수행함으로써 이루어진다.  

$$\theta_{\text{new}} := \theta_{\text{old}} - \eta \nabla J(\theta_{\text{old}})$$

이 수식은 다음과 같은 선형대수학의 벡터 연산으로 해석할 수 있다.  

1.  $\theta_{\text{old}}$: 파라미터 공간에서 현재 위치를 나타내는 **위치 벡터**이다.  
2.  $\eta \nabla J(\theta_{\text{old}})$: 오르막길 방향을 나타내는 그라디언트 벡터 $\nabla J(\theta_{\text{old}})$에 학습률(보폭 크기)이라는 **스칼라** $\eta$를 곱하는 **스칼라 곱** 연산이다.  
이는 '이동할 단계(step)'에 해당하는 벡터를 계산하는 과정이다.  
3.  $\theta_{\text{old}} - (\dots)$: 현재 위치 벡터에서 위에서 계산한 '이동할 단계' 벡터를 빼는 **벡터 뺄셈** 연산이다.  
이로써 비용이 더 낮은 방향으로 이동한 새로운 위치 벡터 $\theta_{\text{new}}$가 결정된다.  

### 4. 결론

결론적으로 경사 하강법은 모델의 성능을 정량화한 비용 함수 $J(\theta)$를 최소화하기 위한 알고리즘이다.  
이 과정은 모델의 모든 파라미터를 하나의 벡터 $\theta$로 간주하고, 비용 함수의 값이 가장 가파르게 감소하는 방향을 알려주는 그라디언트 벡터 $\nabla J(\theta)$를 계산하여, 현재 위치에서 다음 위치로 이동하는 반복적인 벡터 연산을 통해 수행된다.  
더 궁금하면, 아래 논문 참고하자  
[An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)  
[Support Vector Machine Classifier viaL0/1 Soft-Margin Loss](https://arxiv.org/abs/1912.07418)  
