---
layout: single
title:  "L1, L2 Norm"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## L1, L2 Norm 규제

머신러닝, 특히 선형 회귀 모델에서 정규화는 모델의 과적합을 방지하기 위해 사용되는 핵심 기법이다.  
과적합은 모델이 훈련 데이터에 너무 복잡하게 적응하여, 가중치(weights) $\mathbf{w}$가 비정상적으로 큰 값을 갖게 되는 현상이다.  
이게 과해지면 모델을 사용할 때 학습되지 않은 데이터가 들어오면 재대로 대응하지 못하고 이상한 값을 출력하는 문제가 발생한다.  

L1, L2 정규화는 이런 과적합을 막기 위해 손실 함수(Loss Function)에 가중치 벡터 $\mathbf{w}$의 크기를 측정하는 페널티 항(Penalty Term)**을 추가함으로써 가중치가 너무 커지는 것을 제한한다.  
이때 페널티 항으로 L1 Norm 또는 L2 Norm이 주로 사용된다.  
L infinity 는 L1, L2 와 함께 나오는 개념이지만, 사각형의 특징을 가져 정규화에 사용되지 않으며 손실함수 계산기에만 사용된다.  

기본적인 선형 회귀의 비용 함수(잔차 제곱합, RSS)는 다음과 같이 벡터와 행렬로 표현된다.  
$$J_{RSS}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$
여기서 $\mathbf{X}$는 특성 행렬, $\mathbf{y}$는 타겟 벡터, $\mathbf{w}$는 가중치 벡터이다.  


## L2 정규화 (Ridge 회귀)

L2 정규화는 페널티 항으로 가중치 벡터 $\mathbf{w}$의 L2 Norm의 제곱을 사용한다.  
L2 Norm은 유클리드 거리(Euclidean distance)를 의미한다.  

* L2 Norm (제곱): $R_{L2}(\mathbf{w}) = \|\mathbf{w}\|_2^2 = \sum_{i=1}^{n} w_i^2 = \mathbf{w}^T \mathbf{w}$
* Ridge 비용 함수:
    $$J_{Ridge}(\mathbf{w}) = J_{RSS}(\mathbf{w}) + \lambda R_{L2}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$
    $$J_{Ridge}(\mathbf{w}) = (\mathbf{X}\mathbf{w} - \mathbf{y})^T (\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}^T \mathbf{w}$$

여기서 $\lambda$는 정규화의 강도를 조절하는 하이퍼파라미터이다.  

## 선형대수학적 증명

Ridge 회귀의 최적 가중치 $\mathbf{w}$는 비용 함수 $J_{Ridge}(\mathbf{w})$를 $\mathbf{w}$에 대해 미분하여 그레디언트를 0으로 설정함으로써 찾을 수 있다.  

1.  비용 함수 전개:  
    $J_{Ridge}(\mathbf{w}) = (\mathbf{w}^T \mathbf{X}^T - \mathbf{y}^T)(\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}^T \mathbf{w}$
    $J_{Ridge}(\mathbf{w}) = \mathbf{w}^T \mathbf{X}^T \mathbf{X}\mathbf{w} - \mathbf{w}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X}\mathbf{w} + \mathbf{y}^T \mathbf{y} + \lambda \mathbf{w}^T \mathbf{w}$

2.  그레디언트 계산:  
    $\nabla_{\mathbf{w}} J_{Ridge}(\mathbf{w}) = \frac{\partial J_{Ridge}}{\partial \mathbf{w}}$
    * $\frac{\partial}{\partial \mathbf{w}} (\mathbf{w}^T \mathbf{A} \mathbf{w}) = 2\mathbf{A}\mathbf{w}$ (단, $\mathbf{A}$는 대칭 행렬)
    * $\frac{\partial}{\partial \mathbf{w}} (\mathbf{b}^T \mathbf{w}) = \mathbf{b}$
    * $\mathbf{w}^T \mathbf{X}^T \mathbf{y}$와 $\mathbf{y}^T \mathbf{X}\mathbf{w}$는 스칼라이며 서로 동일한 값(전치 관계)이다.
    * 따라서 $\nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{X}^T \mathbf{X}\mathbf{w}) = 2\mathbf{X}^T \mathbf{X}\mathbf{w}$ (이때 $\mathbf{X}^T \mathbf{X}$는 대칭 행렬이다.)  
    * $\nabla_{\mathbf{w}} (-2\mathbf{y}^T \mathbf{X}\mathbf{w}) = -2\mathbf{X}^T \mathbf{y}$ (스칼라의 전치를 이용)
    * $\nabla_{\mathbf{w}} (\lambda \mathbf{w}^T \mathbf{w}) = 2\lambda \mathbf{I} \mathbf{w}$

    이를 조합하면 그레디언트는 다음과 같다.  
    $$\nabla_{\mathbf{w}} J_{Ridge}(\mathbf{w}) = 2\mathbf{X}^T \mathbf{X}\mathbf{w} - 2\mathbf{X}^T \mathbf{y} + 2\lambda \mathbf{I} \mathbf{w}$$

3.  최적 해 유도:  
    그레디언트를 0으로 설정한다.  
    $2\mathbf{X}^T \mathbf{X}\mathbf{w} - 2\mathbf{X}^T \mathbf{y} + 2\lambda \mathbf{I} \mathbf{w} = \mathbf{0}$
    $\mathbf{X}^T \mathbf{X}\mathbf{w} + \lambda \mathbf{I} \mathbf{w} = \mathbf{X}^T \mathbf{y}$
    $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{X}^T \mathbf{y}$

    최종적인 Ridge 해 (정규 방정식):  
    $$\mathbf{w}_{Ridge} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

## 의미

* 역행렬의 존재 보장:  
    표준적인 OLS(Ordinary Least Squares)의 해는 $\mathbf{w}_{OLS} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$이다.  
    하지만 특성 간 다중공선성(Multicollinearity)이 높거나 특성의 수(n)가 샘플 수(m)보다 많으면 $\mathbf{X}^T \mathbf{X}$ 행렬이 특이 행렬(Singular Matrix)이 되어 역행렬이 존재하지 않거나 매우 불안정해진다.  
* $\mathbf{X}^T \mathbf{X}$는 항상 양의 준정부호 행렬이다 (모든 고유값이 0 이상).  
* L2 정규화는 $\mathbf{X}^T \mathbf{X}$의 대각 성분에 $\lambda$ (양수)를 더하는 $\lambda \mathbf{I}$ 항을 추가한다.  
* $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ 행렬은 모든 고유값이 $\lambda$만큼 이동하여 항상 양의 정부호 행렬(Positive Definite)이 된다.  
* 양의 정부호 행렬은 항상 역행렬이 존재한다. 따라서 L2 정규화는 $\lambda > 0$인 한 항상 안정적이고 유일한 해를 보장한다.  
* $\lambda$가 커질수록 $\mathbf{w}$의 값들은 0에 가까워지지만, 0이 되지는 않는다.  


## L1 정규화 (Lasso 회귀)

L1 정규화는 페널티 항으로 가중치 벡터 $\mathbf{w}$의 L1 Norm을 사용한다.  
L1 Norm은 맨해튼 거리(Manhattan distance)를 의미한다.  

* L1 Norm: $R_{L1}(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum_{i=1}^{n} |w_i|$
* Lasso 비용 함수:  
    $$J_{Lasso}(\mathbf{w}) = J_{RSS}(\mathbf{w}) + \lambda R_{L1}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \sum_{i=1}^{n} |w_i|$$

#### 선형대수학적 해석 (기하학적 접근)

L1 정규화는 L2와 달리 폐쇄형 해(Closed-form solution)가 존재하지 않는다.  

* 미분 불가능성: L1 Norm의 절댓값 항 $|w_i|$는 $w_i = 0$ 지점에서 미분이 불가능하다.  
* 해석: 이 미분 불가능성 때문에 $\mathbf{w}_{Ridge}$와 같은 간단한 행렬식으로 해를 표현할 수 없다. 해를 구하기 위해서는 좌표 하강법이나 LARS와 같은 수치적 최적화 알고리즘이 필요하다.  

L1의 효과는 기하학적으로 해석할 수 있다.  
정규화는 $\lambda$를 사용하는 대신, 특정 상수 $t$에 대해 $\|\mathbf{w}\| \le t$ 라는 제약 조건을 두는 것과 동일하다.  

1.  제약 조건의 형태:
    * L2 (Ridge): $\|\mathbf{w}\|_2^2 \le t$ $\Rightarrow$ $\sum w_i^2 \le t$. 이는 원점 중심의 원(Circle) 또는 초구(Hypersphere) 형태의 제약 영역을 만든다.  
    * L1 (Lasso): $\|\mathbf{w}\|_1 \le t$ $\Rightarrow$ $\sum |w_i| \le t$. 이는 원점 중심의 마름모(Diamond) 또는 초팔면체(Cross-polytope) 형태의 제약 영역을 만든다.  

    

2.  최적 해의 위치:  
    * 최적 해 $\mathbf{w}$는 RSS의 등고선(Contour lines)**(타원 형태)과 페널티 제약 영역이 처음 만나는 지점이다.  
    * L2: 원 형태의 제약 영역은 뾰족한 모서리가 없다. 따라서 등고선(타원)과 만나는 지점은 일반적으로 축(axis) 위가 아니다. 즉, 가중치 $w_i$가 0이 되기 어렵다.  
    * L1: 마름모 형태의 제약 영역은 각 축 위에 뾰족한 모서리(Corner)를 가진다. RSS 등고선(타원)이 확장되다가 이 제약 영역과 만날 때, 뾰족한 모서리에서 만날 확률이 매우 높다.  

3.  희소성(Sparsity) 발생:  
    * L1 제약 영역의 모서리는 특정 가중치 $w_i$가 정확히 0이 되는 지점이다 (예: 2차원에서 $(t, 0), (0, t)$ 등).  
    * 따라서 L1 정규화는 최적 해를 찾을 때 많은 가중치를 정확히 0으로 만드는 경향이 있다.  
    * 이는 불필요한 특성을 모델에서 완전히 제거하는 특성 선택(Feature Selection) 효과를 가진다.  


## 결론

L2 정규화 (Ridge): $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ 행렬의 역행렬 존재를 보장하여 안정적인 해를 도출한다.  
모든 가중치를 0에 가깝게 축소시키지만 0으로 만들지는 않는다.  
L1 정규화 (Lasso): 절댓값 항으로 인해 미분이 불가능하여 폐쇄형 해가 없다.  
기하학적으로 마름모 형태의 제약 조건이 뾰족한 모서리를 가지므로, 일부 가중치를 정확히 0으로 만들어 희소성(Sparsity)을 유도하고 특성 선택을 수행한다.  