---
layout: single
title:  "Yolo모델 아키텍쳐"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 간단하게, 욜로 모델의 흐름을 정리했다
욜로 모델의 아키텍쳐 흐름을 정리하였다.  
입력 이미지를 하나의 행렬 $X_{input}$이라 가정하고 가보자.  

## 1. 백본 (Backbone): 특징 추출 
백본의 목표는 입력 행렬 $X_{input}$에 연속적인 연산을 적용하여 유용한 특징을 담은 여러 개의 작은 행렬(특징 맵)을 만드는 것이다.

#### 1. Convolution 연산
모든 기본 연산은 합성곱(Convolution)이다.  
이는 이미지의 특정 영역에 필터(가중치 행렬 $W$)를 이동시키며 곱하고 더하는(Dot Product) 과정이다.  

* $X$: 입력 특징 맵 (m x n 행렬)  
* $W$: 컨볼루션 필터 (k x k 행렬)  
* $b$: 편향 (스칼라)  
* $Y$: 출력 특징 맵  

연산의 한 지점(i, j)에서의 계산은 다음과 같다.  
$$Y_{i,j} = \sum_{a=0}^{k-1} \sum_{b=0}^{k-1} (W_{a,b} \cdot X_{i+a, j+b}) + b$$

#### 2. C2f (Cross Stage Partial Faster) 모듈
C2f는 입력된 특징 맵 $X_{C2f\_in}$을 처리하여 더 풍부한 특징 맵 $X_{C2f\_out}$을 만든다.  

1.  Split: 입력 $X_{C2f\_in}$을 두 개의 행렬 $X_1$, $X_2$로 나눈다.  
    $$X_{C2f\_in} \rightarrow [X_1, X_2]$$
2.  Bottleneck 연산: $X_2$는 여러 개의 Bottleneck 함수($f_{bottle}$)를 순차적으로 통과한다. 각 Bottleneck 함수는 내부적으로 여러 Convolution 연산의 조합이다.  
    $$X_{2_1} = f_{bottle1}(X_2)$$   $$X_{2_2} = f_{bottle2}(X_{2_1})$$   $$\vdots$$
3.  Concatenate: 모든 중간 결과와 $X_1$을 다시 하나로 합친다.  Concatenate는 행렬들을 채널 차원에서 쌓는 것이다.  
    $$X_{C2f\_out} = \text{Concat}([X_1, X_{2_1}, X_{2_2}, \dots])$$

#### 3. SPPF (Spatial Pyramid Pooling Fast)
SPPF는 입력 특징 맵 $X_{SPPF\_in}$을 받아 다양한 크기의 특징을 통합한다.  

1.  Max Pooling: $f_{maxpool}$ 함수는 특정 영역에서 가장 큰 값만 남기는 연산이다. SPPF는 이 연산을 순차적으로 적용한다.  
    $$P_1 = f_{maxpool}(X_{SPPF\_in})$$   $$P_2 = f_{maxpool}(P_1)$$   $$P_3 = f_{maxpool}(P_2)$$
2.  Concatenate: 원본과 풀링된 결과들을 모두 합친다.  
    $$X_{SPPF\_out} = \text{Concat}([X_{SPPF\_in}, P_1, P_2, P_3])$$


## 2. 넥 (Neck): 특징 융합  

넥은 백본에서 나온 서로 다른 크기의 특징 맵들($F_{small}, F_{medium}, F_{large}$)을 융합한다.  

#### 1. 하향식 경로 (Top-down)
해상도가 낮은 큰 특징 맵($F_{small}$)을 키워 작은 특징 맵($F_{medium}$)과 합친다.  

1.  Upsampling: 특징 맵의 크기를 키우는 연산으로, 전치 컨볼루션(Transposed Convolution), $W^T$을 사용하여 표현할 수 있다.  
    $$F_{small\_up} = W^T \cdot F_{small}$$
2.  Concatenate: 업샘플링된 결과와 백본의 중간 특징 맵을 합친다.  
    $$F_{fused1} = \text{Concat}([F_{small\_up}, F_{medium}])$$

#### 2. 상향식 경로 (Bottom-up)  
하향식 경로에서 만들어진 특징 맵($F_{fused1}$)을 다시 다운샘플링하여 다른 특징 맵과 합친다.  

1.  Downsampling: 일반적인 컨볼루션 연산(Stride > 1)을 통해 특징 맵의 크기를 줄인다.  
    $$F_{fused1\_down} = W * F_{fused1} + b$$
2.  Concatenate: 다운샘플링된 결과와 다른 경로의 특징 맵을 합친다.  
    $$F_{fused2} = \text{Concat}([F_{fused1\_down}, F_{another}])$$


## 3. 헤드 (Head): 최종 예측  

헤드는 넥에서 나온 최종 특징 맵($F_{final}$)을 입력받아 예측을 수행한다.  
이 단계는 2D 행렬을 1D 벡터로 펼친(Flatten) 후, 완전 연결 계층(Fully Connected Layer)을 통과시키는 것과 같다.  

* $X_{flat}$: $F_{final}$을 1차원으로 펼친 벡터  
* $W_{bbox}$, $b_{bbox}$: 위치 예측을 위한 가중치 행렬과 편향 벡터  
* $W_{cls}$, $b_{cls}$: 종류 예측을 위한 가중치 행렬과 편향 벡터  

- 분리형 헤드 (Decoupled Head)  
1.  위치(Bounding Box) 예측:
    $$Y_{bbox} = W_{bbox} \cdot X_{flat} + b_{bbox}$$
2.  종류(Class) 및 신뢰도(Confidence) 예측:
    $$Y_{cls} = W_{cls} \cdot X_{flat} + b_{cls}$$


## 4. 역전파 (Backpropagation): 오차 계산 및 가중치 업데이트  
역전파는 예측값과 실제 정답의 차이(오차)를 계산하고,  
이 오차를 기반으로 각 레이어의 가중치를 거꾸로 업데이트하는 과정이다.  

#### 1. 손실(Loss) 계산
먼저 손실 함수 $L$을 통해 최종 예측값($Y_{bbox}, Y_{cls}$)과 정답($Y_{true\_bbox}, Y_{true\_cls}$) 간의 오차를 계산한다.  
$$L = L_{bbox}(Y_{bbox}, Y_{true\_bbox}) + L_{cls}(Y_{cls}, Y_{true\_cls})$$

#### 2. 헤드(Head)에서의 역전파
손실 $L$을 각 가중치 행렬($W_{bbox}, W_{cls}$)로 미분하여 기울기(gradient)를 계산한다.  
이는 연쇄 법칙(Chain Rule)을 통해 이루어진다.  

1.  가중치 기울기 계산:  
    $$\frac{\partial L}{\partial W_{bbox}} = \frac{\partial L}{\partial Y_{bbox}} \cdot \frac{\partial Y_{bbox}}{\partial W_{bbox}} = \frac{\partial L}{\partial Y_{bbox}} \cdot X_{flat}^T$$   $$\frac{\partial L}{\partial W_{cls}} = \frac{\partial L}{\partial Y_{cls}} \cdot \frac{\partial Y_{cls}}{\partial W_{cls}} = \frac{\partial L}{\partial Y_{cls}} \cdot X_{flat}^T$$
2.  이전 레이어로 전달될 오차 계산:  
    헤드의 입력이었던 $X_{flat}$에 대한 기울기를 계산하여 넥(Neck)으로 오차를 전달한다.ㅜㅜ
    $$\frac{\partial L}{\partial X_{flat}} = W_{bbox}^T \cdot \frac{\partial L}{\partial Y_{bbox}} + W_{cls}^T \cdot \frac{\partial L}{\partial Y_{cls}}$$

#### 3. 넥(Neck)과 백본(Backbone)에서의 역전파
헤드에서 전달받은 오차($\frac{\partial L}{\partial X_{flat}}$)를 다시 2D 행렬 형태로 복원하고, 순전파의 역순으로 오차를 계속 전파한다.

 - Concatenate의 역전파: 합쳐졌던 기울기를 원래의 형태로 다시 나눈다.  
 - Convolution의 역전파: 입력 $X$와 가중치 $W$에 대한 기울기를 계산한다. 가중치에 대한 기울기는 가중치 업데이트에 사용되고, 입력에 대한 기울기는 이전 레이어로 전달된다.  
    $$\frac{\partial L}{\partial W} \text{ (계산)}, \quad \frac{\partial L}{\partial X} \text{ (이전 레이어로 전달)}$$

이 과정이 백본의 가장 첫 번째 레이어까지 반복된다.  

## 4. 가중치 업데이트
각 레이어에서 계산된 가중치 기울기를 사용하여 경사 하강법(Gradient Descent)으로 모든 가중치를 업데이트한다.  

- $\eta$: 학습률(Learning Rate)  

$$W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}$$

이 순전파와 역전파 과정을 수없이 반복하며 모델은 점차 정답을 더 잘 예측하도록 학습된다.  

## 결론
대충 이런 느낌으로 진행된다.  
숫자 대입해서 연산하는걸 좋아하는데, 여기서부턴 한계가 느껴진다.  