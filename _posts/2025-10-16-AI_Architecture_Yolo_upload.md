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


## 숫자 넣어서도 해 보자.  
단, 자세한 숫자를 넣어선 예제가 안되므로, 크기만 가지고 계산해 보겠다.  

## 1. 순전파 (Forward Propagation): 데이터의 흐름
순전파는 입력 이미지가 신경망을 통과하며 최종 예측값으로 변환되는 과정이다.  
각 단계에서 데이터(행렬)의 형태가 어떻게 변하는지 자세히 추적한다.  

#### 1. 백본 (Backbone): 특징 추출 및 축소
백본은 고해상도의 원본 이미지에서 저해상도의 핵심 특징 맵을 점진적으로 추출한다.  
- 시작: 입력 이미지 $X_{input}$은 (640, 640, 3) 크기의 행렬이다.  
- Stem Layer (Conv): 첫 번째 컨볼루션 레이어는 큰 보폭(Stride=2)으로 연산을 수행한다.  
    - 입력: $X_{input}$ (640, 640, 3)  
    - 연산: $Y = \text{Conv}(X, W, b)$
    - 출력: 크기는 절반으로 줄고 채널은 증가한 특징 맵 $F_0$ (320, 320, 64)가 생성된다. 이 $F_0$가 다음 레이어의 입력으로 그대로 전달된다.  
- C2f 모듈 및 다운샘플링: C2f 모듈은 특징을 풍부하게 만들고, 중간의 다운샘플링 컨볼루션은 특징 맵의 크기를 계속 줄여나간다.  
    - 과정 1: $F_0$ (320, 320, 64)가 C2f와 다운샘플링 Conv를 거쳐 $F_1$ (80, 80, 256) 으로 변환된다. 이 $F_1$은 나중에 넥(Neck)에서 재사용하기 위해 저장된다.  
    - 과정 2: $F_1$이 다음 C2f와 다운샘플링 Conv를 거쳐 $F_2$ (40, 40, 512) 로 변환된다. 이 $F_2$ 또한 저장된다.  
    - 과정 3: $F_2$가 마지막 C2f와 다운샘플링 Conv를 거쳐 $F_3$ (20, 20, 1024) 로 변환된다.  

#### 넥 (Neck): 양방향 특징 융합 및 재구성
넥은 백본의 여러 단계에서 생성된 특징 맵($F_1, F_2, F_3$)을 입력받아, 정보가 아래위로 오가는 양방향 경로를 통해 특징을 정교하게 융합한다.  
이는 다양한 크기의 객체를 효과적으로 탐지하기 위함이다.

- 하향식 경로 (Top-down): 깊은 층의 의미 정보("이것은 자동차 형태이다")를 얕은 층으로 전달하여 특징을 풍부하게 만든다.
- 상향식 경로 (Bottom-up): 의미 정보가 보강된 얕은 층의 정확한 위치 정보("자동차는 이 픽셀 주변에 있다")를 다시 깊은 층으로 전달해 예측 정확도를 높인다.

##### 1. 하향식 경로 (Top-down Path)
1. 백본의 가장 깊은 특징 맵 $F_3$ (20x20)를 업샘플링(Upsample)하여 해상도를 두 배로 높인 $F_{3\_up}$ (40x40)을 만든다.
2. 이 $F_{3\_up}$을 백본의 중간 특징 맵 $F_2$ (40x40)와 채널 축을 기준으로 결합(Concatenate)한다. 두 정보가 합쳐져 채널의 깊이가 매우 깊어진다.
    $$F_{fused1} = \text{Concat}([F_{3\_up}, F_2]) \rightarrow (40, 40, 1536)$$
3. 깊고 복잡해진 특징 맵 $F_{fused1}$을 C2f 모듈로 처리하여 정보를 효과적으로 압축하고 정제한다. 이를 통해 중간 특징 맵 $N_1$ (40x40)을 생성한다.  
4. 이 과정을 한 번 더 반복($N_1$ 을 업샘플링하여 $F_1$ 과 결합)하여 더 높은 해상도의 중간 특징 맵 $N_2$ (80x80)를 생성한다.

##### 2. 상향식 경로 (Bottom-up Path)
이제 하향식 경로에서 만들어진 특징 맵들을 활용하여 반대 방향으로 정보를 전달한다.  
1. 가장 해상도가 높은 $N_2$ (80x80)를 다운샘플링(Downsample)하여 해상도를 절반으로 줄인 $N_{2\_down}$ (40x40)을 만든다.  
2.  $N_{2\_down}$ 을 하향식 경로의 중간 특징 맵 $N_1$ (40x40)과 결합한다.   
    $$F_{fused2} = \text{Concat}([N_{2\_down}, N_1])$$
3. $F_{fused2}$ 를 다시 C2f 모듈로 처리하여, 최종적으로 헤드(Head)로 전달될 3개의 특징 맵 중 하나인 $P_{medium}$ (40x40)을 생성한다.  
4. 이 과정을 반복하여 최종적으로 $P_{small}$ (80x80), $P_{medium}$ (40x40), $P_{large}$ (20x20) 세 가지 크기의 특징 맵을 완성하며, 이것들이 헤드로 전달될 최종 입력 데이터가 된다.  


#### 헤드 (Head): 최종 예측
헤드는 넥에서 완성된 3개의 특징 맵($P_{small}, P_{medium}, P_{large}$) 각각에 대해 독립적으로 예측을 수행한다.  

- 입력: 예시로 $P_{medium}$ (40x40) 특징 맵을 사용한다.  
- 분리형 헤드 (Decoupled Head): 예측의 정확도를 높이기 위해, 하나의 네트워크가 아닌 위치 예측과 종류 예측을 위한 두 개의 분리된 경로로 특징 맵을 전달한다.  
    - 위치 예측 경로: 바운딩 박스(x, y, w, h) 예측에 특화된 Conv 레이어들을 통과한다.  
        $$Y_{bbox\_feat} = \text{Conv}_{bbox}(P_{medium})$$
    - 종류 예측 경로: 객체의 종류(Class) 예측에 특화된 Conv 레이어들을 통과한다.  
        $$Y_{cls\_feat} = \text{Conv}_{cls}(P_{medium})$$
- 최종 출력: 각 경로는 (40x40) 크기의 최종 특징 맵을 출력한다. 이 맵의 각 격자 셀(grid cell)은 해당 위치에서의 예측값을 담고 있다. 예를 들어, (i, j) 위치의 셀은 바운딩 박스 정보(4개 값)와 클래스 확률(80개 값)을 출력한다. 이는 각 셀의 특징 벡터에 대해 선형 변환을 수행하는 것과 같다.  
    $$Y_{pred} = W \cdot X_{cell\_feat} + b$$


## 역전파 (Backpropagation): 오차의 미분과 전파
역전파는 예측과 정답의 차이(오차)를 계산하고, 이 오차를 신경망에 거꾸로 전파하며 각 가중치를 얼마나 수정해야 할지(기울기) 계산하는 과정이다. 순전파의 모든 연산은 미분 가능하므로 연쇄 법칙(Chain Rule)을 적용할 수 있다.

#### 손실(Loss)과 시작점
손실 함수 $L$을 통해 최종 예측값 $Y_{pred}$와 정답 $Y_{true}$간의 오차를 계산한다.  
이 손실을 최종 예측값으로 미분한 $\frac{\partial L}{\partial Y_{pred}}$
가 역전파의 시작점이 되는 최초의 오차 신호이다.  

#### 헤드(Head)에서의 역전파  
헤드의 최종 연산은 선형 변환 $Y = WX+b$와 같다.  
이 식을 기반으로 역전파를 수행한다.  

1. 가중치($W$)에 대한 기울기 계산:  
    - 목표: 손실에 대한 가중치의 영향력, 즉 $\frac{\partial L}{\partial W}$ 를 구한다.  
    - 연쇄 법칙: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W}$
    - 미분: 순전파 식 $Y=WX+b$를 $W$에 대해 미분하면 $\frac{\partial Y}{\partial W} = X^T$ 이다.  
    - 결과: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot X^T$
    - 의미: 가중치의 기울기는 '출력 오차'에 그 출력을 만드는 데 사용된 '입력 데이터'를 곱한 값이다. 이 기울기는 가중치 업데이트에 직접 사용된다.    

2. 이전 레이어 입력($X$)으로 오차 전파:  
    - 목표: 이전 레이어(넥)로 전달할 오차 신호, 즉 $\frac{\partial L}{\partial X}$ 를 구한다.  
    - 연쇄 법칙: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X}$
    - 미분: 순전파 식 $Y=WX+b$를 $X$에 대해 미분하면 $\frac{\partial Y}{\partial X} = W^T$ 이다.  
    - 결과: $\frac{\partial L}{\partial X} = W^T \cdot \frac{\partial L}{\partial Y}$
    - 의미: 이전 레이어로 전달되는 오차는 '출력 오차'에 '가중치 행렬의 전치'를 곱한 값이다. 이는 각 가중치가 출력 오차에 기여한 만큼을 역으로 계산하여 입력 오차를 재구성하는 과정이다.  

#### 넥(Neck)과 백본(Backbone)에서의 역전파
헤드에서 전파된 오차 신호 $\frac{\partial L}{\partial X}$는 순전파의 역순으로 넥과 백본을 거슬러 올라간다.  
- Concatenate의 역전파: 순전파 때 채널 축으로 합쳐졌던 두 특징 맵 $A, B$가 있었다면, 역전파 시에는 들어온 기울기 텐서를 합쳐지기 전의 채널 크기에 맞게 그대로 분할(Slice)하여 각각의 경로로 전달한다.  
- Convolution의 역전파: 헤드와 마찬가지로, 들어온 출력 오차를 이용해 가중치에 대한 기울기($\frac{\partial L}{\partial W}$)와 입력에 대한 기울기($\frac{\partial L}{\partial X}$)를 계산하여 각각 가중치 업데이트와 이전 레이어로의 오차 전파에 사용한다.  
- 활성화 함수(SiLU)의 역전파: 활성화 함수를 $f(x)$라 하면, 출력 오차에 활성화 함수의 도함수(미분값) $f'(x)$을 곱하여 오차를 전달한다. 이는 특정 뉴런이 활성화된 정도에 따라 오차를 조절하여 전달하는 역할을 한다.  
    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x)} \cdot f'(x)$$

## 4. 최종 가중치 업데이트
모든 레이어에서 계산된 가중치 기울기($\frac{\partial L}{\partial W}$)를 사용하여 경사 하강법(Gradient Descent)으로 모든 가중치를 업데이트한다. ($\eta$는 학습률)  
$$W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}$$
이 순전파와 역전파 과정을 수없이 반복하며, 모델은 오차를 줄이는 방향으로 점차적으로 개선되어 더 정확한 예측을 학습하게 된다.  

## 결론
대충 이런 느낌으로 진행된다.  
숫자 대입해서 연산하는걸 좋아하는데, 여기서부턴 한계가 느껴진다.  