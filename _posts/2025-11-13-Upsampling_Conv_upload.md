---
layout: single
title:  "Upsampling Conv"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Decoder와 해상도 복원의 문제

CNN 기반의 오토인코더(AE), U-Net, GAN과 같은 생성 모델에서 인코더(Encoder)는 입력 이미지의 공간적 해상도(Spatial Resolution)를 줄여가며 핵심 특징을 압축한다.  
반대로, 디코더(Decoder)는 이 압축된 잠재 벡터(Latent Vector)로부터 원본 이미지의 해상도를 다시 복원(Upsampling)하는 임무를 가진다.  

이 'Upsampling'을 수행하는 방식은 크게 두 가지로 나뉜다.  
1.  **역합성곱 (Transposed Convolution)**: 업샘플링 과정을 '학습'하는 단일 연산  
2.  **업샘플링 + 합성곱 (Upsampling + Conv)**: 업샘플링(고정)과 특징 변환(학습)을 분리한 연산  

과거에는 Transposed Convolution이 많이 사용되었지만, 최근의 고성능 모델(StyleGAN 등)들은 대부분 Upsampling + Conv 방식을 채택한다.  
이 글에서는 두 방식의 차이와 왜 후자가 더 선호되는지 그 효율성과 안정성의 관점에서 자세히 알아보자.  


## 2. 역합성곱 (Transposed Convolution)

'Deconvolution'이라는 용어로도 알려져 있으나, 이는 신호 처리의 실제 'Deconvolution'과 의미가 달라 혼동을 준다.  
'Transposed Convolution'이 더 정확한 용어이다.  

### 선형대수학 관점

표준 합성곱(Standard Conv)이 입력 $\mathbf{x}$를 출력 $\mathbf{y}$로 매핑하는 행렬 $\mathbf{C}$를 사용한 선형 변환($\mathbf{y} = \mathbf{C} \mathbf{x}$)이라면,  
Transposed Convolution은 이 변환의 전치(Transpose), 즉 $\mathbf{C}^T$를 사용하여 더 작은 차원의 $\mathbf{y}$를 더 큰 차원의 $\hat{\mathbf{x}}$로 복원($\hat{\mathbf{x}} = \mathbf{C}^T \mathbf{y}$)하는 연산이다.  

### 연산의 문제점: 체크보드 현상 (Checkerboard Artifacts)

Transposed Convolution의 가장 치명적인 단점은 결과물에 체크보드(바둑판) 무늬의 노이즈가 발생한다는 점이다.  



이 문제는 연산 방식 자체의 한계에서 비롯된다.  
Transposed Convolution은 stride($s$) > 1일 때, 입력 피처맵 픽셀 사이에 0을 채워넣고(zero-padding) 여기에 커널(kernel) 연산을 수행한다.  

이때 커널($k$)이 겹치는(overlap) 영역이 발생하는데, 커널의 가중치는 공유된다.  
이로 인해 커널이 "덧칠"되는 중앙 영역과 그렇지 않은 가장자리 영역 간에 값의 불균형이 발생한다.  
이 불균형이 반복적인 패턴으로 나타나는 것이 바로 체크보드 현상이다.  

쉽게 비유하자면:
붓(커널)에 물감을 묻혀 일정한 간격(stride)으로 도화지에 점을 찍으며 그림을 그린다고 상상할 수 있다.  
붓이 찍히는 영역들이 겹칠 때, 중앙은 물감이 두 배로 칠해지고(가중치가 더해짐) 가장자리는 한 번만 칠해질 것이다.  
이 겹침 패턴이 일정한 격자무늬를 만들어내는 것과 같다.  

이 현상은 특히 $k$가 $s$로 나누어떨어지지 않을 때 더욱 심해진다.  
이는 모델이 고주파 노이즈를 생성하게 만들어 이미지의 품질을 심각하게 저하시킨다.  


## 3. 업샘플링 + 합성곱 (Upsampling + Conv)

이 방식은 Transposed Convolution이 한 번에 하려던 두 가지 작업을 명확하게 분리(Decouple)한다.  

1.  **Stage 1: 해상도 증가 (Upsampling)**
    * 먼저, **비-학습(Non-learnable)** 방식의 고정된 알고리즘을 사용해 피처맵의 해상도를 2배(혹은 원하는 배율)로 늘린다.  
    * **최근접 이웃 보간법 (Nearest Neighbor):** 가장 빠르지만, 픽셀이 깨지는 '깍두기' 현상이 발생한다.  
    * **쌍선형 보간법 (Bilinear Interpolation):** 인접 픽셀 간의 값을 선형적으로 보간하여 부드러운 결과물을 만든다. 체크보드 현상을 피하는 데 가장 효과적이며 널리 쓰인다.  

2.  **Stage 2: 특징 변환 (Convolution)**
    * 해상도가 늘어난 피처맵을 입력으로 받아, 일반적인 **표준 `Conv2D` 연산** (주로 `kernel=3x3, stride=1`)을 적용한다.  
    * 이 합성곱 연산의 역할은 해상도를 변경하는 것이 아니라, 업샘플링된 피처맵 위에서 **특징을 다시 정제하고 학습**하는 것이다.  


## 4. 왜 'Upsampling + Conv'가 더 효율적인가?

'효율적'이라는 것은 단순히 속도가 빠르다는 의미를 넘어, **더 적은 문제로 더 높은 품질의 결과를 안정적으로 학습**할 수 있음을 의미한다.  

### 1. 체크보드 문제의 근본적 해결 (품질)  

이것이 가장 결정적인 이유이다.  

Transposed Convolution은 '제로 패딩'된 희소한(sparse) 맵에 학습 가능한 커널을 적용하여 체크보드 문제를 유발했다.  

반면, 'Upsampling + Conv' 방식은 Bilinear Interpolation을 통해 이미 부드럽고 조밀하게(dense) 채워진 맵을 만든다.  
그 후 적용되는 표준 `Conv2D (stride=1)`는 커널 겹침으로 인한 불균형을 야기하지 않는다.  

결과적으로 체크보드 현상 없이 훨씬 깨끗하고 자연스러운 고해상도 결과물을 얻을 수 있다.  

### 2. 학습의 안정성 및 단순화 (학습 효율)

Transposed Convolution 연산은 '해상도를 높이는 방법'과 '특징을 변환하는 방법'이라는 두 가지 복잡한 임무를 하나의 커널이 동시에 학습해야 했다.  

'Upsampling + Conv'는 이 문제를 분리한다.  
- 해상도 증가: Bilinear Interpolation (고정된 연산)이 담당한다. 모델이 신경 쓸 필요가 없다.  
- 특징 변환: 표준 `Conv2D` (학습)가 담당한다.  

모델은 오직 '특징을 잘 정제하는 법'만 배우면 되므로, 학습 목표가 단순해지고(Simpler learning dynamics) 훨씬 더 안정적으로 수렴하며 고품질의 결과에 도달할 수 있다.  

### 3. 계산 및 파라미터 효율성 (자원 효율)  

* Bilinear Upsampling 연산은 학습할 파라미터가 없는(Parameter-free), 매우 가벼운(computationally cheap) 연산이다.  
* 이후 수행되는 `3x3 Conv2D`는 현대 GPU에서 고도로 최적화된 표준 연산이다.  

복잡한 Transposed Convolution 연산을 한 번 수행하는 것보다, 가벼운 Upsampling과 최적화된 Conv 연산을 순차적으로 수행하는 것이 종종 더 빠르거나 비슷한 비용으로 더 나은 결과를 제공한다.  

## 5. 결론

- Transposed Convolution은 업샘플링을 학습 가능한 단일 연산으로 처리하려 했지만, 본질적으로 체크보드 현상(Artifacts)이라는 심각한 품질 저하 문제를 안고 있다.  
- Upsampling + Conv (Bilinear + 3x3 Conv)는 '해상도 복원'과 '특징 변환'이라는 두 개의 작업을 분리(Decouple)한다.
- 이 방식은 체크보드 현상을 원천적으로 방지하고, 학습 과정을 단순화하여 더 안정적이며, 결과적으로 더 높은 품질의 이미지를 효율적으로 생성하는 현대적인 표준 방식이다.  