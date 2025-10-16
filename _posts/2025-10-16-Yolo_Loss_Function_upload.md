---
layout: single
title:  "Yolo Loss Function"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Yolo 모델의 Loss Function
YOLOv8 모델의 손실 함수는 어떤 단일 함수가 아니라, 여러 개의 손실 함수를 조합하여 사용한다.  
전체 손실은 크게 박스(Box) 손실, 분류(Classification) 손실, DFL(Distribution Focal Loss) 손실 세 가지 요소의 가중치 합으로 계산된다.  
기본 설정된 조합이 대부분의 상황에서 좋은 성능을 보이지만, 특정 목표에 따라 일부를 변경하여 성능을 향상시킬 수 있다.  

$$L_{total} = w_1 \cdot L_{box} + w_2 \cdot L_{cls} + w_3 \cdot L_{dfl}$$

## YOLOv8의 기본 손실 함수

YOLOv8에 기본적으로 사용되는 손실 함수는 다음과 같다.  

- 박스 손실 (Box Loss)
    예측한 바운딩 박스의 위치와 크기가 얼마나 정확한지 측정한다. CIoU (Complete Intersection over Union) Loss와 DFL (Distribution Focal Loss)을 함께 사용하여 정확하고 빠른 학습을 돕는다. CIoU는 박스의 겹치는 영역, 중심점 거리, 가로세로 비율을 모두 고려하며, DFL은 박스 좌표를 확률 분포로 학습하여 예측을 더 정밀하게 만든다.  

- 분류 손실 (Classification Loss)
    탐지된 객체가 어떤 클래스에 속하는지 정확하게 분류했는지 측정한다. BCEWithLogitsLoss (Binary Cross-Entropy Loss)를 사용하여 각 클래스에 대한 이진 분류를 수행하고 모델의 예측값과 실제 정답 간의 차이를 계산한다.  


### 손실 함수를 변경하면 좋은 경우  

기본 설정도 우수하지만, 특정 상황에서는 손실 함수를 변경하여 성능을 개선할 수 있다.  

 - 클래스 불균형이 심할 경우
    특정 클래스의 객체 수가 다른 클래스에 비해 압도적으로 많거나 적을 때, 분류 손실을 Focal Loss로 변경하는 것을 고려할 수 있다. Focal Loss는 맞추기 어려운 소수 클래스의 데이터에 더 큰 가중치를 부여하여 학습을 집중시키는 효과가 있다.  

 - 작은 객체 탐지 성능을 높이고 싶을 경우
    박스 손실 함수를 CIoU 대신 WIoU (Wise-IoU)와 같은 최신 IoU 기반 손실 함수로 교체하면 작은 객체나 품질이 낮은 객체에 대한 탐지 성능을 높일 수 있다. WIoU는 객체 품질에 따라 동적으로 가중치를 조절하여 전반적인 일반화 성능을 향상시킨다.  

## 결론

대부분의 경우 YOLOv8의 기본 손실 함수 조합을 그대로 사용하는 것이 가장 좋다.  
하지만 클래스 불균형이나 작은 객체 탐지 성능 개선과 같은 뚜렷한 목표가 있다면, 데이터셋의 특성에 맞춰 Focal Loss나 WIoU 같은 대안적인 손실 함수를 적용하여 모델 성능을 최적화할 수 있다.  