---
layout: single
title:  "F1스코어와 평가지표"
categories: "Deep_Learning"
tag: "model-optimization-deployment"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### F1 스코어와 관련된 평가 지표  

#### 1. F1 스코어란?  
- F1 스코어는 정밀도(Precision)와 재현율(Recall)의 조화 평균을 계산한 값입니다.  
- 정의:  
  \[
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]
- 사용 목적:  
  - 데이터의 불균형이 있는 상황에서 모델 성능 평가.  
  - 정밀도와 재현율 사이의 균형 확인.  


### 2. F1 스코어를 구성하는 주요 요소  
#### (1) 정밀도 (Precision)  
- 정의: 모델이 양성으로 예측한 것 중 실제로 양성인 비율.  
  \[
  \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
  \]
- 목적: 예측 결과의 정확성 평가.  
  - 높은 Precision: 잘못된 양성 예측(FP)이 적음.  

#### (2) 재현율 (Recall)  
- 정의: 실제 양성인 것 중 모델이 양성으로 예측한 비율.  
  \[
  \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]
- 목적: 모델이 얼마나 실제 양성을 놓치지 않았는지 평가.  
  - 높은 Recall: 놓친 양성(FN)이 적음.  

#### (3) F1 스코어의 해석
- F1 스코어는 Precision과 Recall의 균형을 평가:  
  - Precision과 Recall이 비슷할수록 F1 값이 높음.  
  - 두 값 중 하나가 매우 낮으면 F1 스코어도 낮아짐.  
- 완벽한 모델: F1 스코어 = 1 (Precision = Recall = 1).  
- 무의미한 모델: F1 스코어 = 0.  


### 3. F1 스코어와 유사하거나 관련된 다른 평가 지표  

#### (1) Accuracy (정확도)   
- 정의: 전체 데이터 중 모델이 정확히 예측한 비율.  
  \[
  \text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
  \]
- 한계:  
  - 데이터가 불균형일 경우, 정확도가 높아도 의미가 없을 수 있음.  
  - 예: 양성이 1%인 데이터에서 모든 샘플을 음성으로 예측해도 99%의 정확도를 달성.  

#### (2) Specificity (특이도)  
- 정의: 실제 **음성**인 것 중 모델이 음성으로 올바르게 예측한 비율.  
  \[
  \text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}}
  \]
- 목*: 모델이 음성을 얼마나 잘 예측하는지 평가.  

#### (3) Balanced Accuracy  
- 정의: Accuracy의 균형 조정 버전으로, 클래스 불균형 문제를 해결.  
  \[
  \text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}
  \]
- 특징: 데이터 불균형 상황에서 정확한 평가 가능.  

#### (4) ROC-AUC (Receiver Operating Characteristic - Area Under Curve)  
- 정의: TPR(재현율)과 FPR(1 - 특이도) 간의 관계를 나타낸 곡선의 아래 면적.  
  \[
  \text{AUC} = \int_{0}^{1} \text{ROC Curve}
  \]
- 목적: 모델의 전반적인 분류 능력 평가.  
  - AUC = 1: 완벽한 모델.  
  - AUC = 0.5: 랜덤 예측 수준.  

#### (5) Matthews Correlation Coefficient (MCC)  
- 정의: TP, TN, FP, FN 간의 상관관계를 측정.  
  \[
  \text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}
  \]
- 목적: 데이터 불균형 상황에서도 성능 평가 가능.  

#### (6) Fβ 스코어  
- 정의: Precision과 Recall 사이의 가중치를 조정한 F1 스코어.  
  \[
  F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}
  \]
  - \(\beta = 1\): F1 스코어 (Precision과 Recall 동일 가중치).  
  - \(\beta > 1\): Recall에 더 높은 가중치.  
  - \(\beta < 1\): Precision에 더 높은 가중치.  

#### (7) Log Loss (Logarithmic Loss)    
- 정의: 예측 확률과 실제 레이블 간의 오차를 계산.  
  \[
  \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
- 목적: 확률 기반 예측 평가.  

#### (8) G-Mean (Geometric Mean)  
- 정의: 민감도(재현율)와 특이도의 기하 평균.  
  \[
  G\text{-Mean} = \sqrt{\text{Sensitivity} \cdot \text{Specificity}}
  \]
- 목적: 데이터 불균형 문제를 다룰 때 사용.  


### 4. 활용 요약
- F1 스코어: Precision과 Recall 사이의 균형을 강조.  
- ROC-AUC: 전체적인 분류 성능.  
- Specificity와 Sensitivity: 음성과 양성 각각의 예측 능력 평가.  
- MCC: 데이터 불균형 상황에서 전체적인 모델 성능 평가.  
- Fβ 스코어: Precision과 Recall의 가중치 조정 필요 시.  

이 지표들은 문제의 특성과 평가하고자 하는 모델의 목적에 따라 적절히 선택해야 한다.  