---
layout: single
title:  "파이썬으로 만드는 간단한 AI. 로지트틱 회귀"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 파이썬으로 만드는 간단한 로지스틱 회귀 AI 코드 입니다.  
전에 파이썬으로 인공 신경망을 구성해 봤었죠.  
오늘은 로지스틱 회귀로 구조를 변경해서 작성해 보겠습니다.  
간단히 소개 하자면,  
모델 구조: 로지스틱 회귀는 단일 레이어로 이루어져 있으며, 각 입력 특성에 대한 가중치와 바이어스가 있습니다.  
활성화 함수: 시그모이드 함수를 활성화 함수로 사용합니다.  
손실 함수: 로그 손실 함수를 사용합니다.  
학습 알고리즘: 경사 하강법을 사용하여 가중치와 바이어스를 업데이트합니다.  
장점  
간단하고 이해하기 쉽습니다.  
속도가 빠르며 작은 데이터셋에서 잘 작동합니다.  
단점  
비선형적인 관계를 학습할 수 없습니다.  
복잡한 패턴을 모델링하기에는 부족합니다.  


```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = (iris.target != 0) * 1  # 이진 분류를 위해 타겟 레이블을 0과 1로 변환

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# 학습 과정 시각화
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy over Epochs')
plt.show()

# 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')
print('정확도 : {}%'.format(test_accuracy * 100))


```

위 코드를 실행시키려면 다음 명령어를 실행해 다운 받아야 합니다.  
scikitlearn 설치  
pip install scikit-learn  
matplotlib 설치  
pip install matplotlib  
TensorFlow 설치  
pip install tensorflow  
