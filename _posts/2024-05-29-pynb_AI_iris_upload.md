---
layout: single
title:  "파이썬으로 만드는 간단한 AI"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 파이썬으로 만드는 간단한 AI 입니다.  

iris 라는 해더 파일을 DB로 갖는 엄청 간단한 AI 입니다.  
어떤 구성으로 어쩧게 AI가 돌아가는지 간단하게 볼 수 있습니다.  

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 시드 고정
np.random.seed(42)

# 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = (iris.target != 0) * 1  # 이진 분류를 위해 타겟 레이블을 0과 1로 변환

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 가중치와 바이어스 초기화
weights = np.random.rand(X.shape[1])
bias = np.random.rand()

# 하이퍼파라미터 설정
learning_rate = 0.01
epochs = 100

# 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 손실 함수 (로그 손실)
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 학습 과정
losses = []

for epoch in range(epochs):
    # 정방향
    z = np.dot(X_train, weights) + bias
    predictions = sigmoid(z)
    
    # 손실 계산
    current_loss = loss(y_train, predictions)
    losses.append(current_loss)
    
    # 역방향, 오류역전파 - 가중치 업데이트
    error = predictions - y_train
    weights -= learning_rate * np.dot(X_train.T, error) / len(X_train)
    bias -= learning_rate * np.mean(error)
    
    # 에포크별 출력
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}')

# 평가
z_test = np.dot(X_test, weights) + bias
predictions_test = sigmoid(z_test)
predictions_test = [1 if i > 0.5 else 0 for i in predictions_test]

# 정확도 계산
accuracy = np.mean(predictions_test == y_test)
print(f'Test Accuracy: {accuracy:.4f}')
print('정확도 : {}%'.format(accuracy*100))

# 학습 진행 상황 시각화
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

```

# 실행결과  

실행시에는 다음과 같은 결과가 나옵니다.  


Epoch 10/100, Loss: 1.7754  
Epoch 20/100, Loss: 1.3422  
Epoch 30/100, Loss: 0.9370  
Epoch 40/100, Loss: 0.6119  
Epoch 50/100, Loss: 0.4264  
Epoch 60/100, Loss: 0.3556  
Epoch 70/100, Loss: 0.3286  
Epoch 80/100, Loss: 0.3121  
Epoch 90/100, Loss: 0.2983  
Epoch 100/100, Loss: 0.2857  
Test Accuracy: 1.0000  
정확도 : 100.0%  

![결과 표](https://github.com/Gihak111/Gihak111.github.io/assets/162708096/c7339fb3-0feb-4ab7-8511-148c152d4f3c)



다음과 같은 결과를 얻을 수 있습니다.  