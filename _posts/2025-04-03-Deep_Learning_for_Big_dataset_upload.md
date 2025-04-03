---
layout: single
title:  "딥러닝 결과 표시"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## PyTorch로 모델 예측 결과를 분석하고 활용하기

딥러닝 모델을 학습시킨 후, 예측 결과를 분석하고 활용하는 것은 프로젝트의 성공을 결정짓는 중요한 단계이다.  
단순히 모델을 돌리는 것에서 끝나는 것이 아니라, 결과를 해석하고 실무에 적용하는 과정이 필요하다.  
이번에는 PyTorch와 코드를 활용하여 모델 예측 결과를 추출하고, 분석하며, 활용하는 방법을 정리한다.  
예측 성능을 평가하고, 결과를 시각화하며, 실전에서 사용할 수 있는 인사이트를 도출하는 데 초점을 맞춘다.  

### 1. 학습된 모델 로드하기
예측을 시작하려면 학습된 모델을 로드해야 한다.  

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "klue/roberta-large"
model_save_path = './roberta_segment_classifier'

model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
```

저장된 모델과 토크나이저를 로드하여 예측 준비를 한다.  

### 2. 테스트 데이터 준비하기
예측에 사용할 테스트 데이터를 준비한다.  

```python
import pandas as pd

# 예시 테스트 데이터
test_data = {
    'ID': ['cust1', 'cust2', 'cust3'],
    'text': [
        "서비스가 매우 만족스럽습니다",
        "제품이 별로예요",
        "배송이 너무 느립니다"
    ]
}
test_df = pd.DataFrame(test_data)

# 토큰화
def tokenize_function(texts):
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    return encodings

test_encodings = tokenize_function(test_df['text'].tolist())
```

테스트 데이터를 토큰화하여 모델 입력으로 변환한다.  

### 3. 예측 수행하기
모델을 사용해 예측 결과를 생성한다.  

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with torch.no_grad():
    inputs = {key: val.to(device) for key, val in test_encodings.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()

test_df['predicted_label'] = predictions
```

모델을 평가 모드로 전환하고, 예측 레이블을 추출하여 데이터프레임에 추가한다.  

### 4. 레이블 매핑 정의하기
숫자 레이블을 의미 있는 값으로 변환한다.  

```python
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
test_df['predicted_segment'] = test_df['predicted_label'].map(label_map)
print(test_df[['ID', 'text', 'predicted_segment']])
```

숫자 레이블을 A~E로 매핑하여 결과를 해석하기 쉽게 한다.  

### 5. 예측 확률 계산하기
예측의 신뢰도를 확인하려면 확률을 계산한다.   

```python
with torch.no_grad():
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    test_df['confidence'] = [max(prob) for prob in probs]

print(test_df[['ID', 'text', 'predicted_segment', 'confidence']])
```

`softmax`로 확률을 계산하고, 최대 확률을 신뢰도로 추가하여 예측의 확신도를 파악한다.  

### 6. 예측 결과 저장하기
분석 결과를 파일로 저장한다.  

```python
test_df.to_csv('prediction_results.csv', index=False)
print("예측 결과가 'prediction_results.csv'에 저장되었습니다.")
```

CSV 파일로 저장하여 결과를 보존하고 공유한다.  

### 7. 오차 사례 분석하기
잘못된 예측을 찾아 개선점을 도출한다.  

```python
# 실제 레이블이 있다고 가정
true_labels = [0, 1, 2]  # 예시
test_df['true_label'] = true_labels
test_df['is_correct'] = test_df['predicted_label'] == test_df['true_label']

errors = test_df[~test_df['is_correct']]
print("오차 사례:")
print(errors[['ID', 'text', 'predicted_segment', 'true_label']])
```

오차 사례를 필터링하여 모델의 약점을 분석한다.  

### 8. 클래스별 예측 분포 확인하기
예측 결과의 분포를 파악한다.  

```python
from collections import Counter

pred_counts = Counter(test_df['predicted_segment'])
print("클래스별 예측 분포:", dict(pred_counts))
```

`Counter`로 클래스별 빈도를 계산하여 모델의 편향 여부를 확인한다.  

### 9. 혼동 행렬 생성하기
실제와 예측 간의 관계를 분석한다.  

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_df['true_label'], test_df['predicted_label'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('혼동 행렬')
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.show()
```

혼동 행렬을 시각화하여 클래스 간 오차 패턴을 파악한다.  

### 10. 예측 결과를 비즈니스에 활용하기
예측 결과를 실무에 적용한다.  

```python
def recommend_action(segment):
    actions = {
        'A': '프리미엄 서비스 제안',
        'B': '할인 쿠폰 제공',
        'C': '고객 지원 연락',
        'D': '제품 개선 피드백 요청',
        'E': '재구매 유도 캠페인'
    }
    return actions.get(segment, '기본 조치')

test_df['action'] = test_df['predicted_segment'].apply(recommend_action)
print(test_df[['ID', 'text', 'predicted_segment', 'action']])
```

세그먼트별로 맞춤형 액션을 제안하여 비즈니스 가치를 창출한다.  

### 11. 결과 요약 보고서 작성하기
분석 결과를 요약한다.  

```python
summary = {
    '총 예측 수': len(test_df),
    '정확도': test_df['is_correct'].mean() if 'is_correct' in test_df.columns else 'N/A',
    '클래스 분포': dict(pred_counts),
    '오차 사례 수': len(errors) if 'is_correct' in test_df.columns else 'N/A'
}
print("예측 결과 요약:", summary)
```

요약 보고서를 작성하여 결과를 한눈에 이해한다.  

### 결론
모델 예측 결과를 분석하고 활용하는 것은 딥러닝의 실질적인 가치를 실현하는 과정이다.  
학습된 모델을 로드하고, 테스트 데이터를 준비하며, 예측을 수행한다.  
레이블과 확률을 계산하고, 결과를 저장하며, 오차 사례와 분포를 분석한느 것으로 혼동 행렬로 성능을 평가하고, 비즈니스 액션을 제안하며, 요약 보고서로 마무리하면 된다.  
이 과정을 통해 예측 결과를 단순한 숫자에서 실무에 적용 가능한 인사이트로 전환할 수 있다.  
