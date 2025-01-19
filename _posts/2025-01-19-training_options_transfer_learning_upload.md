---
layout: single
title:  "NLP에서 전이 학습 (Transfer Learning) 활용하기"
categories: "Natural Language Processing"
tag: "transfer-learning-nlp"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# NLP에서 전이 학습 (Transfer Learning) 활용하기

자연어 처리(NLP)에서 전이 학습은 사전 학습된 모델을 활용하여 새로운 태스크를 해결하는 강력한 기법이다.  
이 글에서는 전이 학습의 정의, 주요 기법, 활용 사례, 그리고 한계를 다룬다.

---

## 전이 학습이란?

### 정의
전이 학습은 한 태스크에서 학습한 지식을 새로운 태스크에 적용하는 방법이다.  
NLP에서는 대규모 텍스트 코퍼스에서 사전 학습된 모델을 다양한 다운스트림 태스크에 적용한다.

---

## 전이 학습의 과정

### 1. **사전 학습 (Pretraining)**

대규모 코퍼스를 사용하여 일반적인 언어 표현을 학습한다.

- **언어 모델링**: 다음 단어 예측 (예: GPT 시리즈).  
- **마스킹 언어 모델링**: 일부 단어를 마스킹하고 이를 예측 (예: BERT).  

#### 예제
```plaintext
문장: "The cat is on the [MASK]."
예측: "mat"
```

---

### 2. **미세 조정 (Fine-Tuning)**

사전 학습된 모델을 특정 태스크에 맞게 조정한다.  
- **감성 분석**  
- **문서 분류**  
- **질문 응답**  
- **기계 번역**

---

## 주요 전이 학습 기법

### 1. **Feature Extraction**

사전 학습된 모델의 특정 층에서 추출한 벡터를 다운스트림 모델에 입력으로 사용한다.  
- 간단한 분류기와 결합하여 빠르게 태스크 해결 가능.

#### Python 예제
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("This is an example.", return_tensors="pt")
outputs = model(**inputs)
features = outputs.last_hidden_state
```

---

### 2. **Fine-Tuning**

모델 전체를 학습 가능한 상태로 두고 태스크 데이터로 추가 학습을 진행한다.  
- **학습률 조정**과 **정규화**가 성능에 큰 영향을 미친다.

---

### 3. **Adapters**

사전 학습된 모델의 원래 가중치를 동결한 상태로, 소규모 추가 레이어를 학습한다.  
- **효율적인 메모리 사용**과 **빠른 학습** 가능.

---

## 전이 학습 모델의 종류

### 1. **BERT (Bidirectional Encoder Representations from Transformers)**

- 양방향 컨텍스트를 활용하여 문맥적 의미 학습.  
- 대표 태스크: 감성 분석, 질문 응답.

### 2. **GPT (Generative Pretrained Transformer)**

- 순차적 언어 생성에 강점.  
- 대표 태스크: 텍스트 생성, 요약.

### 3. **T5 (Text-To-Text Transfer Transformer)**

- 모든 태스크를 텍스트 변환 문제로 정의.  
- 대표 태스크: 번역, 요약.

### 4. **XLNet**

- BERT의 한계를 보완한 언어 모델.  
- 순서와 관계없는 단어 샘플링으로 학습.

---

## 활용 사례

### 1. **감성 분석 (Sentiment Analysis)**
사전 학습된 BERT 모델을 미세 조정하여 리뷰 데이터의 긍정/부정을 분류.

#### Python 예제
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inputs = tokenizer("I love this product!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

---

### 2. **질문 응답 (Question Answering)**

사전 학습된 모델을 사용하여 특정 문서에서 질문에 대한 답을 추출.

---

### 3. **요약 (Summarization)**

T5와 같은 모델을 사용하여 문서를 요약.

---

## 전이 학습의 한계

1. **대규모 데이터 요구**: 사전 학습에 많은 리소스 필요.  
2. **과적합 가능성**: 다운스트림 데이터가 적을 경우 성능 저하 위험.  
3. **도메인 의존성**: 특정 도메인 데이터에 적합하지 않을 수 있음.  

---

## 결론

전이 학습은 NLP의 혁신을 이끈 핵심 기술로, 다양한 태스크에서 활용 가능하다.  
효율적인 사전 학습과 미세 조정 전략을 통해 더 나은 성능을 달성할 수 있다.
