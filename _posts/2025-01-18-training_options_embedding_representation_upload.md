---
layout: single
title:  "토큰화 이후: 임베딩과 표현 학습"
categories: "Natural_Language_Processing"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 토큰화 이후: 임베딩과 표현 학습

서브워드 토큰화를 통해 텍스트를 처리한 이후, 기계 학습 모델은 이를 수치 데이터로 변환해 학습한다.  
이 과정을 **임베딩(Embedding)**이라고 하며, NLP 모델 성능의 핵심적인 요소로 작용한다.

---

## 임베딩의 개념

### 텍스트를 수치화하는 이유
1. **모델의 입력 요구**: 딥러닝 모델은 숫자 데이터를 입력으로 받음.  
2. **어휘 표현의 일반화**: 유사한 단어를 비슷한 수치로 표현.  
3. **고차원 데이터 표현**: 단어 간의 복잡한 관계를 효율적으로 표현.

### 임베딩의 장점
- **의미 기반 표현**: 단어 간 의미적 유사성을 반영.  
- **차원 축소**: 어휘 크기에 비해 상대적으로 작은 차원으로 단어를 표현.  
- **기계 학습 효율성 증가**: 모델 학습 속도와 성능 향상.

---

## 주요 임베딩 기법

### 1. **Word2Vec**

**Word2Vec**은 단어를 고차원 벡터로 변환하는 기법으로, CBOW와 Skip-Gram 두 가지 모델이 있다.

#### 특징
- **의미 기반 학습**: 유사한 문맥에서 사용되는 단어는 유사한 벡터로 표현.  
- **저차원 벡터 표현**: 수백 차원으로 단어 표현.

#### 예제
```plaintext
단어: "강아지", "고양이"
유사도: cos(강아지, 고양이) ≈ 0.8
```

#### Python 예제
```python
from gensim.models import Word2Vec

sentences = [["I", "love", "cats"], ["I", "love", "dogs"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['cats']
print(vector)
```

---

### 2. **GloVe (Global Vectors)**

**GloVe**는 단어의 동시 등장 행렬을 학습하여 전역적인 문맥 정보를 반영한다.

#### 특징
- **전역적 통계 기반**: 단어 간 동시 등장 확률을 학습.  
- **수학적 안정성**: 최적화 문제로 접근.  

#### 예제
```plaintext
단어: "왕", "여왕", "남자", "여자"
벡터 연산: 왕 - 남자 + 여자 ≈ 여왕
```

#### Python 예제
```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False)
vector = model['king']
print(vector)
```

---

### 3. **FastText**

**FastText**는 단어를 서브워드 단위로 분해하여 벡터를 학습한다.

#### 특징
- **희귀 단어 처리**: 서브워드 조합으로 새로운 단어 표현 가능.  
- **언어 간 유연성**: 언어 구조의 특성을 잘 반영.  

#### 예제
```plaintext
단어: "unbelievable"
서브워드: ["un", "##bel", "##ievable"]
```

#### Python 예제
```python
from gensim.models import FastText

sentences = [["I", "love", "coding"], ["coding", "is", "fun"]]
model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['coding']
print(vector)
```

---

### 4. **Contextual Embeddings**

**Contextual Embeddings**는 단어가 문맥에 따라 다른 벡터로 표현되는 기법이다.  
대표적으로 BERT, GPT와 같은 트랜스포머 기반 모델에서 사용된다.

#### 특징
- **문맥 의존적**: 단어의 의미가 문맥에 따라 변동.  
- **고차원 표현**: 복잡한 문맥 정보를 반영.  

#### 예제
```plaintext
문장 1: "I saw a bat flying in the sky."
문장 2: "I used a bat to play baseball."
벡터: "bat"의 두 문맥에서 서로 다른 벡터로 표현.
```

#### Python 예제
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("I saw a bat flying in the sky.", return_tensors="pt")
outputs = model(**inputs)
vector = outputs.last_hidden_state
print(vector)
```

---

## 임베딩 학습 후의 과정

### 1. **클러스터링 및 시각화**
- 학습된 임베딩을 **t-SNE**나 **PCA**로 시각화하여 구조 확인.  

#### Python 예제
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

vectors = model.wv.vectors
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
plt.show()
```

---

### 2. **특정 태스크 적용**
- **분류**: 감성 분석, 뉴스 카테고리 분류.  
- **유사도 검색**: 검색 엔진, 추천 시스템.  
- **기계 번역**: 다국어 텍스트 처리.

---

## 임베딩의 한계와 개선 방향

1. **대규모 데이터 요구**: 고품질 임베딩 학습에 많은 데이터 필요.  
2. **언어 간 차이**: 각 언어에 특화된 임베딩 필요.  
3. **문맥 처리 한계**: 정적 임베딩의 경우 문맥 의존적 표현 어려움.  

---

## 마무리

서브워드 토큰화 이후 임베딩은 NLP의 핵심 과정을 구성하며, 텍스트 데이터에서 의미를 학습하는 기초를 제공한다.  
다양한 임베딩 기법을 활용하여 텍스트 데이터를 효과적으로 처리하고, 태스크에 맞는 최적의 표현을 선택하는 것이 중요하다.
