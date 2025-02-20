---
layout: single
title:  "NLP에서 데이터 증강 (Data Augmentation)의 역할과 기법"
categories: "Natural_Language_Processing"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# NLP에서 데이터 증강 (Data Augmentation)의 역할과 기법

자연어 처리(NLP)에서 데이터 증강은 모델의 일반화를 돕고 데이터 부족 문제를 해결하는 데 중요한 역할을 한다.  
이 글에서는 데이터 증강의 개념, 다양한 기법, 활용 사례, 그리고 한계를 살펴본다.

---

## 데이터 증강이란?

### 정의
데이터 증강은 기존 데이터를 변형하거나 가공하여 새로운 학습 데이터를 생성하는 과정이다.  
- 데이터가 제한적일 때 모델의 성능을 개선.  
- 학습 데이터의 다양성을 높여 과적합 방지.

---

## 데이터 증강 기법

### 1. **텍스트 교체 기반 기법**

#### 1.1 **Synonym Replacement (유의어 교체)**
단어를 유의어로 대체하여 새로운 문장을 생성.

- 예: "The cat is cute." → "The feline is cute."

#### Python 예제
```python
import random
from nltk.corpus import wordnet

def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_sentence.append(synonyms[0].lemmas()[0].name())
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)

print(synonym_replacement("The cat is cute."))
```

---

#### 1.2 **Random Insertion (무작위 삽입)**
랜덤 단어를 문장에 삽입하여 다양성 확보.

#### 1.3 **Random Deletion (무작위 삭제)**
문장에서 일부 단어를 제거하여 학습 데이터를 확장.

---

### 2. **순서 변경 기반 기법**

#### 2.1 **Sentence Shuffling**
문장 내부 단어의 순서를 무작위로 변경.  
- 예: "The cat is cute." → "cat The cute is."

#### 2.2 **Back Translation (역번역)**
텍스트를 다른 언어로 번역한 후 다시 원래 언어로 번역.  
- 예: "The cat is cute." → "The feline is adorable."

#### Python 예제
```python
from transformers import MarianMTModel, MarianTokenizer

src_text = "The cat is cute."
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
back_translated = tokenizer.decode(translated[0], skip_special_tokens=True)
print(back_translated)
```

---

### 3. **임베딩 기반 기법**

#### 3.1 **Word Embedding Noise**
단어 임베딩 벡터에 노이즈를 추가하여 변형된 데이터를 생성.

#### 3.2 **Contextual Augmentation**
사전 학습된 언어 모델(BERT, GPT)을 사용하여 문맥에 적합한 새로운 단어를 생성.

---

## 활용 사례

### 1. **감성 분석**
- 유의어 교체 및 역번역 기법을 활용하여 감성 레이블 분류 성능 강화.

### 2. **기계 번역**
- 역번역을 통해 다양한 번역 문장을 생성하여 모델 성능 개선.

### 3. **질문 생성 및 응답**
- Contextual Augmentation 기법을 활용하여 질문 및 응답 데이터 생성.

---

## 데이터 증강의 한계

1. **의미 왜곡**: 잘못된 변환으로 데이터 품질 저하 가능.  
2. **도메인 의존성**: 특정 도메인에 적합하지 않을 수 있음.  
3. **증강 데이터 과잉**: 불필요한 데이터가 추가될 위험.  

---

## 결론

데이터 증강은 NLP 모델의 성능을 높이는 데 강력한 도구로, 다양한 기법을 조합하여 활용할 수 있다.  
다만 데이터 품질과 변환의 적절성을 항상 고려해야 한다.