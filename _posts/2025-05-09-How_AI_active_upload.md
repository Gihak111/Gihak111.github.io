---
layout: single
title:  "AI 는 어떻게 작동하는가에 대해 엄청 자세히 알아보자"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## AI 작동 원리 밑 각 레이어의 내부

우리는 이미 소프트 맥스나 워드 인베딩 레이어 같은 많은 층 들을 보며 살아왔다.  
하지만, 공식만 알지 이 내부를 깊게 들여다 본 적은 많지 않은 것 같다.  
그렇다고 그 레이어에 대해 모르는 건 아니다.  
잘 알고 있지만, 그 근본에 대해 깊게 접근하지 못했을 뿐.  
따라서, 오늘은 인코더, 소프트 맥스 등 의 레이어를 진짜 깊게 한번 봐 볼까 한다.  
얼마나 깊은은 내용일 지 한번 들어가 보도록 하자.  

## 간단한 AI
간단한 AI가 존재한다고 치자
인코더, 디코더가 있을텐데,  
워드 임베딩 레이어, 트랜스포머 레이어, 소프트맥스 레이어로만 구성되어 있다 치고,  
앞선 3개의 레이어에 어떤 차이가 있고, 어떤 노드간의 연결 방식 차이로 인해 레이어의 차이가 생기는지 알아보자.  

## 1. 레이어의 구성 – AI는 결국 노드들의 연결이다  

일단 딥러닝 모델은 전부 '레이어(layer)'라는 구조로 이뤄져 있다.  
그리고 이 레이어는 '노드(node)'들의 집합이다.  
여기서 노드는 '뉴런(neuron)'이라고도 부르며, 수학적으로는 어떤 값을 받아서 가공한 뒤 출력하는 함수다.  

AI는 기본적으로 아래의 구조를 갖는다:  

* 인풋 레이어: 데이터를 받아들이는 입구  
* 인코더: 데이터를 "의미 있는 벡터"로 바꾸는 역할  
* 트랜스포머 레이어: 입력 간 관계를 파악하고, 중요도를 조절  
* 디코더: 벡터를 다시 사람이 이해할 수 있는 언어나 구조로 복원  
* 소프트맥스: 결과를 확률로 바꿔주는 역할  
* 출력 레이어: 최종 결과를 보여주는 문, 혹은 행동  

### 노드 연결의 차이

* 인코더 내부 노드는 **셀프 어텐션** 구조로 연결되어 있다. 각 노드는 모든 다른 노드들과 연결된다.  
* 디코더 내부 노드는 **마스크드 어텐션**으로 연결되어 있어, 미래 노드를 참조하지 못하게 막는다.  
* 트랜스포머는 인코더-디코더 어텐션을 통해 인코더의 출력 벡터와 연결된다.  
* 소프트맥스는 마지막 출력을 확률로 매핑하기 위해 모든 출력 노드와 1:1로 연결된다.  

## 2. 인코더 – 입력을 벡터로 바꾸는 장인

인코더는 말 그대로 ‘코드로 바꿔주는 기계’다.  
입력값(예: 단어)을 숫자 벡터로 바꿔주는 역할을 한다.  

### 대표적인 인코더 레이어들

* **워드 임베딩 (Word Embedding)**
  단어를 고정된 길이의 벡터로 바꿔줌.  
  예: "고양이" → `[0.22, -0.44, 0.78, ...]`  

  ```python
  from tensorflow.keras.layers import Embedding

  embedding = Embedding(input_dim=10000, output_dim=512)
  embedded = embedding(input_sequence)  # input_sequence는 정수 인덱스 벡터
  ```  

* **포지셔널 인코딩 (Positional Encoding)**  
  문장의 순서를 알려주는 역할.  
  위치 정보를 벡터에 더함.  

```python
import numpy as np

def positional\_encoding(max\_len, d\_model):  

    PE = np.zeros((max\_len, d\_model))
    for pos in range(max\_len):
        for i in range(0, d\_model, 2):
            PE\[pos, i] = np.sin(pos / (10000 \*\* ((2 \* i)/d\_model)))
            PE\[pos, i + 1] = np.cos(pos / (10000 \*\* ((2 \* (i+1))/d\_model)))
    return PE

```

- **셀프 어텐션 (Self-Attention)**  
문장 내 단어들 간의 관계를 파악.  
```python
import tensorflow as tf

def scaled_dot_product_attention(Q, K, V):
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
```  

## 3. 디코더 – 벡터를 다시 의미로 바꾸는 번역가  

디코더는 반대로 숫자 벡터 → 사람이 이해할 수 있는 단어, 문장으로 바꿔준다.  

### 디코더에서 쓰이는 주요 레이어  

* **마스킹된 어텐션 (Masked Attention)**  
  미래 단어를 못 보게 막는 구조로, 현재 시점까지만 봄.  

* **디코더 어텐션 (Encoder-Decoder Attention)**  
  인코더의 출력을 참조하여 번역이나 생성에 반영.  

## 4. 인코더만 쓰는 경우 – BERT처럼 생각만 할 때  

문장 분류, 감정 분석 등에 사용.  
대표적 예시: BERT  

## 5. 디코더만 쓰는 경우 – GPT처럼 혼잣말할 때  

앞 문장만 보고 다음 단어를 예측.  
대표적 예시: GPT, LLaMA 등  

## 6. 인코더-디코더 모두 쓰는 경우 – 번역처럼 변환이 필요할 때  

대표적 예시: 번역 (Transformer), 요약 (T5)  

## 7. 여러 개의 인코더/디코더가 있는 경우  

인코더가 6개, 디코더가 6개처럼 겹겹이 쌓이면 더 깊은 관계를 학습 가능.  
깊을수록 '저수준 특징 → 고수준 의미'를 학습한다.  

## 8. 어텐션 레이어 – AI의 ‘집중력’  

### 핵심 개념  

Query, Key, Value 벡터를 만들고:  

* Query \* Key → 유사도 (Score)  
* softmax → 중요도 가중치  
* 중요도 \* Value → 새로운 표현  

## 9. 각 레이어의 순서와 조합 효과  

* 임베딩 → 포지셔널 인코딩 → 어텐션 → FFN → 출력  
* 순서를 바꾸면 학습 안 되거나 문맥이 깨진다.  

## 결론
결국 수많은 벡터 간의 연결이다.  
AI는 복잡해 보이지만, 결국은  

* 숫자를 벡터로 바꾸고  
* 벡터끼리 비교하고  
* 중요한 벡터를 강조한 다음  
* 다시 의미 있는 것으로 복원  

초심자도 각 레이어 역할을 이해하면, AI가 문장을 만들고 감정을 판단하고 번역하는 과정을 쉽게 이해할 수 있다.  
