---
layout: single
title:  "직접 AI 모델 만들어 보기 with 수학 개념들"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# AI 직접 만들기
사용만 해 왔던 AI를 직접 만들어 보자.  
Llama 모델의 제작접이 공개되지 않았지만, 이거랑 비슷한 느낌으로 해 보자.  
데이터 준비, 모델 정의, 훈련 루프가 핵심적인 부분이다.  
지금부터 만들 모델은  Hugging Face Transformers 라이브러리에서 제공되는 GPT 계열 모델과 매우 유사하므로, GPT 계열 모델을 훈련하는 방법을 기반으로 설명하겠다.  

## 1. 라이브러리 설치  
Hugging Face Transformers, Datasets, PyTorch 등 여러 라이브러리가 필요하다.  
```bash
pip install transformers datasets torch
```  

## 2. 데이터 준비  
텍스트 데이터셋을통해 딥러닝 하자.  
대형 데이터셋은 Hugging Face Datasets 라이브러리에서 로드할 수 있다.  
```python
from datasets import load_dataset

# 예시: 'wikitext' 데이터셋을 로드
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset['train']
val_data = dataset['validation']

```

## 3. 모델정의  
Transformers 라이브러리를 사용하여 모델을 정의할 수 있다.  
GPT 계열과 비슷하다.  
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2를 LLaMA 모델처럼 사용
model_name = "gpt2"  # 실제 LLaMA 모델 이름으로 대체
model = GPT2LMHeadModel.from_pretrained(model_name)

# 토크나이저 설정
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```

## 4. 데이터 전처리  
텍스트 데이터를 모델에 맞게 tokenizer를 통해 전처리한다.  
```python
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
```

## 5. 훈련 설정  
Hugging Face의 Trainer API를 사용하여 간단히 모델 훈련설정을 한다.  
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # 결과를 저장할 디렉토리
    evaluation_strategy="epoch",     # 평가 전략
    learning_rate=2e-5,              # 학습률
    per_device_train_batch_size=4,   # 훈련 배치 크기
    per_device_eval_batch_size=8,    # 평가 배치 크기
    num_train_epochs=3,              # 훈련 에폭 수
    weight_decay=0.01,               # 가중치 감소
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_data, 
    eval_dataset=val_data
)
```

## 6. 훈련 시작  
훈련 돌리자.  
```python
trainer.train()
```

## 7. 모델 저장  
이어서, 훈련한 모델을 저장하는 것으로 쉽게 만들 수 있다.  
```python
model.save_pretrained("./llama_model")
tokenizer.save_pretrained("./llama_model")
```  

위 내용을 통해 만들면, 굉장히 구린 성능을 보인다.  
이를 통해, 시중에 풀려잇는 AI들이 얼마나 대단한 건지 체감할 수 있다.  

# 이제, 사용된 수학 기법들을 알아보자.  

위에서 설명한 LLaMA 모델 훈련 과정은 주로 딥러닝, 자연어 처리, 트랜스포머 모델의 기본적인 원리들을 바탕으로 이루어 진다.  
이 과정에서 중요한 수학적 개념들이 여러 가지 방식으로 적용된다.  

## 1. 수학적 개념: 선형 대수  
   행렬 연산:  
   트랜스포머 모델은 행렬과 벡터를 주로 사용한다.  
   모델의 입력(텍스트 데이터)은 토큰화(tokenization) 과정을 거쳐 임베딩(embedding) 벡터로 변환되며, 이후 이 벡터들은 행렬 형태로 모델에 입력된.  
   가중치 행렬(weight matrix)와 편향(bias) 값들은 모델을 훈련하는 동안 선형 변환을 통해 갱신된다.  
   예를 들어, 모델의 주의(attention) 메커니즘이나 출력 단계에서 행렬 곱셈(matrix multiplication)이 이루어 진다.  

   행렬 곱셈:  
   트랜스포머 모델에서 중요한 부분 중 하나는 Self-Attention Mechanism이다.  
   이 메커니즘에서는 Query, Key, Value 행렬이 서로 곱해져 Attention Score가 계산된다.  
   이 계산은 다음과 같은 방식으로 이루어진다: 
   ```python
   \[
   Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V
   \]
   ```  

   여기서:  
   \(Q\)는 Query 벡터,  
   \(K\)는 Key 벡터,  
   \(V\)는 Value 벡터,  
   \(d_k\)는 벡터의 차원이다.
   이 과정에서 행렬의 내적(inner product) 및 softmax 함수가 적용된다.  

## 2. 수학적 개념: 미분 및 최적화 (Differentiation & Optimization)
   경사 하강법 (Gradient Descent):  
   훈련 과정에서 모델의 가중치는 경사 하강법(gradient descent)을 통해 업데이트된다.  
   이 방법은 손실 함수(loss function)의 미분 값을 계산하여 가중치를 업데이트 한다.  
   훈련 과정에서 이 값들이 점차 최솟값으로 수렴하도록 한다.  
   역전파(backpropagation) 알고리즘은 이 경사 하강법을 효율적으로 적용하기 위한 방법이다.  
   각 가중치에 대한 기울기(gradient)를 계산하고 이를 바탕으로 가중치를 조정하여 모델을 점진적으로 최적화한다.  

   손실 함수 (Loss Function):  
   교차 엔트로피 손실(Cross-Entropy Loss) 함수는 자연어 처리에서 자주 사용되는 손실 함수로, 모델의 출력값과 실제 정답 간의 차이를 측정한다.  
   이는 모델이 예측한 확률 분포와 실제 레이블 간의 차이를 최소화하는 방향으로 훈련이 이루어지게 한다.  
   ```python
   \[
   \text{Loss} = - \sum y \log(\hat{y})
   \]
   ```
   여기서 \( y \)는 실제 값, \( \hat{y} \)는 모델이 예측한 확률이다.  

## 3. 수학적 개념: 확률 및 통계 (Probability & Statistics)
   Softmax 함수:  
   Softmax는 출력 벡터의 값을 확률로 변환하는 함수로, 주로 분류 문제에서 사용된다.  
   트랜스포머 모델의 최종 출력은 Softmax 함수에 의해 변환되어, 각 클래스(또는 단어)의 확률을 출력한다.  
   ```python
   \[
   \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
   \]
   ```
   이 함수는 벡터 \( z \)의 각 원소를 확률 분포로 변환한다.  

   확률적 최적화:  
   트랜스포머 모델은 확률적 방식으로 학습되므로, 주어진 입력에 대한 출력을 예측할 때 확률을 사용한다.  
   이 방식은 모델이 정확한 값을 예측하기보다 가능한 결과들에 대해 확률 분포를 예측하게 한다.  

## 4. 수학적 개념: 텍스트 임베딩 (Text Embeddings)
   단어 임베딩(Word Embedding):  
   모델의 입력 데이터는 텍스트로 이루어져 있으므로, 텍스트를 벡터로 변환하는 단어 임베딩(word embeddings) 기법이 필요 하다.  
   임베딩은 고차원 공간에서 각 단어를 벡터로 표현하여, 모델이 이를 처리할 수 있도록 한다. 일반적으로 Word2Vec이나 GloVe와 같은 방법이 사용되지만, 트랜스포머 모델에서는 학습 과정에서 임베딩을 fine-tune 한다.  

## 5. 수학적 개념: 주의 메커니즘 (Attention Mechanism)
   Self-Attention:  
   Self-Attention 메커니즘은 각 입력 토큰이 다른 모든 입력 토큰에 대해 얼마나 중요한지를 계산한다.  
   이를 통해 문맥에 맞는 정보를 동적으로 반영할 수 있다.  
   Self-Attention에서 중요한 계산은 어텐션 점수(attention score)를 계산하는 것인데, 이는 행렬의 내적을 사용하여 수행된다.  
   이를 통해 각 입력 벡터의 중요도를 평가한다.  

## 6. 수학적 개념: 배치 처리 (Batch Processing)
   배치 처리(Batch Processing):  
   모델 훈련은 배치(batch) 단위로 이루어진다.  
   한 번에 여러 개의 데이터를 모델에 입력하여 처리하는 방식이다.  
   이렇게 배치 처리를 통해 병렬 처리(parallel processing)가 가능해지고, 훈련 속도와 효율성이 향상된다.  

위와같은 개념들이 AI에 잘 사용되는 개념들이다.  