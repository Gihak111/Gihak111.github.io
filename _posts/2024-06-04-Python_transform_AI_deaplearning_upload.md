---
layout: single
title:  "파이썬으로 만드는 간단한 AI. 트랜스포머 모듈"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 파이썬으로 만드는 간단한 트랜스포머 모듈 활용 코드 입니다.  
해당 코드를 실행시키려면 다양한 패키지를 통해 환경을 먼저 구성해야 합니다.  
transformers, torch 설치  
```pip install transformers torch```  
datasets 모듈  
```pip install datasets```  
Node.js와 npm 설치  
공식 싸이트에서 다운  
실행시 버젼 간의 오류가 날 경우   
```
pip install jupyterlab==4.0.2 notebook==7.0.8 ipywidgets==7.7.2  
jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0.0  
jupyter lab clean  
jupyter lab build  
```  
어떤 버젼에서든지 사용 가능한 버젼으로 다운 받으면 됩니다.  

# 코드
```python
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 데이터셋 로드 (IMDb 리뷰 데이터셋 사용)
dataset = load_dataset('imdb')

# 데이터셋 분할
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = dataset['train']
test_data = dataset['test']

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 데이터 전처리 적용
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

# 데이터셋 포맷 변환
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 모델 로드 (BERT for sequence classification)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 훈련 인수 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  # eval_strategy로 변경
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# 모델 훈련
trainer.train()

# 모델 평가
eval_result = trainer.evaluate()

print(f"Evaluation results: {eval_result}")

```

코드 실행시 나오는 경고는 전 학습된 모델에서 초기화되지 않았으며 새로 초기화되었음을 알려주는 경고로 자연스러운 것 입니다. 무시하고 전부 러닝하면 문제없이 실행됩니다.  
  