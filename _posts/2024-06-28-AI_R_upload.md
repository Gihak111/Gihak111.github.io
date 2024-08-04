---
layout: single
title:  "파이썬으로 분류 AI 만들기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
# 파이썬으로 BERT 활용 모델 만들기
트랜스포머 아키텍처를 기반으로 하는 BERT 모델을 활용한 AI이다.  
뎃글을 보고 그 뎃글이 긍정적인지, 부정적인지 확인한다.
# 코드



<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
import shutil

# 새로운 로그 디렉토리 경로 설정
log_dir = '/tmp/logs_new'  # 또는 'C:/logs_new' 와 같은 절대 경로 사용

# 로그 디렉토리가 파일로 존재하는지 확인하고, 파일인 경우 삭제
if os.path.isfile(log_dir):
    os.remove(log_dir)
    print(f"{log_dir}는 파일로 존재하여 삭제했습니다.")

# 로그 디렉토리 삭제 (이미 디렉토리가 존재하는 경우)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"{log_dir} 디렉토리를 삭제했습니다.")

# 로그 디렉토리 생성
os.makedirs(log_dir)
print(f"로그 디렉토리를 생성했습니다: {log_dir}")

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
    logging_dir=log_dir,
    logging_steps=10,
    eval_strategy="epoch",
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

<pre>
로그 디렉토리를 생성했습니다: /tmp/logs_new
</pre>
<pre>
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
</pre>

    <div>
      
      <progress value='7500' max='7500' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [7500/7500 5:16:21, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.238600</td>
      <td>0.375999</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.254300</td>
      <td>0.390754</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.183600</td>
      <td>0.624563</td>
    </tr>
  </tbody>
</table><p>



    <div>
      
      <progress value='625' max='625' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [625/625 07:12]
    </div>
    


<pre>
Evaluation results: {'eval_loss': 0.6245630979537964, 'eval_runtime': 432.6937, 'eval_samples_per_second': 11.556, 'eval_steps_per_second': 1.444, 'epoch': 3.0}
</pre>

```python
model_save_path = './trained_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
```

<pre>
Model saved to ./trained_model
</pre>

```python
print(f"Evaluation results: {eval_result}")
```

<pre>
Evaluation results: {'eval_loss': 0.6245630979537964, 'eval_runtime': 432.6937, 'eval_samples_per_second': 11.556, 'eval_steps_per_second': 1.444, 'epoch': 3.0}
</pre>

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 새로운 데이터
new_texts = ["This movie was amazing!", "I didn't like the film at all."]

# 텍스트를 토큰화
inputs = tokenizer(new_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)

# 모델 로드
model = BertForSequenceClassification.from_pretrained(model_save_path)

# 모델 추론
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

print(f"Predictions: {predictions}")
```

<pre>
Predictions: tensor([1, 0])
</pre>

```python
# 레이블을 사람이 읽을 수 있는 형식으로 변환
label_map = {0: "Negative", 1: "Positive"}
readable_predictions = [label_map[prediction.item()] for prediction in predictions]

for text, prediction in zip(new_texts, readable_predictions):
    print(f"Text: {text}\nPrediction: {prediction}\n")
```

<pre>
Text: This movie was amazing!
Prediction: Positive

Text: I didn't like the film at all.
Prediction: Negative

</pre>

```python
```

중간에 나오는 오류는 기존의 자료를 추가 작업 없이 바로 활용해서 나오는 오류로 모델 학습 이후에는 문제가 없다.