---
layout: single
title:  "딥러닝 시 성능과 시간, 자원의 저울질 + 기법 정리
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 딥러닝 시 중단
사실상, Epoch 수가 높다고 해서 무조건 다 좋은 것 같진 않다.  
직전에 했던 립러닝에서도, 소폭 감소하더니 나중가선 미친듯한 발산을 보여주기도 했고, 그 외에도 이것저것 문제가 많다.  

그래서, 우리는 채크썸을 활용해서 중간중간 값을 가지고, 이를 통해서 유동적으로 학습율 값을 변동시키는 드으이 방식으로 딥러닝을 보다 효율적으로 돌아가게 하곤 한다.  
또한, loss값이 개선되지 않으면 조기에 딥러닝을 종료하고 꺼버리는 등의 다양한 방식으로 딥러닝을 진행한다.  

하지만, 체크썸의 데이터가 많아지면 50기가가 넘어갈 떄도 다반수이며, 3070 정도의 GPU를 사용하더라고 몇십 시간식 걸리는 것도 일상 다반사 이다. 이러면 어떻게 해야 이걸 좀 더 효율적으로 딥러닝 할 수 있을까?  

## 딥러닝 시 집어넣을 만한 기믹들
이런거 쑤셔 박으면, 엄청 걸리고, 자원도 많이 먹지만, 그래도 F1 스코어는 엄청 잘 나온다.  


### 1. 드롭아웃 (Dropout)
- **설명**: 모델의 과적합을 방지하기 위해 학습 중 일부 뉴런을 무작위로 비활성화한다.
- **코드**:
  ```python
  model.config.hidden_dropout_prob = 0.3
  model.config.attention_probs_dropout_prob = 0.3
  ```
- **효과**: 히든 레이어와 어텐션 메커니즘에서 각각 30%의 드롭아웃을 적용하여 일반화 성능을 높이다.

---

### 2. 옵티마이저 (Optimizer)
- **설명**: 모델 가중치를 업데이트하는 알고리즘으로, AdamW를 사용하며 레이어별 학습률을 다르게 설정한다.
- **코드**:
  ```python
  def get_optimizer_grouped_parameters(model, base_lr=1e-5, weight_decay=0.1):
      no_decay = ["bias", "LayerNorm.weight"]
      optimizer_grouped_parameters = []
      num_layers = model.config.num_hidden_layers
      for i, (name, param) in enumerate(model.named_parameters()):
          if "classifier" in name:
              lr = base_lr * 2.0
          else:
              layer_idx = min(i // 2, num_layers - 1)
              lr = base_lr * (0.95 ** (num_layers - layer_idx))
          wd = weight_decay if not any(nd in name for nd in no_decay) else 0
          optimizer_grouped_parameters.append({"params": param, "lr": lr, "weight_decay": wd})
      return optimizer_grouped_parameters

  trainer = CustomTrainer(
      optimizers=(torch.optim.AdamW(get_optimizer_grouped_parameters(model), eps=1e-8, amsgrad=True), None),
  )
  ```
- **효과**: Layer-wise Learning Rate Decay(LWLRD)를 통해 깊은 레이어는 빠르게, 얕은 레이어는 천천히 학습하여 최적화를 개선하다.

---

### 3. 체크포인트 (Checkpoint)
- **설명**: 학습 중 모델 상태를 저장하여 최적 모델을 선택하거나 중단 후 재개할 수 있게 한다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      output_dir=output_dir,
      save_strategy="epoch",
      load_best_model_at_end=True,
      metric_for_best_model="f1",
  )
  model.save_pretrained(model_save_path)
  tokenizer.save_pretrained(model_save_path)
  ```
- **효과**: 매 에포크마다 체크포인트를 저장하고, F1 스코어를 기준으로 최적 모델을 로드하여 성능을 극대화하다.

---

### 4. 레이블 인코딩 (Label Encoding)
- **설명**: 범주형 레이블을 숫자로 변환하여 모델이 학습할 수 있도록 준비한다.
- **코드**:
  ```python
  label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
  train_df['label'] = train_df['Segment'].map(label_map)
  ```
- **효과**: A~E를 0~4로 매핑하여 학습 데이터의 일관성을 유지하다.

---

### 5. GPU 체크 및 디바이스 설정 (GPU Check and Device Setup)
- **설명**: GPU 사용 가능 여부를 확인하고, 가능하면 GPU로 학습을 진행한다.
- **코드**:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.is_available():
      logger.info(f"GPU를 사용합니다: {torch.cuda.get_device_name(0)}")
  else:
      logger.warning("GPU에 연결되지 않았습니다. CPU로 전환합니다.")
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)
  ```
- **효과**: GPU를 활용하여 학습 속도를 높이고, 대규모 연산을 효율적으로 처리하다.

---

### 6. 데이터 불균형 처리 (Class Imbalance Handling)
- **설명**: 클래스별 데이터 분포가 불균형할 때 가중치를 적용하여 학습을 조정한다.
- **코드**:
  ```python
  label_counts = Counter(train_df['label'])
  n_samples = len(train_df)
  n_classes = 5
  class_weights = [n_samples / (n_classes * label_counts[i]) for i in range(n_classes)]
  class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

  class CustomTrainer(Trainer):
      def compute_loss(self, model, inputs, return_outputs=False):
          labels = inputs.pop("labels")
          outputs = model(**inputs)
          logits = outputs.logits
          loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
          loss = loss_fct(logits, labels)
          return (loss, outputs) if return_outputs else loss
  ```
- **효과**: 소수 클래스의 손실에 더 큰 가중치를 부여하여 모든 클래스를 고르게 학습하다.

---

### 7. NaN 데이터 처리 (NaN Handling)
- **설명**: NaN 및 이상치를 제거하여 데이터 품질을 개선한다.
- **코드**:
  ```python
  train_df = train_df.dropna(subset=['label']).replace([np.inf, -np.inf], np.nan).dropna()
  train_df['label'] = train_df['label'].astype(int)
  ```
- **효과**: 결측값과 무한대를 제거하여 학습 중 오류를 방지하고 안정성을 높이다.

---

### 8. 혼합 정밀도 학습 (Mixed Precision Training)
- **설명**: FP16(반정밀도)을 사용하여 메모리 사용량을 줄이고 학습 속도를 높인다.ConcurrentModificationException
- **코드**:
  ```python
  training_args = TrainingArguments(
      fp16=True if torch.cuda.is_available() else False,
  )
  ```
- **효과**: GPU에서 연산 효율성을 높여 대규모 모델 학습을 가속화하다.

---

### 9. 코사인 학습률 스케줄러 (Cosine Learning Rate Scheduler)
- **설명**: 학습률을 코사인 곡선에 따라 점진적으로 감소시킨다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      lr_scheduler_type="cosine",
  )
  ```
- **효과**: 초기에는 큰 학습률로 탐색하고, 후기에는 작은 학습률로 수렴하여 최적화를 개선하다.

---

### 10. 그래디언트 클리핑 (Gradient Clipping)
- **설명**: 그래디언트 크기를 제한하여 폭발 문제를 방지한다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      max_grad_norm=0.5,
  )
  ```
- **효과**: 학습 안정성을 유지하며, 특히 초기 학습 단계에서 유리하다.

---

### 11. 레이블 스무딩 (Label Smoothing)
- **설명**: 레이블에 약간의 노이즈를 추가하여 모델의 과신을 방지한다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      label_smoothing_factor=0.05,
  )
  ```
- **효과**: 모델의 일반화 성능을 높이고, 과적합을 줄이다.

---

### 12. 진행도 표시 (Progress Bar with tqdm)
- **설명**: 데이터 로드 및 학습 진행 상황을 시각적으로 표시한다.
- **코드**:
  ```python
  for folder in tqdm(folders, desc="폴더 로드 진행"):
      # ...
  for customer_id in tqdm(all_ids, desc="고객 데이터 통합 진행"):
      # ...
  ```
- **효과**: 사용자 경험을 개선하며, 코드 실행 상태를 모니터링하다.

---

### 13. 데이터 토큰화 및 커스텀 데이터셋 (Tokenization and Custom Dataset)
- **설명**: 텍스트 데이터를 토큰화하고, PyTorch Dataset 클래스를 커스터마이징하여 학습에 사용한다.
- **코드**:
  ```python
  def tokenize_function(texts):
      encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
      return encodings

  class CustomDataset(Dataset):
      def __init__(self, encodings, labels=None):
          self.encodings = {key: val.to(device) for key, val in encodings.items()}
          self.labels = labels
      def __getitem__(self, idx):
          item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
          if self.labels is not None:
              item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
          return item
      def __len__(self):
          return len(self.encodings['input_ids'])
  ```
- **효과**: 효율적인 데이터 전처리와 GPU 메모리 활용으로 학습 속도를 높이다.

---

### 14. 평가 메트릭 (Evaluation Metric)
- **설명**: F1 스코어를 사용하여 모델 성능을 평가한다.
- **코드**:
  ```python
  def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      f1 = f1_score(labels, preds, average='weighted')
      return {"f1": f1}
  ```
- **효과**: 클래스 불균형을 고려한 가중 평균 F1 스코어로 모델 성능을 정확히 측정하다.

---

### 15. 웜업 스텝 (Warmup Steps)
- **설명**: 초기 학습률을 점진적으로 증가시켜 학습을 안정화한다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      warmup_steps=20,
  )
  ```
- **효과**: 초기 불안정한 학습을 방지하고, 최적화 과정을 부드럽게 시작하다.

---

### 16. 그래디언트 누적 (Gradient Accumulation)
- **설명**: 배치 크기를 작게 유지하면서 그래디언트를 누적하여 큰 배치 효과를 얻는다.
- **코드**:
  ```python
  training_args = TrainingArguments(
      gradient_accumulation_steps=8,
  )
  ```
- **효과**: 메모리 제약을 극복하며, 더 큰 배치 크기로 학습한 것과 유사한 효과를 내다.


## 현실
사실, 딥러닝 시간과 성능은 서로 비례한다.  
가법게 한다 접근하면 이게 잘 될 수가 없다.  
더 많으 자원을 소모할수록, 더 많은 시간을 부을수록 완성되는 AI의 F1스코어는 올라간다.  
아쉽게도, 이거 안변하는거 같다.  
여기서 효율 치는거에 성공하면, DeepGEMM처럼 혁신에 성공하는 그런거다.  
역시 집에서의 딥러닝에는 한계가 너무 명확하다.  
연구실 가야하는 이유가 이렇게 또 하나 늘어나는 것 같다.  
