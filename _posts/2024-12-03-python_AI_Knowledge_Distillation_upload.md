---
layout: single
title:  "지식증류를 통해 AI 성능 높이기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 성능저하.  
원래의 모델은 높은 정확도로, 문제 없이 사용 가능한 BERT 모델이였다.  
하지만, 이 모델을 모바일에 탑제하기 위해 모바일 벌트를 활용하고,  
ONNX로 변환하여 사용하였다.  
벌거 아닌 수정내용 같지만, 이 과정을 이행하는 중 엄청난 성능저하를 경험했다.  
정확도가 50%아래로 내려오는, 지도학습을 한게 맞나 싶을 정도의 결과가 나왔다.  
이러한 성능 저하가 발생한 원인을 추려보면, 당연히 베이스가 되는 모델을 BERT에서 MobileBERT로 교체한 점이 될 것이다.  
이 모델은 모바일에 최적화 되어 있지만, 그만큼 성능도 BERT에 비해 떨어진다.  
이러한 경우, 모델의 성능을 올리는데 있어 엄청 흥미로운 방법이 있다.  

## 지식 증류  
이 모델을 만드는 과정에서, 나는 모바일 버젼이 아닌, 그냥 버젼의 BERT 모델 또한 만들었다.  
이 BERT 모델의 경험치를 MobileBERT에게 주어 모델의 성능을 올리는 독특한 방법이 있다.  

### 1. 준비 단계: 교사 모델과 학생 모델 정의  
#### 교사 모델  
- 기존의 BERT 또는 BERT 기반의 미세 조정된(pretrained and fine-tuned) 모델을 사용한다.  
- 교사 모델은 크고 성능이 높은 모델이어야 한다.  

#### 학생 모델  
- MobileBERT 또는 비슷한 소형 모델을 사용한다.  
- MobileBERT는 Bottleneck 구조와 Inverted Bottleneck 구조를 사용하여 효율성을 극대화 한다.  


### 2. 손실 함수 구성
지식 증류에서는 여러 손실 함수의 조합을 사용하여 학생 모델이 교사 모델의 지식을 학습하도록 만든다.  

1. Soft Label Loss (KL Divergence):
   - 교사 모델의 출력 logits을 소프트맥스와 온도 매개변수를 적용해 소프트 레이블로 변환한다.  
   - 학생 모델의 출력과 교사 모델의 소프트 레이블 간의 KL Divergence를 계산한다.  
   ```python
   import torch.nn.functional as F
   
   def distillation_loss(student_logits, teacher_logits, temperature):
       teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
       student_probs = F.log_softmax(student_logits / temperature, dim=1)
       loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
       return loss
   ```  

2. Hidden State Loss:
   - 학생 모델의 각 층의 hidden states를 교사 모델과 정렬하여 학습한다.
   ```python
   def hidden_state_loss(student_hidden_states, teacher_hidden_states):
       loss = sum(F.mse_loss(s, t) for s, t in zip(student_hidden_states, teacher_hidden_states))
       return loss
   ```  

3. Attention Map Loss:
   - 어텐션 맵을 비교하여 학생 모델이 교사 모델의 어텐션 구조를 따라가도록 유도한다.  
   ```python
   def attention_loss(student_attentions, teacher_attentions):
       loss = sum(F.mse_loss(s, t) for s, t in zip(student_attentions, teacher_attentions))
       return loss
   ```  

4. Task-specific Loss:
   - 모델이 수행하는 특정 작업에 대한 크로스 엔트로피 손실이다.

### 3. 학습 과정 설계
1. 데이터 준비:
   - 지식 증류에는 대량의 비레이블 데이터가 유용하다.  
   - 레이블 데이터와 비레이블 데이터를 혼합하여 사용하자.  

2. Progressive Transfer:
   - 첫 단계: 교사 모델의 초기 층(hidden states, attention)을 먼저 증류.
   - 두 번째 단계: 더 깊은 층(hidden states, logits)으로 확장.

3. 학습 루프:
   ```python
   for epoch in range(epochs):
       for batch in dataloader:
           student_outputs = student_model(**batch)
           teacher_outputs = teacher_model(**batch)

           # Calculate losses
           soft_label_loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=4)
           hidden_loss = hidden_state_loss(student_outputs.hidden_states, teacher_outputs.hidden_states)
           attention_loss = attention_loss(student_outputs.attentions, teacher_outputs.attentions)

           # Combine losses
           total_loss = soft_label_loss + hidden_loss + attention_loss

           # Backpropagation and optimization
           optimizer.zero_grad()
           total_loss.backward()
           optimizer.step()
   ```  

### 4. Pre-training Distillation  
MobileBERT 논문에서는 미리 Masked Language Modeling(MLM)과 Next Sentence Prediction(NSP)을 교사 모델에서 학생 모델로 전이하는 Pre-training Distillation을 강조한다.  
- MLM 작업: 입력 문장의 일부를 [MASK]로 바꾸고 이를 예측.  
- NSP 작업: 두 문장이 연결될 확률을 예측.  

### 5. **모델 ONNX 변환**
MobileBERT 모델을 학습한 후 ONNX로 변환할 때, TensorRT와 같은 고성능 엔진을 사용하여 추론 속도와 효율성을 극대화 하자.  
아래 글을 확인하자.  
[ONNX로 하여금 작동속도, 학습속도 올리기](https://gihak111.github.io/pynb/2024/11/14/python_AI_trun_into_onnx_upload.html)  

이 글들도 확인하자. 도음이 된다.  
[MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)  
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  
[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)  
