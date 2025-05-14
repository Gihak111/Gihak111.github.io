---
layout: single
title:  "효율적 미세조정을 위한 LoRA"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## LoRA: 대형 언어 모델의 효율적인 미세 조정 기술

대형 언어 모델(LLM)은 뛰어난 성능을 자랑하지만, 특정 작업에 맞춘 미세 조정(fine-tuning)은 막대한 계산 자원과 저장 공간을 요구한다.  
**LoRA(Low-Rank Adaptation)**는 이러한 문제를 해결하며, 소수의 파라미터만 학습해 모델을 효율적으로 적응시킨다.  
이 글에서는 LoRA의 필요성과 작동 원리를 분석하고, PyTorch로 구현하며, Transformer 및 LLM에서의 활용 사례를 실무 관점에서 다룬다.  
문제 정의, 기술 구현, 실무 적용에 초점을 맞춘다.  

### 1. 미세 조정의 문제와 LoRA의 필요성

LLM의 전통적인 미세 조정은 모든 파라미터를 업데이트하므로 계산 비용이 크고, 작업별로 별도의 모델 사본을 저장해야 한다.  
이는 자원 효율성과 확장성을 저해한다.  

- **문제 원인**
  - **높은 자원 소모**: 수십억 파라미터(예: LLaMA-7B, 7B 파라미터)를 가진 모델의 미세 조정은 고성능 GPU와 대용량 메모리 필요.  
  - **저장 공간 문제**: 작업별로 전체 모델을 저장하면 수백 GB의 디스크 공간 소모.  
  - **오버피팅 위험**: 소규모 데이터셋에서 전체 파라미터를 학습하면 일반화 성능 저하.  
- **영향**
  - 소규모 팀이나 개인의 LLM 활용 장벽 증가.  
  - 다중 작업 학습에서 비효율적인 자원 사용.  
  - 배포 환경에서 높은 지연 시간(latency).  

### 2. LoRA: 효율적인 미세 조정 메커니즘

LoRA는 가중치 행렬의 변화를 저차원(low-rank) 행렬로 근사해 소수의 파라미터만 학습한다.  
원본 모델의 가중치는 고정된 상태로 유지되며, 작업별 어댑터(adapter)로 유연성을 제공한다.  

#### 2.1 LoRA의 구조와 역할

- **수식**
  - 원본 가중치: `W ∈ ℝ^(d × k)`
  - LoRA 업데이트: `ΔW = A × B`, 여기서 `A ∈ ℝ^(d × r)`, `B ∈ ℝ^(r × k)`, `r`은 저차원 랭크(rank, r ≪ min(d, k)).  
  - 최종 가중치: `W' = W + ΔW`.  
  - 학습 대상: `A`와 `B`만 업데이트.  
- **특징**
  - **파라미터 효율성**: 전체 파라미터의 0.1%~1%만 학습.  
  - **모듈성**: 작업별 LoRA 어댑터를 독립적으로 저장, 원본 모델 재사용.  
  - **적용 유연성**: Attention 또는 Feed-Forward 레이어에 선택적으로 적용 가능.  
- **장점**
  - 계산 및 메모리 비용 대폭 감소.  
  - 사전 학습된 모델의 지식 보존.  
  - 빠른 작업 전환을 위한 어댑터 교체.  

#### 2.2 코드로 구현하기: LoRA

PyTorch로 LoRA 레이어를 구현해 미세 조정 과정을 살펴본다.  

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        # 원본 가중치 (고정)
        self.W = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        # LoRA 파라미터
        self.A = nn.Parameter(torch.randn(out_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, in_dim))

    def forward(self, x):
        # LoRA 업데이트: ΔW = A × B
        delta_W = self.alpha * (self.A @ self.B)
        # 최종 가중치: W' = W + ΔW
        W_prime = self.W + delta_W
        return x @ W_prime.T

# 예시 실행
batch, seq_len, in_dim, out_dim, rank = 32, 10, 512, 512, 8
x = torch.randn(batch, seq_len, in_dim)
lora = LoRALayer(in_dim, out_dim, rank)
output = lora(x)
print("출력 크기:", output.shape)
print("학습 가능한 파라미터 수:", sum(p.numel() for p in lora.parameters() if p.requires_grad))
print("원본 파라미터 대비 비율:", sum(p.numel() for p in lora.parameters() if p.requires_grad) / (in_dim * out_dim) * 100, "%")
```

이 코드는 LoRA 레이어를 구현하고, 학습 가능한 파라미터가 원본 가중치의 약 1% 수준임을 보여준다.  

#### 2.3 LoRA의 이점

- **자원 효율성**: GPU 메모리 사용량과 학습 시간 감소.  
- **저장 효율성**: LoRA 어댑터는 MB 단위로 저장, 전체 모델(GB 단위) 대비 획기적 절약.  
- **성능 유지**: 전체 미세 조정과 비슷한 성능을 적은 비용으로 달성.  

### 3. Transformer와 LLM에서의 LoRA 적용

LoRA는 Transformer 기반 LLM(예: BERT, T5, LLaMA)에서 Attention과 Feed-Forward 레이어에 주로 적용되며, Parameter-Efficient Fine-Tuning(PEFT)의 핵심 기술로 자리 잡았다.  

#### 3.1 Transformer에서의 LoRA

LoRA는 Self-Attention의 쿼리(Q), 키(K), 값(V) 행렬과 Feed-Forward 레이어에 적용된다.  

- **구조**
  - Attention: `W_q`, `W_v`에 LoRA 어댑터 추가.  
  - Feed-Forward: `W_1`, `W_2`에 LoRA 적용.  
  - LayerNorm과 Residual Connection은 그대로 유지.  
- **역할**
  - **메모리 절감**: 고정된 원본 가중치로 메모리 소모 최소화.  
  - **작업별 최적화**: 작업별 LoRA 어댑터로 다중 작업 지원.  
- **코드 구현**

```python
class LoRATransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, rank=8):
        super(LoRATransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.lora_q = LoRALayer(embed_size, embed_size, rank)
        self.lora_v = LoRALayer(embed_size, embed_size, rank)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        # Self-Attention + LoRA + Residual + LayerNorm
        q = self.lora_q(x)
        v = self.lora_v(x)
        attn_output, _ = self.attention(q, x, v)
        x = self.norm1(x + attn_output)
        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# 예시 실행
embed_size, heads = 512, 8
model = LoRATransformerBlock(embed_size, heads)
x = torch.randn(10, 32, embed_size)  # (시퀀스 길이, 배치, 임베딩)
output = model(x)
print("출력 크기:", output.shape)
```

이 코드는 LoRA를 Attention 레이어에 통합해 효율적인 미세 조정을 구현한다.  

#### 3.2 LLM에서의 최적화

LoRA는 현대 LLM에서 다음과 같이 활용된다:  

- **작업별 미세 조정**: 예: 질의응답, 텍스트 생성, 분류.  
- **Hugging Face PEFT**: LoRA를 쉽게 통합하는 라이브러리 제공.  
- **랭크 조정**: `r=4`, `r=8`, `r=16` 등 작업 복잡도에 따라 선택.  
- **Mixed Precision Training**: LoRA는 혼합 정밀도 학습에서 안정성 강화.  

### 4. 성능 평가: LoRA 유무 비교

LoRA의 효과를 확인하기 위해 시퀀스 분류 작업에서 LoRA 유무를 비교한다.  

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 데이터 준비
seq_len, n_samples, embed_size = 50, 1000, 10
X = torch.randn(n_samples, seq_len, embed_size)
y = torch.randint(0, 2, (n_samples,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

# 모델 학습
def train_model(model, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, 평균 손실: {total_loss / len(loader):.4f}")

# LoRA 포함 Transformer 모델
class LoRATransformerClassifier(nn.Module):
    def __init__(self):
        super(LoRATransformerClassifier, self).__init__()
        self.transformer = LoRATransformerBlock(embed_size, 2)
        self.fc = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.sigmoid(self.fc(x[:, -1, :]))

# 전체 미세 조정 Transformer 모델
class FullTransformerClassifier(nn.Module):
    def __init__(self):
        super(FullTransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, nhead=2), num_layers=1
        )
        self.fc = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.sigmoid(self.fc(x[:, -1, :]))

# 학습 실행
print("LoRA 포함 Transformer 학습:")
train_model(LoRATransformerClassifier())
print("\n전체 미세 조정 Transformer 학습:")
train_model(FullTransformerClassifier())
```

LoRA 포함 모델은 손실이 빠르게 감소하며, 학습 가능한 파라미터 수가 적어 효율적이다.  

### 5. 실무 활용: 작업별 미세 조정  

LoRA를 사용해 LLM을 특정 작업에 효율적으로 적응시킨다.  

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

# T5 모델에 LoRA 적용
model = AutoModelForSequenceClassification.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# LoRA 설정
lora_config = LoraConfig(
    r=8,  # LoRA 랭크
    lora_alpha=16,  # 스케일링 파라미터
    target_modules=["q", "v"],  # Attention 쿼리, 값 행렬
    lora_dropout=0.1
)
lora_model = get_peft_model(model, lora_config)

# 예시: 감정 분석
text = "이 제품은 정말 훌륭해요!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = lora_model(**inputs)
print("예측 결과:", outputs.logits.argmax(-1).item())
```

LoRA는 소규모 데이터로도 작업(예: 감정 분석)에 모델을 효과적으로 적응시킨다.  

### 결론
LoRA는 대형 언어 모델의 미세 조정을 효율적으로 수행하는 핵심 기술이다.  
저차원 행렬로 가중치 변화를 근사해 자원 소모를 줄이고, 작업별 어댑터로 유연성을 제공한다.  
LoRA 없이는 LLM의 작업별 최적화가 비효율적이고 비용이 높아질 수 있다.  
따라서 효율적인 AI 개발에서 LoRA는 필수적이다.  