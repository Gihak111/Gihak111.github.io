---
layout: single
title:  "AI 경량화를 위한 SSM, SLM, FlashAttention"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## SSM, SLM, FlashAttention: 효율적인 AI 모델의 핵심 기술

현대 AI, 특히 대형 언어 모델(LLM)은 높은 성능을 자랑하지만, 계산 비용과 메모리 사용량이 문제로 남는다. **Structured State Space Models (SSM)**, **Small Language Models (SLM)**, 그리고 **FlashAttention**은 이러한 문제를 해결하며 효율성과 성능을 동시에 잡는 기술이다. 이 글에서는 SSM, SLM, FlashAttention의 필요성과 작동 원리를 분석하고, PyTorch로 구현하며, Transformer 및 LLM에서의 활용 사례를 실무 관점에서 다룬다. 문제 정의, 기술 구현, 실무 적용에 초점을 맞춘다.

### 1. 효율성 문제와 기술의 필요성

Transformer 기반 LLM은 Self-Attention 메커니즘으로 강력한 성능을 발휘하지만, 시퀀스 길이에 따라 시간과 메모리 복잡도가 **이차적(quadratic, O(N²))**으로 증가한다. 이는 긴 시퀀스 처리와 대규모 모델 학습에서 병목 현상을 일으킨다. 또한, 자원 효율성을 높이고 소형 디바이스에서의 배포를 가능하게 하는 경량 모델의 필요성도 커지고 있다.

- **문제 원인**
  - **Attention의 비효율성**: Self-Attention은 모든 토큰 쌍을 계산하므로 시퀀스 길이(N)에 따라 메모리와 계산량이 급증.
  - **대규모 모델의 자원 소모**: LLM은 수십억 개의 파라미터로 인해 GPU 메모리와 전력을 과도하게 요구.
  - **소형 디바이스 제약**: 모바일 기기나 엣지 디바이스에서는 대규모 모델을 실행하기 어려움.
- **영향**
  - 긴 시퀀스 처리 시 학습 속도 저하 및 메모리 초과(OOM) 문제.
  - 고비용으로 인해 소규모 팀이나 개인 개발자의 접근성 제한.
  - 실시간 응용에서의 지연(latency) 증가.

### 2. SSM, SLM, FlashAttention: 기술적 개요

이 세 기술은 각각 독특한 방식으로 AI 모델의 효율성을 높인다. SSM은 Attention을 대체하는 선형 복잡도 메커니즘을 제공하고, SLM은 경량화된 모델로 성능을 유지하며, FlashAttention은 Attention 연산을 최적화한다.

#### 2.1 Structured State Space Models (SSM)

SSM은 시퀀스 데이터를 선형 시간 복잡도(O(N))로 처리하는 대안으로, Transformer의 Attention을 대체한다. Mamba와 같은 SSM 모델은 **상태 공간(state space)**을 활용해 입력 데이터를 압축된 상태로 처리한다.

- **수식**
  - 상태 전이: `h(t) = A h(t-1) + B x(t26;`
  - 출력: `y(t) = C h(t) + D x(t)`
  - 여기서 `A`, `B`, `C`, `D`는 학습 가능한 파라미터, `x(t)`는 입력, `h(t)`는 상태, `y(t)`는 출력.
- **특징**
  - **선형 복잡도**: Attention의 O(N²)에 비해 O(N)으로 긴 시퀀스 처리에 적합.
  - **선택적 메커니즘**: 입력에 따라 동적으로 상태를 조정(Mamba의 S6).
  - **하드웨어 최적화**: GPU 메모리 계층(SRAM, HBM)을 활용한 병렬 스캔 알고리즘.

#### 2.2 Small Language Models (SLM)

SLM은 파라미터 수가 적은 경량 언어 모델로, 대규모 LLM의 성능을 유지하면서 자원 소모를 줄인다. 예: Phi-3, TinyLlama.

- **특징**
  - **효율적 학습**: 적은 데이터와 자원으로 고성능 달성.
  - **지식 증류**: 대규모 모델의 지식을 소형 모델로 전이.
  - **엣지 디바이스 배포**: 모바일, IoT 기기에서 실행 가능.
- **장점**
  - 빠른 추론 속도와 낮은 전력 소모.
  - 소규모 팀에서도 활용 가능한 저비용 모델.

#### 2.3 FlashAttention

FlashAttention은 Attention 연산의 메모리와 계산 효율성을 높이는 IO-aware 알고리즘이다.[](https://arxiv.org/abs/2205.14135)

- **메커니즘**
  - **타일링(Tiling)**: 입력 행렬(Q, K, V)을 블록으로 분할해 SRAM에서 연산.
  - **재계산(Recomputation)**: 중간 결과를 저장하지 않고 필요 시 재계산.
  - **커널 퓨전(Kernel Fusion)**: 여러 연산을 단일 커널로 통합해 메모리 접근 감소.
- **효과**
  - 메모리 사용량: O(N²) → O(N).
  - 속도: 기존 Attention 대비 2~4배 빠름.

#### 2.4 코드로 구현하기: SSM과 FlashAttention

PyTorch로 간단한 SSM과 FlashAttention 기반 Transformer 블록을 구현한다.

```python
import torch
import torch.nn as nn
from flash_attn import flash_attention  # FlashAttention 패키지 가정

# 간단한 SSM 구현
class SSM(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(SSM, self).__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))
        self.C = nn.Parameter(torch.randn(input_dim, state_dim))
        self.D = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, input_dim = x.shape
        state_dim = self.A.shape[0]
        h = torch.zeros(batch, state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = h @ self.A + x[:, t] @ self.B
            y = h @ self.C + x[:, t] @ self.D
            outputs.append(y)
        return torch.stack(outputs, dim=1)  # (batch, seq_len, input_dim)

# FlashAttention 기반 Transformer 블록
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.head_dim = embed_size // heads
        self.heads = heads

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        batch, seq_len, _ = x.shape
        # FlashAttention 적용
        qkv = x.view(batch, seq_len, self.heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_output = flash_attention(
            q, k, v, causal=True, dropout=0.1
        ).view(batch, seq_len, -1)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

# 예시 실행
batch, seq_len, embed_size, heads = 32, 128, 512, 8
x = torch.randn(batch, seq_len, embed_size)
ssm = SSM(embed_size, 64)
transformer = TransformerBlock(embed_size, heads)
ssm_output = ssm(x)
trans_output = transformer(x)
print("SSM 출력 크기:", ssm_output.shape)
print("Transformer 출력 크기:", trans_output.shape)
```

이 코드는 SSM의 상태 전이와 FlashAttention 기반 Transformer 블록을 구현한다. FlashAttention은 실제 패키지를 가정했으며, 메모리 효율성을 높인다.

### 3. Transformer와 LLM에서의 적용

SSM, SLM, FlashAttention은 Transformer와 LLM에서 다음과 같이 활용된다.

#### 3.1 SSM in Transformer

SSM은 Attention을 대체하거나 하이브리드 형태로 사용된다(예: Jamba).[](https://qdata.github.io/deep2Read/fmefficient/L26/)

- **구조**
  - Mamba 블록: SSM과 MLP를 결합.
  - Residual Connection과 LayerNorm으로 안정성 유지.
- **역할**
  - 긴 시퀀스 처리: 선형 복잡도로 최대 256K 토큰 처리.
  - 효율적 추론: KV 캐시가 없어 배치 크기 제약 감소.

#### 3.2 SLM in LLM

SLM은 LLM의 대안 또는 보완으로 사용된다.

- **Phi-3 예시**: 3.8B 파라미터로 LLaMA-7B와 유사한 성능.
- **지식 증류**: GPT-4 등의 대규모 모델에서 학습.
- **코드 구현**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phi-3 모델 로드
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 텍스트 생성
text = "AI의 미래는..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### 3.3 FlashAttention in Transformer

FlashAttention은 모든 현대 LLM(GPT-4, LLaMA 등)에 필수적이다.[](https://medium.com/%40lixue421/paper-explained-2-flashattention-172bc2e90440)

- **적용**
  - Self-Attention 레이어에 통합.
  - Pre-LayerNorm과 결합해 안정성 강화.
- **효과**
  - 학습 속도: BERT-large 15% 향상.
  - 메모리 절감: 시퀀스 길이 4K에서 20배 절약.

### 4. 성능 평가: 비교 실험

SSM, SLM, FlashAttention의 효과를 확인하기 위해 간단한 시퀀스 분류 작업에서 비교한다.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 데이터 준비
seq_len, n_samples, embed_size = 128, 1000, 512
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

# SSM 기반 모델
class SSMClassifier(nn.Module):
    def __init__(self):
        super(SSMClassifier, self).__init__()
        self.ssm = SSM(embed_size, 64)
        self.fc = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ssm(x)
        return self.sigmoid(self.fc(x[:, -1, :]))

# FlashAttention 기반 Transformer 모델
class TransformerClassifier(nn.Module):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerBlock(embed_size, 8)
        self.fc = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        return self.sigmoid(self.fc(x[:, -1, :]))

# 학습 실행
print("SSM 기반 모델 학습:")
train_model(SSMClassifier())
print("\nFlashAttention 기반 Transformer 학습:")
train_model(TransformerClassifier())
```

SSM 기반 모델은 선형 복잡도로 빠른 학습을, FlashAttention 기반 모델은 메모리 효율성과 안정성을 보여준다.

### 5. 실무 활용: 효율적인 문맥 이해

SLM과 FlashAttention을 활용해 긴 문맥을 효율적으로 처리한다.

```python
def predict_context(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 예시: Phi-3로 문맥 예측
text = "회의는 내일 오전 10시에 시작되는데..."
prediction = predict_context(model, tokenizer, text)
print("예측된 문맥:", prediction)
```

SLM은 소규모 리소스로도 긴 문맥을 안정적으로 처리하며, FlashAttention은 메모리 효율성을 높인다.

### 결론
SSM, SLM, FlashAttention은 AI 모델의 효율성을 혁신적으로 개선한다. SSM은 선형 복잡도로 긴 시퀀스를 처리하고, SLM은 경량화로 엣지 디바이스 배포를 가능케 하며, FlashAttention은 Attention 연산을 최적화한다. 이 기술들 없이는 현대 LLM의 성능과 확장성을 보장하기 어렵다. 따라서 효율적인 AI 개발을 위해 이들은 필수적이다. 