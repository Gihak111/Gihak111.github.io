---
layout: single
title:  "왜 AI의 모든 레이어에는 NormLayer로 정규화가 필요한가"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 왜 AI의 모든 레이어에는 NormLayer로 정규화가 필요한가

딥러닝 모델에서 정규화(Normalization)는 학습의 안정성과 성능을 높이는 핵심 기법이다. 특히, Layer Normalization(LayerNorm)은 현대 AI 모델, 특히 Transformer 기반 대형 언어 모델(LLM)에서 필수적으로 사용된다. 하지만 왜 모든 레이어에 NormLayer가 필요할까? 이번 글에서는 정규화의 필요성을 분석하고, LayerNorm의 역할과 구현을 코드로 살펴보며, 이를 LLM과 실무에 어떻게 적용하는지 상세히 다룬다. 문제 분석, 해결 구현, 실무 활용에 초점을 맞춘다.

### 1. 정규화가 필요한 이유

딥러닝 모델에서 레이어를 쌓을수록 입력 분포가 변동하며 학습이 불안정해지는 문제가 발생한다. 이를 **내부 공변량 이동(Internal Covariate Shift)**이라고 부르며, 특히 깊은 네트워크에서 학습 속도를 늦추고 성능을 저하시킨다. 정규화는 이러한 문제를 해결하기 위해 각 레이어의 출력 분포를 안정화한다.

- **문제 원인**
  - **분포 변동**: 레이어의 가중치 업데이트로 인해 각 레이어의 입력 분포가 달라짐. 예를 들어, 첫 번째 레이어의 출력이 다음 레이어의 입력인데, 이 분포가 계속 변하면 학습이 어려워짐.
  - **그라디언트 불안정**: 분포가 극단적으로 변하면 그라디언트가 폭발하거나 소실될 가능성 증가.
  - **느린 수렴**: 분포 변동으로 인해 적절한 학습률을 찾기 어려워 학습이 느려짐.
- **영향**
  - 모델이 수렴하지 않거나 성능이 낮아짐.
  - 긴 시퀀스나 깊은 네트워크에서 특히 심각한 성능 저하.

### 2. LayerNorm: 정규화의 핵심 메커니즘

LayerNorm은 각 레이어의 출력(또는 입력)을 정규화하여 평균을 0, 분산을 1로 맞춘 뒤, 학습 가능한 스케일(γ)과 이동(β) 파라미터를 적용한다. 이를 통해 학습 안정성을 높이고, 특히 Transformer와 같은 모델에서 필수적인 역할을 한다.

#### 2.1 LayerNorm의 구조와 역할

- **수식**
  - 입력: `x` (레이어 출력 벡터)
  - 평균: `μ = mean(x)`
  - 분산: `σ² = var(x)`
  - 정규화: `x_norm = (x - μ) / √(σ² + ε)`
  - 스케일링: `y = γ * x_norm + β`
  - 여기서 `ε`은 분모가 0이 되는 것을 방지하는 작은 상수.
- **특징**
  - **레이어별 독립성**: 배치 크기와 상관없이 각 샘플의 레이어 출력에 독립적으로 적용.
  - **그라디언트 안정화**: 정규화로 인해 출력 분포가 안정되어 그라디언트 흐름 개선.
  - **적응성**: 학습 가능한 `γ`와 `β`로 모델이 정규화된 출력을 조정 가능.

#### 2.2 코드로 구현하기: LayerNorm

PyTorch로 간단한 LayerNorm을 구현해 정규화 과정을 확인한다.

```python
import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 스케일
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 이동

    def forward(self, x):
        # x: (batch, seq_len, features)
        mean = x.mean(dim=-1, keepdim=True)  # 마지막 차원에서 평균
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 분산
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # 정규화
        return self.gamma * x_norm + self.beta  # 스케일링 및 이동

# 예시 실행
batch, seq_len, features = 32, 10, 512
x = torch.randn(batch, seq_len, features)
norm = CustomLayerNorm(features)
output = norm(x)
print("출력 크기:", output.shape)
print("출력 평균:", output.mean(dim=-1).abs().mean().item())  # 평균 ≈ 0
print("출력 분산:", output.var(dim=-1).mean().item())  # 분산 ≈ 1
```

이 코드는 입력 텐서를 정규화하고, 학습 가능한 파라미터로 출력 분포를 조정한다. 평균과 분산이 각각 0과 1에 가까워지는 것을 확인할 수 있다.

#### 2.3 LayerNorm의 이점

- **학습 안정성**: 출력 분포를 일정하게 유지해 학습률 선택이 쉬워짐.
- **그라디언트 흐름 개선**: 정규화로 인해 그라디언트가 폭발하거나 소실되지 않음.
- **배치 독립성**: BatchNorm과 달리 배치 크기에 영향을 받지 않아 소규모 배치나 시퀀스 데이터에 적합.

### 3. Transformer와 LLM에서의 LayerNorm 적용

LayerNorm은 Transformer의 핵심 구성 요소로, Self-Attention과 Feed-Forward 레이어 직후에 적용된다. 이를 통해 깊은 네트워크에서도 안정적인 학습이 가능하다.

#### 3.1 Transformer에서의 LayerNorm

Transformer는 Residual Connection과 LayerNorm을 결합해 학습 안정성을 극대화한다.

- **구조**
  - Self-Attention 후: `LayerNorm(x + Attention(x))`
  - Feed-Forward 후: `LayerNorm(x + FFN(x))`
- **역할**
  - **Residual Connection과의 시너지**: 덧셈 연산으로 그라디언트를 보존하고, LayerNorm으로 출력 분포를 안정화.
  - **긴 시퀀스 처리**: 시퀀스 길이에 상관없이 각 토큰의 출력이 일정한 분포를 유지.
- **코드 구현**

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Self-Attention + Residual + LayerNorm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# 예시 실행
embed_size, heads = 512, 8
model = TransformerBlock(embed_size, heads)
x = torch.randn(10, 32, embed_size)  # (시퀀스 길이, 배치, 임베딩)
output = model(x)
print("출력 크기:", output.shape)
```

`norm1`과 `norm2`는 각 레이어의 출력 분포를 정규화하여 학습을 안정화한다.

#### 3.2 LLM에서의 최적화

현대 LLM(예: GPT, LLaMA)에서는 LayerNorm을 다음과 같이 활용한다:

- **Pre-LayerNorm vs. Post-LayerNorm**: Transformer 초기 모델은 Attention 후 LayerNorm(Post-LayerNorm)을 사용했지만, 최신 모델은 Attention 전 LayerNorm(Pre-LayerNorm)을 선호. Pre-LayerNorm은 그라디언트 흐름을 더 안정화.
- **RMSNorm**: LayerNorm의 변형으로, 평균을 계산하지 않고 분산만으로 정규화. 계산 효율성을 높이며 성능 유지.
- **Mixed Precision Training과의 조합**: LayerNorm은 부동소수점 연산의 정밀도 문제를 완화해 혼합 정밀도 학습에서 안정성 제공.

### 4. 성능 평가: LayerNorm 유무 비교

LayerNorm의 효과를 확인하기 위해 간단한 시퀀스 분류 작업에서 LayerNorm 유무를 비교한다.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 데이터 준비
seq_len, n_samples = 50, 1000
X = torch.randn(n_samples, seq_len, 10)
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

# Transformer 모델 (LayerNorm 포함)
class TransformerClassifier(nn.Module):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerBlock(10, 2)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.sigmoid(self.fc(x[:, -1, :]))

# Transformer 모델 (LayerNorm 제외)
class TransformerNoNormClassifier(nn.Module):
    def __init__(self):
        super(TransformerNoNormClassifier, self).__init__()
        self.attention = nn.MultiheadAttention(10, 2)
        self.ffn = nn.Sequential(
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x + self.ffn(x)
        return self.sigmoid(self.fc(x[:, -1, :]))

# 학습 실행
print("LayerNorm 포함 Transformer 학습:")
train_model(TransformerClassifier())
print("\nLayerNorm 제외 Transformer 학습:")
train_model(TransformerNoNormClassifier())
```

LayerNorm 포함 모델은 손실이 더 빠르게 감소하며 안정적인 학습을 보여준다.

### 5. 실무 활용: 안정적인 문맥 이해

LayerNorm을 활용해 LLM의 문맥 이해 성능을 개선한다.

```python
def predict_context(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return tokenizer.decode(outputs.logits.argmax(-1), skip_special_tokens=True)

# 예시: Transformer 기반 모델로 문맥 예측
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")  # 예시 모델
tokenizer = AutoTokenizer.from_pretrained("t5-small")
text = "회의는 내일 오전 10시에 시작되는데..."
prediction = predict_context(model, tokenizer, text)
print("예측된 문맥:", prediction)
```

LayerNorm은 긴 시퀀스의 문맥을 안정적으로 처리하도록 돕는다.

### 결론
LayerNorm이 왜 모든 AI 레이어에 필수적인지, 그 역할과 구현을 코드와 함께 풀어봤다. 정규화 없이는 깊은 네트워크의 학습이 불안정해지고, Transformer와 LLM의 성능도 보장할 수 없다. 이해가 잘 안 되면 코드 돌려보면서 확인해보자. 딱 기억만 해 주면 돼!