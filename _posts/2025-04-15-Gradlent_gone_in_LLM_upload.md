---
layout: single
title:  "그라디언트 증발 문제와 LSTM 및 LLM에서의 해결 방안"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 그라디언트 증발 문제 이해와 LSTM, LLM에서의 해결 방안

그라디언트 증발(Vanishing Gradient)은 딥러닝 모델, 특히 순환 신경망(RNN)을 학습시킬 때 자주 마주치는 문제다.  
긴 시퀀스 데이터를 처리할 때, 그라디언트가 점차 작아지며 학습이 어려워지는 현상이다.  
이 문제를 해결하기 위해 LSTM(Long Short-Term Memory)이 고안되었으며, 현대 대형 언어 모델(LLM)에서도 이러한 개념이 발전적으로 적용되고 있다.  
이번 글에서는 그라디언트 증발 문제를 분석하고, LSTM에서의 덧셈 기반 해결 방안을 코드로 구현하며, 이를 LLM에 어떻게 녹였는지 상세히 다룬다.  
문제 분석, 해결 구현, 실무 활용에 초점을 맞춘다.  

### 1. 그라디언트 증발 문제란?

그라디언트 증발은 역전파 과정에서 그라디언트가 시간 스텝을 거슬러 올라갈수록 급격히 작아지는 현상이다.  
특히, 기본 RNN에서는 시그모이드나 tanh 같은 활성화 함수의 특성상 그라디언트가 0에 가까워지며 긴 시퀀스의 초기 정보를 학습하지 못한다.  

- **원인**
  - 활성화 함수의 기울기 감소: 시그모이드의 경우 최대 기울기가 0.25로, 반복 곱셈으로 값이 급감.  
  - 긴 의존성: 문장에서 "나는 어제 책을 읽었는데..."와 같은 긴 문맥을 이해하려면 초기 단어의 정보가 필요하지만, 그라디언트가 소실되며 이를 반영하지 못함.  
- **영향**
  - 모델이 초기 시퀀스 정보를 잊어 문맥 이해 실패.  
  - 학습이 정체되거나 수렴하지 않음.  

### 2. LSTM에서의 해결: 덧셈 기반 메커니즘

LSTM은 그라디언트 증발 문제를 해결하기 위해 게이트 메커니즘과 덧셈 연산을 도입했다.  
셀 상태(Cell State)를 통해 정보를 장기적으로 전달하며, 곱셈 대신 덧셈을 활용해 그라디언트 흐름을 유지한다.  

#### 2.1 LSTM 구조와 역할

- **셀 상태 (Cell State)**
  - **역할**: 긴 시퀀스의 정보를 보존하는 메모리 역할.  
  - **구성**: `C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`  
  - **특징**: 덧셈 연산(`+`)으로 새로운 정보를 추가하므로 그라디언트가 곱셈으로만 축소되지 않음.  
- **입력 게이트 (Input Gate)**
  - **역할**: 새로운 정보를 얼마나 반영할지 결정.  
  - **구성**: `i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)`  
- **망각 게이트 (Forget Gate)**
  - **역할**: 이전 셀 상태에서 불필요한 정보를 제거.  
  - **구성**: `f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)`  
- **출력 게이트 (Output Gate)**
  - **역할**: 현재 시점의 출력(은닉 상태)을 결정.  
  - **구성**: `o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)`  

#### 2.2 코드로 구현하기: LSTM의 덧셈 연산

간단한 LSTM 셀을 PyTorch로 구현해 덧셈 기반 그라디언트 흐름을 확인한다.  

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 게이트 가중치
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # 입력 게이트
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # 망각 게이트
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # 출력 게이트
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # 셀 상태 후보

    def forward(self, x, hidden, cell):
        # x: (batch, input_size), hidden: (batch, hidden_size), cell: (batch, hidden_size)
        combined = torch.cat((x, hidden), dim=1)

        # 게이트 계산
        i_t = torch.sigmoid(self.W_i(combined))  # 입력 게이트
        f_t = torch.sigmoid(self.W_f(combined))  # 망각 게이트
        o_t = torch.sigmoid(self.W_o(combined))  # 출력 게이트
        c_tilde = torch.tanh(self.W_c(combined)) # 셀 상태 후보

        # 셀 상태 업데이트 (덧셈 연산)
        cell = f_t * cell + i_t * c_tilde
        hidden = o_t * torch.tanh(cell)

        return hidden, cell

# 예시 실행
input_size, hidden_size = 10, 20
model = SimpleLSTM(input_size, hidden_size)
x = torch.randn(32, input_size)  # 배치 크기 32
h, c = torch.zeros(32, hidden_size), torch.zeros(32, hidden_size)
h, c = model(x, h, c)
print("은닉 상태 크기:", h.shape, "셀 상태 크기:", c.shape)
```  

이 코드에서 `cell = f_t * cell + i_t * c_tilde`는 덧셈 연산을 통해 셀 상태를 업데이트하며, 그라디언트가 곱셈으로만 축소되지 않도록 보장한다.  

#### 2.3 덧셈의 이점

- **그라디언트 보존**: 곱셈 연산(`f_t * cell`)은 망각 게이트로 정보 선택을 조절하지만, 덧셈(`+ i_t * c_tilde`)은 새로운 정보 추가 시 그라디언트를 직접 전달.  
- **장기 의존성 학습**: 초기 시퀀스 정보가 셀 상태에 누적되어 긴 문맥을 학습 가능.  

### 3. LLM에서의 발전: Transformer와 그라디언트 관리

LSTM의 덧셈 기반 해결은 현대 LLM의 Transformer 구조에서도 영향을 미쳤다.  
Transformer는 RNN을 대체하며 그라디언트 증발 문제를 더욱 효과적으로 해결했다.  
이를 LLM에 어떻게 녹였는지 살펴본다.

#### 3.1 Transformer의 핵심 메커니즘

- **Self-Attention**
  - **역할**: 시퀀스 내 모든 토큰 간 관계를 병렬적으로 계산.  
  - **구성**: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`  
  - **그라디언트 이점**: RNN처럼 순차적 곱셈이 없어 그라디언트가 시간 스텝에 따라 축소되지 않음.  
- **Residual Connection**
  - **역할**: 레이어 입력을 출력에 직접 더함 (`x + F(x)`).  
  - **구성**: `LayerNorm(x + Attention(x))`  
  - **그라디언트 이점**: 덧셈 연산으로 그라디언트가 입력까지 직접 전달되어 소실 방지.  
- **Layer Normalization**
  - **역할**: 출력 분포를 안정화해 학습 속도를 높임.  
  - **구성**: `LayerNorm(x) = (x - μ) / σ * γ + β`  
  - **그라디언트 이점**: 정규화로 그라디언트 폭발/소실을 완화.  

#### 3.2 LSTM의 덧셈에서 Transformer로의 연결

LSTM의 셀 상태 덧셈은 Transformer의 Residual Connection으로 발전했다.  
Residual Connection은 다음과 같이 구현된다.  

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
        # Self-Attention + Residual Connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)  # 덧셈 연산
        # Feed-Forward + Residual Connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # 덧셈 연산
        return x

# 예시 실행
embed_size, heads = 512, 8
model = TransformerBlock(embed_size, heads)
x = torch.randn(10, 32, embed_size)  # (시퀀스 길이, 배치, 임베딩)
output = model(x)
print("출력 크기:", output.shape)
```

`x + attn_output`와 `x + ffn_output`에서 덧셈 연산은 LSTM의 셀 상태 업데이트와 유사하게 그라디언트를 보존한다.  

#### 3.3 LLM에서의 적용

현대 LLM(예: GPT, LLaMA)에서는 Transformer를 기반으로 추가적인 최적화가 적용된다.  

- **Scaled Dot-Product Attention**: `√d_k`로 나눠 그라디언트 안정화.  
- **Gradient Clipping**: 그라디언트 크기를 제한해 폭발 방지.  
- **Mixed Precision Training**: 계산 효율성을 높이며 그라디언트 손실 최소화.  
- **Positional Encoding**: LSTM처럼 순서 정보를 추가하되, 고정된 사인/코사인 함수로 그라디언트 문제 완화.  

### 4. 성능 평가: LSTM vs. Transformer

LSTM과 Transformer의 그라디언트 흐름을 비교하기 위해 간단한 시퀀스 분류 작업을 수행한다.  

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
            output, _ = model(data, model.init_hidden(data.size(0)))
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, 평균 손실: {total_loss / len(loader):.4f}")

# LSTM 모델
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = SimpleLSTM(10, 20)
        self.fc = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        h, c = hidden
        for t in range(x.size(1)):
            h, c = self.lstm(x[:, t, :], h, c)
        return self.sigmoid(self.fc(h)), (h, c)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, 20), torch.zeros(batch_size, 20)

# Transformer 모델
class TransformerClassifier(nn.Module):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerBlock(10, 2)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.sigmoid(self.fc(x[:, -1, :])), None

    def init_hidden(self, batch_size):
        return None

# 학습 실행
print("LSTM 학습:")
train_model(LSTMClassifier())
print("\nTransformer 학습:")
train_model(TransformerClassifier())
```

Transformer는 병렬 처리와 Residual Connection으로 더 안정적인 그라디언트 흐름을 보여준다.  

### 5. 실무 활용: 문맥 이해 개선

LSTM과 Transformer의 그라디언트 관리 기법을 활용해 문맥 이해를 개선한다.  

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
text = "나는 어제 책을 읽었는데, 오늘은..."
prediction = predict_context(model, tokenizer, text)
print("예측된 문맥:", prediction)
```

Transformer의 안정적인 그라디언트 흐름은 긴 문맥을 더 잘 이해하도록 돕는다.  


### 결론
LSTM에서 덧셈으로 시작된 아이디어가 Transformer와 LLM까지 어떻게 이어졌는지 알기 쉽게 풀어봤다.  
이해가 잘 안 되면 코드 돌려보면서 확인해보자.  