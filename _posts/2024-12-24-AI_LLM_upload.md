---
layout: single
title:  "LLM 완벽이해하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Transformer가 시대를 바꾼다.

### 대화 생성형 AI 모델 설계 가이드

대화 생성형 AI 모델은 복잡한 설계와 학습 과정을 거쳐 사용자의 요구를 정확히 이해하고 자연스러운 응답을 생성할 수 있도록 개발된다.  
이를 위해 다양한 기술적 요소를 체계적으로 설계해야 한다.  
먼저, 은닉층 구성, 활성화 함수, 학습 알고리즘, 상황별 활용 방법, 그리고 구현 예제까지 자세히 살펴보자.  

### 1. 모델 아키텍처 설계

#### 1.1 데이터 전처리
- 데이터 수집: 모델 학습에 적합한 대규모 데이터셋이 필요하다. 예: Reddit 대화 데이터, 영화 대본, SNS 채팅 데이터.  
- 데이터 클리닝: 불완전하거나 잘못된 데이터를 제거한다. 중복 제거, 문법 교정, 비속어 필터링 등도 포함된다.  
- 토큰화(Tokenization): 텍스트 데이터를 숫자 벡터로 변환한다. Transformer 기반 모델에서는 Byte Pair Encoding(BPE), WordPiece 등을 활용한다.  

#### 1.2 은닉층 구성
은닉층은 데이터의 패턴을 학습하는 신경망의 핵심 구성 요소다.  

1. 피드포워드 신경망(Feedforward Neural Network):  
   - 은닉층 뉴런들은 이전 층의 모든 뉴런과 연결된다.  
   - 복잡한 패턴을 학습하지만, 시퀀스 데이터를 다루기에는 적합하지 않다.  

2. RNN (Recurrent Neural Networks):
   - 시퀀스 데이터(텍스트, 오디오 등)를 처리하기 위해 설계되었다.  
   - 이전 상태(hidden state)를 다음 시점으로 전달하여 데이터 간 연속성을 학습한다.  

3. LSTM (Long Short-Term Memory):  
   - RNN의 단점을 보완하여 장기적인 의존성을 학습할 수 있다.  
   - 셀 상태(cell state)를 유지하여 중요한 정보를 장기적으로 보존한다.  

4. Transformer:  
   - Self-Attention 메커니즘을 기반으로 시퀀스 전체에서 중요한 정보를 동적으로 학습한다.  
   - 병렬 처리가 가능하여 빠른 학습 속도를 자랑한다.  
   - GPT, BERT 같은 최신 대화 모델의 핵심 구조이다.  

#### 1.3 활성화 함수  
활성화 함수는 뉴런의 출력을 변환하여 다음 층으로 전달한다.  

1. ReLU (Rectified Linear Unit):  
   - 양수는 그대로 출력하고, 음수는 0으로 변환한다.  
   - 기울기 소실 문제를 완화하며, 계산 효율성이 높다.  

2. Leaky ReLU:  
   - 음수 영역에 대해 작은 기울기를 남겨 뉴런이 "죽는(dead neurons)" 문제를 방지한다.  

3. 소프트맥스(Softmax):  
   - 주로 출력층에서 사용되며, 각 클래스에 대한 확률 분포를 출력한다.  


### 2. 모델 학습 과정  

#### 2.1 손실 함수(Loss Function)  
모델의 예측과 실제 값 간의 차이를 측정한다.  
- 크로스 엔트로피 손실(Cross-Entropy Loss): 분류 문제에 사용.  
- MSE (Mean Squared Error): 회귀 문제에 사용.  

#### 2.2 최적화 알고리즘  
모델의 가중치를 업데이트하여 손실을 최소화한다.  
- Adam: 학습률을 동적으로 조정하며 널리 사용된다.  
- SGD: 단순하지만 강력한 성능을 제공한다.  

#### 2.3 학습 파라미터  
- 에포크(Epoch): 전체 데이터셋을 몇 번 반복 학습할지 결정.  
- 배치 크기(Batch Size): 한 번의 업데이트에 사용할 데이터 수.  
- 학습률(Learning Rate): 가중치 업데이트의 크기를 조정.  


### 3. 상황별 모델 활용  

1. 대규모 데이터셋:  
   - Transformer 기반 모델(GPT, BERT 등)을 활용한다.  
   - 병렬 처리가 가능하므로 빠른 학습이 가능하다.  

2. 긴 시퀀스 데이터:  
   - LSTM, GRU 같은 RNN 변형 모델을 사용한다.  

3. 빠른 학습과 안정성:  
   - 미니배치 SGD와 ReLU 활성화 함수 조합을 사용한다.  

4. 다양한 클래스 분류:  
   - 소프트맥스 출력층을 활용하여 확률 분포를 출력한다.  


### 4. 파이썬 코드 예제: Transformer 모델 처음부터 구현  

### 1. 모델 아키텍처 설계  
Transformer는 `Self-Attention` 메커니즘과 병렬 처리를 기반으로 문맥을 한다.  
모델을 처음부터 설계하려면 기본 구성 요소부터 만들어야 한다.  

#### 1.1 기본 구성 요소
1. Scaled Dot-Product Attention:  
   - Self-Attention 메커니즘의 핵심.  
   - 쿼리(Q), 키(K), 값(V)의 가중치를 계산하여 문맥 정보를 결합.  

2. 멀티헤드 어텐션 (Multi-Head Attention):  
   - 여러 개의 Self-Attention을 병렬로 실행해 더 많은 문맥 정보를 학습.  

3. 포지셔널 인코딩 (Positional Encoding):  
   - 시퀀스 데이터에서 순서를 나타내기 위해 추가되는 벡터.  

4. 피드포워드 신경망 (Feedforward Neural Network):  
   - 어텐션 결과를 처리하여 다음 층에 전달.  

#### 1.2 Transformer 블록  
Transformer 블록은 여러 개의 Multi-Head Attention과 Feedforward Neural Network로 구성된다. 이를 쌓아 모델을 만든다.  


### 2. 구현: Transformer 기본 구조 설계  

#### 1. 주요 특징  
1. Transformer 설계: Self-Attention, Multi-Head Attention, Feedforward Neural Network 등을 직접 구현.  
2. 확장 가능: 여러 레이어를 쌓아 더 깊은 네트워크를 구성할 수 있음.  
3. 포지셔널 인코딩: 시퀀스의 순서를 학습 가능하도록 추가.  

Transformer 아키텍처를 처음부터 설계하는 완벽한 예제이다.  
이제 이 모델을 바탕으로 언어 생성, 번역, 요약 등 다양한 작업에 활용할 수 있는 AI를 만들 수 있다.  

```python 
import torch
import torch.nn as nn
import math

# 1. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, mask=None):
        # query, key, value: Attention 메커니즘의 입력 텐서들
        # mask: 특정 위치를 무시하기 위해 사용하는 텐서 (예: 패딩 위치) 😊

        # 어텐션 스코어 계산: Q x K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None:
            # 패딩 마스크 위치를 매우 작은 값(-∞)으로 설정해 softmax에서 제외
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 소프트맥스를 통해 확률 분포 계산 (어텐션 가중치)
        attention_weights = torch.softmax(scores, dim=-1)

        # 어텐션 결과 = 가중치 x V
        return torch.matmul(attention_weights, value), attention_weights

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads  # 헤드 개수
        self.head_dim = embed_size // num_heads  # 각 헤드의 차원
        
        # 임베딩 크기가 헤드 수로 나누어 떨어지는지 확인
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"
        
        # 쿼리(Q), 키(K), 값(V)를 생성하는 선형 변환
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # 최종 출력 선형 변환
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, mask=None):
        N = query.shape[0]  # 배치 크기
        
        # Q, K, V를 각각 (배치, 헤드 수, 문장 길이, 헤드 차원)으로 변환
        Q = self.query(query).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention 수행
        attention, _ = ScaledDotProductAttention()(Q, K, V, mask)
        
        # 어텐션 결과를 결합하고 선형 변환
        attention = attention.transpose(1, 2).contiguous().view(N, -1, self.num_heads * self.head_dim)
        return self.fc_out(attention)

# 3. Feedforward Neural Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        # 첫 번째 선형 계층: 임베딩 크기를 확장
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        # 두 번째 선형 계층: 원래 임베딩 크기로 축소
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        # 활성화 함수로 ReLU 사용
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 입력 x에 대해 ReLU를 거친 뒤 다시 축소
        return self.fc2(self.relu(self.fc1(x)))

# 4. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout):
        super().__init__()
        # 멀티헤드 어텐션 레이어
        self.attention = MultiHeadAttention(embed_size, num_heads)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Feedforward Network
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        # 드롭아웃: 과적합 방지
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        # 어텐션 + 잔차 연결 + LayerNorm
        attention = self.attention(query, key, value, mask)
        x = self.norm1(attention + query)
        # Feedforward Network + 잔차 연결 + LayerNorm
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# 5. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super().__init__()
        # 최대 길이와 임베딩 크기에 대한 포지셔널 인코딩 초기화
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False  # 학습되지 않도록 설정

        # 포지션 행렬 생성
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 주기를 계산하는 항
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        
        # 짝수 인덱스에 대해 사인 적용
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스에 대해 코사인 적용
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 배치 차원 추가

    def forward(self, x):
        # 입력에 포지셔널 인코딩을 추가
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# 6. Transformer Model
class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, num_layers, vocab_size, max_len, dropout):
        super().__init__()
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 포지셔널 인코딩
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        # Transformer 블록을 여러 층으로 구성
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )
        # 최종 출력 레이어
        self.fc_out = nn.Linear(embed_size, vocab_size)
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 임베딩 + 포지셔널 인코딩 적용
        out = self.embedding(x)
        out = self.positional_encoding(out)
        # 각 Transformer 블록 통과
        for layer in self.layers:
            out = layer(out, out, out, mask)
        # 최종 결과 반환
        return self.fc_out(out)

# 7. 모델 초기화
embed_size = 512  # 임베딩 차원
num_heads = 8  # 멀티헤드 어텐션의 헤드 수
ff_hidden_size = 2048  # 피드포워드 네트워크의 히든 레이어 크기
num_layers = 6  # Transformer 블록 수
vocab_size = 30522  # 어휘 크기
max_len = 100  # 문장의 최대 길이
dropout = 0.1  # 드롭아웃 비율

model = Transformer(embed_size, num_heads, ff_hidden_size, num_layers, vocab_size, max_len, dropout)

# 샘플 입력 데이터
sample_input = torch.randint(0, vocab_size, (2, 20))  # 배치 크기: 2, 문장 길이: 20
output = model(sample_input, mask=None)
print(output.shape)  # 출력 크기: (2, 20, vocab_size)

```

위 코드에 들어간 개념들을 간단히 보자.  

## 1. Attention Mechanism (어텐션 메커니즘)  

입력 데이터의 모든 요소를 살펴보고 중요도를 계산하여 더 중요한 부분에 집중하도록 설계된 Transformer의 핵심기술이다.  

### Scaled Dot-Product Attention  
- 입력: Query(Q), Key(K), Value(V)  
- 출력: 중요도를 반영한 가중치가 적용된 값(Value)  
- 과정:  
  1. Query와 Key의 내적(dot product)을 계산해 중요도(score)를 구함.  
  2. Query와 Key의 차원이 클수록 내적 값이 커지므로, \(\sqrt{d_k}\)로 나누어 크기를 조정.  
  3. Softmax 함수로 확률 분포로 변환.  
  4. 확률을 Value에 곱해 가중합을 계산.  


### Multi-Head Attention  
- Attention 메커니즘을 여러 개(헤드)로 병렬 실행.  
- 각 헤드가 입력의 다른 부분에 주목할 수 있음.  
- 헤드 결과를 다시 결합해 정보를 종합.  


## 2. Positional Encoding  

Transformer는 입력 데이터의 순서를 고려하지 않는다(순서에 대한 고정된 구조 없음).  
따라서 위치 정보를 추가적으로 제공해야 한다.  

- 위치 정보는 사인(sine)과 코사인(cosine) 함수를 기반으로 계산.  
- 입력 데이터의 각 차원에 대해 다른 주기로 위치 정보를 생성.  
- 이를 입력 임베딩에 더하여 순서를 반영.  


## 3. Feedforward Neural Network  

- Fully Connected Layer의 확장판.  
- Transformer에서 각 단어 위치에 독립적으로 적용됨.  
- 비선형 활성화 함수(ReLU)를 사용해 복잡한 관계를 학습.  


## 4. Layer Normalization  

- 입력의 평균과 분산을 기준으로 데이터를 정규화.  
- 학습 안정성을 향상시키고, 더 빠른 수렴을 가능하게 함.  


## 5. Residual Connection (잔차 연결)  

- 입력 데이터를 레이어의 출력에 더하여 네트워크가 원본 정보를 보존할 수 있도록 함.  
- 학습 안정성을 높이고, 정보 손실을 방지.  


## 6. Dropout  

- 신경망에서 특정 비율의 노드를 랜덤하게 꺼서 학습이 과적합(overfitting)되지 않도록 방지.  

## 7. Transformer Block  

- Transformer의 기본 단위. 두 가지 주요 구성 요소로 이루어짐:  
  1. Multi-Head Attention + Residual Connection + LayerNorm  
  2. Feedforward Network + Residual Connection + LayerNorm  

## 8. Embedding Layer  

- 정수형 단어 ID를 고차원 벡터로 변환.  
- 단어 간의 의미적 관계를 수치로 표현.  


## 9. Vocabulary Size and Sequence Length  

- Vocabulary Size: 모델이 처리할 수 있는 단어 또는 토큰의 총 개수. (예: `30522`)  
- Sequence Length: 한 문장에서 모델이 처리할 수 있는 최대 단어 수. (예: `100`)  


## 10. Hyperparameters in Transformer

### 1) Embedding Size
- 각 단어를 표현하는 벡터의 크기.  
- 예: `512` → 512차원 벡터로 표현.  

### 2) Number of Heads
- Multi-Head Attention에서 병렬 실행되는 Attention의 개수.  
- 예: `8` → 8개의 헤드.  

### 3) Feedforward Hidden Size
- Feedforward Neural Network의 중간 계층 크기.  
- 예: `2048` → 활성화 함수 적용 전/후의 중간 출력 크기.  

### 4) Number of Layers
- Transformer Block의 개수.  
- 예: `6` → 6층 모델.  

### 5) Dropout Rate  
- 노드를 랜덤하게 비활성화하는 비율.  
- 예: `0.1` → 10% 드롭아웃.  


## 11. Masking  

- 어텐션 계산 중 특정 입력을 무시하기 위해 사용.  
- 예: 패딩 토큰이나 미래 토큰을 무시.  


## 12. PyTorch Basics  

- `torch.nn.Module`: PyTorch에서 모델을 정의하는 기본 클래스.  
- `Linear`: Fully Connected Layer.  
- `softmax`: 확률로 변환.  
- `Dropout`: 드롭아웃 레이어.  
- `LayerNorm`: 입력을 정규화.  


## 13. Training Process

1. 입력 데이터 준비:  
   - 텍스트 데이터를 단어 토큰으로 변환.  
   - 각 단어를 정수로 인코딩.  

2. 입력 임베딩:  
   - 각 단어를 고차원 벡터로 변환.  
   - 포지셔널 인코딩 추가.  

3. Transformer 블록 통과:  
   - 입력 데이터를 Transformer Block에 전달.  
   - 여러 층을 거치며 복잡한 관계를 학습.  

4. 출력 계산  
   - 최종 출력에서 소프트맥스를 적용해 확률 분포로 변환.  


## 14. 출력 크기와 의미  

- 출력 텐서의 크기: `(batch_size, sequence_length, vocab_size)`  
- 각 단어 위치에서 다음 단어의 확률 분포를 제공.  


## 15. 학습 목표  

- 모델이 문맥에 따라 다음 단어를 정확히 예측하도록 학습.  
- 손실 함수로 Cross-Entropy Loss를 사용.  


위 개념을 충분히 이해하면 Transformer 모델 코드와 그 동작 원리를 완벽히 이해할 수 있다.  

앞으로, 위 개념들에 대해 전부 자세히 들어다 보고, 위에서 본 코드를 다시한번 알아보는 시간을 가질 것이다.  
