---
layout: single
title:  "나만의 대화 생성형 모델 처음부터 만들기: 레이어 설계와 구현"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## PyTorch로 대화 생성형 AI 모델을 처음부터 설계하고 구현하기

대화 생성형 AI를 처음부터 만들어보면 어떨까?  
그냥, 어떻게 말하는지 궁금해서 레이어를 하나씩 쌓아보려고 한다.  
솔직히 다들 써보기만 하고 직접 만들려고 시도한 적은 없지 않는가?
원리도 모르면서 사용하는건 바람직하지 않으므로, 내부 구조에 대해 자세히 알아보자.  
이번 글에서는 PyTorch로 대화 생성형 AI를 처음부터 설계하고, 어떤 레이어를 조합해서 모델을 만드는지 정리한다.  
레이어 설계, 구현 과정, 그리고 간단한 테스트까지 다룬다.  

### 1. 기본 환경 설정하기
모델을 만들기 위한 기본 설정을 잡는다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 하이퍼파라미터
vocab_size = 5000  # 단어 사전 크기
embed_dim = 256    # 임베딩 차원
n_layers = 6       # Transformer 층 수
n_heads = 8        # Attention 헤드 수
ffn_dim = 1024     # FFN 차원
max_len = 128      # 최대 시퀀스 길이
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

간단한 모델을 위해 적당한 크기로 설정한다.

### 2. 입력 레이어 설계하기
대화의 단어를 벡터로 바꾸는 입력 레이어를 만든다.

```python
class InputLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.word_embedding(x) + self.pos_embedding(positions)
        return self.norm(x)

input_layer = InputLayer(vocab_size, embed_dim, max_len).to(device)
```

- **Word Embedding**: 단어를 벡터로 변환.
- **Positional Embedding**: 단어 위치를 추가.
- **LayerNorm**: 학습 안정성을 위해 정규화.

### 3. Transformer Decoder 레이어 설계하기
대화를 생성하는 핵심인 Decoder를 만든다.

```python
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)
```

- **Self-Attention**: 문맥을 이해해서 단어 간 관계 파악.
- **LayerNorm**: Attention 후 정규화.
- **FFN**: 의미를 깊게 처리.
- **LayerNorm**: FFN 후 정규화.

### 4. 전체 모델 조합하기
입력 레이어와 Decoder를 쌓아서 모델을 완성한다.

```python
class DialogueModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, ffn_dim, max_len):
        super().__init__()
        self.input_layer = InputLayer(vocab_size, embed_dim, max_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embed_dim, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.input_layer(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return self.output_layer(x)

model = DialogueModel(vocab_size, embed_dim, n_layers, n_heads, ffn_dim, max_len).to(device)
```

- Decoder를 여러 층으로 쌓아서 복잡한 대화 패턴을 학습하게 함.
- 출력 레이어로 단어 점수를 뽑아냄.

### 5. 테스트 데이터 준비하기
간단한 대화 데이터로 학습과 테스트를 준비한다.

```python
# 가상의 단어 사전과 데이터
vocab = {"<pad>": 0, "안녕": 1, "나": 2, "기분": 3, "좋아": 4, "<eos>": 5}  # 예시
train_data = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)  # "안녕 나 기분 좋아 <eos>"
target_data = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.long).to(device)  # "나 기분 좋아 <eos> <pad>"

# 마스크 생성 (미래 단어 가리기)
mask = nn.Transformer.generate_square_subsequent_mask(max_len).to(device)
```

Causal Mask를 써서 과거 단어만 보고 다음 단어를 예측하게 한다.

### 6. 모델 학습하기
간단히 학습시켜본다.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = model(train_data, mask)
    loss = criterion(output.view(-1, vocab_size), target_data.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

손실을 줄여가며 단어 예측 능력을 키운다.

### 7. 대화 생성하기
학습된 모델로 답변을 만들어본다.

```python
model.eval()
with torch.no_grad():
    input_seq = torch.tensor([[1]], dtype=torch.long).to(device)  # "안녕"
    for _ in range(5):
        output = model(input_seq)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
        input_seq = torch.cat([input_seq, next_token], dim=1)

print("생성된 시퀀스:", input_seq)
```

"안녕"에서 시작해서 단어를 하나씩 뽑아본다.

### 8. 레이어 구조 분석하기: 입력 단계
내가 어떻게 시작되는지 보자.

- **Word Embedding**
  - **구성**: `input_layer.word_embedding` [5000, 256]
  - **역할**: "안녕"을 256차원 벡터로 변환.
- **Positional Embedding**
  - **구성**: `input_layer.pos_embedding` [128, 256]
  - **역할**: 단어 순서를 기억.
- **LayerNorm**
  - **구성**: `input_layer.norm` [256]
  - **역할**: 입력을 안정화.

### 9. 레이어 구조 분석하기: Transformer Decoder
대화의 핵심 처리 단계다.

- **Self-Attention**
  - **구성**: `decoder_layers.self_attn` [256, 256]
  - **역할**: "나"와 "기분"의 관계를 파악.
- **LayerNorm**
  - **구성**: `decoder_layers.norm1` [256]
  - **역할**: Attention 결과 정리.
- **Feed-Forward Network**
  - **구성**: `decoder_layers.ffn` [256 -> 1024 -> 256]
  - **역할**: 단어 의미를 깊이 처리.
- **LayerNorm**
  - **구성**: `decoder_layers.norm2` [256]
  - **역할**: FFN 결과 정리.

### 10. 레이어 구조 분석하기: 출력 단계
마지막으로 단어를 뱉어낸다.

- **Linear Layer**
  - **구성**: `output_layer` [256, 5000]
  - **역할**: 벡터를 단어 점수로 변환.
- **Softmax (별도 연산)**
  - **역할**: 점수를 확률로 바꿔서 "좋아" 같은 단어 선택.

### 11. 결과 저장하기
생성된 시퀀스를 저장한다.

```python
import pandas as pd
result_df = pd.DataFrame({'input': ['안녕'], 'output': [input_seq.tolist()]})
result_df.to_csv('generated_dialogue.csv', index=False)
print("생성 결과가 'generated_dialogue.csv'에 저장되었습니다.")
```

### 12. 결과 요약 보고서 작성하기
만든 모델을 평가해본다.

```python
summary = {
    '레이어 수': n_layers,
    '최종 손실': loss.item(),
    '생성된 토큰 수': input_seq.size(1)
}
print("모델 생성 요약:", summary)
```


### 결론
위와 같은 구조로 엄청 간단한 생성형 AI를 만들 수 있다.  
간단한 구조로 만들었지만, 이해를 돕는데 쓸모가 있다.  
실제로 grok 나 ChatGPT 등은 더욱 다양한 기술을 우겨넣어 만들어 졌을 것이다.  
