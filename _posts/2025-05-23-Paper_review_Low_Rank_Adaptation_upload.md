---
layout: single
title: "[논문 리뷰] LoRA: Low-Rank Adaptation of Large Language Models"
categories: "AI"
tag: "review"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 논문 리뷰
AI를 공부하다 보면 대규모 언어 모델(LLM) 같은 거대한 모델을 만드는 것이 쉽지 않음을 깨닭게 되는데,  
문제는 이 모델들을 파인튜닝하는 게 엄청난 컴퓨터 자원을 잡아먹는다는거다.  
오늘 들고온 논문은, 이 문제점을 경쟁령있게 해결한 방법인인 LoRA를 다룬다.  

논문 링크: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)  

이 논문은 대규모 언어 모델을 효율적으로 튜닝할 수 있는 LoRA(Low-Rank Adaptation)라는 기법을 제안한다.  
쉽게 말해, 모델 전체를 다시 훈련시키지 않고, 아주 적은 파라미터만 조정해서 원하는 작업에 맞게 모델을 빠르게 적응시키는 방법이다.  


## 대규모 언어 모델과 튜닝의 고민
대규모 언어 모델, 그러니까 GPT나 BERT 같은 모델들은 수십억 개의 파라미터를 가지고 있다.  
이 파라미터들은 모델이 문장을 이해고, 질문에 대한 답 같은 "지식"을 담고 있는 거다.  
근데 이 모델을 특정 작업, 예를 들어 영화 리뷰 감성 분석이나 고객 서비스 챗봇처럼 특정 용도에 맞게 조정하려면 다음과 같은 과정을 거친다.  

보통은 **파인튜닝(fine-tuning)**이라는 과정을 통해서 모델의 모든 파라미터를 새 데이터로 조금씩 업데이트한다.  
문제는 이 과정이 많은 자원을 잡아먹는다.  
수십억 개의 파라미터를 다 조정하려면 GPU도 많이 필요하고, 메모리도 많이 먹는다. 추가로, 특정 작업마다 모델을 새로 저장해야 하니까 저장 공간도 많이 부족하다.    

LoRA는 이 문제를 깔끔하게 해결한다.  
모델의 모든 파라미터를 건드리지 않고, 아주 적은 양의 새로운 파라미터만 추가해서 모델을 튜닝하는 거다.  
비유하자면, 거대한 배(모델)를 통째로 개조하는 대신, 배에 작은 추가 부품(새 파라미터)를 붙여서 원하는 방향으로 움직이게 만드는 그런 느낌인 거다.  


## LoRA의 핵심 아이디어
LoRA는 "Low-Rank Adaptation"의 약자로, **저차원 행렬(low-rank matrix)**을 활용해 모델을 효율적으로 튜닝한다.  

1. **모델의 가중치 변화는 작다**  
   대규모 언어 모델의 가중치(파라미터 값)는 원래 학습된 상태에서 특정 작업에 맞게 튜닝할 때 아주 조금만 바뀐다.  
   이 방식으로, 가중치의 변화량을 저차원 행렬로 표현한다.  
   이 행렬은 원래 가중치보다 훨씬 작아서 메모리와 계산량을 엄청 줄여준다.  

2. **가중치에 작은 행렬 추가**  
   LoRA는 모델의 기존 가중치 행렬에 아주 작은 두 개의 행렬(A와 B)을 곱한 결과(AB)를 더한다.  
   이 A와 B는 원래 가중치보다 훨씬 적은 파라미터를 가지고 있어서, 계산이 훨씬 가볍다.  
   예를 들어, 원래 가중치 행렬이 1억 개의 숫자를 포함하고 있다면, LoRA는 그 1% 정도만 다루는 거다.  
   이렇게 하면 메모리 사용량이 확 줄어들고, 속도도 빨라진다.  

3. **기존 모델은 그대로!**  
   LoRA의 가장 큰 장점은 원래 모델의 가중치를 전혀 건드리지 않는다는 거다.  
   원래 모델은 고정된 상태로 두고, LoRA가 추가한 작은 행렬만 학습시키는 거기 때문에,  
   한 모델을 여러 작업에 맞게 튜닝할 때, 각 작업마다 작은 LoRA 행렬만 저장하면 된다.  
   이건 저장 공간도 엄청 아껴주고, 이 방법이야 말로 파인튜닝을 올바르게 해석한 게 아닐까 한다.  



## LoRA의 장점

1. **메모리 효율성**  
   LoRA는 전체 모델의 파라미터를 업데이트하는 대신, 아주 적은 양의 파라미터만 학습시킨다.  
   논문에 따르면, LoRA를 사용하면 파인튜닝에 필요한 메모리가 기존 방식 대비 최대 3배 이상 줄어든다  
   예를 들어, 10GB 메모리가 필요했던 작업이 LoRA로는 3GB 정도로 가능해진다는 그런 거다.  

2. **빠른 튜닝 속도**  
   적은 파라미터만 다루니까 계산 속도도 빨라진다.  
   GPU를 덜 써도 되고, 학습 시간도 단축된다.  
   그러니까 연구자나 개발자들이 더 빠르게 실험하고 결과를 볼 수 있는 거다.  

3. **모델 공유 쉬움**  
   LoRA는 원래 모델을 건드리지 않고 작은 행렬만 저장하니까, 특정 작업에 맞춘 LoRA 행렬만 공유하면 다른 사람들이 그걸 바로 적용할 수 있다.  
   이건 오픈소스 커뮤니티에서 특히 유용하다.

4. **성능은 그대로**  
   놀랍게도, LoRA는 이렇게 효율적인 방법임에도 불구하고 기존 파인튜닝과 비슷한 성능을 낸다.  
   논문에서 여러 태스크(예: 텍스트 분류, 질문 답변)로 테스트해봤는데, LoRA로 튜닝한 모델이 기존 방식과 거의 차이 없는 정확도를 보여줬다.  


## 실제로 어땠을까?
논문에서는 LoRA를 GPT-2, GPT-3 같은 대규모 언어 모델에 적용해서 테스트한다.  
결과는 기존 파인튜닝과 거의 동일한 성능을 내면서 메모리 사용량은 훨씬 줄어들었다.  

게다가 LoRA는 여러 작업에 동시에 적용할 수도 있다.  
예를 들어, 한 모델을 영어 번역, 감성 분석, 질의응답에 각각 튜닝하고 싶다면, 각 작업마다 LoRA 행렬을 따로 만들면 끝이다.  
이 행렬들은 크기가 작아서 저장도 쉽고, 모델 간 전환도 빠르다.  

## 오픈소스 링크
- GitHub: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

## 간단한 LoRA 구현 예제
다음은 PyTorch를 사용한 간단한 LoRA 레이어 구현 예제이다.  
초보자도 이해할 수 있게 간단하게 구성하였다.  

```python
import torch
import torch.nn as nn
import torch.optim as optim

# LoRA를 적용한 간단한 Transformer 레이어 정의
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=4, alpha=1):
        super(LoRALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha

        # 고정된 사전 학습된 가중치
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim), requires_grad=False)
        
        # LoRA 행렬
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, output_dim))
        
        # 스케일링 계수
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # 원래 가중치로 계산된 결과
        result = torch.matmul(x, self.weight.T)
        
        # LoRA 업데이트
        lora_update = torch.matmul(torch.matmul(x, self.A), self.B) * self.scaling
        return result + lora_update

# 예제 사용
input_dim = 64
output_dim = 64
batch_size = 32
seq_length = 10

# 더미 입력
x = torch.randn(batch_size, seq_length, input_dim)

# LoRA 레이어 초기화
lora_layer = LoRALayer(input_dim, output_dim, rank=4, alpha=4)

# LoRA 파라미터만 최적화하는 옵티마이저
optimizer = optim.Adam(lora_layer.parameters(), lr=0.001)

# 더미 훈련 루프 (간소화)
for epoch in range(10):
    optimizer.zero_grad()
    output = lora_layer(x)
    loss = torch.mean(output ** 2)  # 예제 손실 함수
    loss.backward()
    optimizer.step()
    print(f"에포크 {epoch}, 손실: {loss.item()}")

# 추론을 위해 가중치 병합 (선택 사항)
merged_weight = lora_layer.weight + torch.matmul(lora_layer.B.T, lora_layer.A.T) * lora_layer.scaling
```

위 코드는 간단한 LoRA 레이어를 구현한 것이다.  
모델의 원래 가중치는 고정하고, 작은 행렬 A와 B만 학습시키는 구조이다.  
잘 읽어보면 LoRA가 어떻게 작동하는지 감이 잡힌다.  

자 이제, 그게 뭐냐 깃허브 페이지에 있는 진짜 레이어 코드를 한번 봐 보자.  
위 코드는 다음 링크에서 원본을 볼 수 있다.  
[https://github.com/microsoft/LoRA/blob/main/loralib/layers.py](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)  
 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

# LoRA 레이어의 기본 클래스 정의
class LoRALayer():
    def __init__(
        self, 
        r: int,              # LoRA의 순위(rank), 낮은 차원의 행렬 크기를 결정
        lora_alpha: int,     # LoRA의 스케일링 팩터, 업데이트 강도를 조절
        lora_dropout: float, # 드롭아웃 비율, 과적합 방지를 위해 사용(0이면 비활성화)
        merge_weights: bool, # 가중치를 병합할지 여부(추론 시 성능 최적화)
    ):
        self.r = r                    # LoRA의 순위 저장, 작은 값으로 메모리 절약
        self.lora_alpha = lora_alpha  # 스케일링 팩터 저장, 학습 안정성에 기여
        # 드롭아웃이 0보다 크면 Dropout 레이어 생성, 아니면 동일 출력 함수 사용
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)  # 과적합 방지용 드롭아웃
        else:
            self.lora_dropout = lambda x: x                 # 드롭아웃 없이 입력 그대로 반환
        # 가중치가 병합되지 않은 상태로 초기화
        self.merged = False           # 현재 가중치가 병합되지 않았음을 표시
        self.merge_weights = merge_weights  # 병합 여부 플래그 저장

# Embedding 레이어에 LoRA 적용
class Embedding(nn.Embedding, LoRALayer):
    # Dense 레이어에 LoRA를 구현
    def __init__(
        self,
        num_embeddings: int,  # 임베딩 벡터의 총 개수(단어 사전 크기)
        embedding_dim: int,   # 각 임베딩 벡터의 차원
        r: int = 0,          # LoRA 순위, 0이면 LoRA 비활성화
        lora_alpha: int = 1, # LoRA 스케일링 팩터, 기본값 1
        merge_weights: bool = True,  # 가중치 병합 여부, 기본값 True
        **kwargs             # 추가적인 nn.Embedding 인자
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)  # 기본 Embedding 초기화
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,  # LoRA 초기화, 드롭아웃은 0
                           merge_weights=merge_weights)
        # LoRA가 활성화되면 학습 가능한 파라미터 생성
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))  # A 행렬, 초기값 0
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))   # B 행렬, 초기값 0
            self.scaling = self.lora_alpha / self.r  # 스케일링 팩터 계산, 업데이트 크기 조절
            # 사전 학습된 가중치 고정
            self.weight.requires_grad = False  # 기존 가중치는 학습에서 제외
        self.reset_parameters()  # 파라미터 초기화 호출

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)  # Embedding의 기본 파라미터 초기화
        if hasattr(self, 'lora_A'):  # LoRA가 활성화된 경우
            # A는 0으로, B는 정규 분포로 초기화
            nn.init.zeros_(self.lora_A)      # A 행렬 0으로 초기화
            nn.init.normal_(self.lora_B)     # B 행렬 정규 분포로 초기화

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)  # Embedding의 train 모드 설정
        if mode:  # 훈련 모드일 때
            if self.merge_weights and self.merged:  # 병합된 상태에서 훈련 시작 시
                # 가중치를 분리
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling  # LoRA 업데이트 제거
                self.merged = False  # 병합 해제 표시
        else:  # 평가 모드일 때
            if self.merge_weights and not self.merged:  # 병합되지 않은 상태에서 평가 시작 시
                # 가중치 병합
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling  # LoRA 업데이트 추가
                self.merged = True  # 병합 완료 표시

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:  # LoRA 활성화되고 병합되지 않은 경우
            result = nn.Embedding.forward(self, x)  # 기본 Embedding 출력
            after_A = F.embedding(  # A 행렬을 이용해 입력 변환
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling  # B와 곱해 LoRA 업데이트 추가
            return result
        else:
            return nn.Embedding.forward(self, x)  # 병합된 경우 기본 출력

# Linear 레이어에 LoRA 적용
class Linear(nn.Linear, LoRALayer):
    # Dense 레이어에 LoRA 구현
    def __init__(
        self, 
        in_features: int,    # 입력 피처 수
        out_features: int,   # 출력 피처 수
        r: int = 0,         # LoRA 순위, 0이면 비활성화
        lora_alpha: int = 1, # LoRA 스케일링 팩터
        lora_dropout: float = 0.,  # 드롭아웃 비율
        fan_in_fan_out: bool = False,  # 가중치가 (fan_in, fan_out) 형태인지 여부
        merge_weights: bool = True,  # 가중치 병합 여부
        **kwargs             # 추가 인자
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)  # 기본 Linear 초기화
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,  # LoRA 초기화
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out  # 가중치 형태 플래그
        # LoRA가 활성화되면 학습 가능한 파라미터 생성
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))  # A 행렬, 초기값 0
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))  # B 행렬, 초기값 0
            self.scaling = self.lora_alpha / self.r  # 스케일링 팩터 계산
            # 사전 학습된 가중치 고정
            self.weight.requires_grad = False
        self.reset_parameters()  # 파라미터 초기화
        if fan_in_fan_out:  # 가중치 형태 조정
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)  # Linear의 기본 파라미터 초기화
        if hasattr(self, 'lora_A'):  # LoRA가 활성화된 경우
            # A는 Kaiming 초기화, B는 0으로 초기화
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 행렬 초기화
            nn.init.zeros_(self.lora_B)  # B 행렬 0으로 초기화

    def train(self, mode: bool = True):
        def T(w):  # 가중치 전치 함수
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)  # Linear의 train 모드 설정
        if mode:  # 훈련 모드
            if self.merge_weights and self.merged:  # 병합된 상태 해제
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling  # LoRA 업데이트 제거
                self.merged = False
        else:  # 평가 모드
            if self.merge_weights and not self.merged:  # 병합 수행
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling  # LoRA 업데이트 추가
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):  # 가중치 전치 함수
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:  # LoRA 활성화되고 병합되지 않은 경우
            result = F.linear(x, T(self.weight), bias=self.bias)  # 기본 Linear 출력
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling  # LoRA 업데이트 추가
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)  # 병합된 경우 기본 출력

# 병합된 Linear 레이어에 LoRA 적용
class MergedLinear(nn.Linear, LoRALayer):
    # Dense 레이어에 LoRA 구현
    def __init__(
        self, 
        in_features: int,    # 입력 피처 수
        out_features: int,   # 출력 피처 수
        r: int = 0,         # LoRA 순위
        lora_alpha: int = 1, # LoRA 스케일링 팩터
        lora_dropout: float = 0.,  # 드롭아웃 비율
        enable_lora: List[bool] = [False],  # LoRA 활성화 여부 목록
        fan_in_fan_out: bool = False,  # 가중치 형태
        merge_weights: bool = True,  # 가중치 병합 여부
        **kwargs             # 추가 인자
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)  # 기본 Linear 초기화
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,  # LoRA 초기화
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'enable_lora 길이는 out_features로 나눠야 함'  # 유효성 검사
        self.enable_lora = enable_lora  # LoRA 활성화 플래그
        self.fan_in_fan_out = fan_in_fan_out  # 가중치 형태 플래그
        # LoRA가 활성화되면 학습 가능한 파라미터 생성
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))  # A 행렬
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))  # B 행렬
            self.scaling = self.lora_alpha / self.r  # 스케일링 팩터
            # 사전 학습된 가중치 고정
            self.weight.requires_grad = False
            # LoRA 적용 인덱스 계산
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)  # 인덱스 배열
            self.lora_ind[enable_lora, :] = True  # 활성화된 부분 표시
            self.lora_ind = self.lora_ind.view(-1)  # 1D로 변환
        self.reset_parameters()  # 파라미터 초기화
        if fan_in_fan_out:  # 가중치 형태 조정
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)  # Linear의 기본 파라미터 초기화
        if hasattr(self, 'lora_A'):  # LoRA가 활성화된 경우
            # A는 Kaiming 초기화, B는 0으로 초기화
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 행렬 초기화
            nn.init.zeros_(self.lora_B)  # B 행렬 0으로 초기화

    def zero_pad(self, x):  # 0 패딩 함수
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))  # 전체 크기로 0 배열 생성
        result[self.lora_ind] = x  # LoRA 적용 부분에 값 삽입
        return result

    def merge_AB(self):  # A와 B 행렬 병합
        def T(w):  # 가중치 전치 함수
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(  # 1D 컨볼루션으로 A와 B 곱셈
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))  # 전치 및 패딩 적용

    def train(self, mode: bool = True):
        def T(w):  # 가중치 전치 함수
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)  # Linear의 train 모드 설정
        if mode:  # 훈련 모드
            if self.merge_weights and self.merged:  # 병합 해제
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling  # LoRA 업데이트 제거
                self.merged = False
        else:  # 평가 모드
            if self.merge_weights and not self.merged:  # 병합 수행
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling  # LoRA 업데이트 추가
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):  # 가중치 전치 함수
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:  # 병합된 경우
            return F.linear(x, T(self.weight), bias=self.bias)  # 기본 Linear 출력
        else:  # 병합되지 않은 경우
            result = F.linear(x, T(self.weight), bias=self.bias)  # 기본 출력
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling  # LoRA 업데이트 추가
            return result

# Conv 레이어에 LoRA 적용
class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()  # 기본 nn.Module 초기화
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)  # 기본 컨볼루션 레이어
        for name, param in self.conv.named_parameters():  # 컨볼루션 파라미터 등록
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,  # LoRA 초기화
                           merge_weights=merge_weights)
        assert isinstance(kernel_size, int)  # 커널 크기가 정수인지 확인
        # LoRA가 활성화되면 학습 가능한 파라미터 생성
        if r > 0:
            self.lora_A = nn.Parameter(  # A 행렬, 입력 채널 및 커널 크기 반영
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(  # B 행렬, 출력 채널 및 커널 크기 반영
                self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r  # 스케일링 팩터
            # 사전 학습된 가중치 고정
            self.conv.weight.requires_grad = False
        self.reset_parameters()  # 파라미터 초기화
        self.merged = False  # 병합되지 않은 상태로 시작

    def reset_parameters(self):
        self.conv.reset_parameters()  # 컨볼루션 레이어 파라미터 초기화
        if hasattr(self, 'lora_A'):  # LoRA가 활성화된 경우
            # A는 Kaiming 초기화, B는 0으로 초기화
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 행렬 초기화
            nn.init.zeros_(self.lora_B)  # B 행렬 0으로 초기화

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)  # 기본 train 모드 설정
        if mode:  # 훈련 모드
            if self.merge_weights and self.merged:  # 병합 해제
                if self.r > 0:
                    # LoRA 업데이트 제거
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:  # 평가 모드
            if self.merge_weights and not self.merged:  # 병합 수행
                if self.r > 0:
                    # LoRA 업데이트 추가
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:  # LoRA 활성화되고 병합되지 않은 경우
            return self.conv._conv_forward(  # 컨볼루션 연산, LoRA 업데이트 포함
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)  # 병합된 경우 기본 컨볼루션 출력

# 2D, 1D, 3D 컨볼루션에 대한 확장
class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)  # 2D 컨볼루션에 LoRA 적용

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)  # 1D 컨볼루션에 LoRA 적용

# 다른 컨볼루션 레이어(예: 3D)에도 확장 가능
class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)  # 3D 컨볼루션에 LoRA 적용
```

## 왜 LoRA가 중요한가?
LoRA는 대규모 언어 모델을 더 쉽게, 더 저렴하게, 더 빠르게 튜닝할 수 있게 해준다.  
이건 특히 자원이 제한된 연구자나 회사들에게 큰 도움이 된다.  

## 결론
LoRA는 대규모 언어 모델을 튜닝하는 데 있어 혁신적인 해결책을 제시한다.  
기존의 파인튜닝 방식이 자원 소모가 크고 비효율적이었다면, LoRA는 적은 파라미터만으로도 비슷한 성능을 내면서 자원 사용량과 시간을 획기적으로 줄여준다.  
이 기술은 특히 대규모 모델을 다루는 연구자와 개발자들에게 큰 도움이 될 뿐만 아니라, 환경적 지속 가능성 측면에서도 긍정적인 영향을 미친다.  

논문에서 제공한 오픈소스 코드를 활용하면 누구나 LoRA를 직접 적용해볼 수 있다.  
저번 논문과 달리, 확실하게 오픈소스라서 더욱 신뢰가 가는 논문이다.  