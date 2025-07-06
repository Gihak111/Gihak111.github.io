---
layout: single
title:  "[논문 리뷰] Transformers without Normalization"
categories: "AI"
tag: "review"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Transformers without Normalization

지금까지의 트랜스 포머 모델에선 정규화가 필수였다.  
트랜스포머는 LayerNorm 없으면 망할 거라 생각했는데, 이 논문은 **Dynamic Tanh (DyT)**라는 새로운 기술로 정규화 없이도 성능을 챙긴다고 주장한다.  
논문 링크: [https://arxiv.org/abs/2410.03646](https://arxiv.org/abs/2410.03646)  

요즘 AI 모델은 점점 커지면서 계산 비용이 점점 올라가고 있는데, 이 논문은 LayerNorm의 통계 계산 오버헤드를 없애고 더 간단한 방법으로 비슷한 성능을 낸다고 한다.  


## 1. 정규화 계층, 왜 문제일까?

트랜스포머에서 **LayerNorm**(LN)이나 **RMSNorm**은 거의 신성한 존재로 여겨졌다. 이 녀석들은 왜 쓰냐면:  
- 입력 데이터 분포가 훈련 중에 바뀌는 **Internal Covariate Shift**를 줄여줌.  
- 극단적인 값(너무 크거나 작은 값) 억제.  
- 학습을 안정화하고 수렴 속도를 빠르게 함.  

근데, 이 정규화 계층도 문제점이 있다:  
- 평균, 분산 계산하느라 **컴퓨팅 비용**이 제법 듬.  
- 배치 크기나 시퀀스 길이에 따라 성능이 달라질 때가 있음(병렬화에도 방해).  
- 특히 GPU 최적화에서 병목이 될 수 있음.  

그래서 이 논문은 정규화 없이도 할 수 있지 않을까? 라는 도발적인 질문을 던진다.  
그리고 답은 **Dynamic Tanh (DyT)**라는 새로운 모듈이다.  


## 2. LayerNorm

먼저 LayerNorm이 실제로 뭘 하는지 알아보자.  

### 2.1 LayerNorm의 입출력 패턴
논문의 **Figure 2**에서 ViT, wav2vec2.0, DiT 같은 모델의 LayerNorm 입출력을 시각화했는데:  
- **초반 레이어**: 입력과 출력이 거의 선형(입력이 1이면 출력도 거의 1).  
- **후반 레이어**: 출력이 S자 곡선, 마치 **tanh** 함수처럼 생김.  

### 2.2 토큰 vs 채널 분석  
**Figure 4**에서 더 파고들었는데:  
- **토큰 단위**로 보면 LayerNorm은 그냥 선형 변환(거의 y=x).  
- **채널 단위**로 보면 특정 채널의 값이 극단적으로 커지거나 작아짐.  
- 이걸 합치면 전체적으로 S자 곡선이 나온다.  

결론? LayerNorm은 사실상 tanh 같은 비선형 함수처럼 작동하고, 값 스케일링도 같이 해준다는 거다.  
이걸 알았으니, LayerNorm을 대체할 Dynamic Tanh를 알아보자  


## 3. Dynamic Tanh  

논문은 LayerNorm을 **Dynamic Tanh (DyT)**라는 모듈로 바꿔버린다
즉, Dynamic Tanh게 더 효율이 좋아야 한다.  

### 3.1 DyT의 수식
DyT는 간단한 수식으로 정의된다:  
```
DyT(x) = γ · tanh(α·x) + β
```
- **α**: 학습 가능한 스칼라. 입력의 스케일을 자동으로 조정(대충 1/표준편차 역할).  
- **γ, β**: 채널별로 적용되는 스케일링과 이동 파라미터. LayerNorm의 scale/shift와 비슷.  

### 3.2 왜 tanh?
논문의 **Figure 3**에서 tanh, hardtanh, sigmoid를 비교했는데:  
- **tanh**는 부드럽고, 0을 중심으로 대칭이라 값이 균형 잡힘.  
- 극단값을 억제하면서도 중앙 부분은 선형에 가까워 학습이 안정적.  
- sigmoid는 0 중심이 아니라 약간 불리, hardtanh는 덜 부드러움.  

tanh가 이 둘의 장점을 잘 섞어서 최고의 선택지로 뽑혔다.  


## 4. DyT 적용 방법  

DyT는 트랜스포머에서 LayerNorm이나 RMSNorm이 있던 자리에 그냥 넣으면 된다. 별도의 통계 계산(평균, 분산) 필요 없고, 기존 활성화 함수(GELU, ReLU)나 모델 구조도 안 바꿔도 된다.  

### DyT 코드 구현
논문에 나온 DyT를 PyTorch로 구현해 봤다. 오픈소스 코드는 못 찾았지만, 논문 설명대로 따라해봤다.  

```python
import torch
import torch.nn as nn

class DyT(nn.Module):
    def __init__(self, channels, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)  # Learnable scaling
        self.gamma = nn.Parameter(torch.ones(channels))       # Per-channel scale
        self.beta = nn.Parameter(torch.zeros(channels))       # Per-channel shift

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
```

이 코드는 ViT나 LLaMA 같은 모델의 LayerNorm 자리에 바로 끼워넣을 수 있다. 실행은 안 해봐서, 교차검증은 필요하다.  


## 5. 실험 결과  

논문은 DyT를 여러 태스크에서 테스트했다.  
ViT, LLaMA, wav2vec2, DiT, DNA 시퀀스 모델에 대한 결과는 아래 표로 정리했다.  

| 태스크             | 모델            | 정규화       | DyT         | 차이         |
|-------------------|----------------|-------------|------------|-------------|
| Vision 분류       | ViT-B/L       | 82.3%/83.1% | 82.5%/83.6% | +0.2/+0.5%p |
|                   | ConvNeXt-B/L  | 83.7%/84.3% | 83.7%/84.4% | 0/+0.1%p    |
| Self-Supervised   | MAE, DINO     | 83.2–85.5%  | 83.2–85.5%  | ±0.4%p      |
| Diffusion (DiT)   | B/L/XL        | FID 64.9/45.9/19.9 | 63.9/45.7/20.8 | -1.0/-0.2/+0.9 |
| LLM (LLaMA 7–70B) | RMSNorm       | loss 1.45–1.59 | 1.45–1.60  | ±0.01       |
| Speech (wav2vec2) | Base/Large    | loss 1.95/1.92 | 1.95/1.91 | -0/-0.01    |
| DNA 시퀀스        | HyenaDNA, Caduceus | 85.2%/86.9% | 85.2%/86.9% | –           |

- **비전 태스크**: ViT에서 살짝 성능 향상, ConvNeXt는 비슷.  
- **LLM**: LLaMA 7B~70B에서 손실 거의 차이 없음. 임베딩 뒤에 α 추가로 약간 튜닝.  
- **DiT, Speech, DNA**: 거의 동률이거나 살짝 나음.  

놀라운 건, 하이퍼파라미터 거의 안 건드리고 LayerNorm을 DyT로 바꿨을 뿐인데 이 성능을 낸다는 거다.  


## 6. Ablation Study  

### 6.1 스쿼싱 함수 비교
논문은 tanh 대신 다른 함수를 썼을 때를 테스트했다:
- **Identity(정규화 없음)**: 학습 발산, 완전 망함.  
- **tanh**: 최고 성능, ViT-B에서 82.5%.  
- **hardtanh**: 살짝 성능 ↓ (82.2%).  
- **sigmoid**: 중심이 0이 아니라 성능 더 떨어짐 (81.6%).  

tanh가 안정성과 성능 면에서 더 좋다.  

### 6.2 α의 역할  
- α는 입력 분포의 표준편차(1/std)랑 비슷하게 학습됨 (**Figure 8**).  
- α만으로는 비선형 효과 못 내니까 tanh랑 같이 써야 함.  

### 6.3 α 초기화  
- **비-LLM**: α₀=0.5~1.2 사이에서 안정. 0.5 추천.  
- **LLM**: 모델 크기에 따라 다름.  
  - Attention 블록: 7B는 α₀=0.8, 70B는 0.2.  
  - FFN/최종 레이어: 더 낮게(7B는 0.2, 70B는 0.05).  
 

## 7. 기존 방법들과 비교

정규화 없는 다른 방법들(Fixup, SkipInit, oReparam)과 비교해봤다  

| 방법      | ViT-B  | ViT-L  | MAE ViT-B | MAE ViT-L |
|-----------|-------|-------|-----------|-----------|
| Fixup     | 77.2% | 78.1% | 73.7%     | 74.1%     |
| SkipInit  | 74.1% | 75.6% | 73.1%     | 74.0%     |
| oReparam  | 82.5% | 83.0% | 83.2%     | 85.4%     |
| **DyT**   | **82.8%** | **83.6%** | **83.7%** | **85.8%** |

DyT가 최소 변경으로 최고 성능! Fixup이나 SkipInit은 학습률 낮춰야 해서 귀찮고, oReparam은 비슷하지만 DyT가 더 간단하다.  

---

## 8. 효율성  

DyT는 통계 계산이 없어서 빠르다. H100 GPU에서 테스트 결과  

| 설정             | RMSNorm | DyT   | 전체 모델 | 개선율 |
|-----------------|---------|-------|----------|-------|
| Uncompiled (BF16) |         |       |          |       |
| Inference (100단계) | 2.1s    | 1.0s  | 14.1s→13.0s | 15–52%↓ |
| Train (100단계)     | 8.3s    | 4.8s  | 42.6s→39.1s | 18–42%↓ |
| Compiled (torch.compile) |         |       |          |       |
| Inference        | 0.3s    | 0.3s  | 12.3s    | ≒0%   |
| Train            | 3.9s    | 3.9s  | 38.9s    | ≒0%   |

- **Uncompiled**: DyT가 훨씬 빠름.  
- **Compiled**: 컴파일러가 최적화해서 차이 거의 없음. 그래도 DyT가 더 가볍다.  

---

## 9. 한계와 앞으로의 방향

DyT는 트랜스포머에선 쓸모 있지만, 한계도 있다  
- **ConvNet(BN)**: ResNet-50, VGG19에서 성능 떨어짐. BatchNorm 대체는 실패.  
- **하드웨어**: reduction 연산이 병목인 특수 장치에서 DyT가 더 유리할 가능성.  
- **미래 연구**: BN 대체, DyT와 연산 융합(fuse), 메모리 최적화.  


## 10. 결론  

Dynamic Tanh는 LayerNorm의 본질을 “비선형 스쿼싱 + 스케일링”으로 재정의했다. 통계 계산 없이도:  
- S자 곡선으로 극단값 억제.  
- 선형 구간에서 학습 역동성 유지.  
- 비전, 언어, 음성, DNA까지 다양한 태스크에서 성능 비슷하거나 더 나음.  

이 논문은 트랜스포머의 정규화 계층을 다시 생각하게 만든다.  
DyT는 간단하면서도 강력해서 앞으로 더 많은 모델에 적용될 가능성이 크다.  

근데, 솔직히 말하면, 이거 다른 정규화 대체 방법 oReparam 같은 거와 섞어서 쓰면 더 쎌 것 같기도 하다.  
예를 들어, oReparam의 스펙트럼 제어랑 DyT의 비선형을 합치면 더 좋지 않을까.  
아직 테스트 안 해봐서 모르겠지만, 뭔가 터질 것 같은 느낌이다.  
