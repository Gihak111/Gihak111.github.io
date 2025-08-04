---
layout: single
title:  "[논문리뷰] How LLMs are Shaping the Future of Virtual Reality"
categories: "AI"
tag: "review"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Consistency Models  

AI를 공부하다 보면, 생성 모델(Generative Model), 특히 Diffusion 모델이 이미지 생성에서 얼마나 강력한지 알게 된다.  
Stable Diffusion이나 DALL·E와 같은 모델은 매우 정교한 이미지를 생성할 수 있지만, 그 이면에는 매우 느린 추론 과정이 존재한다.  
이러한 비효율성을 극복하기 위해 제안된 것이 바로 Consistency Models이다.  

Consistency Models는 기존의 Diffusion 방식과는 다른 단 한 번의 샘플링으로 고품질 이미지를 생성할 수 있는 새로운 패러다임이다.  
논문링크 : [Consistency Models (arXiv:2508.00737)](https://arxiv.org/abs/2508.00737)  
위 내용을 바탕으로 Consistency Models의 개념, 구조, 장점, 구현 방법 등을 상세히 설명한다.  


## 생성 모델과 느린 샘플링의 문제점  

Diffusion 기반 생성 모델은 매우 뛰어난 이미지 품질을 제공하지만, 한 장의 이미지를 생성하기 위해 수십\~수백 번의 추론이 필요하다는 단점이 있다.  
예를 들어, Stable Diffusion은 보통 50\~100회 이상의 반복 추론이 요구되며, 이는 연산량이 많고 실행 시간이 오래 걸리는 문제를 유발한다.  

이러한 방식은 고성능 서버 환경에서는 감당할 수 있지만, 모바일 기기나 실시간 응용에서는 사실상 사용이 불가능하다.  
Consistency Model은 이러한 문제를 해결하기 위해 고안된 방식으로, Diffusion의 장점을 유지하면서도 속도를 획기적으로 개선한다.  


## Consistency Model의 핵심 아이디어  

기존 Diffusion 모델은 매 시간 단계 $t$마다 중간 이미지를 점진적으로 복원해 나가는 방식으로 작동한다.  
반면, Consistency Model은 노이즈가 얼마나 섞여 있든 관계없이 항상 최종 이미지 $x_0$를 정확하게 예측하도록 학습된다.  
즉, 모든 시간 단계에 대해 ‘일관된(consistent)’ 출력을 내는 하나의 모델을 학습하는 것이 목표이다.  

이러한 일관성을 학습하기 위한 핵심 기법이 바로 다음과 같다.  

### Consistency Distillation  

* 먼저, 기존의 Diffusion 모델을 Teacher 모델로 설정하고,  
* Consistency Model을 Student 모델로 설정한다.  
* 다양한 시간 단계 $t$에서 노이즈가 섞인 샘플 $x_t$를 만들고,
* Teacher가 예측한 결과 $x_0$를 Student가 정확히 모방하도록 학습한다.  
* 결과적으로, 학습된 Consistency Model은 단 한 번의 샘플링만으로도 고품질 이미지를 생성할 수 있게 된다.  
나처럼 생성형 AI 공부를 시작한지 얼마 안된 사람이면, 이게 뭔 소리지 할꺼다.  

이걸 더 설명해 보자면,  
기존의 Diffusion 모델은 천천히 이미지를 만드는 능력은 매우 뛰어나다.  
새로 만들고자 하는 Consistency Model은 빠르게 이미지를 만들려고 하지만, 성능은 떨어진다.  
따라서, Consistency Model이 Diffusion를 따라 배우는 구조로 훈련을 진행한다.  

기존의 Diffusion 모델은 다음과 같은 방식으로 이미지를 만든다.  
1. 아무런 정보도 없는 노이즈 이미지에서 시작한다.  
2. 아주 천천히, 1000번에 걸쳐서 노이즈를 조금씩 제거하면서 깨끗한 이미지를 만든다. 천천히, 반복해서 정제해 나간다.  

이 엄청 긴 과정을 Consistency Model은 단 1번만에 하려고 하는거다.  
최종 목표는, 과정을 하나씩 다 따라가는 것이 아니라, Diffusion 모델의 최종 결과물만 보고 바로 흉내내는 것을 목표로 한다는거다.  

## Consistency Model의 구조

구조 자체는 기존의 UNet, Transformer 기반 Diffusion 모델과 유사하게 구성된다.
하지만 **훈련 목표는 완전히 다르다.**

* 다양한 노이즈 레벨의 입력 $x_t$에 대해
* 항상 동일한 깨끗한 결과 $x_0$을 예측하도록 학습된다.
* 따라서 학습이 완료되면 시간 단계와 관계없이 **즉시 결과를 생성할 수 있는** 특성을 갖게 된다.

이러한 구조는 **Diffusion 모델의 고품질 생성 능력**은 유지하면서도, **추론 속도는 수십 배 이상 향상**되는 결과를 만든다.


## 훈련 과정 요약

훈련 방식은 다음과 같은 순서로 이루어진다.

1. 사전 학습된 Diffusion 모델 $D$를 Teacher로 사용한다.
2. 데이터 $x_0$에 노이즈를 주입하여 $x_t$를 생성한다.
3. Teacher는 $x_t$로부터 $x_0^{(D)}$를 예측한다.
4. Consistency Model $C$는 동일한 $x_t$로부터 $\hat{x}_0$를 예측한다.
5. 두 결과가 같아지도록 **MSE 기반 손실 함수**로 Student 모델을 학습한다.

$$
\mathcal{L}_{\text{consistency}} = \| C(x_t, t) - x_0^{(D)} \|^2
$$

이 손실 함수를 통해 **일관성(consistency)을 갖는 모델**이 완성된다.


## 실험 결과

Consistency Model은 CIFAR-10, ImageNet 등 다양한 데이터셋에서 실험되었으며, **단 1\~4회의 추론만으로도 기존 DDIM 수준의 이미지 품질을 달성**하였다.

### FID 성능 비교 (낮을수록 품질이 우수함)

| 모델                    | 추론 스텝 수 | FID (CIFAR-10 기준) |
| --------------------- | ------- | ----------------- |
| DDPM                  | 1000    | 3.17              |
| DDIM                  | 50      | 4.16              |
| **Consistency Model** | **1**   | **4.10**          |

* Consistency Model은 DDIM보다 빠르면서도 유사하거나 더 나은 품질을 제공한다.
* 특히 **1\~4 스텝만으로 이미지를 생성할 수 있으므로**, 모바일, 엣지, 실시간 응용에서 활용 가능성이 크다.


## 간단한 PyTorch 예제 코드

```python
import torch
import torch.nn as nn

# ConsistencyModel 정의 (단순화된 예시)
class ConsistencyModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

    def forward(self, x, t=None):  # 실제로는 t도 인코딩함
        return self.net(x)

# 샘플 입력 데이터
x_t = torch.randn(64, 128)

# Consistency Model 인스턴스화 및 학습 루프
model = ConsistencyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    x_0_teacher = torch.randn_like(x_t)  # 실제는 Diffusion 모델의 출력
    output = model(x_t)
    loss = torch.nn.functional.mse_loss(output, x_0_teacher)
    loss.backward()
    optimizer.step()
    print(f"에포크 {epoch}, 손실: {loss.item()}")
```

이 예제는 Consistency Model의 핵심 아이디어인 **다양한 노이즈 레벨에서 항상 같은 결과를 내는 구조**를 단순화하여 구현한 것이다.


## Consistency Model의 장점

| 항목  | 장점                                       |
| --- | ---------------------------------------- |
| 속도  | 최대 100배 빠른 생성 속도                         |
| 품질  | Diffusion 기반 생성 품질 유지                    |
| 효율성 | 추론 횟수 감소로 계산 자원 절약                       |
| 유연성 | 기존 사전 학습된 Diffusion 모델에서 distillation 가능 |
| 호환성 | 다른 조건 생성 모델(ControlNet 등)과 통합 가능         |

Consistency Model은 속도, 정확도, 메모리, 확장성의 균형을 모두 달성한 모델이라 할 수 있다.


## 결론

Consistency Models는 생성 모델의 속도 한계를 극복한 새로운 접근 방식이다.
기존의 느린 샘플링을 없애고, 단 1회 추론으로도 고품질 이미지를 생성할 수 있는 기술을 실현하였다.
이러한 속성과 성능은 모바일 디바이스, 실시간 애플리케이션, 자원 제한 환경에서 매우 유용하게 활용될 수 있다.

앞으로의 생성 모델 연구 및 응용에 있어 Consistency Models는 하나의 중요한 전환점이 될 것으로 기대된다.