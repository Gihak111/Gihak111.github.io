---
layout: single
title: "AI 아키텍쳐 12. MoE"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## MoE
Mixture of Experts는 모델의 성능을 높이기 위해 파라미터 수를 무작정 늘릴 때 발생하는 연산 비용의 선형적 증가 문제를 해결하기 위해 부활한 프레임워크이다.  
기존의 밀집 모델은 입력 토큰 하나를 처리하기 위해 수천억 개의 파라미터를 전부 가동해야 했다.  
이는 마치 1 + 1을 계산하기 위해 뇌의 모든 뉴런을 동시에 발화시키는 것과 같은 비효율이다.  
MoE는 희소 활성화 원리를 도입하여, 입력 데이터의 특성에 따라 필요한 일부 전문가(Expert) 네트워크만을 조건부로 실행한다.  
이를 통해 모델의 총 용량은 수조 개로 늘리면서도, 추론 시 연산량(FLOPs)은 1/10 수준으로 억제하는 '거대함과 민첩함의 공존'을 가능케 했다.  

## 2. 수식적 원리: 조건부 연산과 부분 공간 분할
MoE의 이론적 핵심은 입력 벡터 $x$를 고차원 공간상에서 분석하여, 가장 적합한 함수(Expert) $E_i$로 라우팅(Routing)하는 **게이팅 메커니즘(Gating Mechanism)**에 있다.
출력 $y$는 활성화된 전문가들의 가중 합으로 표현된다.

$$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$

여기서 $G(x)$는 라우터(Router)가 출력하는 확률 분포이며, 일반적으로 **Top-k** 방식이 적용된다.
$$G(x) = \text{Softmax}(\text{TopK}(W_g x + \epsilon))$$

**증명(Proof)의 관점**에서 볼 때, MoE는 복잡한 비선형 함수를 여러 개의 **국소 선형 근사(Piecewise Linear Approximation)** 혹은 더 단순한 비선형 함수의 조합으로 분해하는 과정이다.
데이터의 매니폴드(Manifold) $\mathcal{M}$을 $N$개의 부분 공간(Subspace) $\{\mathcal{S}_1, ..., \mathcal{S}_N\}$으로 분할하고, 각 $E_i$가 특정 $\mathcal{S}_i$ 내에서의 매핑을 전담하도록 학습한다.
이는 전체 파라미터 $\Theta$를 키우더라도, 실제 경사 하강법이 작용하는 활성 파라미터 $\Theta_{active}$는 고정되므로, **VC 차원(VC Dimension)**은 증가시키면서 일반화 오차(Generalization Error)의 발산은 막을 수 있음을 시사한다.

## 3. 학습 방법론: Load Balancing과 Auxiliary Loss
MoE 학습의 최대 난제는 특정 전문가에게만 데이터가 쏠리는 **붕괴(Collapse)** 현상이다.
라우터가 "Expert 1"만 계속 선택하면, 나머지 전문가들은 학습 기회를 박탈당하고 모델은 단순한(그리고 비효율적인) Dense Model로 전락한다.

이를 방지하기 위해 주 손실 함수(Cross-Entropy) 외에 **로드 밸런싱 손실(Load Balancing Loss)**을 추가해야 한다.

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{aux} N \sum_{i=1}^{N} f_i P_i$$

여기서 $f_i$는 해당 배치(Batch)에서 Expert $i$가 실제로 선택된 비율(Fraction), $P_i$는 라우터가 Expert $i$를 선택할 평균 확률이다.
이 수식은 변분법(Calculus of Variations)적으로 $f_i$와 $P_i$가 균등 분포(Uniform Distribution)일 때 최솟값을 가지도록 유도한다.
1.  **Capacity Factor**: 각 전문가가 처리할 수 있는 토큰의 최대 개수(Buffer Capacity)를 제한한다. 이를 초과하는 토큰은 과감하게 버려지거나(Token Dropping), 다음 레이어로 우회된다.
2.  **Jittering**: 라우팅 과정에 가우시안 노이즈를 주입하여 탐색(Exploration)을 유도하고 조기 수렴을 방지한다.
3.  **최적화**: Switch Transformer(Top-1)나 GShard(Top-2) 방식이 주류이며, 최근에는 전문가 간의 역할을 명시적으로 구분하지 않고 공유하는 Shared Expert 기법도 사용된다.

## 4. 핵심 기술: 전문가 병렬화와 통신 최적화
MoE가 실제로 하드웨어 위에서 돌아가게 만드는 것은 시스템 엔지니어링의 영역이다.

* **전문가 병렬화 (Expert Parallelism)**:
    모델의 레이어(Attention 등)는 모든 GPU에 복제(Data Parallel)하고, 전문가(FFN) 부분만 여러 GPU에 나누어 저장한다.
    입력 토큰은 라우팅 결과에 따라 다른 GPU에 있는 전문가에게 전송(**All-to-All Communication**)되어 처리된 후, 다시 원래 GPU로 복귀한다.
* **토큰 라우팅 최적화**:
    네트워크 대역폭이 병목이 되는 것을 막기 위해, 통신과 연산을 겹쳐서 수행하는 **Overlap Communication and Computation** 기술이 필수적이다.
    메모리 대역폭 효율을 위해 인접한 토큰들이 같은 전문가를 찾도록 유도하는 지역성(Locality) 제약 조건을 추가하기도 한다.

## 5. 아키텍처 확장: Soft MoE와 Mixture of Depths
전통적인 Hard Routing(이산적 선택)은 미분 불가능성 문제와 전문가 활용도 저하를 야기했다. 이를 개선한 확장 아키텍처들이 등장하고 있다.

* **Soft MoE**:
    토큰을 특정 전문가에게 할당하는 대신, 여러 토큰을 섞어서(Mix) 전문가에게 입력하고 그 결과를 다시 분배한다.
    $$E_i(X) = \text{FFN}_i(\sum_t w_{i,t} x_t)$$
    이는 완전 미분 가능(Fully Differentiable)하며 모든 전문가를 고르게 활용하게 만든다.
* **Mixture of Depths (MoD)**:
    전문가 선택뿐만 아니라, 토큰마다 통과하는 레이어의 깊이(Depth)를 다르게 설정한다. 쉬운 토큰은 얕게, 어려운 토큰은 깊게 처리하여 시간 효율성을 극한으로 끌어올린다.

## 6. 추론 및 VRAM Bottleneck
MoE의 추론은 '연산'보다는 '메모리'와의 싸움이다.

1.  **High Parameter, Low FLOPs**: GPT-4 수준의 MoE는 파라미터가 1.8조 개에 달하지만, 한 번의 추론에는 그중 1% 미만만 사용한다.
2.  **Memory Bandwidth Bound**: 하지만 1.8조 개의 파라미터가 VRAM 어딘가에는 로드되어 있어야 한다. 매 토큰마다 서로 다른 전문가 가중치를 VRAM에서 칩(HBM)으로 퍼 날라야 하므로, 연산 속도보다 메모리 대역폭이 전체 성능을 좌우한다.
3.  **Quantization**: 이 때문에 MoE는 가중치를 int4, int8로 깎는 양자화(Quantization)가 선택이 아닌 필수다.

이러한 특성 때문에 MoE는 단일 GPU에서는 거의 구동이 불가능하며, 고속 인터커넥트(NVLink)로 연결된 GPU 클러스터가 강제된다.

## 결론
MoE는 "공짜 점심은 없다"는 AI 판에서 유일하게 "더 많이 먹어도 살이 안 찌는" 마법 같은 아키텍처처럼 포장되어 있다.
구글과 OpenAI가 이 기술로 스케일링의 벽을 넘었으니, 학술적으로는 의심할 여지 없는 SOTA(State-of-the-Art) 기술이다.
하지만 실무 엔지니어에게 MoE는 **'배포의 재앙'**이다.
학습시킬 때는 로드 밸런싱 깨져서 툭하면 발산하고, 서빙할 때는 그 무식한 파라미터 크기 때문에 VRAM 비용이 천문학적으로 깨진다.
토큰 하나 찍을 때마다 수 테라바이트짜리 모델이 메모리 버스를 꽉 틀어막는 꼴을 보면 속이 터진다.
결국 남들이 "우리도 MoE 쓴다"고 자랑할 때, 현명한 회사는 잘 깎은 7B짜리 Dense 모델 쓴다.
그게 돈 버는 길이다.