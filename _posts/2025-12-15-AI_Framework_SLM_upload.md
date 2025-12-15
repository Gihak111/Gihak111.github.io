---
layout: single
title: "SLM"
categories: "AI"
tag: "Framework"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## SLM
Small Language Models는 거대 언어 모델의 비대해진 파라미터 수로 인한 비효율적인 연산 비용과 배포의 제약을 해결하기 위해 대두된 프레임워크이다.  
기존 LLM들은 Scaling Law에 따라 모델 크기를 키우는 데 집중했지만, 이는 추론 시 막대한 GPU 자원을 요구하고 온디바이스(On-device) 환경에서의 구동을 불가능하게 만들었다.  
SLM은 모델의 크기를 줄이면서도 데이터의 질과 학습 밀도를 극한으로 높여, 적은 파라미터로도 LLM에 준하는 성능을 내는 파라미터 효율성의 정점을 보여준다.  

## 2. 지식 증류와 정보 엔트로피
SLM의 이론적 토대는 거대 모델의 암묵적 지식을 작은 모델로 전이하는 지식 증류에 있다.  
단순히 정답 레이블을 맞추는 것이 아니라, Teacher 모델이 출력하는 확률 분포 자체를 근사하게 만든다.  

학습 목표는 Student 네트워크 $S$와 Teacher 네트워크 $T$ 간의 쿨백-라이블러 발산(KL Divergence)을 최소화하는 것이다.  

$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \tau^2 D_{KL}(\sigma(z_t/\tau) || \sigma(z_s/\tau))$$

여기서 $z_t, z_s$는 로짓(Logits), $\tau$는 분포를 부드럽게 만드는 온도(Temperature) 계수이다.  
증명(Proof)의 관점에서 볼 때, LLM의 파라미터 공간 $\Theta_{L}$은 실제 언어 작업에 필요한 본질적 차원 $d$보다 훨씬 큰 과매개변수화 상태이다 ($|\Theta_{L}| \gg d$).  
SLM은 이 거대한 공간 속에 존재하는 복권 당첨 티켓과 같은 최적의 부분 공간 $\Theta_{S}$를 찾아내는 과정으로 볼 수 있으며,  
이는 정보 이론적으로 Teacher가 가진 엔트로피 $H(T)$를 손실 없이 $S$에 압축하는 것이 가능함을 시사한다.  

## 3. 학습 방법론: Curriculum Learning과 Data Pruning
SLM은 제한된 용량(Capacity) 내에 정보를 욱여넣어야 하므로, 데이터의 양보다 질이 압도적으로 중요하다.  
이를 위해 Textbook Quality Data만을 선별하여 학습하는 전략을 사용한다.  

학습 과정은 데이터의 난이도와 품질에 따라 가중치를 조절하는 커리큘럼 학습으로 공식화된다.  

$$\min_\theta \sum_{i=1}^{N} w_i(t) \mathcal{L}(f_\theta(x_i), y_i)$$

여기서 $w_i(t)$는 학습 단계 $t$에서 데이터 샘플 $x_i$의 중요도를 나타내는 가중치 함수이다.  
1.  Filtering : 교과서, 논문, 양질의 코드 등 정보 밀도가 높은 데이터를 필터링하여 노이즈를 제거한다. 웹 크롤링 데이터의 90%는 SLM 학습에 오히려 해가 된다.  
2.  Synthetic Data Generation : LLM을 활용하여 논리적 추론 과정이 포함된 합성 데이터를 생성하고, 이를 SLM에게 학습시킨다(Phi-series 접근법).  
3.  최적화 : 양자화를 염두에 둔 QAT(Quantization-Aware Training)를 수행하여, 추후 int4/int8로 압축 시 성능 저하를 방지한다.  

## 4. 핵심 기술: 파라미터 공유와 희소 활성화
SLM이 작지만 강력할 수 있는 비결은 연산 효율성을 극대화하는 아키텍처 설계에 있다.  

* **그룹 쿼리 어텐션 (Grouped-Query Attention, GQA)**:  
    MHA(Multi-Head Attention)의 메모리 대역폭 문제를 해결하기 위해, Key와 Value 헤드를 여러 Query 헤드가 공유하도록 설계한다.  
    이는 KV 캐시 크기를 줄여 긴 문맥 처리를 가능하게 하고 추론 속도를 비약적으로 높인다.  
* **희소 전문가 모델 (Sparse MoE)**:  
    SLM의 전체 파라미터 수는 늘리되, 추론 시에는 일부 전문가 네트워크만 활성화한다.  
    $$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$
    게이팅 네트워크 $G(x)$가 입력에 따라 필요한 $E_i$만 선택하므로, 연산량은 작은 모델 수준으로 유지하면서 지식의 총량은 늘릴 수 있다.  

## 5. 아키텍처 확장: SLM-Speculative
SLM의 빠른 추론 속도는 거대 모델의 보조 엔진으로 활용될 때 극대화된다.  
이를 구현한 **Speculative Decoding** 아키텍처는 SLM을 '초안 작성자'로 활용한다.  

$$x_{draft} \sim P_{SLM}(x|prefix), \quad \text{Verify: } \frac{P_{LLM}(x_{draft})}{P_{SLM}(x_{draft})}$$

* **Drafting**: 가볍고 빠른 SLM이 먼저 $K$개의 토큰을 빠르게 생성한다.  
* **Verification**: 무거운 LLM은 생성된 $K$개의 토큰을 한 번의 전방 전달로 검증한다.  
* **Accept/Reject**: LLM의 확률 분포와 비교하여 허용 가능한 토큰은 채택하고, 나머지는 수정한다.  
    이 구조는 단독 SLM의 성능 한계를 LLM으로 보완하면서, LLM의 느린 속도를 SLM으로 가속화하는 상호보완적 확장을 가능케 한다.  

## 6. 추론 및 On-device Optimization
SLM의 추론은 클라우드가 아닌 엣지 디바이스(Edge Device) 내에서 완결되는 것을 목표로 한다.  

1.  **Graph Folding**: 모델의 연산 그래프를 분석하여 Conv와 BatchNorm, 혹은 Linear와 Activation을 하나의 연산으로 합쳐(Fusion) 메모리 접근 횟수를 줄인다.  
2.  **Memory Mapping**: 가중치 데이터를 메모리에 모두 로드하지 않고, mmap을 통해 필요한 부분만 스트리밍하거나 NPU(Neural Processing Unit) 전용 메모리 레이아웃으로 변환한다.  
3.  **Execution**: CPU/GPU/NPU 간의 이기종 컴퓨팅을 통해 전력 소모를 최소화하며 토큰을 생성한다.  

이 방식은 인터넷 연결 없이도 프라이버시를 보장하며 AI 기능을 수행할 수 있다는 강력한 장점이 있다.

## 결론
SLM은 "거거익선(Bigger is Better)"을 외치던 AI 업계에 "작은 고추가 맵다"는 것을 수식적으로 증명해낸 효율성의 승리이다.
전력 소모와 비용 문제로 인해 모든 곳에 GPT-4를 심을 수 없는 현실에서, 스마트폰, 자동차, 가전제품에 탑재될 AI의 표준은 결국 SLM이 될 것이다.
하지만 기업들이 SLM을 미친 듯이 미는 진짜 이유는 '온디바이스 AI의 혁신'이나 '프라이버시 보호' 때문이 아니다.
그냥 클라우드 API 호출 비용 아끼고 싶어서다.
사용자 입장에서는 내 폰 배터리 녹여가며 멍청한 모델 돌리느니, 그냥 서버에 있는 똑똑한 놈 부르는 게 낫다.