---
layout: single
title: "AI 아키텍쳐 4. GPT-oss-20B"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## GPT-oss
얼마 전에 공개된 신상 모델이다.  
GPT-5가 나오면서 허깅페이스에 오픈소스로 올라온 모델인데, 압도적인 성능으로 조금 이슈가 되었던 녀석이다.  
특히, GPT-4o mini와 비슷한 수준의 성능을 낸다는 점이 큰 강점으로 떠올랐다.  
이게 왜 이슈냐면, 기존의 클로즈드 모델들처럼 클라우드에 의존하지 않고 로컬에서 돌릴 수 있는 오픈 웨이트 모델인데, 21억 파라미터 규모지만 MoE 덕에 액티브 파라미터가 3.6억 정도로 가볍게 느껴지면서도 고성능을 내기 때문이다.  
모델은 20B(실제 21B)와 120B, 두 가지로 공개되었는데, 120B는 좀 무리지만 20B 모델은 맥에서도 돌릴 수 있을 정도로 접근성이 좋아 이번 공개가 나름 큰 의미를 가진다고 생각한다.  
개인적으로 보자면, 이 모델이 오픈소스로 풀린 덕에 연구자들이나 개발자들이 직접 커스터마이징해서 실험할 수 있게 됐고, 특히 MoE 구조의 효율성을 보여주면서 AI 민주화에 한 발짝 다가선 느낌이다.  

모델 링크는 다음과 같다.  

20B: [https://huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)  
120B: [https://huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)  

120B는 가정에선 돌릴수도 없고, 텐서 뜯는것도 쉽지 않으니 20B 기준으로 글을 작성하였다.  

아무튼, 모델이 공개되었다고 해서 당연히 가중치 값이 더 중요하지만,  
일단 모델 구조라도 뜯어보는 게 재미있지 않겠나.  
뜯는 방법은 간단하다. 허깅페이스에 들어가서 `config.json`을 열어보고, 모델 레이어 텐서를 들여다보면 된다.  
여기서 config.json을 보면 architectures가 "GptOssForCausalLM"으로 되어 있고, hidden_size 2880, num_hidden_layers 24 등 세부 스펙이 나와 있어서, 깊이 파고들기 딱 좋다.  
더 나아가, 모델 카드나 깃헙 리포를 보면 MXFP4 양자화나 harmony response format 같은 추가 세팅이 나오니, 그걸 참고하면 실제 구현할 때 유용할 거다.  

## MoE, YaRN
들어가긱에 앞서, 이 두 기술 자세히 보고 가자  
내가 따로 다룬 적이 없는 것 같아서 적고 들어가야 할 것 같다.  

### 1. Mixture of Experts (MoE)  
MoE는 쉽게 말해서, 여러 고성능 조수를 두고, 필요할 때 적제적소의 조수를 부르는 방식이다.  
전통적인 트랜스포머는 모든 파라미터를 한꺼번에 써서 계산하는데, 이건 계산량이 어마어마해서 GPU 메모리가 많이 부족해진다.  
MoE는 이걸 해결한 기술로, **전체 파라미터를 다 쓰지 않고 일부만 선택적으로 활성화**해서 성능은 유지하면서 효율성을 챙긴다.  
GPT-oss-20b를 보면, 21억 파라미터 중 실제로 추론할 때 쓰이는 액티브 파라미터는 3.6억 정도라고 나와있다.  
이게 바로 MoE의 효과이다.  

### MoE의 작동 원리
MoE는 트랜스포머의 피드 포워드 네트워크(FFN)를 대체하는 구조로, 여러 개의 전문가라는 작은 네트워크를 두고, 입력에 따라 어떤 전문가를 쓸지 동적으로 결정한다.  
GPT-oss에서는 `num_local_experts` 32개, `experts_per_token` 4로 설정되어 있다. 즉, 토큰 하나당 32개의 전문가 중 4개를 골라서 쓴다는 거다.  

- **라우터(Router)**: 입력 토큰을 보고 어떤 전문가를 쓸지 결정하는 게이트 네트워크다.  
  수식으로 보면, 입력 벡터 $$\mathbf{x}$$가 들어오면 라우터가 $$\mathbf{g} = \text{softmax}(\mathbf{x} W_g)$$를 계산해서 각 전문가에 대한 확률 분포를 만든다. 여기서 $$W_g$$  는 학습 가능한 가중치 행렬.  
  그리고 top-k(4) 방식으로 상위 4개 전문가를 선택해서, $$\mathbf{y} = \sum_{i \in \text{topk}} g_i \cdot E_i(\mathbf{x})$$로 출력 벡터를 계산한다. $$E_i$$는 i번째 전문가의 FFN 출력.  
  이 과정에서 `router_aux_loss_coef` 0.9로 설정된 보조 손실 $$mathcal{L}_{aux} = 0.9 cdot text{load_balance_loss}$$를 추가해 전문가들이 골고루 쓰이도록 학습 안정성을 높인다.  

- **전문가(Expert)**: 각 전문가는 독립적인 FFN으로, GPT-oss에서는 `hidden_act` "silu"와 `swiglu_limit` 7.0을 사용한다.  
  SwiGLU는 $$\text{SwiGLU}(\mathbf{x}) = \mathbf{x} \odot \text{SiLU}(\mathbf{x} W_1) W_2$$ 로, SiLU 활성화 함수를 변형해 비선형성을 강화한다.  
  전문가는 $$\mathbf{x} \to \text{SiLU}(\mathbf{x} W_1 + b_1) W_2 + b_2$$처럼 동작하며, `intermediate_size` 2880으로 계산량을 최적화한다.  

- **효율성 증명**: MoE의 강점은 전체 파라미터(21억)를 다 쓰지 않고, 토큰당 4개 전문가만 활성화해서 계산량을 확 줄이는 거다.  
  예를 들어, 32개 전문가 중 4개만 쓰면 파라미터 사용률은 대략 $$4/32 = 12.5\%$$ 수준.  
  이 덕에 GPT-oss-20b는 3.6억 액티브 파라미터로 120B급 모델과 비슷한 성능을 내면서도 맥북 같은 로컬 장비에서 돌릴 수 있는거다.  
  게다가 `quantization_config` mxfp4로 전문가 가중치를 4비트로 양자화해서 메모리도 아꼈다. 단, `model.layers.*.self_attn`이나 `model.embed_tokens` 같은 핵심 모듈은 양자화 제외해서 정밀도를 유지한다.  

- **왜 대박이냐?**: MoE는 계산 효율성을 높이면서도 성능 손실을 최소화한다.  
  대학원생 입장에서 보면, MoE 라우팅 알고리즘을 뜯어보면서 $$text{load_balance_loss}$$  나 top-k 선택 전략 같은 걸 실험해보면 논문 한 편 뽑기 딱 좋다.  
  실제로 `output_router_logits` false로 라우터 로짓은 출력 안 하지만, 디버깅용으로 이걸 true로 바꿔서 전문가 선택 패턴을 분석해보면 재밌는 인사이트가 나올 거다.  

## 2. YaRN RoPE 스케일링
YaRN(Yet another RoPE-based Network)은 GPT-oss가 긴 문맥을 처리할 수 있게 해주는 비밀 병기다.  
기존 트랜스포머는 max_position_embeddings가 고정되어 있어서, 예를 들어 GPT-2는 1024 토큰 정도로 제한됐다.  
근데 GPT-oss는 `max_position_embeddings` 131,072로, 거의 소설 한 권 분량의 문맥을 다룰 수 있다. 이게 다 YaRN 덕분이다.  
YaRN은 RoPE(Rotary Position Embedding)를 스케일링해서 긴 시퀀스에서도 위치 정보를 안정적으로 유지하는 기술이다.  

### YaRN의 작동 원리
RoPE는 위치 임베딩을 학습 가능한 벡터 대신 회전 행렬로 처리하는 방식이다.  
일반적인 위치 임베딩은 $$\mathbf{p}_m$$처럼 각 위치 $$m$$에 고정 벡터를 할당하는데, 이건 긴 시퀀스에서 메모리 비효율적이고 extrapolation이 약하다.  
RoPE는 쿼리 $$\mathbf{q}$$와 키 $$\mathbf{k}$$를 회전 변환해 위치 정보를 내재화한다.  

- **RoPE 기본 수식**:  
  위치 $$m$$의 토큰에 대해 쿼리 벡터 $$\mathbf{q}$$를 $$\mathbf{q}_m = \mathbf{q} \odot \exp(i m \theta)$$로 변환. 여기서 $$\theta$$는 회전 각도, GPT-oss에서는 `rope_theta` 150,000으로 설정.  
  키 $$\mathbf{k}$$도 마찬가지로 $$\mathbf{k}_m = \mathbf{k} \odot \exp(i m \theta)$$.  
  어텐션 스코어는 $$\mathbf{q}_m^T \mathbf{k}_n = (\mathbf{q} \odot \exp(i m \theta))^T (\mathbf{k} \odot \exp(i n \theta))$$로 계산되며, 상대적 위치 $$m-n$$에 따라 회전 각도가 반영된다.  
  이 방식은 $$\exp(i (m-n) \theta)$$처럼 상대적 거리를 자연스럽게 학습한다.  

- **YaRN의 스케일링**:  
  YaRN은 RoPE를 확장해 긴 시퀀스에서도 성능 저하 없이 extrapolation을 가능하게 한다.  
  GPT-oss의 `rope_scaling`은 `factor` 32, `beta_fast` 32, `beta_slow` 1, `original_max_position_embeddings` 4096으로 설정.  
  YaRN은 $$\theta_k = \theta_0 / (k+1)^\alpha$$로 회전 각도를 동적으로 조절하며, $$\alpha$$는 beta_fast와 beta_slow로 조합된다.  
  예를 들어, $$\theta_k = 150000 / (k+1)^{1/\text{factor}}$$처럼 스케일링해 긴 시퀀스에서 안정성을 유지한다.  
  `truncate` false로 전체 시퀀스를 유지하며, `initial_context_length` 4096에서 131,072까지 확장 가능.  

- **효율성 증명**:  
  YaRN은 기존 RoPE 대비 긴 컨텍스트에서 어텐션 스코어의 붕괴를 막는다.  
  예를 들어, 4096 토큰 이상의 시퀀스에서 일반 RoPE는 $$\theta$$가 커지면서 수치 불안정성을 유발하지만, YaRN은 $$\text{factor} 32$$로 스케일링해 $$\theta$$를 조절한다.  
  수식적으로, $$\mathbf{q}_m^T \mathbf{k}_n \approx \mathbf{q}^T \mathbf{k} \cos((m-n)\theta)$$로 근사되며, YaRN은 $$\theta$$의 분포를 조정해 긴 거리에서도 정보 손실을 최소화한다.  
  이 덕에 GPT-oss는 소설 쓰기, 긴 코드 생성, 심지어 논문 요약 같은 작업에서 문맥 일관성을 유지한다.  

- **왜 대박이냐?**:  
  YaRN은 긴 문맥 처리의 한계를 깨고, 메모리 효율적으로 extrapolation을 가능하게 한다.  
  대학원생 입장에서, YaRN의 $$\theta$$ 조절 로직이나 $$\beta$$ 하이퍼파라미터를 실험해보면 긴 컨텍스트 모델링 논문에 좋은 소재가 된다.  
  `rope_scaling` 설정을 바꿔가며 131,072 이상으로 확장해보거나, sliding_window 128과 결합해 메모리 효율성을 분석해보면 재밌을 거다.  

## 아키텍처 구조
GPT-oss의 아키텍처는 트랜스포머 기반의 디코더 구조를 중심으로 설계되었으며, 특히 **Mixture of Experts(MoE)** 방식을 도입해 효율성과 성능을 극대화한 점이 눈에 띈다.  
이는 GPT-2의 단방향 자기회귀(autoregressive) 방식에서 더 발전시킨 형태로, 문맥을 기반으로 다음 단어를 예측하면서도 대규모 데이터 처리와 생성 능력을 강화한 구조다.  
이 모델은 기존 GPT 시리즈와 비슷한 철학을 공유하지만, **슬라이딩 어텐션**과 **풀 어텐션**을 번갈아 사용하는 독특한 레이어 구성과 **YaRN RoPE 스케일링**을 통해 긴 문맥을 효과적으로 처리할 수 있도록 설계되었다.  
구체적으로, layer_types가 sliding_attention과 full_attention이 번갈아 12개씩 총 24개로 구성되어 있어서, 메모리 효율과 문맥 이해를 균형 잡은 게 인상적이다.  
또한, MoE를 통해 num_local_experts 32개 중 experts_per_token 4개를 선택적으로 활성화하니, 전체 파라미터를 다 쓰지 않고도 고성능을 내는 게 핵심 포인트다.  

GPT-oss-20b 모델의 주요 구조는 다음과 같다:  

| 모델 이름 | 파라미터 수 | 디코더 레이어 | 임베딩 차원 | 어텐션 헤드 수 | 키-값 헤드 수 | 헤드 차원 | 중간 크기 | 로컬 전문가 수 | 토큰당 전문가 수 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-oss-20b | 21억 (액티브 3.6억) | 24 | 2880 | 64 | 8 | 64 | 2880 | 32 | 4 |

이 테이블에서 보듯, 임베딩 차원이 2880으로 꽤 크고, 키-값 헤드가 8개로 그룹 쿼리 어텐션(GQA)을 암시하니, 추론 속도가 빨라질 거다.  
또, max_position_embeddings가 131072로 YaRN 스케일링 덕에 긴 컨텍스트를 지원한다.  

이제 GPT-oss의 데이터 처리 흐름을 단계별로 살펴보자.  
이 흐름은 기본적으로 트랜스포머 디코더의 반복 루프지만, MoE와 슬라이딩 어텐션이 더해져 효율이 업그레이드됐다.  

1. **입력 표현 (Input Representation)**: 입력 텍스트는 토큰 시퀀스로 변환된다.  
GPT-oss는 BPE 기반 토크나이저를 사용해 단어와 서브워드를 효율적으로 처리하며, 201,088개의 어휘를 지원한다.  
   이 어휘 크기는 다국어 지원을 고려한 거로, 예를 들어 한국어 같은 언어에서 서브워드 분해가 잘 돼서 OOV 문제를 최소화한다.  
   토큰화 과정은 $$\text{Token Sequence} = \text{BPE}( \text{Input Text} )$$로 표현할 수 있고, pad_token_id 199999, eos_token_id 200002처럼 특수 토큰이 정의되어 있다.  

2. **임베딩 (Embedding)**: 토큰과 위치 정보를 결합한 벡터로 변환된다.  
특히, **YaRN RoPE 스케일링**을 통해 최대 131,072 토큰의 긴 문맥을 처리할 수 있도록 설계되었다.  
   RoPE는 회전 행렬을 이용한 위치 인코딩으로, $$\mathbf{x}_i = \mathbf{W} \mathbf{e}_i$$에서 회전 각도 $$\theta$$를 rope_theta 150000으로 설정해 extrapolation을 좋게 한다.  
   YaRN은 factor 32, beta_fast 32, beta_slow 1로 스케일링되어 긴 시퀀스에서 성능 저하를 방지한다.  

3. **트랜스포머 디코더 스택 (Transformer Decoder Stack)**: 24개의 디코더 레이어가 슬라이딩 어텐션과 풀 어텐션을 번갈아 사용하며 문맥을 점진적으로 심화시킨다.  
각 레이어는 MoE 구조를 활용해 32개의 로컬 전문가 중 4개를 선택해 처리한다.  
   이 번갈아 구성은 sliding_window 128로 로컬 어텐션을 제한해 메모리를 아끼고, 풀 어텐션으로 글로벌 컨텍스트를 잡는다.  
   전체 스택은 $$\mathbf{H}^{l+1} = \text{DecoderLayer}(\mathbf{H}^l)$$로 반복되며, l=1 to 24.  

4. **출력 및 샘플링 (Output & Sampling)**: 마지막 레이어의 출력은 어휘 크기의 로짓 벡터로 변환되고, 소프트맥스를 통해 다음 단어의 확률 분포를 생성한다.  
이 분포를 기반으로 단어를 샘플링한다.  
   로짓은 $$\mathbf{o} = \mathbf{H}^{24} \mathbf{W}_{out} + \mathbf{b}_{out}$$로 계산되며, 확률은 $$\mathbf{p} = \text{softmax}(\mathbf{o}/T)$$로, T는 temperature로 샘플링 다양성을 조절한다.  
   output_router_logits false로 라우터 로짓을 출력하지 않지만, router_aux_loss_coef 0.9로 보조 손실을 사용해 학습 안정성을 높인다.  

5. **자기회귀 (Autoregression)**: 예측된 단어를 입력 시퀀스에 추가해 다음 단어를 예측하는 과정을 반복하며 문장을 생성한다.  
   이는 $$\mathbf{x}_{t+1} = \arg\max \mathbf{p}(\mathbf{x}_{t+1} | \mathbf{x}_{1:t})$$로 표현되며, 빔 서치나 nucleus 샘플링 같은 기법을 추가로 쓸 수 있다.  
   긴 컨텍스트에서 use_cache true로 KV 캐싱을 활용해 추론 속도를 높인다.  

이제 각 구성 요소의 역할과 특징을 좀 더 깊이 파헤쳐 보자.  

## 각 레이어별 역할

GPT-oss의 강력함은 단순한 다음 단어 예측을 넘어, 복잡한 문맥을 이해하고 생성하는 데 있다.  
이를 위해 각 레이어가 유기적으로 협력하며 문맥의 뉘앙스를 정교하게 다듬는다.  
특히, MoE와 YaRN의 결합이 모델의 스케일링 능력을 높여주니, 이 부분을 중점적으로 봐보자.  

### 1. 임베딩 레이어 (Embedding Layer)  
모델의 첫걸음은 텍스트를 숫자 벡터로 변환하는 것이다.  
GPT-oss는 단어의 의미와 순서 정보를 결합해 강력한 초기 표현을 만든다.  
이 레이어는 tie_word_embeddings false로 임베딩과 출력이 공유되지 않아, 더 유연한 학습이 가능하다.  

- **토큰 임베딩 (Token Embeddings)**:  
  BPE 토크나이저를 사용해 201,088개의 어휘를 처리한다.  
  자주 등장하는 단어와 서브워드를 학습해 새로운 단어도 유연하게 표현할 수 있다.  
  예를 들어, "unbelievable"은 "un", "believe", "able"로 나누어 처리되며, 각 토큰은 2880차원의 벡터로 변환된다.  
  임베딩은 $$\mathbf{e}_i = \text{EmbeddingLookup}(t_i)$$로, initializer_range 0.02로 가중치 초기화되어 학습 안정성을 준다.  
  BPE의 병합 과정은 빈도 기반으로, $$\text{merge}(a,b) = \arg\max \text{freq}(ab)$$처럼 반복되어 어휘를 구축한다.  

- **위치 임베딩 (Position Embeddings)**:  
  트랜스포머는 병렬 처리를 위해 순서 정보를 별도로 학습해야 한다.  
  GPT-oss는 **YaRN RoPE 스케일링**을 사용해 최대 131,072 토큰의 위치 정보를 효과적으로 처리한다.  
  이는 기존 GPT 모델 대비 긴 문맥을 다룰 수 있게 해주는 핵심 기술이다.  
  RoPE는 $$\mathbf{q}_i = \mathbf{q} \odot \exp(i \theta m)$$처럼 회전 벡터를 적용하며, YaRN은 $$\theta_k = \theta_0 / (k+1)^\alpha$$로 스케일링해 extrapolation을 개선한다.  
  original_max_position_embeddings 4096에서 factor 32로 확장되며, truncate false로 전체를 유지한다.  
  $$ \text{Input Embedding} = \text{Token Embedding} + \text{RoPE Position Embedding}$$


  이 덕에 모델은 긴 소설이나 코드 생성에서 문맥 일관성을 유지한다.  

### 2. 트랜스포머 디코더 레이어 (Transformer Decoder Layer)  

GPT-oss-20b는 24개의 디코더 레이어로 구성되며, 각 레이어는 **마스크드 셀프 어텐션**, **MoE 기반 피드 포워드 네트워크**, 그리고 **잔차 연결 및 레이어 정규화**로 이루어진다.  
hidden_act "silu"로 SiLU 활성화 함수를 쓰며, attention_bias true, attention_dropout 0.0으로 안정적이다.  

#### 2-1. 마스크드 멀티-헤드 셀프 어텐션 (Masked Multi-Head Self-Attention)  

- **역할**: 문맥의 과거 정보만을 참고해 다음 단어를 예측한다.  
- **작동 원리**: 64개의 어텐션 헤드와 8개의 키-값 헤드를 사용해 문맥을 병렬적으로 분석한다.  
  슬라이딩 윈도우(128 토큰)와 풀 어텐션을 번갈아 사용해 계산 효율성과 문맥 이해를 균형 있게 유지한다.  
  마스크는 미래 토큰의 정보를 차단해 자기회귀 방식의 예측을 보장한다.  
  어텐션 스코어는 $$\text{Attn}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V$$로, M은 look-ahead mask (상삼각에 -\infty).  
  멀티헤드는 $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$로 h=64.  
  GQA로 KV 헤드 8개로 메모리 절감하며, $$\sqrt{d_k} = \sqrt{64}$$로 스케일링.  
- **특징**: 슬라이딩 어텐션은 긴 시퀀스에서 메모리 사용량을 O(n)으로 줄이고, 풀 어텐션은 중요한 문맥 정보를 정밀하게 포착한다.  
  이 번갈아 구성은 layer_types 배열로 구현되며, sliding_window 128로 로컬 컨텍스트를 제한해 O(n^2) 문제를 완화한다.  

#### 2-2. MoE 피드 포워드 네트워크 (MoE Feed-Forward Network)  
- **역할**: 각 토큰의 표현을 비선형적으로 변환해 문맥을 심화시킨다.  
- **작동 원리**: 32개의 로컬 전문가 중 4개를 선택해 처리하며, **Swiglu 활성화 함수**(제한값 7.0)를 사용한다.  
  이는 기존 GELU 대비 더 복잡한 패턴을 학습할 수 있게 한다.  
  MoE 구조는 계산 효율성을 높이며, 특정 작업에 최적화된 전문가를 동적으로 선택한다.  
  라우팅은 $$\mathbf{g} = \text{softmax}(\mathbf{x} W_g)$$로 게이트 계산 후 top-k(4) 선택, $$\mathbf{y} = \sum_{i \in \text{topk}} g_i \cdot E_i(\mathbf{x})$$로 출력.  
  swiglu_limit 7.0으로 SwiGLU $$\text{SwiGLU}(\mathbf{x}) = \mathbf{x} \odot \text{SiLU}(\mathbf{x} W_1) W_2$$ 변형 사용.  
  quantization_config mxfp4로 MoE 웨이트만 양자화, self_attn과 router 등은 제외해 정밀도 유지.  
- **증명**: MoE는 모든 파라미터를 항상 사용하는 대신 선택적으로 활성화해, 21억 파라미터 모델이 더 큰 모델과 유사한 성능을 낼 수 있게 한다.  
  router_aux_loss_coef 0.9로 $$\mathcal{L} {aux} = 0.9 \cdot \text{load balance loss}$$처럼 균형 손실 추가해 전문가 이용률 균등화.  

#### 2-3. 잔차 연결 및 레이어 정규화 (Add & Norm)  
- **역할**: 깊은 네트워크의 학습 안정성을 보장한다.  
- **작동 원리**: 각 레이어의 입력과 출력을 더하는 잔차 연결은 그래디언트 소실을 방지하고, RMS 정규화(엡실론 1e-05)는 출력 분포를 안정화시킨다.  
  잔차는 $$\mathbf{x}' = \mathbf{x} + \text{Sublayer}(\mathbf{x})$$로, RMSNorm은 $$\text{RMSNorm}(\mathbf{x}) = \mathbf{x} / \sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}$$로 구현.  
  이는 24개 레이어의 깊은 구조에서 필수적이다.  
  rms_norm_eps 1e-05로 수치 안정성 보장하며, 깊은 레이어에서 그래디언트 흐름을 $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 1 + \frac{\partial \text{Sublayer}}{\partial \mathbf{x}}$$처럼 유지.  

## GPT-oss의 학습 방식과 활용  
GPT-oss의 진가는 대규모 사전 학습과 효율적인 MoE 구조에서 나온다.  
harmony response format으로 학습되어, 추론 시 이를 따라야 하며, reasoning level (low, medium, high)을 프롬프트로 조절할 수 있다.  

### 1. 표준 언어 모델링 (Standard Language Modeling)  
- **목표**: 주어진 문맥에서 다음 단어를 예측한다.  
- **방식**: 방대한 웹 데이터(예: WebText와 유사한 데이터셋)로 사전 학습되었다.  
  Cross-Entropy Loss를 사용해 예측 정확도를 높이며, 문법, 의미, 논리적 흐름을 학습한다.  
  손실 함수는 $$\mathcal{L} = -\sum_{t} \log p(\mathbf{x}_t | \mathbf{x}_{<t})$$로, MoE 라우팅 손실 추가.  
  학습 데이터는 고품질 웹 스크랩으로, 40GB 규모 추정되며, BPE 토크나이저로 전처리.  
  post-training MXFP4 양자화로 효율화되었고, consumer hardware에서 fine-tuning 가능.  
- **효과**: 단순한 목표지만, MoE와 YaRN RoPE를 통해 긴 문맥과 복잡한 패턴을 효과적으로 학습한다.  
  예를 들어, high reasoning 모드에서 chain-of-thought를 내재화해 복잡한 문제 해결.  

### 2. 제로샷/퓨샷 학습 (Zero/Few-shot Learning)  
- **제로샷**: 추가 학습 없이 프롬프트만으로 번역, 요약 등 다양한 작업을 수행한다.  
  예: "Translate to Korean: Hello world"처럼 지시만으로 동작.  
- **퓨샷**: 몇 개의 예시를 제공하면 성능이 더욱 향상된다.  
  예시 3~5개로 패턴 학습해 정확도도 올라간다.  
- **의미**: GPT-oss는 언어의 일반적 패턴을 학습해 프롬프트 기반 상호작용에서 강력한 유연성을 보여준다.  
  tool use 지원으로 web browsing, Python 실행, function calling 가능하며, structured output으로 JSON 등 출력.  
  reasoning access로 내부 생각 과정 디버깅 가능, Apache 2.0 라이선스로 상업 이용 자유.  

## 결론  
GPT-oss는 트랜스포머 디코더와 MoE, YaRN RoPE 스케일링을 결합해 효율성과 성능을 동시에 잡은 모델이다.  
GPT-2가 생성 AI의 서막을 열었다면, GPT-oss는 긴 문맥 처리와 계산 효율성을 강화해 로컬 모델의 접근성 개선에 크게 기여한다.  
특히, 20B 모델은 개인 장비에서도 실행 가능해 아무나 들고가서 광고 엄청 하면서 한국어로 대답하라고 프롬프트 조금만 던져두면, 뤼튼마냥 바로 돈방석에 앉을 수 있는거다.  
이상 마치겠다 오늘도 좋은 하루 보내길 바란다.  