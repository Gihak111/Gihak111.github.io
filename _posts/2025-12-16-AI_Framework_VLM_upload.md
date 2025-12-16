---
layout: single
title: "VLM"
categories: "AI"
tag: "Framework"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## VLM
Vision-Language Models는 텍스트라는 단일 모달리티에 갇혀있던 거대 언어 모델(LLM)의 한계인 시각적 문맹을 해결하기 위해 제안된 프레임워크이다.  
기존 LLM은 세상의 지식을 글로만 배웠기에, "이 사진 속 남자가 들고 있는 물건이 위험한가?"와 같은 직관적인 질문에 답할 수 없었다.  
반면 비전 모델(CV)은 분류는 잘하지만 추론과 설명 능력이 결여되어 있었다.  
VLM은 시각 정보(Pixel)를 언어 정보(Token)와 동일한 임베딩 공간으로 투영하여, AI가 세상을 보고 이해하며 말하는 멀티모달 추론의 시대를 열었다.  

## 2. 수식적 원리: 모달리티 정렬과 교차 어텐션
VLM의 이론적 토대는 서로 다른 물리적 차원을 가진 이미지 특징 공간과 텍스트 임베딩 공간을 수학적으로 정렬하는 데 있다.  
이미지 $x_v$와 텍스트 지시문 $x_t$가 주어졌을 때, 응답 $y$를 생성하는 조건부 확률 분포를 모델링한다.  

$$p_\theta(y | x_v, x_t) = \prod_{i=1}^{T} p_\theta(y_i | y_{<i}, \mathcal{P}(Enc_v(x_v)), x_t)$$

여기서 $Enc_v$는 비전 인코더(ViT 등)이며, 핵심은 투영 함수 $\mathcal{P}$이다.  
이 함수는 비전 인코더가 추출한 연속적인 특징 벡터(Continuous Feature Vector)를 LLM이 이해할 수 있는 이산적인 토큰 임베딩(Token Embedding) $H_v$로 변환한다.  
수학적으로 이는 두 매니폴드(Manifold) 간의 선형 또는 비선형 매핑(Mapping) 문제로 귀결되며,  
이를 통해 픽셀 데이터는 언어 모델 내부에서 마치 외국어 단어처럼 처리된다.  

## 3. 학습 방법론: Visual Instruction Tuning
VLM은 이미지와 텍스트의 연관성을 학습하는 동시에, 사용자의 지시를 따르는 능력을 배양해야 한다.  
이를 위해 Pre-training과 Instruction Tuning의 2단계 최적화 전략을 주로 사용한다(예: LLaVA 방식).  

손실 함수는 투영된 이미지 토큰을 조건으로 하는 다음 토큰 예측의 음의 로그 유도(NLL)이다.  

$$\mathcal{L}_{VLM} = - \sum_{i=1}^{L} \log P_\theta(y_i | y_{<i}, H_v, x_{instruct})$$

1.  **Feature Alignment (Pre-training)**:  
LLM과 비전 인코더를 고정하고, 둘 사이를 연결하는 투영 레이어(Projector)만 학습시킨다.  
이때 이미지-캡션 쌍(Image-Caption Pairs)을 사용하여 $H_v$가 텍스트 공간 내에서 올바른 의미 위치에 안착하도록 유도한다.  
2.  **Visual Instruction Tuning (Fine-tuning)**:  
투영 레이어와 LLM을 함께(또는 일부만) 학습시키며, 복잡한 질문과 답변이 포함된 데이터셋을 통해 시각적 추론 능력을 극대화한다.  
3.  **최적화**:  
이미지 해상도가 높아질수록 토큰 수가 급증하므로, 토큰을 압축하는 C-Abstractor나 Spatial Pooling 기법을 적용하여 연산 효율을 확보한다.  

## 4. 핵심 기술: 커넥터와 비전 인코더
VLM이 단순히 이미지를 캡셔닝하는 것을 넘어 복합적인 대화를 할 수 있는 비결은 아키텍처를 구성하는 핵심 모듈에 있다.  

* **비전 인코더 (Vision Encoder)**:
    CLIP이나 SigLIP과 같이 대규모 이미지-텍스트 대조 학습(Contrastive Learning)으로 훈련된 모델을 사용하여, 이미지의 의미론적 특징(Semantic Feature)을 추출한다.  
    단순히 엣지나 텍스처가 아닌 '객체와 관계'를 본다.  
* **모달리티 커넥터 (Modality Connector)**:  
    비전 특징을 언어 공간으로 이어주는 '다리' 역할을 한다.  
    * **Linear/MLP**: LLaVA와 같이 단순하지만 강력한 방식. 시각 정보를 손실 없이 그대로 LLM에 밀어넣는다.  
    * **Q-Former**: BLIP-2와 같이 학습 가능한 쿼리를 사용하여 비전 특징 중 텍스트와 관련된 정보만 선택적으로 추출(Bottleneck)하여 LLM에 전달한다.  

## 5. 아키텍처 확장: Any-to-Any (Omni)
VLM의 개념은 이미지에 국한되지 않고 비디오, 오디오를 포함하는 **Omni-modal Model**로 확장된다.  
이를 구현한 **Unified Transformer** 아키텍처는 입력과 출력을 모두 토큰화하여 처리한다.  

$$Token_{out} = \text{Transformer}([Token_{img}, Token_{audio}, Token_{text}])$$

* **시간적 확장 (Temporal Extension)**: 비디오 처리를 위해 프레임 간의 시간적 상관관계를 분석하는 3D Conv나 Temporal Attention을 도입하지 않고, 단순히 프레임을 나열하여 LLM의 긴 문맥 처리(Long Context) 능력에 의존하는 경향이 강해지고 있다.  
* **인터리브드 처리 (Interleaved Processing)**: "텍스트-이미지-텍스트-이미지"가 섞여 있는 게시글 전체를 하나의 시퀀스로 이해하고, 텍스트 중간에 이미지를 생성해 넣는 양방향 생성 능력으로 진화하고 있다(예: GPT-4o, Gemini).  

## 6. 추론 및 Resolution Strategy
VLM의 추론 성능은 입력 이미지의 해상도 처리 방식인 **Resolution Strategy**에 크게 좌우된다.

1.  **Dynamic Resolution**: 고해상도 이미지를 있는 그대로 넣으면 연산량이 폭발하므로, 이미지를 여러 개의 패치(Patch)로 자른 뒤, 축소된 전체 이미지와 함께 입력한다.
2.  **AnyRes Encoding**: 다양한 종횡비를 가진 이미지를 왜곡 없이 처리하기 위해, 패치 그리드를 동적으로 구성하고 위치 인코딩을 2차원으로 보간하여 적용한다.
3.  **Chain-of-Thought**: 시각적 추론 시에도 "이미지의 왼쪽 상단을 보면..."과 같이 시선의 흐름을 텍스트로 풀어내며 추론하도록 유도하여 환각을 줄인다.

## 결론
VLM은 인간의 오감 중 시각이라는 가장 강력한 정보 채널을 AI에게 부여함으로써, 진정한 의미의 인공지능(AGI)으로 가는 핵심 퍼즐을 맞춘 아키텍처이다.  
이제 AI는 냉장고 속 재료를 보고 요리를 추천하거나, 손글씨 회로도를 보고 코드를 짤 수 있게 되었다.  
학술적으로도 모달리티 간의 경계를 허물었다는 점에서 거대한 이정표임이 분명하다.  
하지만 치명적인 단점은 환각도 멀티모달로 한다는 것이다.  
없는 사물을 있다고 하거나, 이미지 속 글자를 엉터리로 읽는(OCR 오류) 경우가 허다하다.  
결국 현업에서 영수증 처리나 문서 인식을 할 때는 최첨단 VLM 안 쓴다.  
그냥 구관이 명관인 OCR 엔진 따로 돌려서 텍스트만 뽑아 LLM에 넣는 게 훨씬 정확하다.  
염병하지 말고 텍스트나 잘 뽑는 게 최고다.  