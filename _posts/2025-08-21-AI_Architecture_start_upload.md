---
layout: single
title: "다양한 AI의 아키텍텨를 알아보자"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## AI 모델
지금까지, 이미 블로그에서 AI를 많이 다루었다.  
하지만, 아직 갈 길이 멀다.  
이번에는, 다음 AI의 아키텍쳐를 한번 뜯어보는 시간을 가져볼까 한다.  
물론, 어떤 모델들은 미공개라서 대충 추축할 수 밖에 없거나 하지만,  
그래도 한번 알아보자.  

너가 인코더, 디코더 등등 AI 기본에 대해 잘 알고 있다면 쉽게 이해할 수 있을 것이다.  

## 목록
일단, 이정도만 먼저 다루어 볼까 한다  
### **I. 언어 모델 (Language Models)**

#### **1) 기반 아키텍처 (Foundational Architectures)**
* **BERT (`bert-base-uncased`)**
    * **역할:** 문맥 이해 및 임베딩 추출  
    * **분석 포인트:** 최초의 양방향 구조. 트랜스포머 **인코더(Encoder)** 스택의 작동 방식과 MLM(Masked Language Model) 사전 학습 기법을 이해하기 위한 필수 모델.  
    * **링크:** [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)  

* **BART (`BART-Large`)**
    * **역할:** 텍스트 생성 및 변환(특히 요약에 강하다)  
    * **분석 포인트:** 양방향 인코더와 자기회귀 디코더를 물리적으로 결합한 독특한 구조. 소스 텍스트는 양방향으로 완벽하게 이해, 타겟 텍스트는 순차적으로 생성  
    * **링크:** [ https://huggingface.co/facebook/bart-large]( https://huggingface.co/facebook/bart-large)  

* **GPT-2 (`gpt2`)**
    * **역할:** 자기회귀 텍스트 생성  
    * **분석 포인트:** 트랜스포머 **디코더(Decoder)** 스택의 표준. 인과적 마스킹(Causal Masking)을 통해 어떻게 미래 정보를 보지 않고 다음 단어를 예측하는지 분석의 핵심.  
    * **링크:** [https://huggingface.co/gpt2](https://huggingface.co/gpt2)  

* **gpt-oss-20b (`gpt-oss-20b]`)**
    * **역할:** 자기회귀 텍스트 생성  
    * **분석 포인트:** 트랜스포머 **디코더(Decoder)** 스택의 표준. 인과적 마스킹(Causal Masking)을 통해 어떻게 미래 정보를 보지 않고 다음 단어를 예측하는지 분석의 핵심.  
    * **링크:** [https://huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)  

* **T5 (`t5-base`)**
    * **역할:** 텍스트-투-텍스트(Text-to-Text) 변환  
    * **분석 포인트:** 표준적인 **인코더-디코더(Encoder-Decoder)** 구조의 정석. 모든 NLP 문제를 텍스트 변환으로 통일하는 접근 방식과 구조의 관계를 파악하기 좋다.  
    * **링크:** [https://huggingface.co/t5-base](https://huggingface.co/t5-base)  

#### **2) 효율성 개선 아키텍처 (Efficiency-Improved Architectures)**
* **Mixtral 8x7B (`mistralai/Mixtral-8x7B-Instruct-v0.1`)**
    * **역할:** 고성능 LLM의 계산 효율화  
    * **분석 포인트:** **희소 전문가 혼합(Sparse Mixture-of-Experts, SMoE)** 아키텍처. 모든 파라미터를 사용하는 대신, 입력에 따라 활성화되는 '전문가' 네트워크(Feed-Forward)를 선택하여 계산량을 줄이는 라우팅 메커니즘이 핵심.  
    * **링크:** [https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)  

* **Mamba (`state-spaces/mamba-2.8b-hf`)**
    * **역할:** 트랜스포머의 대안 (비(非)어텐션)  
    * **분석 포인트:** 어텐션의 제곱 연산(quadratic complexity) 문제를 해결하는 **상태 공간 모델(State Space Model, SSM)**. RNN처럼 선형적으로 연산하면서도 장기 의존성을 효과적으로 포착하는 선택적 스캔 메커니즘(Selective Scan Mechanism, S6)을 분석할 수 있다.  
    * **링크:** [https://huggingface.co/state-spaces/mamba-2.8b-hf](https://huggingface.co/state-spaces/mamba-2.8b-hf)  

* **RecurrentGemma (`google/recurrentgemma-2b-it`)**
    * **역할:** 고정된 메모리로 긴 시퀀스 처리  
    * **분석 포인트:** **순환 신경망(RNN)과 트랜스포머의 하이브리드**. 블록 단위로 정보를 처리하고 압축된 '상태(state)'를 다음 블록으로 넘겨주는 방식으로, 어텐션 윈도우의 한계를 극복하는 구조를 분석할 수 있습니다.  
    * **링크:** [https://huggingface.co/google/recurrentgemma-2b-it](https://huggingface.co/google/recurrentgemma-2b-it)  

#### **3) 특수 목적 아키텍처 (Specialized Architectures)**
* **StarCoder2 (`bigcode/starcoder2-15b`)**
    * **역할:** 코드 생성 및 완성  
    * **분석 포인트:** 코드의 특성을 고려한 아키텍처. **그룹화된 쿼리 어텐션(GQA)**, **슬라이딩 윈도우 어텐션(SWA)**과 함께 코드 중간을 채우는 **Fill-in-the-Middle (FIM)** 사전 학습 목표를 분석할 수 있다.  
    * **링크:** [https://huggingface.co/bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b)  

* **LayoutLMv3 (`microsoft/layoutlmv3-base`)**
    * **역할:** 문서 이해 (텍스트 + 이미지 + 레이아웃)  
    * **분석 포인트:** 텍스트와 이미지 임베딩뿐만 아니라, 단어의  **위치 정보(bounding box coordinates)를 함께 입력**받아 문서의 시각적 구조를 이해하는 멀티모달 퓨전(Fusion) 아키텍처가 특징이다.  
    * **링크:** [https://huggingface.co/microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)  

### **II. 비전 모델 (Vision Models)**

#### **1) 이미지 (Image)**
* **ViT (Vision Transformer) (`google/vit-base-patch16-224`)**
    * **역할:** 이미지 분류  
    * **분석 포인트:** 이미지를 16x16 **패치(patch)로 분할**하여 토큰 시퀀스처럼 처리하는 방식. CNN의 합성곱(convolution)을 대체한 트랜스포머 인코더 구조 분석.  
    * **링크:** [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)  

* **Swin Transformer (`microsoft/swin-base-patch4-window7-224`)**
    * **역할:** 고해상도 이미지 처리 및 분할  
    * **분석 포인트:** **계층적(Hierarchical) 구조**와 **이동 윈도우(Shifted-Window) 어텐션**. 전체 이미지가 아닌 작은 윈도우 내에서 어텐션을 계산하여 연산 효율을 높이고, 윈도우를 겹치고 이동시키며 정보를 교환하는 메커니즘이 핵심.  
    * **링크:** [https://huggingface.co/microsoft/swin-base-patch4-window7-224](https://huggingface.co/microsoft/swin-base-patch4-window7-224)  

* **SegFormer (`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`)**
    * **역할:** 이미지 분할 (Semantic Segmentation)  
    * **분석 포인트:** 계층적 트랜스포머 인코더로 다양한 크기의 특징(feature)을 추출하고, 복잡한 디코더 대신 **가벼운 MLP(Multi-Layer Perceptron) 디코더**로 효율성과 성능을 모두 잡은 구조.  
    * **링크:** [https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)  


#### **2) 비디오 (Video)**
* **Timesformer (`facebook/timesformer-base-finetuned-k400`)**
    * **역할:** 비디오 분류  
    * **분석 포인트:** ViT를 비디오로 확장. 각 프레임 내의 공간적(spatial) 어텐션과 프레임 간의 시간적(temporal) 어텐션을 분리하여 계산하는 **'분할 시공간 어텐션(Divided Space-Time Attention)'** 아키텍처를 분석할 수 있다.  
    * **링크:** [https://huggingface.co/facebook/timesformer-base-finetuned-k400](https://huggingface.co/facebook/timesformer-base-finetuned-k400)  


### **III. 기타 데이터 및 작업 특화 모델**

#### **1) 시계열 (Time-Series)**
* **Chronos-T5 (`amazon/chronos-t5-large`)**
    * **역할:** 시계열 예측  
    * **분석 포인트:** 시계열 데이터를 스케일링과 양자화(quantization)를 통해 **이산적인 토큰 시퀀스로 변환**하고, 이를 T5 인코더-디코더 아키텍처로 학습하는 독특한 접근 방식.  
    * **링크:** [https://huggingface.co/amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large)  

#### **2) 그래프 (Graph)**
* **Graphormer (`clefourrier/graphormer-base-pcqm4mv2`)**
    * **역할:** 그래프 구조 데이터 분석  
    * **분석 포인트:** 표준 트랜스포머에 그래프의 구조적 정보를 추가하는 **Centrality Encoding, Spatial Encoding, Edge Encoding** 방식이 핵심. 노드 간의 관계를 어텐션으로 학습하는 GNN(Graph Neural Network)과의 차이점 분석.  
    * **링크:** [https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2](https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2)  

#### **3) 멀티모달 (Multimodal)**
* **IDEFICS2 (`HuggingFaceM4/idefics2-8b`)**
    * **역할:** 이미지-텍스트 혼합 입력 처리 (VLM)  
    * **분석 포인트:** 이미지 시퀀스와 텍스트 시퀀스를 유연하게 인터리빙(interleaving)하여 처리하는 구조. 비전 인코더와 언어 모델을 연결하는 **Perceiver Resampler**와 **Gated Cross-Attention** 레이어를 분석할 수 있다.  
    * **링크:** [https://huggingface.co/HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b)  

#### **4) 오디오 및 음악 (Audio & Music)**
* **MusicGen (`facebook/musicgen-small`)**
    * **역할:** 텍스트 기반 음악 생성  
    * **분석 포인트:** 오디오를 이산적인 토큰으로 변환하는 **EnCodec**과, 이 토큰 시퀀스를 생성하는 **단일 스테이지(single-stage) 자기회귀 트랜스포머**의 결합 구조.  
    * **링크:** [https://huggingface.co/facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small)  

#### **5) 강화학습 (Reinforcement Learning)**
* **Decision Transformer (`edbeeching/decision-transformer-gym-hopper-medium`)**
    * **역할:** 강화학습 문제 해결  
    * **분석 포인트:** 강화학습을 가치 함수나 정책 경사를 학습하는 대신, **'원하는 보상(return), 상태(state), 행동(action)'의 시퀀스를 예측**하는 조건부 시퀀스 모델링 문제로 재구성. GPT와 유사한 자기회귀 아키텍처를 사용.  
    * **링크:** [https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium](https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium)  

## 결론
목록에 있는 모델들 아키텍쳐 분석해서 올려볼까 한다  
아마 이거 정리하다 보면 나도 많이 늘지 않을까  
일단 여기에 있는거 다 한 다음, oss 모델 마냥 최신에 나온 것들 다 들고와서 정리할까 한다.  
아키텍쳐 재미있다.  