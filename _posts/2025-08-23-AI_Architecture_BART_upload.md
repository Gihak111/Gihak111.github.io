---
layout: single
title: "AI 아키텍쳐 2. BART"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## BART

기존의 BERT 모델에는 인코더만 엄청나게 있었다.  
하지만, BART에선 인코더 + 디코더로, BERT와는 완전히 다른 모델이다.  
gpt2에 더 가까운 모델이며, 암튼 좋은 모델이였다.  

## 아키텍쳐 구조

BART(Bidirectional and Auto-Regressive Transformers)는 이름에서 알 수 있듯이 \*\*BERT의 양방향 인코더(Bidirectional Encoder)\*\*와 **GPT의 단방향 디코더(Auto-Regressive Decoder)** 구조를 결합한, 매우 유연한 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델이다.  
기존의 BERT가 문장의 숨은 의미를 파악하고 분류하는 등 문맥을 '이해'하는(Natural Language Understanding, NLU) 데 극도의 강점을 보이고, GPT가 주어진 프롬프트를 바탕으로 다음 단어를 예측하며 문장을 '생성'하는(Natural Language Generation, NLG) 데 특화되었다면, BART는 이 두 가지 상반된 장점을 하나의 모델에 통합하려는 시도에서 탄생했다.  
핵심 아이디어는 멀쩡한 텍스트에 의도적으로 노이즈(noise)를 주어 망가뜨린 후, 모델이 이 손상된 텍스트를 원래의 깨끗한 텍스트로 복원하는 '디노이징(denoising)' 과정을 학습하는 것이다. 수학적으로 이 목표는 손상된 텍스트 $\\tilde{x}$가 주어졌을 때 원본 텍스트 $x$의 조건부 확률 $p(x | \\tilde{x})$를 최대화하는 모델을 학습하는 것과 같다.

BART의 핵심적인 아키텍처는 트랜스포머의 인코더와 디코더를 모두 사용하는 표준적인 인코더-디코더 구조이다. 이는 BERT(인코더-only)나 GPT(디코더-only)와 명확히 구분되는 지점이다.  
기본 모델인 `BART-base`는 6개의 인코더 레이어와 6개의 디코더 레이어를, 더 큰 모델인 `BART-large`는 12개의 인코더와 12개의 디코더 레이어를 사용한다.

BART의 전체적인 데이터 처리 흐름을 수식과 함께 표현하면 다음과 같다.

1.  **입력 손상 (Input Corruption)**: 원본 텍스트 시퀀스 $x = (x\_1, ..., x\_n)$에 임의의 손상 함수 $g$를 적용하여 손상된 시퀀스 $\\tilde{x} = g(x)$를 생성한다.
2.  **인코딩 (Encoding)**: 손상된 시퀀스 $\\tilde{x}$가 **양방향 인코더**를 통과하여, 각 토큰에 대한 문맥 정보를 압축한 은닉 상태(hidden states) 시퀀스 $H\_{enc} = (h\_1, ..., h\_n)$를 출력한다.
    $$H_{enc} = \text{Encoder}(\tilde{x})$$
3.  **디코딩 (Decoding)**: **자기회귀(Auto-Regressive) 디코더**가 인코더의 출력 $H\_{enc}$와 이전에 자신이 생성한 토큰들 $y\_{\<t}$를 입력으로 받아, 다음 토큰 $y\_t$의 확률 분포를 예측한다. 이 과정은 문장의 끝을 나타내는 `[EOS]` 토큰이 생성될 때까지 반복된다.
    $$p(y | \tilde{x}) = \prod_{t=1}^{m} p(y_t | y_{<t}, H_{enc})$$
4.  **출력 (Output)**: 최종적으로 생성된 텍스트 시퀀스 $y = (y\_1, ..., y\_m)$이 모델의 출력이 되며, 학습 시에는 이 $y$가 원본 텍스트 $x$와 같아지도록 모델의 파라미터가 최적화된다.

이러한 구조 덕분에 BART는 번역, 요약과 같이 입력 시퀀스를 다른 형태의 출력 시퀀스로 변환하는 모든 태스크에 자연스럽게 적용될 수 있다.

## 각 레이어별 역할

BART의 아름다움은 손상된 정보를 깊이 있게 이해하는 인코더와, 그 이해를 바탕으로 논리 정연하고 유창한 문장을 생성하는 디코더의 유기적인 협업에 있다.  
손상된 입력 텍스트가 깨끗한 출력 텍스트로 변환되기까지의 다단계 과정을 각 레이어별로 훨씬 더 상세히 알아보자.

### **1. 트랜스포머 인코더 레이어 (Transformer Encoder Layer)**

BART의 인코더는 그 구조와 역할 면에서 BERT의 인코더와 거의 동일하다. 하지만 학습 목표의 관점에서 보면 그 역할이 미묘하게 다르다. BERT 인코더의 목표가 주로 분류(Classification)나 질의응답을 위한 '표현'을 만드는 것이라면, BART 인코더의 목표는 손상된 텍스트 속에서 원본의 '정수(essence)'를 추출하여 디코더가 사용할 완벽한 '설계도'를 만드는 것이다.

  * **구조**: 각 인코더 레이어 $L$은 입력으로 이전 레이어의 은닉 상태 $H^{(l-1)}$를 받는다. 이 입력은 멀티-헤드 어텐션과 피드 포워드 네트워크를 순차적으로 통과하며, 각 하위 모듈은 잔차 연결(residual connection)과 레이어 정규화(Layer Normalization)로 감싸여 있다.
    $$H'_{enc} = \text{LayerNorm}(H^{(l-1)}_{enc} + \text{MultiHeadAttention}(H^{(l-1)}_{enc}))$$
    $$H^{(l)}_{enc} = \text{LayerNorm}(H'_{enc} + \text{FeedForwardNetwork}(H'_{enc}))$$
  * **역할 상세**:
      * **멀티-헤드 셀프 어텐션**: 손상된 텍스트 내에서 토큰 간의 관계를 양방향으로 파악한다. 어텐션 함수는 쿼리(Q), 키(K), 값(V)을 사용하여 각 토큰의 표현을 문장 내 다른 모든 토큰의 정보로 가중합하여 업데이트한다.
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
        여러 개의 어텐션 헤드는 이러한 관계를 통사적, 의미론적 등 다양한 각도에서 동시에 분석하여 종합적인 이해를 돕는다.
      * **피드 포워드 신경망**: 셀프 어텐션을 통해 얻은 문맥 정보를 비선형적으로 변환하여 더 풍부하고 복잡한 특징을 추출한다. GELU 활성화 함수를 사용하는 두 개의 선형 변환으로 구성된다.
        $$\text{FFN}(h) = \text{GELU}(hW_1 + b_1)W_2 + b_2$$
        이는 단순히 정보를 섞는 것을 넘어, 각 토큰의 표현을 고차원 공간에서 재조합하여 디코딩에 필요한 추상적인 정보를 만들어내는 과정이다.

결론적으로, 인코더는 손상된 입력에도 불구하고 원본 텍스트를 복원하는 데 필요한 모든 정보를 담고 있는, 강력하고 압축된 표현 $H\_{enc}$를 만들어 디코더에 전달하는 책임을 진다.

### **2. 트랜스포머 디코더 레이어 (Transformer Decoder Layer)**

BART의 디코더는 GPT와 유사한 자기회귀(Auto-Regressive) 방식을 사용하지만, 한 가지 결정적인 차이점이 있다. 바로 인코더의 출력을 참조하는 **교차 어텐션(Cross-Attention)** 메커니즘이 추가되었다는 점이다. 이 교차 어텐션이 인코더의 '이해'와 디코더의 '생성'을 연결하는 핵심적인 다리 역할을 한다.

디코더의 각 레이어는 세 개의 하위 레이어로 구성된다.

#### **2-1. 마스크된 멀티-헤드 셀프 어텐션 (Masked Multi-Head Self-Attention)**

  * **역할**: 디코더가 텍스트를 생성할 때, 이미 생성된 앞부분의 단어들만 참조하여 다음 단어를 예측하도록 한다. 이는 생성 과정의 일관성과 문법적 유창함을 담당한다.
  * **수학적 원리**: 일반 셀프 어텐션과 동일하게 작동하지만, 어텐션 점수 행렬에 하삼각 행렬 형태의 마스크 $M$을 더하여 미래 시점의 정보가 현재 시점의 예측에 영향을 주지 못하도록 한다. 마스크 행렬의 값은 참조해서는 안 되는 위치에 $-\\infty$를, 참조 가능한 위치에 $0$을 가진다.
    $$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
    소프트맥스 함수를 통과하면 $-\\infty$는 $0$이 되어 해당 토큰에 대한 어텐션 가중치가 $0$이 된다. 이는 디코더가 왼쪽에서 오른쪽으로 순차적으로 문장을 생성하는 자기회귀
    ($$p(y\_t | y\_{\<t}, ...)$)속성을 수학적으로 보장하는 필수 장치다.

#### **2.2. 인코더-디코더 교차 어텐션 (Encoder-Decoder Cross-Attention)**

  * **역할**: 디코더가 단어를 생성하는 매 시점마다, 인코더가 만들어낸 '손상된 텍스트 전체의 문맥'을 집중적으로 참고하게 한다. 이것이 BART의 핵심적인 연결고리이자, 단순한 언어 모델을 넘어선 조건부 생성(Conditional Generation)을 가능하게 하는 원동력이다.
  * **작동 원리 및 증명**:
    이 어텐션 메커니즘에서는 정보의 흐름이 명확히 구분된다.
      * **쿼리(Query, $Q\_{dec}$)**: 디코더의 이전 하위 레이어(마스크된 셀프 어텐션)의 출력에서 온다. 즉, 현재까지 생성된 내용을 바탕으로 "다음 단어를 위해 어떤 정보가 필요한가?" 라는 질문을 던지는 주체다.
      * **키(Key, $K\_{enc}$) & 값(Value, $V\_{enc}$)**: **인코더의 최종 출력 은닉 상태 $H\_{enc}$ 전체**에서 온다. 이 값들은 한 번 계산된 후 모든 디코더 레이어와 모든 디코딩 타임 스텝에서 재사용된다. 이는 "손상된 원본 문장에 대한 나의 완전한 이해"라는 정적인 참조 정보를 제공한다.
        $$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$
        **증명 (필요성)**: 디코더는 매 스텝마다 자신의 현재 상태($Q\_{dec}$)를 가지고 인코더의 전체 출력($K\_{enc}$)을 훑어보며 어떤 부분에 집중할지를 결정한다. 이 과정이 없다면 디코더는 인코더의 입력을 무시하고 그저 학습 데이터에서 배운 가장 일반적인 문장을 생성할 뿐, 입력에 기반한 정교한 복원은 불가능하다. 이 어텐션 모듈이 있기에 조건부 확률 $p(y | \\tilde{x})$의 '조건($\\tilde{x}$)'이 디코딩 과정에 실질적으로 반영될 수 있다.

#### **2-3. 피드 포워드 신경망 (Position-wise Feed-Forward Network)**

  * **역할**: 교차 어텐션을 통해 결합된 정보를 비선형 변환하여 최종적인 토큰 표현을 만든다.
  * **설명**: 인코더의 FFN과 동일한 구조와 역할을 수행한다. 마스크된 셀프 어텐션을 통해 얻은 '내부적 문맥'과 교차 어텐션을 통해 얻은 '외부적(인코더) 문맥'을 종합하여, 다음 토큰을 예측하기 직전의 최종적인 의미 표현으로 가공하는 '정보 처리 장치'라고 할 수 있다.

#### 코드 예시 (`transformers` 라이브러리)

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# 모델과 토크나이저 로드 (요약 태스크에 파인튜닝된 모델)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# 요약할 긴 텍스트
ARTICLE_TO_SUMMARIZE = (
    "BART is a denoising autoencoder for pretraining sequence-to-sequence models. "
    "It is trained by corrupting text with an arbitrary noising function and learning a model to reconstruct the original text. "
    "BART uses a standard Transformer-based neural machine translation architecture which consists of a bidirectional encoder and an autoregressive decoder. "
    "This model is particularly effective for text generation tasks like summarization and translation but also works well for comprehension tasks such as question answering and classification."
)

# 입력 텍스트를 토큰화
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt', truncation=True)

# 요약 생성
# num_beams: 더 나은 품질의 문장을 생성하기 위한 빔 서치(beam search) 크기
# min_length, max_length: 생성할 요약문의 최소/최대 길이
summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=10, max_length=50, early_stopping=True)

# 생성된 ID를 텍스트로 디코딩
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(summary)
# BART is a denoising autoencoder for pretraining sequence-to-sequence models. It uses a standard Transformer-based neural machine translation architecture. This model is particularly effective for text generation tasks like summarization and translation.
```

## 각 구조의 의의와 BART의 학습 방식

BART 아키텍처의 진정한 힘은 **디노이징 사전 학습(Denoising Pre-training)** 방식에서 나온다. BERT가 일부 토큰을 맞추는 데 집중하고 GPT가 다음 단어를 예측하는 데 집중했다면, BART는 '손상된 전체 문서를 원본으로 완벽하게 복원'하는, 훨씬 더 복잡하고 일반적인 과제를 학습한다. 이는 모델이 언어의 표면적인 패턴뿐만 아니라, 더 깊은 구조적, 의미적 속성을 파악하도록 강제한다.

### **디노이징 오토인코더 (Denoising Autoencoder)**

  * **목표**: 모델의 파라미터 $\\theta$를 최적화하여, 손상된 입력 $\\tilde{x}$가 주어졌을 때 원본 $x$의 음의 로그 우도(negative log-likelihood)를 최소화하는 것이다. 이는 원본 텍스트의 복원 확률을 최대화하는 것과 같다.
    $$\mathcal{L}(\theta) = -\sum_{i=1}^{|D|} \log p(x^{(i)} | \tilde{x}^{(i)}; \theta)$$
    여기서 $D$는 전체 학습 데이터셋을 의미한다. 각 시퀀스에 대한 손실은 각 토큰의 예측에 대한 교차 엔트로피 손실의 합으로 계산된다.
    $$\log p(x | \tilde{x}) = \sum_{t=1}^{m} \log p(x_t | x_{<t}, \tilde{x})$$

  * **주요 손상 기법(Corruption Strategies)과 그 효과**:

      * **토큰 마스킹 (Token Masking)**: `the cat sat on the mat` → `the [M] sat on the [M]`. BERT와 유사하며, 모델이 주변 단어를 통해 빈칸을 채우는 능력을 기르게 한다.
      * **토큰 삭제 (Token Deletion)**: `the cat sat on the mat` → `the sat on mat`. 모델은 단순히 빈칸을 채우는 것을 넘어, 단어가 '사라진 위치'까지 파악해야 하므로 더 어렵고, 문장 구조에 대한 이해를 높인다.
      * **텍스트 인필링 (Text Infilling)**: `the cat sat on the mat` → `the cat [M] mat`. 여러 단어로 구성된 구간(span)을 단 하나의 마스크 토큰으로 대체한다. 모델은 사라진 구간의 '길이'와 '내용'을 모두 예측해야 하므로, 장기적인 의존성과 구(phrase) 단위의 의미를 학습하는 데 매우 효과적이다.
      * **문장 순열 섞기 (Sentence Permutation)**: `[문장1] [문장2] [문장3]` → `[문장2] [문장1] [문장3]`. 모델이 문장 간의 논리적, 시간적 관계를 파악하고 글의 전체적인 담화 구조(discourse structure)를 이해하도록 훈련시킨다.
      * **문서 회전 (Document Rotation)**: `A B C D` → `C D A B`. 문서의 시작점이 임의로 변경되었을 때 원본을 복원하게 함으로써, 문서의 전체적인 주제와 구조적 일관성을 파악하는 능력을 기른다.

  * **증명 (학습 효과)**:
    이러한 다양한 손상 기법들은 모델이 특정 유형의 노이즈에만 과적합되는 것을 방지하고, 다방면의 언어적 능력을 균형 있게 발전시킨다. 이 포괄적인 학습 방식 덕분에 BART는 입력의 문맥을 이해하는 능력(인코더)과 유창하고 적절한 문장을 생성하는 능력(디코더)을 동시에 극대화할 수 있다.

## 결론

이러한 사전 학습을 통해 BART는 특정 Task에 국한되지 않은, 매우 유연하고 강력한 언어 모델로 거듭난다. 인코더는 BERT처럼 문장 분류나 개체명 인식 같은 이해(NLU) 태스크에 강점을 보이고, 디코더는 GPT처럼 요약, 번역, 대화 생성 같은 생성(NLG) 태스크에 뛰어난 성능을 발휘한다.  
특히 요약(Summarization)과 같이 긴 문서를 읽고(인코더) 짧은 문장으로 생성(디코더)해야 하는 태스크나, 기계 번역(Machine Translation)과 같이 한 언어를 이해(인코더)하고 다른 언어로 생성(디코더)해야 하는 태스크에서 자연스럽게 최고의 성능을 보여준다.  
즉, 하나의 사전 학습 모델로 사실상 대부분의 NLP 문제를 효과적으로 해결할 수 있는 범용적인 프레임워크를 제시한 것이다. 이것이 바로 BART가 BERT 이후 시퀀스-투-시퀀스 모델의 새로운 표준 중 하나로 자리 잡은 이유이다.  
마지막으로, 성능 좋은 모델 링크나 몇개 남기고 마치겠다.  
[https://huggingface.co/facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)  
[https://huggingface.co/facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)