---
layout: single
title:  "AI 아키텍쳐 1. BERT"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## BERT
분류 AI로 이미 만들어 진지 꽤 된 모델이다  
이미 많은 연구가 되어있기에, 성능이 보장되어 있는 모델들이 많이 있다  
이미, 당신들도 허깅페이스에서 사용해 봤을 수 도 있다  
암튼, 이 아름다운 모델 아키텍쳐나 한번 알아보자.  

## 아키텍쳐 구조
BERT(Bidirectional Encoder Representations from Transformers)는 이름에서 알 수 있듯이 **트랜스포머(Transformer)의 인코더(Encoder) 구조**를 기반으로 만들어진 언어 모델 이다.  
기존의 언어 모델들이 텍스트를 왼쪽에서 오른쪽으로, 혹은 오른쪽에서 왼쪽으로 단방향으로만 처리했던 것과 달리, BERT는 **양방향(Bidirectional)**으로 문맥을 동시에 파악하는 방법을 사용한다.  
이 덕분에 문장의 전체적인 의미를 훨씬 더 깊이 있게 이해할 수 있게 되었다.  

BERT의 핵심적인 아키텍처는 트랜스포머의 인코더를 여러 겹으로 쌓아 올린 구조이다.  
기본 모델인 `BERT-base`는 12개의 인코더 레이어를, 더 큰 모델인 `BERT-large`는 24개의 인코더 레이어를 사용한다.  

BERT의 전체적인 데이터 처리 흐름은 다음과 같다.  

1.  **입력 표현 (Input Representation)**: 문장(들)을 모델이 이해할 수 있는 숫자 형태의 벡터로 변환한다.  
2.  **임베딩 (Embedding)**: 변환된 숫자들을 의미를 함축한 벡터로 만든다. BERT는 세 가지 종류의 임베딩을 합산하여 사용한다.  
3.  **트랜스포머 인코더 스택 (Transformer Encoder Stack)**: 임베딩된 벡터가 여러 개의 인코더 레이어를 순차적으로 통과한다. 각 인코더는 '셀프 어텐션(Self-Attention)'과 '피드 포워드 신경망(Feed-Forward Neural Network)'을 통해 입력된 문장의 문맥적 의미를 더욱 풍부하게 만든다.  
4.  **출력 (Output)**: 마지막 인코더 레이어를 통과한 벡터들은 각 토큰(단어)에 대한 깊은 문맥적 정보를 담고 있으며, 이 벡터들을 다양한 자연어 처리 문제(Task)에 활용할 수 있다.  

이제 각 구성 요소가 어떤 역할을 하는지 더 자세히 살펴보자.

## 각 레이어별 역할

BERT의 아름다움은 각 레이어가 유기적으로 연결되어 문장의 의미를 점진적으로 정교하게 다듬어 나간다는 점에 있다.  
입력 텍스트가 최종 출력 벡터로 변환되기까지의 다단계 과정을 각 레이어별로 상세히 알아보자.  

### **1. 임베딩 레이어 (Embedding Layer)**

모델이 텍스트를 직접 처리할 수는 없다.  
따라서 첫 단계는 텍스트를 숫자 벡터로 변환하는 것이다.  
BERT의 임베딩 레이어는 단순한 단어 사전을 넘어, 문장의 구조와 의미를 파악하기 위한 세 가지 핵심 정보를 벡터에 담아낸다.  

* **토큰 임베딩 (Token Embeddings)**:  
    * **역할**: 각 단어(또는 서브워드)를 고유한 벡터로 변환한다.  
    * **설명**: BERT는 'WordPiece'라는 토크나이저(Tokenizer)를 사용하여 문장을 더 작은 단위인 토큰으로 분리한다. 예를 들어, "playing"이라는 단어는 "play"와 "##ing"로 나뉠 수 있다. 이렇게 하면 사전에 없는 단어(Out-of-Vocabulary, OOV) 문제에 효과적으로 대응할 수 있다. 각 토큰은 사전(Vocabulary)에 있는 고유 ID에 매핑되고, 이 ID에 해당하는 학습된 벡터가 바로 토큰 임베딩이다. `bert-base-uncased` 모델의 경우 약 3만 개의 토큰 사전을 가지며, 각 토큰은 768차원의 벡터($d_{model}=768$)로 표현된다.  

* **세그먼트 임베딩 (Segment Embeddings)**:
    * **역할**: 두 개의 문장을 입력으로 받을 때, 각 토큰이 첫 번째 문장에 속하는지 두 번째 문장에 속하는지를 구분해 준다.  
    * **설명**: BERT는 질의응답(Question Answering)이나 문장 간 관계 추론(Natural Language Inference)과 같이 두 문장을 함께 이해해야 하는 Task를 처리할 수 있도록 설계되었다. 이를 위해 입력의 시작 부분에 `[CLS]` (Classification) 토큰을, 문장과 문장 사이 그리고 문장의 끝에 `[SEP]` (Separator) 토큰을 추가한다. 세그먼트 임베딩은 각 토큰이 첫 번째 문장(Sentence A)에 속하면 $E_A$ 벡터를, 두 번째 문장(Sentence B)에 속하면 $E_B$ 벡터를 더해주는 방식으로 문장을 구분한다.  

* **위치 임베딩 (Position Embeddings)**:  
    * **역할**: 문장 내에서 각 토큰의 위치(순서) 정보를 알려준다.  
    * **증명 (필요성)**: 트랜스포머 구조의 핵심인 셀프 어텐션 메커니즘은 단어들의 순서를 고려하지 않는다. 입력 토큰들의 순서를 바꾸어 넣어도 연산 과정(행렬 곱)의 특성상 동일한 결과가 나온다. "나는 너를 사랑해"와 "너는 나를 사랑해"는 구성 단어는 같지만 의미는 완전히 다르다. 이처럼 순서 정보가 매우 중요하기 때문에, 각 토큰의 위치(0번, 1번, 2번, ...)에 해당하는 고유한 벡터를 학습시켜 더해준다. 이를 통해 모델은 단어의 상대적, 절대적 위치를 파악할 수 있게 된다.  

최종적으로 이 **세 가지 임베딩 벡터가 모두 합산(element-wise sum)**되어 해당 토큰의 초기 입력 벡터가 완성된다.  
$$ \text{Input Embedding} = \text{TokenEmbedding} + \text{SegmentEmbedding} + \text{PositionEmbedding} $$
이 합산이 가능한 이유는 세 임베딩이 모두 동일한 차원($d_{model}=768$)을 갖기 때문이며, 역전파 과정에서 각 임베딩 테이블은 자신에게 해당하는 그래디언트(gradient)를 받아 독립적으로 학습된다.  
이는 마치 하나의 벡터 공간 안에 '의미', '문단 소속', '순서'라는 세 가지 축을 동시에 부여하는 것과 같다.  

#### 코드 예시 (`transformers` 라이브러리)

```python
from transformers import BertTokenizer, BertModel
import torch

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 입력 문장
sentence_a = "The quick brown fox"
sentence_b = "jumps over the lazy dog."

# 토크나이저가 알아서 [CLS], [SEP] 추가 및 ID로 변환
# token_type_ids: 세그먼트 구분을 위함 (0: 문장 A, 1: 문장 B)
encoded_input = tokenizer(sentence_a, sentence_b, return_tensors='pt')
print(encoded_input)
# {'input_ids': tensor([[ 101, 1996, 4248, 2829, 4419,  102, 14523, 2058, 1996, 13971, 3899,  102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

# 임베딩 레이어의 출력 확인
# model.embeddings는 세 가지 임베딩을 모두 합산한 결과를 출력합니다.
with torch.no_grad():
    embedding_output = model.embeddings(
        input_ids=encoded_input['input_ids'],
        token_type_ids=encoded_input['token_type_ids']
    )

print(embedding_output.shape)
# torch.Size([1, 12, 768])  # (batch_size, sequence_length, embedding_dim)
```

### **2. 트랜스포머 인코더 레이어 (Transformer Encoder Layer)**

임베딩 레이어를 통과한 벡터 시퀀스($X \\in \\mathbb{R}^{n \\times d\_{model}}$, n은 시퀀스 길이)는 이제 BERT의 심장부인 트랜스포머 인코더 스택으로 들어간다.  
각 인코더 레이어는 동일한 두 개의 하위 레이어(Sub-layer)로 구성되어 있으며, 입력 벡터를 받아 더 정교한 문맥 정보가 담긴 벡터로 변환하는 역할을 반복한다. (`BERT-base`는 이 과정을 12번 반복한다.)  

#### **2-1. 멀티-헤드 셀프 어텐션 (Multi-Head Self-Attention)**

  * **역할**: 문장 내의 단어들이 서로에게 얼마나 '주의(Attention)'를 기울여야 하는지를 계산하여, 각 단어의 의미를 주변 단어와의 관계 속에서 재정의한다.  

  * **작동 원리 및 증명**:  

    1.  **Q, K, V 벡터 생성**: 입력 시퀀스 $X$의 각 토큰 벡터로부터 세 개의 다른 벡터, 즉 **쿼리(Query, Q)**, **키(Key, K)**, \*\*값(Value, V)\*\*을 생성한다. 이는 학습 가능한 가중치 행렬 $W^Q, W^K, W^V$를 각각 곱하여 이루어진다.
        $$ Q = XW^Q, \quad K = XW^K, \quad V = XW^V $$

          * **Query**: "내가 다른 단어들에게 물어볼 질문" (정보를 요청하는 주체)  
          * **Key**: "다른 단어들이 자신을 설명하는 푯말" (Query와 비교될 대상)   
          * **Value**: "다른 단어들이 실제로 가지고 있는 의미/정보" (최종적으로 조합될 내용)  

    2.  **어텐션 점수(Attention Score) 계산**: 특정 단어(쿼리)가 문장 내 다른 모든 단어(키)와 얼마나 연관이 있는지를 계산한다. 이는 Q와 K 벡터의 내적(Dot Product)을 통해 이루어잔다. 내적은 두 벡터의 유사도(유사한 방향을 가리킬수록 값이 커짐)를 측정하는 효과적인 방법이기 때문이다.  
        $$ \text{Scores} = QK^T $$

    3.  **스케일링(Scaling)**: 내적 값은 벡터의 차원($d\_k$)이 커질수록 그 값이 너무 커져서 소프트맥스 함수의 기울기를 0에 가깝게 만드는 문제(gradient vanishing)가 발생할 수 있다. 이를 방지하기 위해 키 벡터 차원의 제곱근($\\sqrt{d\_k}$)으로 나누어 분포를 안정화시킨다.
        $$ \text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}} $$

    4.  **소프트맥스(Softmax) 적용**: 스케일링된 점수에 소프트맥스 함수를 적용하여 합이 1이 되는 확률 분포, 즉 \*\*어텐션 가중치(Attention Weights)\*\*를 만든다. 이 가중치는 특정 단어가 다른 단어들의 정보를 얼마나 '참조'할지에 대한 비율을 나타낸다.

    5.  **최종 출력 계산**: 계산된 어텐션 가중치를 각 단어의 V(Value) 벡터에 곱한 후 모두 더한다. 가중치가 높은 단어의 Value 값이 최종 결과에 더 큰 영향을 미치게 된다.
        $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

    <!-- end list -->

      * **멀티-헤드(Multi-Head)의 의의**:
        이러한 어텐션 과정을 한 번만 수행하는 대신, 여러 개의 '헤드'를 만들어 병렬적으로 수행한다. `BERT-base`는 12개의 헤드를 사용한다. 각 헤드는 독립적인 $W^Q, W^K, W^V$ 가중치를 가지므로, 서로 다른 관점의 관계를 학습할 수 있다.
        $$ \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V) $$
        예를 들어, 어떤 헤드는 동사-목적어 관계를, 다른 헤드는 수식 관계를, 또 다른 헤드는 인접한 단어의 관계를 집중적으로 학습할 수 있다. 이는 마치 한 사람이 여러 전문가의 의견을 종합하여 결론을 내리는 것과 같아서 훨씬 풍부하고 다각적인 문맥 정보를 포착하게 된다.
        각 헤드에서 나온 결과 벡터들을 모두 연결(concatenate)한 후, 마지막으로 출력 가중치 행렬 $W^O$를 곱하여 최종 출력을 만든다.  
        $$ \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$

#### **2-2. 피드 포워드 신경망 (Position-wise Feed-Forward Network)**

  * **역할**: 셀프 어텐션을 통해 얻은 문맥 정보를 더욱 복잡하게 조합하고 비선형 변환(non-linear transformation)을 추가하여 표현력을 높인다.

  * **작동 원리 및 증명**:
    셀프 어텐션 레이어는 주로 토큰 간의 정보를 '섞어주는' 역할을 한다. 즉, 다른 토큰들의 Value 벡터를 선형 결합(linear combination)하는 과정이다. 여기에 비선형성(non-linearity)을 추가하고 각 토큰의 표현을 더 깊이 있게 처리하기 위해 피드 포워드 신경망이 필요하다.
    이 신경망은 각 토큰의 위치마다(Position-wise) 독립적으로, 하지만 동일한 가중치($W\_1, b\_1, W\_2, b\_2$)를 사용하여 적용된다. 구조는 다음과 같다.

    1.  선형 변환 (차원 증가): $d\_{model} \\rightarrow d\_{ff}$ (BERT-base: $768 \\rightarrow 3072$)  
    2.  GELU 활성화 함수: 비선형성 추가  
    3.  선형 변환 (차원 복원): $d\_{ff} \\rightarrow d\_{model}$ (BERT-base: $3072 \\rightarrow 768$)  

    $$ \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2 $$
    이 FFN 레이어는 셀프 어텐션으로 재조합된 문맥 정보를 바탕으로 각 토큰이 가진 특징을 추출하고 변환하는 역할을 한다. 차원을 일시적으로 크게 확장했다가 다시 줄이는 구조는 모델이 더 복잡하고 추상적인 특징을 학습할 수 있는 '작업 공간'을 제공하는 것과 같다. 이 과정이 없다면 BERT는 단순히 토큰 정보를 섞는 얕은 모델에 그쳤을 것이다.  

#### **2-3. 잔차 연결 및 레이어 정규화 (Add & Norm)**

  * **역할**: 깊은 신경망의 학습을 안정화하고 최적화한다.  

  * **작동 원리 및 증명**:
    각 인코더 레이어의 두 하위 레이어(셀프 어텐션, 피드 포워드 신경망)는 각각 \*\*잔차 연결(Residual Connection)\*\*과 \*\*레이어 정규화(Layer Normalization)\*\*를 거친다.  

    1.  **잔차 연결 (Residual Connection)**:
        하위 레이어의 입력($x$)을 출력($\\text{Sublayer}(x)$)에 그대로 더해주는 기법이다.  
        $$ \text{Output} = x + \text{Sublayer}(x) $$
        **증명 (필요성)**: BERT처럼 12개, 24개의 레이어를 쌓으면 학습 과정에서 기울기가 소실(vanishing gradient)되거나 폭발(exploding gradient)하는 문제가 발생하기 쉽다. 잔차 연결은 역전파 시 그래디언트가 하위 레이어를 건너뛰고 직접 상위 레이어로 전달될 수 있는 '지름길(shortcut)'을 만들어 준다. 이는 정보의 손실을 막고 깊은 네트워크의 학습을 원활하게 한다.  

    2.  **레이어 정규화 (Layer Normalization)**:  
        잔차 연결 후, 레이어의 출력값 분포를 각 토큰별로 정규화하여 학습을 안정시키고 속도를 높인다. 특정 샘플의 모든 특성(feature)에 대해 평균($\\mu$)과 분산($\\sigma^2$)을 계산하고 이를 이용해 정규화를 수행한다. 학습 가능한 파라미터 $\\gamma$와 $\\beta$를 통해 스케일과 이동을 조절한다.
        $$ \text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$
        **증명 (필요성)**: NLP에서는 문장의 길이가 가변적이어서 배치(batch) 단위로 특성의 통계를 계산하는 배치 정규화(Batch Normalization)는 불안정할 수 있다. 레이어 정규화는 각 시퀀스 데이터 내에서 독립적으로 정규화를 수행하므로 시퀀스 길이나 배치 크기에 영향을 받지 않아 트랜스포머 모델에 더 적합하다.  

## 각 구조의 의의와 BERT의 학습 방식

BERT의 아키텍처는 그 자체로도 혁신적이지만, 진정한 힘은 **사전 학습(Pre-training)** 방식에서 나온다. BERT는 대규모의 텍스트 데이터(예: 위키피디아, 책 코퍼스)를 사용하여 두 가지 비지도 학습(unsupervised learning) 과제를 수행하며 언어 자체에 대한 깊은 이해를 얻는다.  

### **1. 마스크 언어 모델 (Masked Language Model, MLM)**  

  * **목표**: 문장의 빈칸에 들어갈 단어를 맞추게 함으로써, 단어의 양방향 문맥을 이해하는 능력을 학습시킨다.  
  * **방식**: 입력 문장의 토큰 중 15%를 무작위로 선택하여 다음과 같이 처리한다.  
      * 80%의 확률로 `[MASK]` 토큰으로 바꾼다.  
      * 10%의 확률로 어휘 사전의 임의의 단어로 바꾼다.  
      * 10%의 확률로 원래 단어를 그대로 둔다.  
  * **증명 (학습 효과)**:  
    이 전략은 모델이 단순히 `[MASK]` 토큰의 위치만 외우는 것을 방지하고, 모든 입력 토큰에 대해 신뢰할 수 있는 문맥적 표현을 학습하도록 강제한다. 모델은 특정 토큰이 진짜인지, 바뀌었는지 알 수 없으므로 모든 토큰의 표현을 주변 문맥을 기반으로 재구성하는 법을 배우게 된다. 손실 함수로는 마스킹된 위치의 토큰 예측에 대해서만 Cross-Entropy Loss를 계산한다.
    $$ \mathcal{L}_{MLM} = -\sum_{i \in M} \log p(t_i | \hat{T}) $$
    여기서 $M$은 마스킹된 토큰의 인덱스 집합, $t\_i$는 실제 토큰, $\\hat{T}$는 손상된 입력 시퀀스이다.  

### **2. 다음 문장 예측 (Next Sentence Prediction, NSP)**

  * **목표**: 두 문장이 실제로 이어지는 문장인지 아닌지를 맞추게 함으로써, 문장 간의 관계를 이해하는 능력을 학습시킨다.  
  * **방식**: 학습 데이터에서 두 문장(A, B)을 가져올 때, 50%의 확률로 실제 이어지는 문장 쌍을, 나머지 50%는 전혀 관련 없는 문장을 B로 가져와 쌍을 만든다. 그리고 모델은 입력의 맨 앞에 있는 `[CLS]` 토큰의 최종 출력 벡터($C \\in \\mathbb{R}^{d\_{model}}$)를 이용하여 이 두 문장이 이어지는 관계인지(`IsNext`) 아닌지(`NotNext`)를 이진 분류(Binary Classification)하도록 학습된다.  
    $$ p = \text{softmax}(CW^T_{cls}) $$
  * **증명 (학습 효과)**:  
    NSP 과제를 통해 BERT는 단순한 문장 내 문법 구조를 넘어, 문단 수준의 논리적 흐름이나 주제의 일관성과 같은 더 큰 단위의 언어 구조를 이해하게 된다. 이는 질의응답이나 자연어 추론과 같이 문장 간의 관계 파악이 중요한 다운스트림 태스크에서 높은 성능을 발휘하는 기반이 된다.  

## 결론

이러한 사전 학습을 통해 BERT는 특정 Task에 국한되지 않은, 범용적인 언어 이해 능력을 갖추게 된다.  
그리고 이렇게 잘 학습된 BERT 모델을 가져다가 우리가 풀고 싶은 특정 문제(예: 감성 분석, 개체명 인식 등)에 맞게 약간의 추가 학습(미세 조정, Fine-tuning)만 시키면 매우 높은 성능을 얻을 수 있다.  
이것이 바로 BERT가 NLP 분야에 혁명을 가져온 이유이다.  
마지막으로, 성능 좋은 모델 링크나 몇개 남기고 마치겠다.  
[https://huggingface.co/deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)  
https://huggingface.co/google-bert/bert-base-uncased[](https://huggingface.co/google-bert/bert-base-uncased)  
