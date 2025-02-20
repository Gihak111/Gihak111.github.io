---
layout: single
title:  "서브워드 토큰화: 자연어 처리의 기초 기술"
categories: "Natural_Language_Processing"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 서브워드 토큰화 (Subword Tokenization)

자연어 처리(NLP)에서 텍스트 데이터를 효율적으로 다루기 위해 사용하는 핵심 기술 중 하나가 **서브워드 토큰화(Subword Tokenization)**이다.  
단어를 작은 단위로 분해하여 희귀 단어 문제를 해결하고, 모델의 일반화를 돕는다.

---

## 서브워드 토큰화가 필요한 이유

### 문제의 본질
1. **희귀 단어 문제**: 어휘 크기가 제한적일 경우 OOV(Out-of-Vocabulary) 문제가 발생.  
2. **어휘 크기의 한계**: 단어 단위로 토큰화 시 어휘 크기가 너무 커질 수 있음.  
3. **모델 학습의 비효율성**: 어휘가 클수록 학습과 추론이 느려짐.

### 서브워드 토큰화의 장점
- **희귀 단어 처리**: 새로운 단어를 기존 서브워드 조합으로 표현.  
- **어휘 크기 감소**: 상대적으로 작은 어휘 크기로 모든 단어 표현 가능.  
- **언어 간 유연성**: 다양한 언어에서 공통적으로 사용 가능.

---

## 주요 서브워드 토큰화 기법

### 1. **Byte Pair Encoding (BPE)**

**BPE**는 가장 자주 등장하는 문자 쌍을 병합하는 방식으로 어휘를 점진적으로 확장한다.  
GPT 및 RoBERTa와 같은 대규모 언어 모델에서 주로 사용된다.

#### 특징
- **병합 규칙 기반**: 자주 등장하는 문자 또는 서브워드 쌍을 병합.  
- **희귀 단어 처리**: 희귀 단어도 여러 서브워드 조합으로 표현 가능.  

#### 예제
```plaintext
초기 어휘: {"l", "o", "w", "lo", "low"}
병합 규칙 적용: "l"+"o" → "lo"
최종 어휘: {"lo", "w", "low"}
```

#### Python 예제
```python
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=5000)
tokenizer.train(["data.txt"], trainer)
```

---

### 2. **WordPiece**

**WordPiece**는 각 서브워드의 확률을 계산하고, 이를 최대화하는 방식으로 어휘를 구성한다.  
BERT와 같은 언어 모델에서 사용된다.

#### 특징
- **확률 기반 병합**: 병합할 서브워드 쌍을 확률적으로 선택.  
- **작은 어휘 크기**: 효율적인 어휘 구성.

#### 예제
```plaintext
단어: "unbelievable"
서브워드: ["un", "##bel", "##ievable"]
```

#### Python 예제
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unbelievable")
print(tokens)
# 출력: ['un', '##bel', '##ievable']
```

---

### 3. **SentencePiece**

**SentencePiece**는 언어에 독립적인 서브워드 토큰화를 목표로 하며, 입력 텍스트의 전처리를 생략할 수 있다.  
Google의 T5 및 ALBERT 모델에서 사용된다.

#### 특징
- **언어 독립적**: 공백과 문장을 포함한 모든 텍스트를 하나의 스트림으로 처리.  
- **단순한 설정**: 텍스트 전처리가 필요 없음.

#### 예제
```plaintext
문장: "안녕하세요"
서브워드: ["▁안녕", "하세요"]
```

#### Python 예제
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(input='data.txt', model_prefix='spm', vocab_size=5000)
sp = spm.SentencePieceProcessor(model_file='spm.model')
tokens = sp.encode("안녕하세요", out_type=str)
print(tokens)
# 출력: ['▁안녕', '하세요']
```

---

### 4. **Unigram Language Model**

**Unigram Language Model**은 단어를 독립적으로 평가하며, 가장 적합한 서브워드를 선택한다.  
XLNet과 같은 모델에서 사용된다.

#### 특징
- **확률 기반 선택**: 각 서브워드의 확률에 따라 어휘 구성.  
- **최적화된 어휘 크기**: 비효율적인 서브워드는 자동 제거.

#### 예제
```plaintext
단어: "복잡성"
서브워드: ["복", "잡", "성"]
```

#### Python 예제
```python
from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor(model_file='unigram.model')
tokens = sp.encode("복잡성", out_type=str)
print(tokens)
# 출력: ['복', '잡', '성']
```

---

## 서브워드 토큰화의 한계

1. **의미 해석의 어려움**: 서브워드 단위로 나뉘어 의미가 불분명해질 수 있음.  
2. **모델 복잡성 증가**: 추가적인 병합 규칙과 어휘 관리 필요.  
3. **언어 의존성 문제**: 언어에 따라 최적의 서브워드 크기와 규칙이 다름.

---

## 서브워드 토큰화의 응용

- **대규모 언어 모델**: BERT, GPT 등에서 표준적으로 사용.  
- **다국어 모델**: 서브워드를 활용하여 여러 언어를 통합적으로 처리.  
- **모바일 및 Edge 환경**: 작은 어휘 크기로 메모리 효율성 확보.  

---

## 마무리

서브워드 토큰화는 NLP 모델의 효율성을 극대화하는 핵심 기술이다.  
BPE, WordPiece, SentencePiece 등 다양한 접근 방식이 있으며, 모델과 데이터의 특성에 따라 적절한 방법을 선택해야 한다.  
