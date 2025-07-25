---
layout: single
title: "[논문 리뷰] Dropout Reduces UnderfittingLanguage"
categories: "AI"
tag: "review"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 논문 리뷰

만일, 너가 딥러닝 파인튜닝 이런걸 꾸준히 해 왔더라면, 서브워드 토크나이징, 드롭아웃 같은 단어들을 많이 들어보았을 것이다.  
오늘 알아볼 논문은, 이 중 하나인 드롭아웃에 관한 논문이다.  


## 논문 링크
- 논문: [https://aclanthology.org/2020.emnlp-demos.6/](https://aclanthology.org/2020.emnlp-demos.6/)  

## 트랜스포머 라이브러리의 핵심 아이디어
**Transformers** 라이브러리는 NLP 작업을 간소화하고, 대규모 모델을 효율적으로 다룰 수 있게 설계되었다.  
논문에서 소개된 주요 아이디어는 다음과 같다:

1. **모듈화된 구조**  
   트랜스포머 모델은 세 가지 구성 요소로 나뉜다:  
   - **토크나이저(Tokenizer)**: 텍스트를 인덱스 형태로 변환.  
   - **트랜스포머(Transformer)**: 문맥 정보를 가진 임베딩 생성.  
   - **헤드(Head)**: 작업별 예측을 수행(예: 분류, 생성, 번역 등).  
   이런 구조 덕분에 모델을 쉽게 바꾸거나 확장할 수 있다. Auto 클래스를 사용하면 모델 간 전환도 두 줄 코드로 끝난다.  

2. **사전 학습과 파인튜닝**  
   트랜스포머는 대규모 텍스트 코퍼스에서 사전 학습된 모델을 제공하며, 이를 특정 작업에 맞게 파인튜닝할 수 있다.  
   예를 들어, BERT는 언어 모델링과 문장 예측 헤드로 사전 학습되고, 이를 GLUE나 SQuAD 같은 작업에 맞게 조정 가능하다.  

3. **커뮤니티 모델 허브**  
   라이브러리는 2,097개 이상의 사전 학습 및 파인튜닝 모델을 제공하는 모델 허브를 운영한다.  
   사용자는 간단한 명령어로 모델을 다운로드하고, 바로 파인튜닝이나 추론에 사용할 수 있다.  
   모델 카드를 통해 학습 데이터, 성능, 편향 정보 등을 투명하게 확인 가능하다.  

4. **배포 효율성**  
   트랜스포머는 PyTorch와 TensorFlow를 모두 지원하며, ONNX나 CoreML 같은 포맷으로 모델을 변환해 배포 효율성을 높였다.  
   특히 ONNX를 사용하면 추론 속도가 최대 4배 빨라진다.  

## 트랜스포머의 장점
- **확장성**: 연구자와 개발자가 새로운 모델을 쉽게 추가하거나 수정 가능.  
- **호환성**: PyTorch, TensorFlow 간 모델 변환과 다양한 배포 환경 지원.  
- **커뮤니티 주도**: 400명 이상의 외부 기여자와 함께 지속적으로 발전.  
- **사용 편의성**: 두 줄 코드로 사전 학습 모델을 로드하고 바로 사용 가능.  

## 실제로 어땠을까?
논문에서는 트랜스포머 라이브러리의 실제 사례를 몇 가지 소개한다:  
- **SciBERT**: AllenAI가 생의학 텍스트 추출을 위해 PubMed 데이터로 학습한 모델. 모델 허브를 통해 쉽게 배포되었다.  
- **Jiant**: NYU 연구진이 다양한 트랜스포머 모델을 비교하고 파인튜닝하는 데 사용.  
- **DistilBART**: Plot.ly가 문서 요약을 위해 빠르고 간단하게 배포한 사례.  

이런 사례들은 트랜스포머가 연구와 산업 모두에서 얼마나 유연하고 강력한지를 보여준다.  

## 오픈소스 링크
- GitHub: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)  
- 공식 문서: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)  

## 간단한 Transformers 구현 예제
다음은 PyTorch를 사용해 FlauBERT 모델을 로드하는 간단한 예제 코드다.  
초보자도 이해할 수 있게 간단히 구성했다.  

```python
from transformers import AutoTokenizer, AutoModel

# FlauBERT 모델과 토크나이저 로드
tknzr = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
model = AutoModel.from_pretrained("flaubert/flaubert_base_uncased")

# 텍스트 입력 예제
text = "Bonjour, comment vas-tu?"
inputs = tknzr(text, return_tensors="pt")

# 모델 추론
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

위 코드는 프랑스어 사전 학습 모델인 FlauBERT를 로드하고, 간단한 텍스트를 처리하는 예제다.  
Hugging Face의 Auto 클래스를 사용해 모델과 토크나이저를 쉽게 불러올 수 있다.  
더 자세한 코드는 다음 링크에서 확인할 수 있다: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers).  

## 왜 Transformers가 중요한가?
Transformers 라이브러리는 대규모 언어 모델을 연구자와 개발자가 쉽게 접근하고 활용할 수 있게 해준다.  
특히 자원이 제한된 환경에서도 사전 학습 모델을 빠르게 파인튜닝하고 배포할 수 있어, NLP 분야의 진입 장벽을 낮췄다.  

## 결론
Transformers는 NLP에서 트랜스포머 아키텍처와 사전 학습 모델을 효율적으로 활용할 수 있는 혁신적인 도구다.  
복잡했던 하이퍼파라미터 튜닝, 파인튜닝, 배포 과정을 간소화하며, 커뮤니티 중심의 모델 허브를 통해 다양한 모델을 누구나 쉽게 사용할 수 있게 했다.  
오픈소스라는 점에서 신뢰도가 높고, 연구와 산업 모두에서 강력한 인프라로 자리 잡았다.  
이 논문과 라이브러리를 통해 NLP의 미래가 더 밝아질 거라 확신한다.