---
layout: single
title:  "Falcon에 RAG 추가해서 간의 검색 기능 추가하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# RAG
RAG은 AI환각 등 다양한 오류를 해결하고, 원래의 성능을 간화 할 수 있는 간력한 솔루션이다.  
이를 Llama 모델에 적용시키는 것으로 Llama역시 강해질 수 있다.  
이렇게 RAG로 검색 기능을 추가하면, 본인의 데이터 안에 정보가 없을 경우 여기의 docs에 있는 목록에서 검색해 정보를 찾는다.  
실시간 온라인 검색은 API를 사용해야 하므로 짜치니까 다음에 설명하겠다.   

## 1. 라이브러리 준비
RAG에선 두가지를 사용한다.  
1. 정보 검색 라이브러리 (예: ElasticSearch, FAISS)
2. Llama 모델을 위한 transformers와 datasets 라이브러리
```bash
# 필요한 패키지 설치
pip install transformers datasets faiss-cpu
```

## 2. 대화 시작
다음과 같은 코드를 구성하자.  
```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import torch

# 모델과 토크나이저 로드
model_name = "tiiuae/falcon-7b"  # Falcon7B 모델 경로
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# CSV 파일에서 문서 불러오기
def load_docs_from_csv(file_path):
    data = pd.read_csv(file_path)
    docs = data['content'].tolist()
    return docs

# CSV 파일 경로 설정
csv_file_path = "documents.csv"
docs = load_docs_from_csv(csv_file_path)

# Step 1: 문서 임베딩 및 FAISS 인덱스 생성
index = faiss.IndexFlatL2(4096)  # Falcon7B 임베딩 크기에 맞게 수정 (4096 차원)
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]
    return embeddings.numpy()

doc_embeddings = get_embeddings(docs)
index.add(doc_embeddings)

# Step 2: 질문을 받아서 관련 문서 검색
def search(query, top_k=3):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [docs[idx] for idx in indices[0]]
    return results

# Step 3: 검색 결과를 바탕으로 답변 생성
def generate_answer(query):
    # 검색 결과 가져오기
    retrieved_docs = search(query)
    context = " ".join(retrieved_docs)
    
    # 모델 입력 생성
    prompt = f"질문: {query}\n컨텍스트: {context}\n답변:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 답변 생성
    output = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer

# Step 4: 대화 인터페이스 생성
def chat():
    print("AI와 대화를 시작합니다. '종료'라고 입력하면 대화가 종료됩니다.")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "종료":
            print("대화를 종료합니다.")
            break
        
        # 답변 생성
        answer = generate_answer(query)
        print("AI의 답변:", answer)

# 대화 시작
chat()

```

위 코드는 csv 파일의 내용을 데이터 베이스로 하여 자신에게 정보가 없는 경우, 이 docs 파일에서 내용을 검색한다.  
이 과정을 통해서 AI 환각 현상을 해결할 수 있고, 딥러닝 없이 AI를 강과할 수 있는 방법이므로 유용하다.  
그냥, 저 코드 실행하면 AI와 대화할 수 있는 점도 간단해서 좋은 것 같다.  

csv 파일 경우 다음과 같이 한다.  
간단하게, 50줄만 갈겨봤다. 물론 AI가
```csv
title,content
"Python 소개","Python은 인기 있는 프로그래밍 언어입니다. 높은 가독성과 다양한 라이브러리 지원으로 데이터 분석, 웹 개발, 머신러닝 등 다양한 분야에서 사용됩니다."
"RAG 개념","RAG는 Retrieval-Augmented Generation의 약자로, 검색과 생성 모델을 결합하여 더 정확한 정보 생성을 목표로 하는 기술입니다."
"Lambda 함수","Lambda 함수는 Python에서 익명 함수를 만들 때 사용되는 키워드입니다. 주로 한 줄로 간단한 작업을 처리할 때 유용합니다."
"텐서플로우(TensorFlow)","텐서플로우는 Google에서 개발한 오픈소스 머신러닝 라이브러리입니다. 딥러닝 모델을 구축하는 데 널리 사용됩니다."
"딥러닝의 개념","딥러닝은 인공신경망을 기반으로 한 기계 학습의 한 분야입니다. 대량의 데이터와 강력한 계산 자원을 통해 학습을 진행합니다."
"GPU와 CPU 차이","GPU는 수많은 작은 연산을 동시에 처리하는 데 특화된 반면, CPU는 빠른 순차 연산에 강점을 가집니다."
"PyTorch vs TensorFlow","PyTorch는 동적 계산 그래프를 사용하여 직관적인 모델링이 가능하고, TensorFlow는 정적 그래프를 사용하여 대규모 배포에 유리합니다."
"RNN(순환 신경망)","RNN은 시퀀스 데이터를 처리하는 데 특화된 신경망으로, 이전의 출력을 다음 계산에 사용하는 특징이 있습니다."
"배치 학습과 온라인 학습","배치 학습은 모든 데이터를 한 번에 학습하는 방법이며, 온라인 학습은 데이터를 점진적으로 학습하는 방법입니다."
"자연어 처리(NLP)","자연어 처리는 텍스트 데이터를 분석하고 처리하는 기술로, 머신러닝과 딥러닝을 사용하여 문장을 이해하거나 생성합니다."
"학습률(Learning Rate)","학습률은 모델의 가중치를 업데이트할 때 한 번에 조정하는 크기입니다. 너무 크거나 너무 작으면 모델이 잘 학습되지 않습니다."
"트랜스포머(Transformer)","트랜스포머는 자연어 처리 모델에서 문맥을 파악하는 데 뛰어난 성능을 보이는 구조로, Attention Mechanism을 사용합니다."
"컴퓨터 비전","컴퓨터 비전은 이미지나 비디오 데이터를 통해 의미 있는 정보를 추출하는 기술로, 얼굴 인식, 객체 추적 등에 활용됩니다."
"강화 학습(Reinforcement Learning)","강화 학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방식입니다."
"BERT(Bidirectional Encoder Representations from Transformers)","BERT는 양방향으로 텍스트를 처리하는 모델로, 문장 내 단어들의 관계를 잘 이해할 수 있습니다."
"자율 주행 자동차","자율 주행 자동차는 환경을 인식하고, 판단하며, 자율적으로 주행하는 자동차로, 딥러닝 기술이 핵심입니다."
"추천 시스템","추천 시스템은 사용자의 이전 행동이나 데이터를 바탕으로 관심 있을 만한 제품이나 서비스를 추천해 주는 기술입니다."
"미니 배치(Mini-batch) 학습","미니 배치는 전체 데이터를 작은 크기로 나누어 한 번에 처리하는 방법으로, 학습 속도를 개선할 수 있습니다."
"정규화(Normalization)","정규화는 데이터의 범위를 일정한 값으로 변환하는 기법으로, 모델이 더 빠르고 정확하게 학습할 수 있도록 돕습니다."
"XGBoost","XGBoost는 그래디언트 부스팅 알고리즘을 사용한 머신러닝 라이브러리로, 예측 정확도가 높아 많은 대회에서 우승한 알고리즘입니다."
"AutoML","AutoML은 머신러닝 모델의 설계와 튜닝을 자동화하는 기술로, 비전문가도 고성능 모델을 쉽게 만들 수 있도록 돕습니다."
"YOLO(You Only Look Once)","YOLO는 실시간 객체 인식을 위한 딥러닝 모델로, 하나의 신경망을 사용하여 이미지를 빠르게 분석합니다."
"GAN(Generative Adversarial Network)","GAN은 두 신경망이 경쟁적으로 학습하는 방식으로, 가짜 데이터를 생성하거나 기존 데이터를 개선하는 데 사용됩니다."
"자연어 생성(NLG)","자연어 생성은 컴퓨터가 사람이 이해할 수 있는 텍스트를 생성하는 기술로, 챗봇이나 자동 요약 등에 사용됩니다."
"GPT(Generative Pre-trained Transformer)","GPT는 대규모 데이터로 미리 학습한 후 특정 작업에 맞게 fine-tuning하는 방식으로, 자연어 처리에 강력한 성능을 보입니다."
"이미지 분할(Image Segmentation)","이미지 분할은 이미지 내에서 객체의 경계를 구분하여 각 영역을 분리하는 기술로, 의료 영상 분석 등에 활용됩니다."
"음성 인식(Speech Recognition)","음성 인식은 음성을 텍스트로 변환하는 기술로, 음성 명령이나 회화형 인터페이스에 사용됩니다."
"트랜스퍼 러닝(Transfer Learning)","트랜스퍼 러닝은 이미 학습된 모델을 활용하여 새로운 작업에 맞게 학습을 진행하는 기법입니다."
"전이 학습(Transfer Learning)과 Fine-tuning","전이 학습은 기존 모델을 활용하여 다른 문제를 해결하는 기법으로, Fine-tuning은 그 모델을 미세 조정하는 과정입니다."
"SQL vs NoSQL","SQL은 관계형 데이터베이스 관리 시스템에 사용되며, NoSQL은 비관계형 데이터베이스로 비정형 데이터를 다루는 데 적합합니다."
"감성 분석(Sentiment Analysis)","감성 분석은 텍스트에서 감정이나 의견을 분석하여 긍정적, 부정적 또는 중립적인 감정을 파악하는 기술입니다."
"검색엔진 최적화(SEO)","SEO는 검색엔진에서 웹사이트의 순위를 높이는 기법으로, 적절한 키워드 사용과 콘텐츠 품질이 중요합니다."
"모델 평가 지표","모델 평가 지표는 모델의 성능을 측정하는 데 사용되며, 정확도, 정밀도, 재현율, F1-score 등이 있습니다."
"기계 학습(Machine Learning)","기계 학습은 데이터를 기반으로 알고리즘이 자동으로 학습하여 예측하거나 결정을 내리는 기술입니다."
"AI 윤리","AI 윤리는 인공지능이 사회에 미치는 영향과 그에 따른 윤리적 문제를 다루는 분야입니다."
"클라우드 컴퓨팅","클라우드 컴퓨팅은 인터넷을 통해 데이터를 저장하고 처리하는 기술로, 다양한 서비스 모델이 있습니다."
"기계 번역(Machine Translation)","기계 번역은 한 언어를 다른 언어로 변환하는 기술로, 자동 번역기에서 널리 사용됩니다."
"자연어 이해(NLU)","자연어 이해는 컴퓨터가 인간의 언어를 이해하는 기술로, 질문 답변 시스템이나 음성 비서를 구현하는 데 사용됩니다."
"네이티브 앱 vs 웹 앱","네이티브 앱은 특정 운영 체제에서 실행되는 앱이며, 웹 앱은 인터넷을 통해 실행되는 애플리케이션입니다."
"클라우드 데이터베이스","클라우드 데이터베이스는 클라우드에서 제공되는 데이터베이스로, 물리적인 서버 없이 온라인에서 데이터 저장 및 관리가 가능합니다."
"딥러닝 프레임워크","딥러닝 프레임워크는 신경망 모델을 구축하고 학습시키는 데 사용되는 소프트웨어 라이브러리입니다."
"코드 최적화","코드 최적화는 프로그램의 실행 속도나 메모리 사용량을 개선하는 과정을 말합니다."
"로봇공학(Robotics)","로봇공학은 로봇의 설계, 제작, 운영 등을 연구하는 학문으로, 자동화와 기계 학습 기술이 결합된 분야입니다."
"AI와 빅데이터","AI와 빅데이터는 상호 보완적인 기술로, 빅데이터에서 유의미한 정보를 추출하여 AI 모델을 학습시킵니다."
"모바일 앱 개발","모바일 앱 개발은 스마트폰과 태블릿을 위한 애플리케이션을 설계하고 구현하는 과정입니다."
"소프트웨어 개발 생명 주기(SDLC)","소프트웨어 개발 생명 주기는 소프트웨어를 개발하는 전체 과정을 설명하는 모델로, 계획, 설계, 구현, 테스트, 유지보수 등의 단계가 포함됩니다."

```

위 처럼 백과사전마냥 내용을 집어넣어 두면, AI가 자신에게 없는 내용을 이 백과 사전에서 찾는다.  
물론, 백과사전 대신에 웹에서 검색하게끔 할 수 도 있지만, 이는 API를 활용하며, 다음에 설명하겠다.  
