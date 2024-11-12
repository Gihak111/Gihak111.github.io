---
layout: single
title:  "Falcon에 RAG에서 실시간 웹 검색 기능 API 없이 구현하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 외부 검색 크롤링을 통해 구현
웹 크롤링을 통해서 실시간 검색 기능을 만들 수 있다.  
웹에서 실시간으로 데이터를 크롤링 하거나, 로컬 문서 데이터를 사용하는 것으로 비슷하게 검색 시스템을 구현해 보자.  
이어서, 이 데이터를 바탕으로 RAG을 구성할 수 있다.  

## 1. 기능 요약  
1. 웹 크롤링  
Python 라이브러리인 BeautifulSoup 또는 Scrapy를 사용하여 웹 페이지에서 정보를 실시간으로 크롤링한다.  

2. 검색 인덱스 구축
크롤링한 데이터를 임베딩하고, 이를 FAISS와 같은 벡터 검색 라이브러리를 사용하여 검색 인덱스를 구축한다  

3. 실시간 쿼리 처리
사용자가 입력한 쿼리와 관련된 정보를 크롤링된 데이터에서 검색하고, 해당 정보를 바탕으로 언어 모델을 사용해 답변을 생성한다.  

단계를 보면, 웹 크롤링을 진행, 일르 통해 FAISS 인덱스 구축, 검색쿼리 처리, 답변 생성이 된다.  

코드로 보자.  
1. 웹 크롤링  

```python
import requests
from bs4 import BeautifulSoup

data_cvs = r'your/cvs file/location'
# 웹 페이지에서 실시간 데이터 크롤링
def crawl_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 예: 웹 페이지에서 모든 텍스트 추출
    texts = soup.get_text()
    return texts

# 예시 URL
url = data_cvs
crawled_data = crawl_website(url)
print(crawled_data[:500])  # 크롤링한 데이터 일부 확인

```

2. 크롤링 데이터로 FAISS 인덱스 생성  

```python
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델과 토크나이저 로드
model_name = "./Falcon_korean_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# 문서 임베딩 함수
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]
    return embeddings.numpy()

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(768)  # Llama 모델의 임베딩 차원
doc_embeddings = get_embeddings([crawled_data])  # 크롤링한 텍스트 임베딩
index.add(doc_embeddings)  # FAISS에 데이터 추가

```

3. 실시간 검색 및 답변 생성  

```python
# 검색 함수: 쿼리에 대해 FAISS에서 유사한 문서 검색
def search(query, top_k=3):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    # 유사한 문서들을 반환
    results = [crawled_data[idx] for idx in indices[0]]
    return results

# 검색된 문서 바탕으로 답변 생성
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

```

4. 실시간 대화 인터페이스  

```python
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

위 코드를 합쳐 실행하는 것으로, 웹 크롤링을 통한 검색기능을 추가할 수 있다.  
이는 API를 사용하지 않고 구현할 수 있지만, 성능은 조금 부족할 수 있다.  

개선사항은 다음과 같다.  
1. 크롤링 주기
웹 크롤링은 실시간으로 정보를 가져오는 방법을 사용하지만, 특정 주기로 데이터를 업데이트하거나 크롤링해야 할 수 있다.  

2. 데이터 저장
롤링한 데이터를 매번 다시 크롤링하는 대신 로컬에 저장하고, 필요한 경우에만 업데이트하는 방식으로 효율성을 높일 수 있다.  

3. URL 목록 관리
크롤링할 사이트나 페이지 URL 목록을 관리하여 원하는 정보를 주기적으로 크롤링할 수 있다.  