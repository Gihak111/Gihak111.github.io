---
layout: single
title:  "Falcon에 RAG에서 실시간 웹 검색 기능 구현하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 외부 검색 API를 통해 구현
가장 간단하게 실시간 웹 검색 기능을 구현할 수 있다.  
이는 API키를 요구하며, 이거 없이 구현 가능한지는 잘 모르겠다.  
OpenAI의 browser 도구나 Python의 requests 라이브러리를 사용해 Google Custom Search API를 통해 실시간 검색을 구현한다.  
Google Custom Search API를 사용하려면 Google Cloud Console에서 Custom Search JSON API를 활성화하고 API 키를 발급받자.  

```python
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Falcon 7B 모델과 토크나이저 불러오기
model_name = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Google Custom Search API 설정
API_KEY = "YOUR_GOOGLE_API_KEY"  # 발급받은 Google API 키
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"  # 발급받은 검색 엔진 ID

def google_search(query, num_results=3):
    """Google Custom Search API로 검색하여 상위 결과를 반환."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
    }
    response = requests.get(url, params=params)
    results = response.json()
    return [item['snippet'] for item in results.get("items", [])]

def generate_answer(query):
    # 웹 검색을 통한 정보 수집
    retrieved_docs = google_search(query)
    context = " ".join(retrieved_docs)

    # 모델 입력 생성
    prompt = f"질문: {query}\n컨텍스트: {context}\n답변:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 답변 생성
    output = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

# 실시간 대화 인터페이스
def live_chat():
    print("실시간 AI 대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.")
    while True:
        try:
            query = input("질문을 입력하세요: ")
            if query.lower() == "종료":
                print("대화를 종료합니다.")
                break
            
            # 응답 생성
            answer = generate_answer(query)
            print(f"AI의 답변: {answer}\n")

        except KeyboardInterrupt:
            print("대화가 중단되었습니다.")
            break

# 실시간 대화 시작
live_chat()

```

google_search 함수를 통해서 Google Custom Search API에 쿼리를 보내고, 검색 결과에서 상위 3개의 요약(snippet)을 가져와 retrieved_docs에 저장한다.  
API_KEY와 SEARCH_ENGINE_ID를 설정해야 API가 정상적으로 작동한다.  
다음에는 API 없이, 웹 크롤링을 통해서 실시간 웹 검색 기능을 구현해 보자.  
