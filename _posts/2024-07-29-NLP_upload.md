---
layout: single
title:  "NLP"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
# 파이썬으로 자연어 처리(NLP) 기초 다져보기

텍스트 데이터를 다루는 데 유용한 기본적인 예제이다.  

## 예제: 텍스트 전처리와 빈도 분석
`nltk`는 자연어 처리를 위한 다양한 기능을 제공하는 라이브러리이다.  
이걸로 택스트 전처리, 비교분석 할 수 있다.  

```bash
pip install nltk
```

# 코드
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

# nltk 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # 소문자로 변환
    text = text.lower()
    
    # 문장 부호 제거
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 단어 토큰화
    words = word_tokenize(text)
    
    # 불용어(stopwords) 제거
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return filtered_words

def analyze_frequency(words):
    # 단어 빈도 분석
    word_freq = Counter(words)
    return word_freq

# 샘플 텍스트
sample_text = """
Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.
"""

# 텍스트 전처리
processed_words = preprocess_text(sample_text)

# 단어 빈도 분석
word_frequencies = analyze_frequency(processed_words)

print("단어 빈도 분석 결과:")
for word, freq in word_frequencies.items():
    print(f"{word}: {freq}")

```  