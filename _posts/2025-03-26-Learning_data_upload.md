---
layout: single
title:  "딥러닝 데이터셋 얻는 법"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 데이터셋은 어디서 구할까? CC-100과 AI 허브 탐색

데이터셋은 머신러닝과 AI 모델을 만드는 데 필수적인 재료이다.  
좋은 데이터셋을 구하면 모델의 성능을 높일 수 있고, 연구나 프로젝트를 더 효율적으로 진행할 수 있다.  
오늘은 데이터셋을 어디서 구할 수 있는지 알아보자.  
특히 `CC-100`과 한국의 `AI 허브`를 중심으로 살펴보자.

### 1. CC-100 데이터셋 구하기

`CC-100`은 다국어 데이터셋의 보물창고이다.  
이 데이터셋은 100개 이상의 언어로 된 단일 언어 데이터를 제공한다.  
XLM-R 모델을 학습시키기 위해 만들어진 이 데이터셋은 웹 크롤링 데이터(Common Crawl)에서 추출된 것이다.  
로마자 표기 언어 데이터(예: `*_rom`)도 포함되어 있어서 다양한 언어 모델을 훈련시키기에 적합하다.  

#### 어디서 구할까?
`CC-100`은 `https://data.statmt.org/cc-100/`에서 쉽게 구할 수 있다.  
이 사이트는 에든버러 대학의 Statistical Machine Translation 팀이 제공하는 곳이다.  
사이트에 들어가면 언어별로 정리된 데이터 파일을 다운로드할 수 있다.  
각 파일은 텍스트로 되어 있고, 문서는 이중 개행(`\n\n`)으로, 문단은 단일 개행(`\n`)으로 구분된다.  
진짜 이만한데이터 어디가도 없다. 한국어 데이터가 13기가. 압풀 풀면 53기가  
그것도 광기의 txt 파일 하나로 53기가다.  
존나커서 열리지도 않는, 개지리는 데이터셋이다.  
진짜 GOAT  

#### 어떻게 활용할까?
언어 모델을 사전 학습시키거나 단어 표현을 만들 때 이 데이터를 사용해보자.  
예를 들어, 한국어 데이터(`ko.txt`)를 다운받아서 토크나이저를 훈련시키거나, 다국어 모델을 만들 때 여러 언어 데이터를 섞어서 써보자.  
다만, Common Crawl에서 온 데이터라 개인 정보가 섞여 있을 수 있으니, 민감한 작업에 쓸 때는 주의하자.  


### 2. AI 허브에서 데이터셋 찾기

한국에서 데이터셋을 구하고 싶다면 `AI 허브`가 좋은 선택이다.  
`I 허브`는 `https://aihub.or.kr/`에서 접근할 수 있는 공공 데이터 플랫폼이다.  
이곳은 한국어 중심의 다양한 데이터셋을 제공하며, AI 개발을 지원하기 위해 정부가 운영한다.  
이미지, 텍스트, 음성 등 여러 종류의 데이터가 준비되어 있다.  

#### 어떤 데이터가 있을까?
`AI 허브`에는 800개가 넘는 데이터셋이 있다.  
예를 들어, 한국어-영어 병렬 코퍼스(`aihub_translation`)는 번역 모델을 만들 때 유용하다.  
또 다른 예로는 인도 보행 데이터(`NIA Sidewalk dataset`)가 있는데, 장애인의 이동 문제를 해결하려는 공공 목적의 데이터이다.  
이런 데이터는 실무에서 바로 써볼 수 있는 실용적인 자료이다.  

#### 구하는 방법
사이트에 접속해서 회원가입을 하고, 원하는 데이터셋을 검색해보자.  
다운로드는 PC에서만 가능하니, 노트북이나 데스크톱으로 접속해야 한다.  
데이터셋 페이지에서 `다운로드` 버튼을 누르면 파일을 받을 수 있다.  
예를 들어, 스테레오 매칭 데이터는 `https://aihub.or.kr/aidata/136/download`에서 구할 수 있다.  
필요한 데이터를 골라서 바로 프로젝트에 적용해보자.  


### 3. 두 곳 비교하며 선택하기

`CC-100`과 `AI 허브`는 목적에 따라 다르게 쓰인다.  
`CC-100`은 다국어 텍스트 데이터가 필요할 때 최고의 선택이다.  
전 세계 언어를 다루고 싶다면 `https://data.statmt.org/cc-100/`으로 가서 데이터를 가져오자.  
반면, `AI 허브`는 한국어 중심이거나 공공 목적의 데이터를 원할 때 적합하다.  
한국어 음성 인식 모델을 만들고 싶다면 `https://aihub.or.kr/`에서 관련 데이터를 찾아보자.  

#### 팁
- **대규모 언어 모델**: `CC-100`을 활용해서 사전 학습 데이터를 모아보자.  
- **한국어 특화 작업**: `AI 허브`에서 한국어 텍스트나 음성 데이터를 구하자.  
- **혼합 사용**: 두 곳의 데이터를 섞어서 더 풍부한 데이터셋을 만들어보자.  

#### 주소 하이퍼 링크
[CC-100](https://data.statmt.org/cc-100/)  
[AI 허브](https://aihub.or.kr/)  

### 마무리

데이터셋은 프로젝트의 기반이다.  
`CC-100`은 `https://data.statmt.org/cc-100/`에서, `AI 허브`는 `https://aihub.or.kr/`에서 각각 구할 수 있다.  
이 두 곳을 잘 활용하면 연구나 개발에서 한 발 앞서갈 수 있다.  
진짜 저 두 사이트가 래전드 고트 보물창고다.  