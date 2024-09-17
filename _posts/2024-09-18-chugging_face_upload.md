---
layout: single
title:  "허깅 페이스에 앱 올리기"
categories: "Ai"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
 
Hugging Face에 AI 모델을 업로드하는 과정은 다음과 같다.  
특히 `config.json`에 대한 문제를 해결하는 방법도 포함되어 있다.  

### 1. **Hugging Face 계정 생성 및 로그인**

1. Hugging Face 웹사이트([Hugging Face](https://huggingface.co))에 가서 계정을 생성한다.
2. 로그인 후, 사용자 프로필 메뉴에서 "Settings"를 클릭하여 API 토큰을 생성힌다.

### 2. **Hugging Face Transformers 라이브러리 설치**

```bash
pip install transformers
```

### 3. **모델과 관련 파일 준비**

1. **모델 파일**: `pytorch_model.bin`, `tf_model.h5`, `model.bin` 등
2. **토크나이저 파일**: `tokenizer.json`, `vocab.txt`, `merges.txt` 등
3. **구성 파일**: `config.json`

### 4. **`config.json` 파일 준비**

모델에 필요한 구성 파일을 작성합니다. 일반적으로 다음과 같은 내용이 포함된다:

- BERT 모델 예시:

```json
{
  "architectures": [
    "BertForSequenceClassification"
  ],
  "hidden_size": 768,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

여기서 `architectures`, `hidden_size`, `num_attention_heads`, `num_hidden_layers` 등은 모델의 아키텍처에 따라 다를 수 있다.

### 5. **Hugging Face에 모델 업로드**

1. **Hugging Face CLI 설치**:

```bash
pip install huggingface_hub
```

2. **로그인**:

```bash
huggingface-cli login
```

API 토큰을 입력하여 로그인힌다.

3. **모델 업로드**:

```bash
from huggingface_hub import HfApi, Repository

# API 객체 생성
api = HfApi()

# 모델 저장소 생성
repo_url = api.create_repo(repo_id="your-username/your-model-name", repo_type="model")

# 저장소 클론
repo = Repository(local_dir="model_repo", clone_from=repo_url)

# 모델과 토크나이저 파일 복사
import shutil
import os

# 현재 작업 디렉토리에서 모델 파일 복사
for file_name in ["pytorch_model.bin", "config.json", "tokenizer.json"]:
    shutil.copy(file_name, "model_repo/")

# 커밋 및 푸시
repo.git_add()
repo.git_commit("Initial commit")
repo.git_push()
```

여기서 `your-username/your-model-name`을 자신의 사용자명과 모델명으로 바꿔야 한다.

### 6. **모델을 사용하는 방법**

Hugging Face에 모델이 업로드되면, 모델을 로드하고 사용하는 방법은 다음과 같다:

```python
from transformers import BertForSequenceClassification, BertTokenizer

model_name = "your-username/your-model-name"

# 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 예측
inputs = tokenizer("Sample text", return_tensors="pt")
outputs = model(**inputs)
```

이제 Hugging Face에 모델을 업로드하고, 로드 및 사용할 준비가 끝난다.  
폴더 안에 구성하지 말고, 바로 config.json 이랑 Ai 파이 ㄹ있어야 한다.   