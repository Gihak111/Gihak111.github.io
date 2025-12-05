---
layout: single
title: "GPT oss 모델을 사용해 보자"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## oss - 120B
아마 우리같으 ㄴ일반인이 구할 수 있는 모델 중 최고의 모델이 아닐까 생학해 본다.  
공개된 무료 버젼의 로컬 기동 모델 중 내가 사용해 본 것들 중에선 압도적으로 성능이 잘 나온다.  
이런 아름다운 모델을 파인튜닝으로 마개조 시키기 위해 허깅페이스에 올라와 있는 코드들을 통해 직접 모델을 사용해 보자  

## 환경 설정
우선, 공식 문서에 나와있는대로, 다음 명령어를 실행하자.  
```bash
pip install -U transformers kernels torch 
```

이어서, 이 코드를 실행하자.  
```python
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

```

하지만, 이것만으론 부족하다 오류가 나기 떄문이다.  

## 오류 해결방법
만일, 당신이 이런 무거운 모델의 로드 밑 설치가 처음이라면 라이브버리 없다는 오류가 나올 것이다  
이 오류는 다음 코드를 통해서 해결할 수 있다.  
```bash
pip install accelerate
```

위 라이브러리는 모델이 뒤지게 무거울 떄 cpu, gpu에 나누어서 할당해주는 그런 라이브러리 이다.  

앞선 코드에서,  
```python 
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
```
여기에서 device_map = "auto" 를 사용하기 위해 필요한 라이브러리 이다 암튼 이게 참 중요한 라이브러리라 이거까지 깔아주면 모델을 다운받을 수 있어진다.  

총 62기가 정도의 저장공간을 차지하지만, 직접 사용해 보면, 와 개쩐다 싶은거다  
이거 잘 구워삶고 광고 열심히 하면 뤼튼 같은거 만들어서 돈방석 앉을 수 있게 되는거다 ㅋㅋ