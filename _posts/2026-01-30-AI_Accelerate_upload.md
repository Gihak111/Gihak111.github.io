---
layout: single
title: "Hugging Face Accelerate"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Hugging Face Accelerate

딥러닝을 하다 보면 모델 학습 코드 자체보다 GPU 설정, 멀티 GPU 분산 학습(DDP), 혼합 정밀도(FP16) 설정 등 하드웨어 최적화를 위한 복잡한 코드 작성에 더 많은 시간을 쏟게 되는 경우가 많다.  

Accelerate는 바로 이 문제를 해결해 주는 도구이다.  
파이토치 하드웨어 설정 코드를 걷어내고, 단 몇 줄의 변경만으로 코드를 CPU, GPU, 멀티 GPU, TPU 어디서든 돌아가게 만들어 준다.  


## Accelerate란

기존의 순수 PyTorch로 딥러닝 코드를 짤 때, 하드웨어를 효율적으로 쓰려면 이런거 했어야 했다.  

- .to(device)를 변수마다 붙여줘야 한다 (CPU/GPU 이동)  
- 멀티 GPU를 쓰려면 DistributedDataParallel 설정을 짜야 했다.  
- 메모리를 아끼기 위해 FP16(Mixed Precision)을 쓰려면 GradScaler 등을 써야 함  

Accelerate는 이 모든 과정을 자동화한다.  
진짜 개꿀  

코드 하나로 CPU, GPU(1개), 멀티 GPU, TPU에서 모두 실행 가능해진다.  
또한 복잡한 설정 없이 데이터 분산 학습(DDP)을 바로 적용할 수 있다.  
코드 한 줄 변경 없이 FP16, BF16 연산 수행할 수 있다.  
무엇보다, 새로운 프레임워크가 아니라, 기존 PyTorch 코드 스타일을 그대로 유지하는거라 배우기 개쉽다.  


## 사용 방법 (Step-by-Step)


### 1. 설치

터미널에서 pip를 이용해 설치해보자.  

```bash
pip install accelerate

```

### 2. 환경 설정 

코드를 고치는 대신 터미널에서 환경을 설정한다 ㅋㅋㅋㅋ.  

```bash
accelerate config

```

이 명령어를 입력하면 질문들이 나온다. (예: GPU를 몇 개 쓸 건지, FP16을 쓸 건지 등).  
여기서 설정한 내용은 저장되어 나중에 코드를 실행할 때 자동으로 적용된다.  

### 3. 파이썬 코드 수정

기존 PyTorch 코드에 3가지만 변경하면 된다.  

1. Accelerator 객체 생성  
2. 모델, 옵티마이저, 데이터로더를 prepare()로 감싸기  
3. loss.backward() 대신 accelerator.backward(loss) 사용  

한번 기존 코드랑 비교해 보며 얼마나 사기적인지 한번 봐 보자.  

#### 기존 PyTorch 코드

```python
import torch

# 1. 디바이스 설정 (매번 확인해야 함)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device) # 모델을 GPU로 보냄
optimizer = torch.optim.Adam(model.parameters())

# 데이터 로더
dataloader = ...

for batch in dataloader:
    # 2. 데이터를 일일이 GPU로 보내야 함
    inputs = batch["input"].to(device)
    labels = batch["label"].to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    # 3. 역전파
    loss.backward() 
    optimizer.step()

```

#### Accelerate 적용 코드

```python
from accelerate import Accelerator # 추가

# 1. Accelerator 객체 생성
accelerator = Accelerator()

model = MyModel() # .to(device) 불필요
optimizer = torch.optim.Adam(model.parameters())
dataloader = ...

# 2. 모든 객체를 prepare()로 감싸면 알아서 디바이스 할당 및 분산 처리 설정 완료
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    # .to(device) 제거 (알아서 처리됨)
    inputs = batch["input"]
    labels = batch["label"]
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    # 3. 역전파를 accelerator가 담당
    accelerator.backward(loss) 
    optimizer.step()

```

### 4. 실행 

파이썬 스크립트를 실행할 때 python train.py 대신 아래 명령어를 쓴다.  

```bash
accelerate launch train.py

```

이렇게 하면 2단계에서 설정한 값(GPU 개수, FP16 여부 등)을 불러와서 자동으로 최적화하여 실행한다.  
자동으로 최적화 한다.  


## 언제 사용하면 좋을까

GPU가 여러 개 있는 서버에서 코드를 돌려야 할 때  
Colab이나 Kaggle처럼 GPU 환경이 계속 바뀌는 곳에서 작업할 때  
- 복잡한 파이토치 분산 학습 코드를 공부하기엔 시간이 부족할 때  
- 거대 언어 모델(LLM) 등을 불러올 때 메모리가 부족해서 `Big Model Inference` 기능이 필요할 때 (Accelerate는 큰 모델을 자동으로 CPU RAM과 Disk에 분산시켜 로딩해주는 기능도 있다)  


## 요약

| 특징 | 기존 PyTorch | Hugging Face Accelerate |
| --- | --- | --- |
| **디바이스 지정** | `x.to('cuda')` 수동 지정 | 자동 처리 (`prepare`) |
| **멀티 GPU** | `DistributedDataParallel` 등 복잡한 코드 작성 | 코드 변경 없음 (Config로 해결) |
| **Mixed Precision** | `AutoCast`, `GradScaler` 수동 구현 | Config에서 선택만 하면 됨 |
| **실행 방법** | `python script.py` | `accelerate launch script.py` |

---


## 결론
Accelerate는 딥러닝 엔지니어들이 하드웨어 설정 자동화 해 주는 아주 좋은 그런거다.  
원래는 모델 하나 분할해서 여러 GPU에 올리는게 말도 안되게 어려웠었는데, 이걸로 딸깍이 된다.  
이 얼마나 큰 발전인가 안그래도 랩값 미쳐 날뛰어서 사기 힘든데  
이런걸로라도 허리 비틀어야 하지 않겠는가.  