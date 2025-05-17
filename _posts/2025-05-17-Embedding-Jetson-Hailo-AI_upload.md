---
layout: single
title:  "Embedding 보드에서 AI"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## Embedding 보드에서 AI 모델 구현: Jetson, Hailo

대형 언어 모델(LLM)과 컴퓨터 비전 모델은 뛰어난 성능을 제공하지만, 클라우드 환경에서는 높은 지연 시간과 연결성 문제가 발생할 수 있다.  
**Embedding 보드**(예: NVIDIA Jetson, Hailo-8)는 저전력, 고효율의 에지 컴퓨팅을 통해 이러한 문제를 해결하며, 로봇, IoT, 자율주행 등 실시간 AI 응용에 필수적이다.  
이 글에서는 Embedding 보드에서 AI 모델 구현의 필요성과 방법을 분석하고, PyTorch로 구현하며, 실무에서의 활용 사례를 다룬다.  
문제 정의, 기술 구현, 실무 적용에 초점을 맞춘다.  

### 1. 에지 AI의 문제와 Embedding 보드의 필요성

클라우드 기반 AI는 대규모 계산 자원을 제공하지만, 에지 디바이스에서의 실시간 처리가 필요한 응용에서는 한계가 있다.  
긴 지연 시간, 네트워크 의존성, 그리고 높은 전력 소모는 에지 AI의 주요 장애물이다.  

- **문제 원인**
  - **높은 지연 시간**: 클라우드에서 데이터를 전송하고 처리하는 과정에서 발생하는 딜레이는 자율주행, 로봇 등 실시간 응용에 부적합.  
  - **네트워크 의존성**: 인터넷 연결이 불안정한 환경(예: 공장, 농업)에서 클라우드 기반 AI는 동작 불가.  
  - **전력 소모**: 고성능 GPU는 전력 소모가 크며, 배터리 기반 디바이스에 부적합.  
- **영향**
  - 실시간 응용(예: 객체 탐지, 음성 인식)에서 성능 저하.  
  - 대규모 배포 시 높은 운영 비용.  
  - 데이터 프라이버시 문제로 클라우드 전송이 제한되는 경우.  

### 2. Embedding 보드: 에지 AI의 핵심 플랫폼

Embedding 보드는 저전력, 고성능의 하드웨어로 설계되어 에지에서 AI 모델을 효율적으로 실행한다.  
NVIDIA Jetson은 GPU 기반 병렬 연산을, Hailo-8은 AI 전용 NPU(Neural Processing Unit)를 제공한다.  

#### 2.1 Embedding 보드의 구조와 역할

- **NVIDIA Jetson**
  - **구성**: ARM CPU, CUDA 지원 GPU, JetPack SDK(예: TensorRT, cuDNN).  
  - **특징**: 최대 40 TOPS(Jetson Orin Nano), 다양한 AI 프레임워크(TensorFlow, PyTorch) 지원.  
  - **용도**: 컴퓨터 비전, 로봇, 자율주행.  
- **Hailo-8**
  - **구성**: 26 TOPS NPU, Dataflow Compiler, M.2 폼 팩터.  
  - **특징**: 저전력(2~3W), 모델 최적화(양자화, 컴파일)로 고효율 추론.  
  - **용도**: 스마트 카메라, 산업 자동화, 보안.  
- **공통 장점**
  - **저전력**: 배터리 기반 디바이스에 적합.  
  - **실시간 처리**: 낮은 지연 시간으로 실시간 응용 지원.  
  - **컴팩트 설계**: 소형 디바이스에 통합 가능.  

#### 2.2 코드로 구현하기: Jetson과 Hailo에서 AI 모델 실행

PyTorch로 간단한 객체 탐지 모델(YOLOv5)을 Jetson과 Hailo에 배포하는 예제를 구현한다.  

```python
import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Jetson에서 Faster R-CNN 실행 (TensorRT 최적화 가정)
class JetsonDetector(nn.Module):
    def __init__(self):
        super(JetsonDetector, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            predictions = self.model([img])
        return predictions

# Hailo-8에서 모델 실행 (가정: Hailo Dataflow Compiler로 컴파일된 HEF 파일)
class HailoDetector:
    def __init__(self, hef_path):
        # Hailo-8 초기화 (실제로는 HailoRT 라이브러리 사용)
        self.model = self.load_hailo_model(hef_path)

    def load_hailo_model(self, hef_path):
        # Hailo Executable Format(HEF) 로드 (가정)
        return hef_path

    def predict(self, img):
        # 이미지 전처리 및 Hailo-8 추론 (가정)
        processed_img = self.preprocess(img)
        output = self.model.run(processed_img)
        return self.postprocess(output)

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    def postprocess(self, output):
        # 가정: 바운딩 박스 및 클래스 반환
        return output

# 예시 실행
img = cv2.imread("sample.jpg")
img_tensor = torch.from_numpy(np.transpose(img / 255.0, (2, 0, 1))).float()

# Jetson 추론
jetson_detector = JetsonDetector()
jetson_output = jetson_detector(img_tensor)
print("Jetson 출력:", jetson_output)

# Hailo 추론 (가정)
hailo_detector = HailoDetector("yolov5.hef")
hailo_output = hailo_detector.predict(img)
print("Hailo 출력:", hailo_output)
```

**참고**: Hailo-8은 실제로 `HailoRT`와 Dataflow Compiler를 사용해 모델을 HEF(Hailo Executable Format)로 변환해야 하며, 위 코드는 이를 가정한 예제이다.  
Jetson은 TensorRT로 최적화된 모델을 실행할 수 있다.[](https://www.mdpi.com/2079-9292/13/12/2322)  

#### 2.3 Embedding 보드의 이점

- **저전력 고성능**: Jetson Orin(40 TOPS, 15~40W), Hailo-8(26 TOPS, 2~3W)으로 배터리 기반 디바이스에 적합.[](https://components101.com/articles/popular-development-boards-for-ai-machine-learning)  
- **실시간 처리**: 낮은 지연 시간으로 객체 탐지, 음성 인식 등에 최적.  
- **프라이버시 보장**: 데이터가 디바이스 내에서 처리되어 클라우드 전송 불필요.  

### 3. Transformer와 LLM에서의 Embedding 보드 활용

Embedding 보드는 Transformer 기반 모델과 LLM의 에지 배포를 가능하게 한다.  
예: Jetson에서 경량화된 LLaMA, Hailo-8에서 컴퓨터 비전 모델.  

#### 3.1 Transformer 모델 배포

Transformer 모델은 Attention 메커니즘으로 높은 성능을 제공하지만, 에지 디바이스에서는 메모리와 계산 제약이 있다.  
Embedding 보드는 이를 해결한다.  

- **구조**
  - Jetson: TensorRT로 모델 최적화, FP16/INT8 양자화.  
  - Hailo-8: Dataflow Compiler로 모델을 NPU에 최적화.  
- **역할**
  - **경량화**: LoRA, 양자화로 모델 크기와 계산량 감소.  
  - **실시간 추론**: 이미지 분류, 객체 탐지 등에 적합.  
- **코드 구현**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Jetson에서 경량화된 Transformer 모델 실행
class JetsonTransformer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(JetsonTransformer, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

# 예시 실행
model = JetsonTransformer()
text = "이 제품은 정말 훌륭해요!"
output = model(text)
print("Jetson Transformer 출력:", torch.argmax(output, dim=-1).item())
```

#### 3.2 LLM 최적화

LLM은 에지에서 실행하기 위해 경량화가 필수적이다.  

- **Jetson**: LoRA와 TensorRT로 LLaMA-7B 실행, INT8 양자화로 메모리 절약.  
- **Hailo-8**: 컴퓨터 비전 중심이지만, Hailo-10H는 LLM 추론 지원(최대 40 TOPS).[](https://www.reddit.com/r/LocalLLaMA/comments/1d7shcr/raspberry_pi_goes_all_in_on_ai_with_70_hailo_kit/)  
- **실무 사례**: 스마트 카메라에서 객체 탐지, 로봇에서 음성 명령 처리.  

### 4. 성능 평가: Embedding 보드 비교

Jetson과 Hailo-8의 성능을 비교하기 위해 간단한 이미지 분류 작업을 수행한다.  

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time

# 데이터 준비
n_samples, img_size, channels = 1000, 224, 3
X = torch.randn(n_samples, channels, img_size, img_size)
y = torch.randint(0, 2, (n_samples,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

# Jetson 모델
class JetsonClassifier(nn.Module):
    def __init__(self):
        super(JetsonClassifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# 학습 및 추론 시간 측정
def evaluate_model(model, loader):
    start_time = time.time()
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    for data, target in loader:
        data, target = data.to(model.device), target.to(model.device).float()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        total_loss += loss.item()
    print(f"평균 손실: {total_loss / len(loader):.4f}")
    print(f"추론 시간: {time.time() - start_time:.2f}초")

# 실행
jetson_model = JetsonClassifier()
print("Jetson 성능 평가:")
evaluate_model(jetson_model, loader)
```

**참고**: Hailo-8은 Dataflow Compiler로 컴파일된 HEF 파일을 사용하며, 위 코드는 Jetson 중심으로 작성되었다. Hailo-8은 동일 작업에서 약 2배 빠른 추론 속도(예: 500 FPS)와 낮은 전력 소모를 제공한다.[](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/)  

### 5. 실무 활용: 에지 AI 응용

Embedding 보드를 활용해 실시간 응용을 구현한다.  

```python
from transformers import pipeline

# Jetson에서 실시간 감정 분석
def jetson_sentiment_analysis(text):
    classifier = pipeline("sentiment-analysis", device=0)  # CUDA 사용
    return classifier(text)

# 예시
text = "이 영화는 정말 재미있어요!"
result = jetson_sentiment_analysis(text)
print("감정 분석 결과:", result)
```

- **Jetson**: JetPack SDK로 컴퓨터 비전, NLP 모델 배포(예: 스마트 카메라, 드론).  
- **Hailo-8**: 스마트 감시 카메라에서 객체 탐지, 산업 자동화에서 결함 감지.[](https://www.embedded.com/choosing-an-embedded-ai-platform/)  

### 결론
Embedding 보드는 에지 AI의 핵심 플랫폼으로, Jetson과 Hailo-8은 저전력과 실시간 추론으로 클라우드 의존성을 줄이고 성능을 극대화한다.  
LoRA, 양자화, TensorRT, Dataflow Compiler를 활용해 모델을 최적화하면 다양한 실무 응용이 가능하다.  
Embedding 보드 없이는 에지에서의 효율적인 AI 구현이 어려울 수 있다.  
따라서 에지 AI 개발에서 이들은 필수적이다.  