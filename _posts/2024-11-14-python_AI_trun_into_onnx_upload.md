---
layout: single
title:  "여러 프레임워크에서 AI 호환되도록 하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# ONNX
우리가 일반저긍로 모델을 저장할 때, h5, keras를 사용하거나, torch를 사용하곤 한다.  
하지만, 이는 파이썬 환경이 아닌, RN이나 Springboot에서 불러와 사용하는 것이 불가능 하다.  
물론, Flask 서버를 통해 AI를 배포하고, 이를 가져다가 사용하는 방법이 있지만,  
때론 백엔드 서버를 사용할 수 없거나, 로컬에서만 앱이 기동해야 할 경우가 있기 대문에, 여러 환경에서 사용할 수 있는 onnx 형태로 저장하는 것은 유리한 상황을 만들기 좋다.  

# ONNX 사용법
단계별로 알아보자. 임의의 모델을 만들고, 그걸 onnx 로 변경하는 과정을 한전 순차적으로 보자.  

### PyTorch에서 모델 저장 및 불러오기  
#### 모델 저장  
`state_dict()`로 모델 파라미터를 추출하고, `torch.save`로 파일로 저장한다.  

```python
import torch

# 학습이 완료된 모델 예시
model = ...  # 학습이 완료된 모델
torch.save(model.state_dict(), "model_weights.pth")
```  

#### 모델 불러오기
- `torch.load()`로 저장된 파라미터를 불러와서 모델에 `load_state_dict()`로 적용한다.  

```python
# 모델 인스턴스 생성 후 파라미터 불러오기
model = ...  # 동일한 구조의 모델
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
```  

> `map_location="cpu"` 옵션은 GPU에서 저장된 모델을 CPU에서 로드할 때 사용한다.

### ONNX로 모델 내보내기  
ONNX로 모델을 내보내기 위해 PyTorch 모델을 사용하고, 입력 텐서를 이용해 계산을 한 번 수행한다.  

```python
import torch.onnx

# 더미 입력 데이터 생성 (예: 1 x 3 x 224 x 224 이미지)
dummy_input = torch.randn(1, 3, 224, 224)

# 모델을 ONNX 형식으로 내보내기
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'])
```

### ONNX 파일에 Shape 정보 추가하기  
모델 입출력 크기(shape)를 추가해 모델 구조를 명확히 확인할 수 있다.  

```python
import onnx
from onnx import shape_inference

# ONNX 파일의 shape 정보 추가
onnx_model = onnx.load("model.onnx")
onnx_model_with_shapes = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model_with_shapes, "model_with_shapes.onnx")
```

### ONNX 파일에서 각 Layer 값 추출  
`onnx` 라이브러리와 `numpy_helper`를 사용해 각 레이어의 값을 확인할 수 있다.  

```python
import onnx
from onnx import numpy_helper

# ONNX 모델 불러오기
onnx_model = onnx.load("model_with_shapes.onnx")

# 초기화된 레이어 값 추출 (딕셔너리 형식)
initializers = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
```  

### PyTorch와 ONNX 모델의 레이어 값 비교하기  
ONNX와 PyTorch 모델의 동일 레이어에 대한 값을 비교해 모델이 정확히 변환되었는지 확인한다.  

```python
import numpy as np

def compare_layers(actual, expected, layer_name, rtol=1e-5, atol=1e-7):
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        print(f"{layer_name}: 일치합니다.")
    except AssertionError as e:
        print(f"{layer_name}: 차이 발생")
        print(e)

# PyTorch 모델의 파라미터 불러오기
torch_layers = {name: param.detach().numpy() for name, param in model.named_parameters()}

# 레이어 값 비교
for layer_name, onnx_weights in initializers.items():
    if layer_name in torch_layers:
        compare_layers(onnx_weights, torch_layers[layer_name], layer_name)
```  

이 코드에서는 `compare_layers` 함수를 사용해 각 레이어의 값을 비교하고, 차이가 발생할 경우 에러 메시지를 출력하여 확인할 수 있다.

## 예시를 보자.  
모바일에서 실행 가능한 간단한 BERT 모델을 만들고, React Native에서 실행하는 전체 과정을 이행해보자.  

### 1. BERT 모델 준비하기  
#### 단계 1: Hugging Face에서 사전 학습된 BERT 모델 선택  
- `distilbert-base-uncased`와 같은 경량화된 BERT 모델을 선택하는 것이 좋다.  

#### 단계 2: 모델을 ONNX 형식으로 변환  
- Hugging Face에서 제공하는 모델을 ONNX 형식으로 변환하려면 `transformers` 라이브러리와 `onnx` 라이브러리를 사용한다.  

```python
from transformers import BertModel
from transformers.convert_graph_to_onnx import convert

# 1) 사전 학습된 모델 로드
model = BertModel.from_pretrained("distilbert-base-uncased")

# 2) ONNX로 모델 변환
convert(framework="pt", model=model, output="distilbert_model.onnx")
```  

이제 `distilbert_model.onnx` 파일이 생성되었다.  
이 파일을 모바일에서 사용할 준비가 되었다.  

### 2. React Native에서 ONNX 모델 실행 설정  
ONNX 모델을 React Native에서 실행하려면 `onnxruntime` 라이브러리를 사용한다.  
이 라이브러리는 ONNX 모델을 모바일에서 효율적으로 실행할 수 있게 해준다.  

#### 단계 1: ONNX Runtime 패키지 설치  
React Native에서 ONNX 모델을 사용하려면 `onnxruntime-react-native` 패키지를 설치해야 한다.  

```bash
npm install onnxruntime-react-native
```  

#### 단계 2: ONNX 모델 불러오기  
모델 파일(`distilbert_model.onnx`)을 `assets` 폴더에 추가한다.  
그런 다음, React Native에서 모델을 불러온다.  

```javascript
import * as ort from 'onnxruntime-react-native';

// 앱 초기화 단계에서 ONNX 모델 불러오기
async function loadModel() {
  await ort.init();
  const model = await ort.InferenceSession.create(require('./assets/distilbert_model.onnx'));
  return model;
}
```  

### 3. 예측 실행하기
#### 단계 1: 입력 전처리
BERT 모델은 텍스트 입력을 받기 전에 토큰화가 필요하다.  
`tokenizers` 라이브러리 또는 서버에서 토큰화하는 API를 사용하여 입력 텍스트를 처리할 수 있다.  

#### 단계 2: 예측 실행
모델에 입력을 텐서 형식으로 전달하고 예측을 실행한다.  

```javascript
async function predict(text) {
  const model = await loadModel();

  // 입력 텍스트를 토큰화하여 BERT 입력 형식에 맞는 텐서로 변환
  const inputTensor = new ort.Tensor('float32', [/* 토큰화된 텍스트 배열 */], [1, /* 텍스트 길이 */]);

  // 예측 실행
  const output = await model.run([inputTensor]);
  return output; // 예측 결과
}
```  

이 과정에서는 `run()` 메서드를 통해 모델 예측을 실행하고, 출력 값을 반환받는다.  

### 4. 예측 결과 처리  
모델의 출력은 일반적으로 여러 벡터로 이루어져 있으며, 이를 후처리하여 사용자가 원하는 형태로 변환해야 할 수 있다.  

```javascript
async function processOutput(output) {
  const logits = output['logits'];  // 예측된 로짓 값
  const predictions = logits.argmax(-1); // 예측된 클래스 값 추출
  return predictions;
}
```  

이 과정에서 모델 최적화나 모바일 성능에 맞는 모델 선택이 중요하다.  
예를 들어, 텍스트 전처리나 예측 후 처리는 서버에서 처리할 수 있는 방법을 고려할 수도 있다.  
