---
layout: single
title:  "RN에서 AI 실행해 보기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 1. BERT 모델 준비하기  
모바일에서 실행할 수 있도록 작은 크기의 BERT 모델을 변환해야 한다.  

#### 단계 1: Hugging Face에서 사전 학습된 모델 선택  
작은 사이즈의 BERT 모델을 선택하던가, 직접 bert 모델을 하나 만들자.  
예를 들어 `distilbert-base-uncased`처럼 경량화된 모델을 사용하는 것이 좋다.  
  
#### 단계 2: TensorFlow Lite 변환  
- 모델을 TensorFlow Lite 포맷으로 변환하여 모바일에서 실행 가능하도록 한다.  

```python
from transformers import TFBertModel
import tensorflow as tf

# 1) 사전 학습된 모델 로드
model = TFBertModel.from_pretrained("distilbert-base-uncased")

# 2) 모델을 TensorFlow Lite 형식으로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3) 모델 저장
with open("distilbert_model.tflite", "wb") as f:
    f.write(tflite_model)
```  

이제 `distilbert_model.tflite` 파일이 모바일에서 사용할 준비가 되었다.  

### 2. React Native에서 TensorFlow Lite 설정하기  

#### 단계 1: TensorFlow Lite 패키지 설치  
React Native에서 TensorFlow Lite 모델을 사용하려면 `react-native-tensorflow-lite`와 같은 라이브러리를 사용해야 합니다.  

```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-react-native
```

그리고 아래처럼 TensorFlow Lite를 위한 native 설치가 필요할 수 있다.  

#### 단계 2: tflite 모델을 불러오기  
변환된 모델 파일(`distilbert_model.tflite`)을 `assets` 폴더에 추가하자.  

```javascript
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

// 앱 초기화 단계에서 텐서플로우와 모델 불러오기
async function loadModel() {
  await tf.ready();
  const modelJson = require('./assets/distilbert_model.tflite');
  const model = await tf.loadLayersModel(bundleResourceIO(modelJson));
  return model;
}
```

### 3. 예측 실행하기  

이제 React Native에서 입력 텍스트를 전처리하고 모델을 통해 예측을 실행할 수 있다.  

#### 단계 1: 입력 전처리  
- BERT 입력에 맞게 텍스트를 토큰화해야 한다.  
`tokenizers` 라이브러리를 사용하거나, 서버에서 토큰화하는 API를 구현해 로컬에서 처리량을 줄일 수 있다.  

#### 단계 2: 예측 실행  
```javascript
async function predict(text) {
  const model = await loadModel();
  
  // 전처리한 입력 텐서 준비 (예: inputTensor)
  const inputTensor = tf.tensor([/* 토큰화된 텍스트 텐서 */]);

  const output = model.predict(inputTensor);
  return output.dataSync(); // 예측 결과
}
```  

이 과정을 통해 텍스트 입력에 대해 BERT 모델로 예측 결과를 얻을 수 있다.  

### 요약  
1. 사전 학습된 BERT 모델을 TensorFlow Lite로 변환.  
2. React Native에 TensorFlow Lite 설정 후, 모델을 로드.  
3. 입력 전처리 후, 모델로 예측 수행.  

모바일 성능이 제한적이므로 최적화된 모델을 사용하는 것이 중요하며, 필요한 경우 서버와 상호작용해 사전 처리를 할 수 있다.  
