---
layout: single  
title:  "Expo와 Flask 서버 연동"  
categories: "Expo"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---
# Expo 앱과 Flask 서버 연동
Expo 앱에서 Ai를 사용하는 가장 간단한 방법일 것이다.  
빠르게 알아보자.  
Flask 서버를 사용하여 AI 모델을 REST API로 노출하고, Expo 앱에서 이를 호출하여 결과를 낼 수 있다.  

## Flask 서버 생성
먼저, Flask 서버 와 종속성을 설치 해 주자.  
```pip install Flask torch torchvision``` 이 코드를 통해 할 수 있다.  
아래의 코드로 모델을 불러와 서버를 열어보자.  
간단한 예제로 MNIST 숫자 분류 모델을 사용해 보았다.  
Flask 서버 코드 
```
from flask import Flask, request, jsonify
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# 간단한 모델 정의 (여기서는 MNIST 숫자 분류 모델)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 모델 초기화 및 로드 (여기서는 미리 학습된 가중치가 있다고 가정)
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))  # 모델 가중치를 로드
model.eval()

# 전처리 함수
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])
    image = transform(image_bytes)
    image = image.view(-1, 28*28)  # Flatten the image
    return image

# API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = transform_image(file)
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
    return jsonify({'prediction': prediction.item()})

if __name__ == '__main__':
    app.run(debug=True)
```

위는 간단한 MNIST 숫자 분류 모델 활용 예제이다.  
Flask 서버를 통해 이미지를 받아 숫자를 예측하는 API를 생성한다.  
모델 가중치는 'model.pth' 파일에 저장된 상태라고 가정하고 만든거다.  

이어서, Expo 앱 설정을 하고, Flask 서버와 연동시키자.   

## Expo 앱 설정
먼저, Expo 앱을 생성 한다.  
```bash
npx create-expo-app MyAIApp
cd MyAIApp
```

이어서, 필요한 패키지를 설치해 주자.  
```bash
npx expo install expo-image-picker
npm install axios
```

Expo  앱 코드는 다음과 같이 하자.  
```javascript
import React, { useState } from 'react';
import { Button, Image, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

export default function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const pickImage = async () => {
    // 이미지 선택 요청
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.uri);
    }
  };

  const uploadImage = async () => {
    if (!image) return;

    // Flask 서버에 이미지 업로드
    let formData = new FormData();
    formData.append('file', {
      uri: image,
      name: 'photo.jpg',
      type: 'image/jpg',
    });

    try {
      const response = await axios.post('http://<your-server-ip>:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Button title="Pick an image from camera roll" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
      <Button title="Upload to Flask Server" onPress={uploadImage} />
      {prediction !== null && <Text>Prediction: {prediction}</Text>}
    </View>
  );
}
```

## 실행
먼저, Flask 서버를 실행하자.  
기본적으로 http://localhost:5000 에서 실행된다.  
Expo 앱과 통신하려면 로컬 네트워크에서 서버에 접근 가능하도록 IP 주소로 연결을 설정해야 한다.  
Expo 앱을 실행하고, 이미지를 선택하여 서버에 업로드하면 예측 결과를 받을 수 있다.  

이렇게 하면 Flask 서버와 Expo 앱 간의 통신을 통해 AI 모델을 활용할 수 있습니다.