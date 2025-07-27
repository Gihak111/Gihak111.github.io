---
layout: single
title: "Next.js에서 IndexedDB로 TensorFlow 모델 저장하고 불러오기"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Next.js에서 IndexedDB로 TensorFlow 모델 저장하고 불러오기

Next.js는 쓸모 있다.  
여기에는, 무려 프론트엔드에서 브라우저 딥러닝을 할 수 있는 방법도 있다.  
이건 그냥 main.js에서 use client 갈기고, html에서 tensorflow 불러와서 사용하면 끝인 거지만 이게 은근 좋다.  
완성된 모델도 인덱스 디비에 날먹으로 저장 할 수 있어 더 좋다.  
오늘은 Next.js에서 TensorFlow.js로 만든 모델을 IndexedDB에 저장하고 불러오는 방법을 알아본다.  
이 방법을 알면 모델 로딩 속도가 빨라지고 오프라인에서도 AI를 자유롭게 활용한다.  

## 왜 IndexedDB인가

IndexedDB는 브라우저의 로컬 스토리지보다 훨씬 강력하다.  
특히 TensorFlow.js 모델 같은 큰 데이터를 다룰 때 적합하다. 용량 제한이 로컬 스토리지보다 넉넉하고, 오프라인에서도 데이터를 유지한다.  
Next.js와 TensorFlow.js를 조합해 모델을 저장하고 불러오면 사용자 경험이 좋아지고 네트워크 부담도 줄어든다.  
무엇보다 서버의 부담이 줄어드는, 사용자의 컴이 구리면 딥러닝이 느려지는 ㅋㅋㅋ 그런 상황을 만들 수 있다.  


## 1. Next.js 프로젝트 설정한다
먼저 Next.js 프로젝트를 준비한다.  
터미널을 열고 다음 명령어로 새 프로젝트를 만들거나 기존 프로젝트를 연다.

```bash
npx create-next-app@latest my-ai-app
cd my-ai-app
```

다음으로, TensorFlow.js를 설치한다.  
터미널에서 다음 명령어를 입력한다:

```bash
npm install @tensorflow/tfjs
```

이제 Next.js에서 TensorFlow.js를 사용할 준비가 끝났다

## 2. TensorFlow.js 모델 생성하거나 로드한다
TensorFlow.js로 모델을 만들거나 이미 훈련된 모델을 가져온다.  
간단한 예제로 Sequential 모델을 만든다.  
페이지 컴포넌트(예: `pages/index.js`)에 다음 코드를 추가한다.  

```javascript
import * as tf from '@tensorflow/tfjs';

const createModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [5] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
  return model;
};
```

이것은 간단한 이진 분류 모델이다.  
실제로는 더 복잡한 모델을 사용하거나 사전 훈련된 모델을 가져온다(예: `tf.loadLayersModel('https://.../model.json')`).  

## 3. IndexedDB에 모델 저장한다
이제 모델을 IndexedDB에 저장한다.  
TensorFlow.js는 `model.save('indexeddb://my-model')`로 IndexedDB에 모델을 쉽게 저장한다.  
Next.js 컴포넌트에서 이를 구현하려면 useEffect를 사용해 저장 로직을 추가한다.  

```javascript
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Home() {
  const [model, setModel] = useState(null);

  useEffect(() => {
    const initModel = async () => {
      let myModel = await tf.loadLayersModel('indexeddb://my-model').catch(() => {
        console.log('IndexedDB에 모델이 없다. 새로 만든다!');
        return createModel(); // 위에서 만든 모델 생성 함수
      });
      setModel(myModel);
      await myModel.save('indexeddb://my-model'); // 모델을 IndexedDB에 저장
      console.log('모델이 IndexedDB에 저장되었다!');
    };
    initModel();
  }, []);

  return (
    <div>
      <h1>Next.js + TensorFlow.js + IndexedDB는 최고다!</h1>
      <p>모델이 준비되었다!</p>
    </div>
  );
}
```

이 코드는 페이지가 로드될 때 IndexedDB에서 모델을 먼저 불러오려 시도하고, 없으면 새로 만들어 저장한다.  
`indexeddb://my-model`는 TensorFlow.js가 IndexedDB에 모델을 저장할 때 사용하는 URL 스키마다.  

## 4. IndexedDB에서 모델 불러와 예측한다
저장한 모델을 불러와 예측한다. 예를 들어, 사용자가 입력한 데이터를 기반으로 예측하는 버튼을 추가한다.  

```javascript
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const initModel = async () => {
      let myModel = await tf.loadLayersModel('indexeddb://my-model').catch(() => {
        console.log('IndexedDB에 모델이 없다. 새로 만든다!');
        return createModel();
      });
      setModel(myModel);
      await myModel.save('indexeddb://my-model');
      console.log('모델이 IndexedDB에 저장되었다!');
    };
    initModel();
  }, []);

  const predict = async () => {
    if (!model) return;
    const input = tf.tensor2d([[0.1, 0.2, 0.3, 0.4, 0.5]]); // 예제 입력
    const result = model.predict(input);
    const prediction = await result.data();
    setPrediction(prediction[0]);
  };

  return (
    <div>
      <h1>Next.js + TensorFlow.js + IndexedDB는 최고다!</h1>
      <button onClick={predict}>예측한다</button>
      {prediction && <p>예측 결과: {prediction}</p>}
    </div>
  );
}
```

이 코드는 버튼을 누르면 모델이 입력 데이터를 기반으로 예측하고 결과를 화면에 보여준다.  
IndexedDB 덕분에 모델을 매번 네트워크에서 불러오지 않고 빠르게 로드한다.  

## 5. 꿀팁과 주의사항
- **IndexedDB 용량**: 브라우저마다 용량 제한이 다르다. 너무 큰 모델은 서버에 저장하고 필요한 부분만 IndexedDB에 캐싱한다.  
- **오프라인 모드**: IndexedDB는 오프라인에서도 동작한다. PWA(Progressive Web App)와 함께 사용하면 정말 강력하다.  
- **에러 핸들링**: 모델 로딩 실패 같은 에러를 잘 처리한다. 위 코드에서 `.catch`로 처리한 부분을 참고한다.  
- **VS Code 터미널**: Next.js 개발 시 터미널에서 `npm run dev`를 실행하면 로컬 서버가 바로 띄워진다. 콘솔 로그도 쉽게 확인한다.  

## 결론
Next.js와 TensorFlow.js의 조합은 정말 아름다운 그런거다.  
IndexedDB까지 더하면 웹앱이 날아다닌다.  
모델을 저장하고 불러오는 과정이 이렇게 간단하다.  
아무튼, 다들 한번 해 보기 바란다.  