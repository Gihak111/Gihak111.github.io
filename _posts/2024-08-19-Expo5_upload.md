---
layout: single
title:  "Expo 프로젝트 백앤드 재대로 구현하기"
categories: "Expo"
tag: "code, Expo"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Expo앱 백앤드 구현
Expo 앱으 ㅣ백앤드를 만들기 위해선, Express.js와 같은 Node.jjs 기반의 백엔드 프레임워크를 사용하여 백엔드를 구축하면 된다.  
이를 Expo로 만든 프론트엔드와 연결할 수 있다.  

다음 코드를 통해서 프로젝트를 만들고, 링수 라이브러리를 설치하자.  
```cmd
# Expo CLI를 이용해 새로운 프로젝트 생성
npx create-expo-app MyApp
cd MyApp

# 백엔드용 디렉토리 생성
mkdir backend
cd backend

# npm 초기화
npm init -y

# Express.js와 필요한 라이브러리 설치
npm install express cors body-parser

# 개발용 라이브러리 설치
npm install --save-dev nodemon

```  

백앤드를 Expo 앱에 집어넣는 방법은 다음과 같은 폴더 구조로 만들면 된다.  
```arduino
MyApp/
├── backend/
│   ├── node_modules/
│   ├── src/
│   │   ├── controllers/
│   │   │   └── exampleController.js
│   │   ├── routes/
│   │   │   └── exampleRoutes.js
│   │   ├── models/
│   │   │   └── exampleModel.js
│   │   ├── app.js
│   │   └── server.js
│   ├── package.json
│   └── ...
├── node_modules/
├── App.js
├── ...
└── package.json

```  

다음과 같은 코드를 집어넣어 보자.  
backend/src/app.js  
```javascriot
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const exampleRoutes = require('./routes/exampleRoutes');

const app = express();

// 미들웨어 설정
app.use(cors()); // Cross-Origin Resource Sharing 설정
app.use(bodyParser.json()); // JSON 바디 파서 설정

// 라우터 설정
app.use('/api/example', exampleRoutes);

module.exports = app;

```

backend/src/server.js  
```javascriot
const app = require('./app');
const port = process.env.PORT || 5000;

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

```

backend/src/routes/exampleRoutes.js  
```javascriot
const express = require('express');
const router = express.Router();
const exampleController = require('../controllers/exampleController');

// GET 요청을 처리하는 예제 라우트
router.get('/', exampleController.getExample);

module.exports = router;

```

backend/src/controllers/exampleController.js  
```javascriot
// 예제 컨트롤러
exports.getExample = (req, res) => {
  res.json({ message: 'Hello from the backend!' });
};

```

위 코드를 다 채우고, 백앤드 서벌르 실행하자.  
backend/package.json  
```javascriot
"scripts": {
  "start": "node src/server.js",
  "dev": "nodemon src/server.js"
}

```

개발 모드로 백엔드를 실행하자.  
```sh
npm run dev
```

프론트 엔드 설정
Axios를 설치해서 프론트 엔드 API를 호출하자.  
```sh
cd ..
npm install axios
```

MyApp/App.js  
```javascriot
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import axios from 'axios';

export default function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    // 백엔드 API 호출
    axios.get('http://localhost:5000/api/example')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.error('There was an error!', error);
      });
  }, []);

  return (
    <View style={styles.container}>
      <Text>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

```  
위 코드로 프론트, 백앤드를 구분해서 만들 수 있다.  
테스트를 해보자.  
1. 백엔드 서버 실행: ```cd backend && npm run dev```
2. Expo 앱 실행: ```npm start (or expo start)```

위 방법으로 백앤드, 프론트 앤드를 설정할 수 잇으며, 프론트에서 백앤드의 API를 호출하여 데이터를 받아올 수 있다.  


빌드 시 백앤드 서버와 연동 해야 한다. 방법은 다음과 같다.  
1. 백엔드 서버를 로컬에서 실행시키고, 모바일 디바이스가 같은 네트워크에 연결되어 있는지 확인한다.  
  ```bash
  # 백엔드 디렉토리에서 서버 실행
  cd backend
  npm run dev
  ```
2. 백엔드 서버를 로컬에서 실행한 후, 프론트엔드의 API 호출 URL을 로컬 네트워크 IP 주소로 설정한다.  
  ```bash
  // App.js에서 axios 호출 URL을 로컬 네트워크 IP로 변경
  axios.get('http://192.168.X.X:5000/api/example')
  ```

빌드된 앱은 백엔드 서버가 실행 중이지 않아도 작동한다.  
이는 백앤드롸 프론트 앤드가 독립적으로 만들어 졌기 때문이다.  
단, API 호출이 실패할 수 있으므로 적절한 오류 처리가 필요하다.  

