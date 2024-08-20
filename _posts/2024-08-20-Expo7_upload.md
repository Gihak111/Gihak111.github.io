---
layout: single
title:  "Expo 프로젝트 백앤드에 어울리는 구조"
categories: "Expo"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 백앤드와 프론트엔드
Expo는 프론트엔드 모바일 애플리케이션을 개발하는 도구이다.  
주로 React Native를 기반으로 앱을 개발하게 되며,  
백엔드는 따로 구축하는 것이 일반적이다.  
이번엔, 어떤 구조로 백앤드를 구축하는게 좋은지 알아보자.  

## 백앤드 구조
### 모놀리식 백엔드 vs. 마이크로서비스 백엔드
1. 모놀리식 백엔드  
  하나의 서버에서 모든 기능을 제공하는 구조이다.  
  이 구조에서는 모든 API가 하나의 프로젝트 내에서 관리되며, 스프링 부트 또는 Express.js 같은 프레임워크를 사용하여 MVC 패턴을 적용할 수 있다.  

2. 마이크로서비스 구조
  개별 서비스들이 독립적으로 운영되는 구조이다.  
  각의 서비스는 독립적인 API를 가지며, 이 API들은 서로 통신하거나 API 게이트웨이를 통해 클라이언트와 통신할 수 있다.  

## Expo 앱에서 여러 서비스 통신 방법
  각 서비스는 별도의 API 엔드포인트를 가질 수 있으며, Expo 앱에서 각각의 서비스와 통신하기 위해 HTTP 클라이언트(예: fetch 또는 Axios)를 사용할 수 있다.  
  API 게이트웨이를 설정하여, 모든 API 호출을 중앙에서 관리할 수 있다.  
  이렇게 하면, 클라이언트는 API 게이트웨이를 통해 서비스에 접근하게 된다.  

일반적으론 Node.js/Express.js 기반의 MVC 구조로 만드는게 가장 무난하다.  
  Express.js는 Node.js 환경에서 MVC 패턴을 구현하기에 적합한 프레임워크이다.  
  스프링 부트와 유사하게 모델(Model), 뷰(View), 컨트롤러(Controller) 구조를 사용할 수 있다.  
  예를 들어, 각 서비스마다 Express.js로 작성된 독립적인 서버를 구성할 수 있습니다. 이는 마이크로서비스 아키텍처와 잘 맞는다.  

## 여러개의 서비스 병합
  Expo로 앱을 개발하고 여러 서비스를 통합하는 구조를 설정하는 가장 적합한 방법은 마이크로서비스 아키텍처를 사용하는 것이다.  
  이렇게 하면, 각 서비스는 독립적으로 개발되고 배포될 수 있으며, 유지보수 및 확장성에서 이점을 가질 수 있다.  

  API 게이트웨이를 중심으로 여러 서비스를 연결하고, 클라이언트 앱에서는 API 게이트웨이를 통해 각 서비스에 접근하는 방식을 추천한다.  
  백엔드는 각각의 서비스마다 Express.js를 사용하여 개발할 수 있으며, 이 때 각 서비스는 독립적인 MVC 구조를 가질 수 있다.  

  구조는 다음과 같은 형태이다.

  서비스 A (유저 관리 서비스)
  Node.js + Express.js로 구현
  /users 엔드포인트 제공
  데이터베이스 연동하여 사용자 정보 관리

  서비스 B (상품 관리 서비스)
  Node.js + Express.js로 구현
  /products 엔드포인트 제공
  데이터베이스 연동하여 상품 정보 관리

  API 게이트웨이
  Express.js 기반의 API 게이트웨이 서버
  클라이언트로부터의 요청을 각 서비스로 라우팅
  예: /api/users -> 유저 관리 서비스, /api/products -> 상품 관리 서비스

  Expo 앱
  fetch나 Axios를 통해 API 게이트웨이와 통신
  예: GET /api/users를 통해 사용자 정보 가져오기, GET /api/products로 상품 정보 가져오기

  위와 같은 구조로 동작하게 할 수 있다.  

  Express.js 기반의 마이크로서비스 구조가 Expo 앱과 통합하기에 가장 적합한 방식임을 알 수 있다.  
  빌드할 떄는, 프론트 엔드만 빌드가 되고, 이가 서버 API와 동작하는 것으로 실행중인 API 서버와 작동한다.  

## 직접 Express.js로 백엔드 서버 구현하기
### 먼저, Express.js로 간단한 REST API 서버를 만들어 보자.  
  다음 코드를 통해서 프로젝틀르 생성하자.  
  ```bash
  mkdir express-backend  # 프로젝트 디렉토리 생성
  cd express-backend     # 생성한 디렉토리로 이동
  npm init -y            # 기본 설정으로 package.json 파일 생성
  npm install express    # Express.js 설치
  ```

  이제, 서버 코드를 만들어 보자.  
  express-backend 디렉토리 내에 index.js 파일을 생성하고, 다음과 같이 작성한다.  
  ```javascript
  const express = require('express');  // express 모듈을 불러온다.
  const app = express();               // express 애플리케이션을 생성한다.
  const port = 3000;                   // 서버가 동작할 포트를 설정한다.

  // 사용자의 데이터를 가정한 배열을 만든다.
  const users = [
    { id: 1, name: 'Alice' },          // 사용자 1
    { id: 2, name: 'Bob' },            // 사용자 2
    { id: 3, name: 'Charlie' }         // 사용자 3
  ];

  // "/users" 엔드포인트를 통해 사용자 목록을 반환하는 API를 만든다.
  app.get('/users', (req, res) => {
    res.json(users);                   // JSON 형식으로 사용자 데이터를 클라이언트에 응답한다.
  });

  // 서버를 설정한 포트에서 시작한다.
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });
  ```

  위 만든 코드의 서버를 실행시켜보자.  
  ```bash
  node index.js  # 서버 실행
  ```

  서버가 성공적으로 실행되면 http://localhost:3000/users 에 접속했을 때 JSON 형태의 사용자 목록을 확인할 수 있다.  

### Expo로 프론트엔드 앱 구현
  Expo를 사용하여 React Native 앱을 만들어서 Express.js 서버에서 데이터를 가져오는 기능을 구현하자.  
  다음 코드를 통해서 프로젝틀르 생성하자.  
  ```bash
  npm install -g expo-cli             # Expo CLI 전역 설치
  expo init expo-frontend             # 새로운 Expo 프로젝트 생성
  cd expo-frontend                    # 프로젝트 디렉토리로 이동
  ```

  항상 하던데로, App.js 파일을 수정하자.  
  ```javascript
  import React, { useState, useEffect } from 'react';  // React와 관련된 훅을 불러온다.
  import { StyleSheet, Text, View, FlatList } from 'react-native';  // 필요한 컴포넌트와 스타일시트를 불러온다.

  export default function App() {
    const [users, setUsers] = useState([]);  // 사용자 데이터를 저장할 상태를 선언한다.

    useEffect(() => {
      // 컴포넌트가 마운트될 때 서버에서 사용자 데이터를 가져온다.
      fetch('http://localhost:3000/users')
        .then((response) => response.json())   // 서버의 응답을 JSON으로 변환한다.
        .then((data) => setUsers(data))        // 변환된 데이터를 상태에 저장한다.
        .catch((error) => console.error(error));  // 오류 발생 시 콘솔에 출력한다.
    }, []);  // 빈 배열을 두 번째 인수로 전달하여, 이 효과가 한 번만 실행되도록 한다.

    return (
      <View style={styles.container}>
        <Text style={styles.title}>User List</Text>  {/* 제목 텍스트를 표시합니다. */}
        <FlatList
          data={users}                             // 상태에 저장된 사용자 데이터를 목록에 바인딩 한다.
          keyExtractor={(item) => item.id.toString()}  // 각 항목의 키를 설정한다.
          renderItem={({ item }) => (
            <Text style={styles.item}>{item.name}</Text>  // 각 사용자 이름을 텍스트로 표시한다.
          )}
        />
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
    title: {
      fontSize: 24,
      marginBottom: 20,
    },
    item: {
      fontSize: 18,
      marginVertical: 10,
    },
  });
  ```

### 서버와 Expo 앱의 통신
네트워크 설정 확인하자.  
Expo 앱은 실제 모바일 기기 또는 에뮬레이터에서 실행된다.  
이를 위해 Express.js 서버가 동일한 네트워크 내에서 접근 가능해야 한다.  

이런식으로 독립적으로 실행시켜야 한다.  
이래야 핸드폰에 다운받은 앱의 크기를 줄일 수 있으며 완활하게 작동시킬수 있고, 유지 보수 또한 쉽게 할 수 있다.  
