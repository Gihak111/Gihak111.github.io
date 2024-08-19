---
layout: single
title:  "Expo 프로젝트 백앤드 서버가 내려가도 프론트가 내려가지 않는 이유"
categories: "Expo"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 백앤드와 프론트의 통신  
우리는 이미 스프링 부트를 공부 하며서, API 게이트 웨이를 통해서 백앤드와 프론트 앤드가 통싱하게 하는 방법을 배웠다.  
Expo도 마찬가지다. 백앤드와 프론트 앤드를 서로 독립적이며, 통신을 하며 작동한다.  
앱의 기본 동작 원리를 보며  자세히 보자.  

#### 앱의 기본 동작 원리  
앞서 말 했다 시피, 프론트 엔드와 백엔드는 서로 독립적이다.  
프론트엔드는 사용자 인터페이스를 담당하고, 백엔드는 데이터 처리 및 API를 제공한다.  
프론트엔드는 네이티브 앱이거나 웹 애플리케이션으로 빌드될 수 있으며, 백엔드는 서버에서 실행되는 애플리케이션이다.  

앱은 기본적으로 이렇게 실행된다.  
Expo 앱(프론트엔드)은 자체적으로 실행되며, 사용자 인터페이스를 렌더링 한다.  
사용자가 앱을 열면, UI 요소들이 표시되고, 사용자는 앱과 상호작용할 수 있다.  
이 과정에서 API 호출은 특정 이벤트(예: 버튼 클릭, 화면 로드 등)가 발생할 때 이루어 진다.  

#### 백엔드 서버와의 연동  
API 호출의 비동기성  
API 호출은 네트워크 요청을 통해 백엔드 서버와 통신한다.  
만약 백엔드 서버가 실행 중이지 않으면, API 호출은 실패하지만, 앱은 계속 실행되는 것이다.  
API 호출이 실패하면 오류 메시지를 표시하거나 기본값을 사용하도록 설정할 수 있다.  
이는 예외처리의 영역인데, 프론트엔드 코드에서 API 호출 실패에 대한 적절한 에러 처리를 해야 한다는 거다.  
예를 들어, API 호출이 실패하면 사용자에게 오류 메시지를 보여주거나, 기본 데이터를 사용하여 앱을 계속 작동시킬 수 있다.  

예시를 통해 알아보자.  
#### 예시
App.js
```javascript
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Alert } from 'react-native';
import axios from 'axios';

export default function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    // 백엔드 API 호출
    axios.get('http://localhost:5000/api/example')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.error('There was an error!', error);
        // 오류가 발생하면 기본 메시지 설정 또는 사용자에게 알림
        setMessage('Failed to load data');
        Alert.alert('Error', 'Failed to connect to the server');
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

위 코드는 백앤드 API를 프론트 엔드에서 호출하는 것을 보여준다.  
위와 같은 ㅣ유로, 백앤드가 실행하지 않아도 프론트 엔드는 자체적으로 실행된다.  