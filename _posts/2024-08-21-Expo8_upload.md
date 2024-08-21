---
layout: single  
title:  "Expo 프로젝트에서 백엔드와 연동된 데이터 처리 및 관리"  
categories: "Expo"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---
# Expo 앱에서 데이터 관리와 상태 처리
Expo 앱이*Express.js 백엔드와 통신하여 데이터를 처리하고, 이를 앱 내에서 효과적으로 관리하는 방법을 알아보자.  
앞선 글에서  Expo와 Express.js를 사용해 간단한 앱과 서버를 구축했으니, 이번에는 그 연장선에서 데이터를 어떻게 다루고 효율적으로 사용할 수 있는지 살펴보자.  

Expo 앱이 백엔드 서버와 통신하여 데이터를 가져오고, 이를 앱 내에서 관리하고 처리하는 방법에 대해 알아보겠다는 거다.  
Expo에서의 데이터 관리는 **React의 상태 관리**와 **API 통신**을 중심으로 이루어진다.  
자세히 알아보자.  

## 데이터 요청과 상태 관리
### useState와 useEffect를 사용한 기본 데이터 처리

Expo 앱에서 백엔드로부터 데이터를 가져오기 위해 React의 `useState`와 `useEffect` 훅을 사용한다.  
이번에는 상태 관리와 데이터 처리 로직을 개선하는 방법을 살펴보자.  

#### 상태 관리와 비동기 데이터 로드

백엔드로부터 데이터를 받아 상태로 관리하는 기본적인 방법은 다음과 같다.  
```javascript
import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, FlatList, ActivityIndicator } from 'react-native';

export default function App() {
  const [users, setUsers] = useState([]);        // 사용자 데이터를 저장할 상태
  const [loading, setLoading] = useState(true);  // 데이터 로드 상태를 관리하는 상태

  useEffect(() => {
    fetch('http://localhost:3000/users')
      .then((response) => response.json())       // JSON으로 변환
      .then((data) => {
        setUsers(data);                          // 데이터를 상태에 저장
        setLoading(false);                       // 로딩 완료로 상태 변경
      })
      .catch((error) => {
        console.error(error);                    // 에러 발생 시 로그 출력
        setLoading(false);                       // 에러 발생 시에도 로딩 상태 종료
      });
  }, []);                                        // 컴포넌트가 마운트될 때만 실행

  if (loading) {
    return <ActivityIndicator size="large" color="#0000ff" />;  // 로딩 중일 때 로딩 스피너 표시
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>User List</Text>
      <FlatList
        data={users}                                // 상태에 저장된 데이터를 목록에 바인딩
        keyExtractor={(item) => item.id.toString()} // 고유 키로 각 아이템을 구분
        renderItem={({ item }) => (
          <Text style={styles.item}>{item.name}</Text>  // 각 사용자의 이름을 텍스트로 표시
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

위 코드를 개선해서 더 상태관리를 잘 해보자.  

#### 오류 처리 추가
데이터를 가져오는 중에 오류가 발생할 수 있다.  
이 경우, 사용자에게 적절한 오류 메시지를 표시하는 것이 중요하다. 다음과 같이 하자.  

```javascript
export default function App() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);  // 에러 상태를 추가로 관리

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch('http://localhost:3000/users');
        const data = await response.json();
        setUsers(data);
      } catch (err) {
        setError('Failed to load data');  // 에러 발생 시 상태 업데이트
      } finally {
        setLoading(false);                // 데이터 로드 완료
      }
    };

    fetchUsers();
  }, []);

  if (loading) {
    return <ActivityIndicator size="large" color="#0000ff" />;
  }

  if (error) {
    return (
      <View style={styles.container}>
        <Text style={styles.error}>{error}</Text>  {/* 에러 메시지 표시 */}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>User List</Text>
      <FlatList
        data={users}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <Text style={styles.item}>{item.name}</Text>
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
  error: {
    fontSize: 18,
    color: 'red',
  },
});
```

### 상태 관리와 데이터 흐름

Expo 앱에서는 상태 관리가 매우 중요하다.  
위에선 `useState`와 `useEffect`를 활용한 기본적인 상태 관리 방법을 보여주었지만, 실제로 더 복잡한 앱에서는 Redux, MobX, Recoil과 같은 상태 관리 라이브러리를 사용하여 상태를 보다 효율적으로 관리할 수 있다.

## API 통신 패턴 및 실시간 데이터 처리

Expo 앱에서 백엔드와의 통신은 단순한 데이터 가져오기뿐만 아니라, 데이터를 업데이트, 삭제하거나, 실시간 데이터를 처리하는 경우에도 사용된다.  
데이터 추가/업데이트/삭제시, 백엔드와의 통신을 통해 데이터를 추가하거나 수정하고, 삭제할 수도 있습니다. 이는 `POST`, `PUT`, `DELETE` 메서드를 사용하여 구현할 수 있습니다.  
예제를 통해 알아보자.  

```javascript
const addUser = async (newUser) => {
  try {
    const response = await fetch('http://localhost:3000/users', {
      method: 'POST',               // POST 메서드를 사용하여 데이터 추가 요청
      headers: {
        'Content-Type': 'application/json',  // JSON 형식으로 데이터를 전송
      },
      body: JSON.stringify(newUser),  // 새 사용자 데이터를 요청 본문에 포함
    });

    if (!response.ok) {
      throw new Error('Failed to add user');
    }

    const addedUser = await response.json();
    setUsers((prevUsers) => [...prevUsers, addedUser]);  // 상태 업데이트
  } catch (err) {
    console.error(err.message);
  }
};
```

### 실시간 데이터 처리
WebSocket이나 SSE(Server-Sent Events)를 사용하여 실시간 데이터를 처리할 수 있다.  
실시간 데이터는 채팅 앱이나 주식 가격 추적 앱과 같이 데이터의 실시간 갱신이 필요한 경우에 중요하다.  
아래 코드는 WebSocket을 사용해 실시간으로 데이터를 수신하는 예제이다.  
```javascript
import React, { useState, useEffect } from 'react';
import { Text, View } from 'react-native';

export default function App() {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3000');  // WebSocket 연결

    ws.onmessage = (event) => {
      setMessages((prevMessages) => [...prevMessages, event.data]);  // 실시간으로 메시지 수신
    };

    ws.onerror = (error) => {
      console.error('WebSocket error: ', error);  // WebSocket 오류 처리
    };

    return () => {
      ws.close();  // 컴포넌트가 언마운트될 때 WebSocket 연결 종료
    };
  }, []);

  return (
    <View>
      {messages.map((msg, index) => (
        <Text key={index}>{msg}</Text>  // 수신된 메시지를 화면에 표시
      ))}
    </View>
  );
}
```  

### Expo앱 관리
앱이 데이터 관리와 상태 처리에서 성능을 유지하면서 원활하게 동작하도록 최적화하는 방법을 알아보자.  

### 메모리 관리

React Native에서 메모리를 효율적으로 관리하는 것은 중요하다.  
불필요한 상태 업데이트나 메모리 누수를 방지하기 위해 `useEffect`의 정리(cleanup) 함수를 사용한다.  

### 배포 준비
Expo 앱을 최적화한 후, 최종적으로 앱 스토어 또는 플레이 스토어에 배포한다.  
배포 과정에서는 앱의 크기를 최소화하고, 필요한 애셋만 포함시켜 빌드하는 것이 중요하다.  

1. **코드 스플리팅**: 불필요한 코드가 포함되지 않도록 코드 스플리팅을 통해 앱의 크기를 최적화 한다.  
2. **이미지 최적화**: 이미지를 WebP 형식으로 변환하여 용량을 줄인다.  
    ```bash
    expo build:android  # Android용 APK 생성
    expo build:ios      # iOS용 IPA 생성
    ```

위의 코드가 동작하는 단걔를 한번 보자.  
위의 구조는 Expo 프론트엔드 앱과 Express.js 백엔드 서버 간의 통신을 통해 데이터를 처리하고 관리하는 방식이다.  
이 구조가 돌아가는 단계는 다음과 같다
### 1. 백엔드 서버 실행
   - Express.js 백엔드 서버는 Node.js 환경에서 실행된다.
   - 서버가 실행되면, 설정된 포트(예: `3000번 포트`)에서 HTTP 요청을 대기하게 된다.
   - 이 서버는 특정 엔드포인트(예: `/users`)에서 데이터를 제공하는 REST API를 제공한다.
   - 서버 실행 명령: `node index.js`

### 2. 프론트엔드 앱 실행
   - Expo 프론트엔드 앱은 React Native 기반의 모바일 애플리케이션 이다.
   - Expo CLI를 통해 앱이 실행되면, 에뮬레이터나 실제 모바일 기기에서 앱이 구동된다.
   - 프론트엔드 앱 실행 명령: `expo start`

### 3. 컴포넌트 마운트 및 데이터 요청
   - App 컴포넌트가 처음 렌더링될 때, `useEffect` 훅이 실행된다.
   - 이 훅은 컴포넌트가 마운트될 때 백엔드 서버에 **HTTP 요청**을 보냅니다. 예를 들어, `http://localhost:3000/users`로 GET 요청을 보내 사용자 데이터를 요청한다.
   - 이 요청은 비동기적으로 이루어지며, 네트워크 요청이 완료될 때까지 앱은 데이터를 기다린다.

### 4. 백엔드 서버의 응답
   - Express.js 서버는 클라이언트의 요청을 받으면, 요청된 엔드포인트에 맞는 데이터를 반환한다.
   - 예를 들어, `/users` 엔드포인트에 대한 GET 요청이 오면 서버는 사용자 목록을 JSON 형식으로 응답한다.

### 5. 데이터 수신 및 상태 업데이트
   - Expo 앱에서 서버로부터 응답을 받으면, `useState` 훅을 통해 받은 데이터를 상태에 저장한다.
   - 이 과정에서 데이터가 성공적으로 받아지면 `loading` 상태가 `false`로 변경되고, 오류가 발생하면 `error` 상태가 업데이트 된다.

### 6. 화면에 데이터 렌더링
   - Expo 앱은 서버로부터 받은 데이터를 상태로 관리하며, 이를 통해 화면에 데이터를 렌더링 한다.
   - 예를 들어, 사용자 목록 데이터를 `FlatList` 컴포넌트를 사용하여 화면에 출력 한다.
   - 만약 데이터 로드 중이라면 `ActivityIndicator`를 통해 로딩 스피너를 표시하고, 에러가 발생하면 오류 메시지를 출력 한다.

### 7. 데이터 추가, 수정, 삭제
   - 사용자가 데이터를 추가하거나 수정, 삭제하려는 요청을 하면, Expo 앱은 해당 요청을 백엔드 서버로 보냅니다. 예를 들어, 새로운 사용자를 추가할 때는 POST 요청이 서버로 전송 된다.
   - 서버는 요청에 따라 데이터를 추가, 수정, 삭제하고, 결과를 다시 클라이언트에 응답 한다.
   - 클라이언트는 응답을 받고 상태를 업데이트하여 화면에 즉시 반영 한다.

### 8. 실시간 데이터 처리 (선택 사항)
   - 만약 앱이 실시간 데이터를 처리해야 한다면, **WebSocket** 또는 **SSE**를 사용해 서버와 지속적인 연결을 유지한다.
   - 서버는 실시간으로 데이터를 클라이언트에 전송하고, 클라이언트는 이를 상태에 반영해 즉시 화면에 표시한다.

### 9. 최종 배포
   - Expo 앱이 완성되면, 앱을 최적화한 후 실제 배포 환경(예: Android, iOS)으로 빌드하여 앱 스토어에 배포한다.  
   - Express.js 서버는 클라우드 서버나 호스팅 서비스에 배포하여 모바일 앱이 언제든지 서버와 통신할 수 있도록 설정한다.  

이러한 단계들이 결합되어 Expo 앱과 Express.js 서버 간의 데이터 처리와 관리가 원활하게 이루어진다.  
각 단계는 독립적으로 작동하지만, 느슨히 연결되어 앱의 전체 기능을 구성한다.  