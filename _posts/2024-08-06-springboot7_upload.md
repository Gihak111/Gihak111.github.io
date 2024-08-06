---
layout: single
title:  "spring boot 7. 모바일 앱 개발"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 자바 앱 모바일로 변환  
자바 애플리케이션을 모바일 앱으로 변환하는 과정은 다음과 같다.  
1. REST API 구축 : 스프링 부트 애플리케이션에서 모바일 앱과 통신할 수 있는 REST API 이다.  
2. 모바일 프론트엔드 개발 : 모바일 프레임워크를 사용하여 모바일 앱의 사용자 인터페이스를 개발   
3. API 통합 : 모바일 앱에서 스프링 부트 API를 호출하여 데이터를 가져오고 전송하는 로직을 구현  
4. 앱 빌드 및 배포 : 모바일 앱을 빌드하고 앱스토어나 플레이스토어에 배포  

먼저,  
Spring Initializ로 스프링 부트 프로젝트를 생성한다.  
다음 조건으로 만들어 보자.  
Project: Maven  
Language: Java  
Group: com.example  
Artifact: mobileapi  
Name: mobileapi  
Package Name: com.example.mobileapi  

Dependencies  
Spring Web  

이제, 각 파일에 다음을 추가해 보자.
생성한 것들은 그대로 두고, controller 폴더에 다음을 만들어 보자.  
ApiController.java  
```java
package com.example.mobileapi.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ApiController {

    @GetMapping("/hello")
    public String sayHello() {
        return "Hello, World!";
    }
}

```  

위의 어플리케이션은 hello world를 출력하는 간단한 앱이다.  
이제 이 앱을 모바일에서 실행할 수 있도록 해 보다.  

# 모바일 API
이 과정에는 npm과 Node.js가 필요하다.  
다음에, 이를 설치하는 과정을 설명하도록 하겠다.  

먼저, 다음 코드를 실행하자.  
```bash
npx react-native init MyMobileApp
```  

해당 프로젝트의 구조는 다음과 같다.  
```lua
MyMobileApp/
├── __tests__/
├── android/
├── ios/
├── node_modules/
├── src/
│   └── App.js
├── .gitignore
├── App.js
├── app.json
├── babel.config.js
├── index.js
├── metro.config.js
└── package.json

```  

App.js 파일을 다음과 같이 수정하자  
```jsx
import React, { useEffect, useState } from 'react';
import { SafeAreaView, Text, StyleSheet } from 'react-native';
import axios from 'axios';

const App = () => {
  // 상태 변수를 선언하여 API 호출 결과를 저장.
  const [message, setMessage] = useState('');

  // 컴포넌트가 마운트될 때 한 번 실행되는 useEffect.
  useEffect(() => {
    // axios를 사용하여 스프링 부트 API를 호출.
    axios.get('http://10.0.2.2:8080/api/hello')
      .then(response => {
        // API 호출에 성공하면 응답 데이터를 상태 변수에 저장.
        setMessage(response.data);
      })
      .catch(error => {
        // 오류가 발생하면 콘솔에 출력.
        console.error(error);
      });
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.message}>{message}</Text>
    </SafeAreaView>
  );
};

// 스타일을 정의.
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  message: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});

export default App;

```  

안드로이드 에뮬레이터 설정
리액트 네이티브 앱이 로컬 호스트에서 실행 중인 스프링 부트 애플리케이션에 접근하려면, 안드로이드 에뮬레이터에서 로컬 호스트를 10.0.2.2로 참조해야한다.  
따라서 위의 코드의 호스트가 http://10.0.2.2:8080/api/hello로 정해져 있다.  

위의 앱을 안드로이드 에뮬레이터에서 실행한다.  
```bash
npx react-native run-android
```  
위의 과정을 통해 모바일에서 작동 가능한 애플리케이션을 만들 수 있다.  

# 앱 배포 및 서명  
Google Play Store에 앱을 배포하려면 앱을 서명해야 한다.  
Android Studio에서 키스토어 파일을 생성하고, 이를 사용하여 앱을 서명하면 된다.  

1. 키스토어 파일 생성  
```bash
keytool -genkey -v -keystore my-release-key.keystore -alias my-key-alias -keyalg RSA -keysize 2048 -validity 10000
```  

2. gradle.properties 파일 설정  
키스토어 파일 정보를 android/gradle.properties 파일에 추가  
```properties
MYAPP_RELEASE_STORE_FILE=my-release-key.keystore
MYAPP_RELEASE_KEY_ALIAS=my-key-alias
MYAPP_RELEASE_STORE_PASSWORD=********
MYAPP_RELEASE_KEY_PASSWORD=********
```  

3. app/build.gradle 파일 수정
서명 설정을 추가
```gradle
android {
    ...
    signingConfigs {
        release {
            if (project.hasProperty('MYAPP_RELEASE_STORE_FILE')) {
                storeFile file(MYAPP_RELEASE_STORE_FILE)
                storePassword MYAPP_RELEASE_STORE_PASSWORD
                keyAlias MYAPP_RELEASE_KEY_ALIAS
                keyPassword MYAPP_RELEASE_KEY_PASSWORD
            }
        }
    }
    buildTypes {
        release {
            ...
            signingConfig signingConfigs.release
        }
    }
}
```

4. 릴리즈 APK 빌드
```bash
cd android
./gradlew assembleRelease
```

위의 과정을 통해서 app-release.apk를 android/app/build/outputs/apk/release/ 위치에서 찾을 수 있다.  
이제, Google Play Store에 앱 배포을 업로드 할 수 있다.  

위의 단계를 통해 스프링 부트 애플리케이션을 REST API로 구축하고, 리액트 네이티브를 사용하여 모바일 앱으로 만든 후, 최종적으로 Google Play Store에 배포할 수 있다.  