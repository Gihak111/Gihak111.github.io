---
layout: single
title:  "리엑트 네이티브 빌드시 패키지에 따른 오류"
categories: "ERROR"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# background fetch
백그라운드 패치.  
보통 2개의 라이브러리로 구현할 수 있다.  
```bash
react-native-background-fetch
react-native-background-task
```
저 두 놈은 항상 엄청난 오류를 가져온다.  
다른 패키지랑 맨날 퉁돌나고, 맨날 빌드 실패하고, 패키지도 다운되고 취약점 엄청나게 나온다.
```bash
up to date, audited 1158 packages in 2s

153 packages are looking for funding
  run `npm fund` for details

# npm audit report

plist  <=3.0.4
Severity: critical
Prototype pollution in Plist before 3.0.5 can cause denial of service - https://github.com/advisories/GHSA-4cpg-3vgw-4877
Depends on vulnerable versions of xmldom
fix available via `npm audit fix`
node_modules/plist
node_modules/simple-plist/node_modules/plist
  react-native-background-fetch  2.0.6 - 2.5.4
  Depends on vulnerable versions of plist
  Depends on vulnerable versions of xcode
  node_modules/react-native-background-fetch
  simple-plist  <=1.3.0
  Depends on vulnerable versions of plist
  node_modules/simple-plist
    xcode  0.8.3 - 1.1.0
    Depends on vulnerable versions of simple-plist
    node_modules/xcode


xmldom  *
Severity: critical
Misinterpretation of malicious XML input - https://github.com/advisories/GHSA-h6q6-9hqw-rwfv
xmldom allows multiple root nodes in a DOM - https://github.com/advisories/GHSA-crh6-fp67-6883
Misinterpretation of malicious XML input - https://github.com/advisories/GHSA-5fg8-2547-mr8q
fix available via `npm audit fix`
node_modules/xmldom

5 critical severity vulnerabilities

To address all issues, run:
  npm audit fix
```


위의 오류처럼, 취약점이 많이 뜨고, 이 취약점은 빌드시 오류를 발생시킬 수 있다.  
저거 링크 들어가서 뭘 한다 해도 뭐가 돌아가는게 없다.  
React Native로 앱을 개발하다 보면 주기적으로 취약점이 발생할 수 있으며, 이러한 취약점은 보안적인 문제뿐만 아니라 앱의 성능에도 영향을 미칠 수 있다.  
또한 백그라운드 작업을 처리해야 하는 경우, 안드로이드와 iOS의 플랫폼별 정책을 이해하고 적절한 방법을 선택하는 것이 중요하다.  
이번 글에서는 ```npm audit```을 통해 발생한 취약점을 수정하는 방법과 React Native에서 백그라운드 작업을 관리하는 방법을 깊이 있게 다뤄보자.  

## 취약점 해결을 위한 npm audit의 역할

Node.js 생태계에서 패키지 관리 도구로 많이 사용하는 npm은 패키지의 보안 취약점을 확인하는 기능을 제공한다.  
특히 `npm audit` 명령어를 통해 프로젝트에 설치된 패키지의 보안 취약점을 검사하고, 이를 `npm audit fix` 명령어로 자동으로 수정할 수 있다.  

### 1. 취약점 자동 수정

대부분의 취약점은 `npm audit fix`를 통해 간단하게 해결할 수 있다.  
이 명령어는 프로젝트의 **package-lock.json** 파일을 수정하여 가능한 경우 취약점이 포함된 패키지의 버전을 업데이트한다.  

```bash
npm audit fix
```

자동으로 수정 가능한 취약점은 대부분 패치된 패키지를 사용함으로써 해결할 수 있으며, 보안 취약점으로부터 프로젝트를 보호할 수 있다.  
당연히, 위 방법만으로 모든 문제르 ㄹ해결할 순 없다.
### 2. 수동으로 취약한 패키지 업데이트

하지만 모든 취약점이 자동으로 수정되지 않을 수 있다. 이 경우 특정 패키지를 직접 수동으로 업데이트해야 한다.  
아래는 자주 발생하는 취약점이 있는 패키지들을 업데이트하는 예시이다:  

- **plist** 패키지를 최신 버전으로 업데이트:

    ```bash
    npm install plist@latest
    ```

- **react-native-background-fetch** 패키지를 수동으로 업데이트:

    ```bash
    npm install react-native-background-fetch@latest
    ```

- **xmldom** 패키지 업데이트:

    ```bash
    npm install xmldom@latest
    ```

이처럼 문제가 발생한 특정 패키지를 수동으로 업데이트한 후, 다시 `npm audit`을 실행해 취약점이 해결되었는지 확인해야 한다.  
하지만 당연히, 해결안된다 진짜 너무 아름다운 패키진것 같다.  

### 3. 취약점 해결이 어려운 경우

모든 패키지를 최신 버전으로 업데이트한 후에도 여전히 취약점이 남아 있다면, 해당 패키지의 최신 버전이 아직 문제를 해결하지 못했을 가능성이 정말 크다.  
이 경우 패키지의 GitHub 저장소에서 관련 이슈를 확인하거나 직접 해결 방안을 찾아보는 것이 좋다.  
솔직히, 패키지가 서로 충돌나면, 관련된 기능과 패키지 내부의 로직을 보고, 가져와서 직접 구현하거나, 대페가 가능한 다른 라이브러리로 바꾸는게 더 좋은것 같다.  
그냥 능력이 되면 직접 로직 구현하자. 그게 좋은것 같다.  

# React Native에서 백그라운드 작업 처리 방법

React Native 앱에서 특정 작업을 백그라운드에서 실행해야 하는 경우가 있다.  
예를 들어, 사용자가 앱을 종료했더라도 중요한 데이터를 서버에 전송하거나 주기적인 동작을 유지해야 할 때가 있다.  
안드로이드와 iOS는 백그라운드 작업을 처리하는 데 있어서 각기 다른 제한 사항을 두고 있으며, 이를 제대로 이해하고 사용하는 것이 필수적이다.  

## 안드로이드에서의 백그라운드 작업 처리

안드로이드는 백그라운드 작업을 지원하는 다양한 메커니즘을 제공한다. 대표적인 방법으로 **Foreground Service**, **Headless JS**, **JobScheduler** 등을 사용할 수 있다.  

### 1. Foreground Service

**Foreground Service**는 사용자가 명시적으로 인식할 수 있는 서비스로, 앱이 백그라운드에서 종료되지 않고 계속 실행되도록 한다.  
특히 안드로이드 8.0(Oreo) 이후로 백그라운드에서 장시간 실행해야 하는 작업은 Foreground Service를 사용해야 한다.  

#### 설정 방법

1. **react-native-android-foreground-service** 라이브러리를 설치한다.

    ```bash
    npm install react-native-android-foreground-service
    ```

2. 서비스 시작 코드를 작성한다:

    ```javascript
    import ForegroundService from 'react-native-android-foreground-service';

    const startService = () => {
      ForegroundService.start({
        id: 144,
        title: 'App Service',
        message: 'Your service is running',
        importance: 'high',
      });
    };
    ```

이렇게 Foreground Service를 통해 백그라운드에서 작업을 유지할 수 있다.

### 2. Headless JS

**Headless JS**는 앱이 백그라운드 상태일 때에도 JavaScript 코드를 실행할 수 있도록 도와주는 기능이다. 이는 주로 Firebase 메시지 처리와 같은 이벤트 기반 작업에 유용하다.

#### Headless Task 등록

1. **Headless JS**를 등록하고 백그라운드에서 작업을 수행할 수 있다.

    ```javascript
    import { AppRegistry } from 'react-native';

    const backgroundTask = async () => {
      console.log("Background task is running");
      // 백그라운드 작업 처리
    };

    AppRegistry.registerHeadlessTask('SomeTaskName', () => backgroundTask);
    ```

이 방식은 앱이 백그라운드에서 종료되어 있더라도 작업을 처리할 수 있게 한다.

### 3. JobScheduler

네이티브 안드로이드에서는 주기적인 작업을 예약하고 관리할 수 있는 **JobScheduler**를 제공한다.  
이를 통해 앱이 비활성화된 상태에서도 주기적인 작업을 스케줄링할 수 있다.  
이는 React Native 외부에서 네이티브 Android로 직접 작업을 예약하고 관리해야 하는 경우에 유용하다.  

---

## iOS에서의 백그라운드 작업 처리

iOS는 백그라운드에서 작업을 실행하는 데 있어 매우 엄격한 제약이 있다.  
예를 들어, **위치 추적**이나 **오디오 재생**, **VoIP**와 같은 특정 작업만이 백그라운드에서 장시간 실행될 수 있다.  
그렇지 않은 경우 대부분의 작업은 iOS에서 자동으로 종료된다.  

### 1. 푸시 알림을 통한 백그라운드 작업

iOS에서 일반적인 백그라운드 작업을 처리하는 방법은 **Firebase Cloud Messaging(FCM)**을 사용하여 푸시 알림을 통해 작업을 트리거하는 것이다.  

#### Firebase Cloud Messaging 설정

1. **@react-native-firebase/messaging** 패키지를 설치한다.  

    ```bash
    npm install @react-native-firebase/app @react-native-firebase/messaging
    ```

2. 푸시 알림을 백그라운드에서 처리하는 코드를 작성한다.  

    ```javascript
    import messaging from '@react-native-firebase/messaging';

    messaging().setBackgroundMessageHandler(async remoteMessage => {
      console.log('Message handled in the background!', remoteMessage);
    });
    ```

iOS는 알림을 통해 백그라운드 작업을 트리거할 수 있으며, 이는 사용자와 상호작용 없이도 서버로부터 받은 데이터를 처리하는 데 적합하다.  

### 2. Background Modes

iOS에서 특정한 **Background Modes**를 사용하여 백그라운드 작업을 수행할 수 있다.  
위치 기반 서비스, VoIP, 블루투스 관련 작업 등이 여기에 포함된다.  
하지만 이 기능을 남용할 경우 Apple의 앱 스토어 심사에서 거절당할 수 있으므로 신중하게 사용해야 한다.  

---

## 결론

React Native 프로젝트에서 발생할 수 있는 보안 취약점 문제와 백그라운드 작업 처리 방법을 이해하는 것은 매우 중요하다.  
npm audit를 통해 취약점을 미리 파악하고 이를 적절히 수정하는 것이 프로젝트의 안정성과 보안을 보장하는 첫 걸음이다.  
또한, 플랫폼별로 적합한 백그라운드 작업 처리 방식을 선택하여 앱의 기능을 극대화할 수 있다.  
안드로이드와 iOS의 특성을 고려한 적절한 구현 방법을 통해 백그라운드 작업을 효율적으로 처리할 수 있을 것이다.  
아 모바일 앱 하나 백그라운드 돌리는거 개어렵다.  