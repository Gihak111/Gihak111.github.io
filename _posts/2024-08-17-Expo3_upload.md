---
layout: single
title:  "Expo 프로젝트 생성 밑 APK 생성"
categories: "Expo"
tag: "code, Expo"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Expo
정말 간단하게 모바일 앱을 만들수 있다.  
간단하게, 어떻게 앱을 만들고, 실행하는지 프로젝트를 만드는 과정을 간단하게 cmd 명령어로 알아보자.  

#### 프로젝트 생성
```cmd
expo init CameraApp
```
위 코드를 통해 CameraApp을 생선할 수 있다.  

```
cd CameraApp
```
위 코드르 통해 폴더 위치를 옯기고,  

```
expo install expo-camera @react-navigation/native
```
이런 식으로 의존성을 추가할 수 있다.  

이어서, App.js를 수정해서 원하는 로직을 집어넣을 수 있다.  

#### 앱 실행
윈도우에서 모바일 앱을 실행하려면 에뮬레이터를 통해서 모바일 가상 환경을 구축해 주야 한다.  
엄청 어렵진 않지만, 오류도 많고 하니 Expo 앱을 통해 진행하는 방법을 사용해 보자.  
이 방법은  자바 스크립트의 번들링 오류를 재현하고, 진단하는데에 효과적이다.  
```cmd
npx expo start --no-dev
```
이 코드를 통해 나오는 Qr코드를 모바일에서 다운받은 Expo 앱으로 실행시키면 된다.  
오류가 나면 오류 로그가 나오기 때문에, 디버깅에도 유용하며, 앱이 잘 싱행되는지 내 핸드폰 기기로 쉽게 확인할 수 있다.  
하지만, 이 방법으로 로컬에서 실행이 잘 되어도, 빌드에 실패할 경우가 있으니, 이럴땐 Expo 사이트에 접속해서 빌드 오류 로그를 통해서 디벅깅 하자.    

#### 앱 빌드
eas.json을 다음과 같이 수정해서 apk 파일로 만들 수 있다.  
```json
{
  "build": {
    "production": {
      "android": {
        "buildType": "apk"
      }
    }
  }
}
```

빌드 하기 위해 필요한 것들을 먼저 받아준다.  
```cmd
npm install -g eas-cli
```

이제 빌드할 수 있다.  
윈도우 환경에서는 슬프게도 iso 빌드는 불가하다.  

안드로이드로 빌드 (에뮬레이터)
```cmd
eas build --platform android
```

안드로이드 빌드 (APK)
```cmd
eas build --profile production --platform android
```

위 두 코드를 통해서 서로 다른 방식으로 빌드 할 수 있다. 
애뮬레이터가 없는 경우, APK를 바로 빌드 하는게 속편하다.  
링크가 나오면 그 링크를 통해서 앱을 다운 받을 수 있다.  
이어서 나오는 선택지로 에뮬레이터로도 실행할 수 있다.  
암튼, 모바일에서 위 링크로 접속해 Apk를 다운 받고, 앱을 설치하고 실행 하는 것으로 쉡게 모바일 앱을 만들 수 있다.  
하지만, 이렇게 만드는 앱은 공갈빵 같은 텅 빈 앱 이므로 src 폴더를 통해 백앤드를 만드는 과정을 나중에 정리하도록 하겠다.  
아 참고로, 구글 앱 스토어 같온 곳에 앱을 출시 할 떄는 보톤 AAB 형태로 만든다. 이것도 나중에 정리하겠다.  