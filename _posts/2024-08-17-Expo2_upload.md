---
layout: single
title:  "Expo 버전 오류 해결"
categories: "Expo"
tag: "code, Expo"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Expo 버젼
Expo에 이것저것 ㅅ설치해서 연동해서 사용하게 되면, 버젼 오류가 날 수 있다.  
이번엔 expo-camera와 @tensorflow/tfjs-react-native의 버젼 간의 호환이 되지 않아서 @tensorflow/tfjs-react-native가 재대로 설치되지 못하였고, 이에 따른 빌드에도 실패 하는 결과가 나왔다. 이를 해결하기 위해선 버젼을 호환되는 것으로 고쳐줄 필요가 있다.  
아 참고로, Expo가 진짜로 좋은게 만일 빌드중에 실패 했어도, 로그인을 하고 진행했다면 그 로그가 남아서,  
나중에 사이트에 들어가 어디서 오류가 난건지 엄청 쉽에 알 수 있다.  

#### expo-camera 버전 조정
먼저, 원래 깔려 있었던 Expo를 지워준다.  
```sh
npm uninstall expo-camera
```
이어서, 호환되는 버젼의 카메라 앱을 깔아준다.  
```sh
npm install expo-camera@13.4.4
```

#### @tensorflow/tfjs-react-native 설치
위 단계를 통해서 버젼이 호환 가능 버젼이 되었으므로, 설티가 완활하게 될 것이다.  
```sh
npm install @tensorflow/tfjs-react-native
```

#### --force 또는 --legacy-peer-deps
만일, 위 단계에서 설치에 실패 하였다면, 이 방법도 있다.  
하지만 이는 의존성을 무시하고 설치한느 것이기 때문에, 빌드에서 실패할 가능성이 높다.  
```sh
npm install @tensorflow/tfjs-react-native --legacy-peer-deps
```

의존성 충돌이 해결되엇는지 확인하는 방법은 간단히 package.json를 확인하면 된다.  
이후, 다시 빌드해서 성공하는지 확인해 보자.  