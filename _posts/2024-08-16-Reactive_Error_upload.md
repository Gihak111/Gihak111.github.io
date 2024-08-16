---
layout: single
title:  "리액트 네이티브 플러그인 버젼 오류"
categories: "spring"
tag: "ERROR"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 리액트 네이티브 플러그인
리액트 네이티브에 의 존성을 추가하고, 라이브러리를 가져오는 가장 간단한 방법은 플러그인을 추가하는 것이다.  
```cmd
npm install @tensorflow/tfjs --legacy-peer-deps
```
위 코드를 통해서 버젼을 무시하고 강제로 플러그인을 추가할 수 있다.  
하지만, 이런식으로 플러그인을 반복해서 집어넣으면, 오류가 생기기 마련이다.  
그렇다고, 플러그인이 요구하는 버젼이 서도 달라 충돌이 날 수 있다.  
그럴때 ㄴ다음과 같은 방법을 통해 해결할 수 있다.  

예를들어, 서로 요구하는 자바의 버젼이 다를경우, 다음과 같은 방법으로 해결할 수 있다.  
- **다른 버젼의 자바 다운**  
    ```cmd
    jenv add C:\Program Files\Java\jdk-11
    jenv add C:\Program Files\Java\jdk-17
    ```
- **프로젝트별 JDK  버젼 설정**
    ```cmd
    jenv local 11
    ```
-**gradle.properties 파일에서 JDK 설정**
    특정 프로젝트에서 사용하는 JDK 버전을 지정할 수 있다.  
    gradle.properties  
    ```properties
    org.gradle.java.home=C:/Program Files/Java/jdk-17
    ```
    위의 과정으로 프로젝트 마다 다른 버젼의 자바를 사용할 수 잇다.  

하지만, 일반적으로 하나의 앱에서 여러 버젼의 자바를 사용하는것은 일반적이지 않으므로 피하는것이 좋다.  