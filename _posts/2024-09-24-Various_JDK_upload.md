---
layout: single  
title: "여러 개의 JDK 버전 활용"  
categories: "ERROR"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---  

# 여러 개의 JDK  
안드로이드 스튜디오, CMake, 스프링부트는 서로 다른 버전의 JDK를 요구할 수 있다.  
하지만, 여러 버전의 JDK를 상황에 따라 다운받거나 환경 변수를 재설정하는 것은 귀찮다.  
간단하게 JDK 버전을 전환할 수 있는 방법을 알아보자.  

### 1. JDK 설치  
오라클에서 원하는 버전의 JDK를 다운로드합니다.  
[https://www.oracle.com/java/technologies/downloads/](https://www.oracle.com/java/technologies/downloads/)  

### 2. 스크립트 폴더 생성  
`C:\Program Files\Java` 이 위치에 다운로드한 여러 버전의 JDK 파일을 모아둔다.  

### 3. 환경 변수 추가  
환경 변수에 방금 만든 스크립트 폴더를 추가해 준다.  

### 4. BAT 파일 추가  
사용하고자 하는 버전만큼 BAT 파일을 생성해 준다.  
8과 17을 예제로 만들면, 다음과 같다.  

- **버전 8**  
`java8.bat`  
```bat
@echo off
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_361
set Path=%JAVA_HOME%\bin;%Path%
echo Java 8 activated
java -version
```

- **버전 17**  
`java17.bat`  
```bat
@echo off
set JAVA_HOME=C:\Program Files\Java\jdk-17
set Path=%JAVA_HOME%\bin;%Path%
echo Java 17 activated
java -version
```

위와 같이 입력한 후, BAT 파일을 CMD에서 실행시키면, 자바 버전이 실행시킨 BAT 파일이 가리키는 버전으로 바뀐다.