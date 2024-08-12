---
layout: single
title:  "한국어 자연어 처리"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 자연어 처리
일반적으로 영어를 기본으로 제공하지만, 어떤 상황에선 한국어를 데이터로 사용해야 할 경우가 있다.  
konlpy의 oky를 통해 쉽게 구현할 수 있지만,  
환경을 구성하는데에 많은 오류가 있어서 이를 정리했다.  
설치 과정은 다음과 같다.  
1. JAVA 설치  
2. JAVA_HOME 환경변수 설정  
3. JPype 다운로드 및 설치  
4. konlpy 설치  

## JAVA 설치
```okt = Okt()``` 자바가 없으면, 이 파트에서 오류가 나곤 한다.  
JVM 파일을 찾을 수 없어서 발생하는 에러이다.  
JVM은 Java Virtual Machine으로, 결국 JAVA를 설치하면 해결되는 오류이다.  
https://www.oracle.com/kr/java/technologies/downloads/  
위 링크에서 JDK를 설치한다.  


## JAVA_HOME 환경변수 설정  
시스템 환경변수 편집에 들어가서, 환경변수를 설정한다.  
java\jdk-버전 이 있는 위치를 환병변수로 지정하면 된다.  

## JPype 다운로드 및 설치  
konlpy는 JPype가 필요하다.  
https://github.com/jpype-project/jpype 이 파일을 다운 받아서 setup.py를 실행하자.  
cd 를 통해 클로한 폴더로 가서
```cmd
python setup.py build
python setup.py install
```
위의 두 코드를 통해 설치할 수 있다.  

## konlpy 설치 
위 과정을 전부 해야 비로소 이걸 설치 할 수가 있다.  
```cmd
pip install konlpy
```
위 코드를 통해 다운로드 하자.  

위 과정을 통해, 다운받고 실행 할 수 있다.  