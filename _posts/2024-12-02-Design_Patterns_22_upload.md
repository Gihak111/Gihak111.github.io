---
layout: single
title:  "디자인 패턴 시리즈 22. 프록시 패턴"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 22: 프록시 패턴 (Proxy Pattern)

프록시 패턴(Proxy Pattern)은 구조적 디자인 패턴으로, 대리인 역할의 객체를 사용하여 실제 객체에 대한 접근을 제어하는 방식이다.  
프록시는 클라이언트와 실제 객체 사이의 인터페이스 역할을 하며, 접근 제어, 캐싱, 로깅, 지연 초기화와 같은 기능을 제공한다.

## 프록시 패턴의 필요성

소프트웨어에서 직접 객체에 접근할 때 다음과 같은 문제가 발생할 수 있다:

1. 보안 및 접근 제어: 민감한 데이터나 리소스에 대한 접근 제한이 필요하다.  
2. 성능 최적화: 불필요한 객체 생성을 피하거나, 리소스를 효율적으로 사용해야 한다.  
3. 추가 작업 처리: 접근 전에 로깅, 인증, 캐싱 등의 추가 작업이 필요하다.  

프록시 패턴은 이러한 문제를 해결하기 위해 프록시 객체를 통해 실제 객체를 간접적으로 제어한다.  

### 예시: 이미지 로드 최적화  

대형 이미지 파일을 표시하는 애플리케이션에서, 이미지를 사용하기 전까지 실제 이미지를 로드하지 않도록 설계할 수 있다.  

## 프록시 패턴의 구조  

1. Subject (주체): 실제 객체와 프록시 객체가 공유하는 인터페이스를 정의한다.  
2. RealSubject (실제 객체): 주된 작업을 수행하는 실제 객체를 구현한다.  
3. Proxy (프록시): 실제 객체에 대한 대리 객체로, 접근 제어 또는 추가 기능을 제공한다.  

### 구조 다이어그램  

```
Client → Proxy → RealSubject
           ↘ Subject
```

### 프록시 패턴 동작 순서  

1. 클라이언트는 `Proxy` 객체를 통해 작업을 요청한다.  
2. `Proxy`는 요청을 처리하거나, 필요시 `RealSubject`에 요청을 전달한다.  
3. 클라이언트는 `Proxy`와 상호작용하며, 실제 객체의 존재를 알 필요가 없다.  

## 프록시 패턴 예시

이번 예시에서는 "이미지 로드 최적화"를 프록시 패턴으로 구현한다.

### Java로 프록시 패턴 구현하기

```java
// Subject 인터페이스
interface Image {
    void display();
}

// RealSubject 클래스
class RealImage implements Image {
    private String fileName;

    public RealImage(String fileName) {
        this.fileName = fileName;
        loadFromDisk();
    }

    private void loadFromDisk() {
        System.out.println(fileName + " 로드 중...");
    }

    @Override
    public void display() {
        System.out.println(fileName + " 화면에 표시 중...");
    }
}

// Proxy 클래스
class ProxyImage implements Image {
    private RealImage realImage;
    private String fileName;

    public ProxyImage(String fileName) {
        this.fileName = fileName;
    }

    @Override
    public void display() {
        if (realImage == null) {
            realImage = new RealImage(fileName);
        }
        realImage.display();
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        Image image1 = new ProxyImage("photo1.jpg");
        Image image2 = new ProxyImage("photo2.jpg");

        // 이미지를 처음 요청할 때만 로드됨
        image1.display();
        System.out.println();

        // 두 번째 호출에서는 로드 과정을 생략
        image1.display();
        System.out.println();

        // 다른 이미지 로드
        image2.display();
    }
}
```

### 코드 설명  

1. Subject (주체): `Image` 인터페이스는 `display` 메서드를 정의한다.  
2. RealSubject (실제 객체): `RealImage` 클래스는 실제 이미지를 로드하고 표시하는 작업을 수행한다.  
3. Proxy (프록시): `ProxyImage` 클래스는 `RealImage` 객체의 로드를 지연시키고, 필요할 때만 로드한다.  
4. Client (클라이언트): `Main` 클래스는 프록시 객체를 사용하여 이미지를 요청한다.  

### 출력 결과  

```
photo1.jpg 로드 중...
photo1.jpg 화면에 표시 중...

photo1.jpg 화면에 표시 중...

photo2.jpg 로드 중...
photo2.jpg 화면에 표시 중...
```

### 프록시 패턴 활용

1. 가상 프록시: 리소스가 무거운 객체의 지연 초기화를 위해 사용.  
2. 보호 프록시: 접근 제어를 위해 사용.  
3. 원격 프록시: 네트워크 상의 객체를 로컬에서 제어할 때 사용.  
4. 캐싱 프록시: 반복된 요청에 대해 결과를 캐싱.  

## 프록시 패턴의 장점  

1. 객체 생성 지연: 필요할 때만 객체를 생성하여 성능 최적화.  
2. 접근 제어: 보안, 인증, 로깅 등의 기능을 추가할 수 있다.  
3. 구조적 유연성: 클라이언트와 실제 객체 간의 결합도를 낮춘다.  

## 프록시 패턴의 단점  

1. 복잡성 증가: 프록시 객체와 실제 객체를 따로 관리해야 하므로 구조가 복잡해질 수 있다.  
2. 성능 문제: 모든 요청이 프록시를 거치기 때문에, 프록시 구현이 비효율적일 경우 성능이 저하될 수 있다.  

### 마무리  

프록시 패턴(Proxy Pattern)은 객체에 대한 접근을 제어하거나, 추가 기능을 캡슐화하는 데 유용하다.  
특히, 리소스가 무거운 객체의 생성을 지연시키거나, 민감한 리소스에 대한 접근을 제한할 때 유용하다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  
