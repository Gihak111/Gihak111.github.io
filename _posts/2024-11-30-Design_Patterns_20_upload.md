---
layout: single
title:  "디자인 패턴 시리즈 20. 퍼사드"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 20: 퍼사드 패턴 (Facade Pattern)

퍼사드 패턴(Facade Pattern)은 구조적 디자인 패턴으로, 복잡한 서브시스템의 인터페이스를 단순화하여 사용자가 서브시스템의 내부 복잡성을 알 필요 없이 사용할 수 있도록 돕는다.  
이 패턴은 여러 객체와 클래스의 관계를 캡슐화하여 클라이언트가 더 쉽게 시스템과 상호작용할 수 있게 한다.  

## 퍼사드 패턴의 필요성  

소프트웨어 시스템은 시간이 지나며 기능이 추가될수록 복잡해진다.  
클라이언트가 서브시스템의 모든 세부 사항을 알아야 한다면 다음과 같은 문제점이 생긴다:  

1. 코드 복잡도 증가: 클라이언트가 복잡한 시스템 세부 사항까지 이해하고 사용해야 한다.  
2. 유지보수 어려움: 클라이언트와 서브시스템 간 결합도가 높아져 변경 시 영향을 받기 쉽다.  

퍼사드 패턴은 다음과 같은 이점을 제공한다:  

1. 단순화된 인터페이스 제공: 복잡한 서브시스템을 감추고 단일 진입점을 제공한다.  
2. 결합도 감소: 클라이언트와 서브시스템 간의 의존성을 줄인다.  

### 예시: 스마트 홈 시스템  

스마트 홈 시스템은 조명, 에어컨, 커튼 등을 제어할 수 있다.  
퍼사드 패턴을 사용하면 단순한 인터페이스로 다양한 기능을 쉽게 제어할 수 있다.  

## 퍼사드 패턴의 구조

1. Facade(퍼사드): 클라이언트가 사용하는 단순화된 인터페이스를 제공한다.  
2. Subsystem(서브시스템): 복잡한 동작을 수행하는 다양한 클래스들.  

### 구조 다이어그램  

```
Client → Facade → SubsystemA
                  → SubsystemB
                  → SubsystemC
```  

### 퍼사드 패턴 동작 순서  

1. 클라이언트는 복잡한 서브시스템을 직접 호출하지 않고 퍼사드를 통해 요청한다.  
2. 퍼사드는 서브시스템의 메서드를 호출하여 작업을 수행한다.  

## 퍼사드 패턴 예시  

이번 예시에서는 "스마트 홈 시스템"을 퍼사드 패턴으로 구현해보겠다.  

### Java로 퍼사드 패턴 구현하기  

```java
// 서브시스템 클래스 1
class Light {
    public void turnOn() {
        System.out.println("조명이 켜졌습니다.");
    }

    public void turnOff() {
        System.out.println("조명이 꺼졌습니다.");
    }
}

// 서브시스템 클래스 2
class AirConditioner {
    public void turnOn() {
        System.out.println("에어컨이 켜졌습니다.");
    }

    public void turnOff() {
        System.out.println("에어컨이 꺼졌습니다.");
    }

    public void setTemperature(int temperature) {
        System.out.println("에어컨 온도가 " + temperature + "도로 설정되었습니다.");
    }
}

// 서브시스템 클래스 3
class Curtain {
    public void open() {
        System.out.println("커튼이 열렸습니다.");
    }

    public void close() {
        System.out.println("커튼이 닫혔습니다.");
    }
}

// Facade 클래스
class SmartHomeFacade {
    private Light light;
    private AirConditioner airConditioner;
    private Curtain curtain;

    public SmartHomeFacade() {
        this.light = new Light();
        this.airConditioner = new AirConditioner();
        this.curtain = new Curtain();
    }

    public void startMorningRoutine() {
        System.out.println("아침 루틴을 시작합니다.");
        light.turnOn();
        curtain.open();
        airConditioner.setTemperature(22);
    }

    public void startNightRoutine() {
        System.out.println("야간 루틴을 시작합니다.");
        light.turnOff();
        curtain.close();
        airConditioner.turnOff();
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        SmartHomeFacade smartHome = new SmartHomeFacade();

        // 아침 루틴 실행
        smartHome.startMorningRoutine();

        System.out.println();

        // 야간 루틴 실행
        smartHome.startNightRoutine();
    }
}
```  

### 코드 설명  

1. Subsystem (서브시스템): `Light`, `AirConditioner`, `Curtain` 클래스가 복잡한 동작을 정의한다.  
2. Facade (퍼사드): `SmartHomeFacade` 클래스는 서브시스템을 통합하고 단순화된 인터페이스를 제공한다.  
3. Client (클라이언트): `Main` 클래스는 퍼사드를 통해 서브시스템을 제어한다.  

### 출력 결과  

```
아침 루틴을 시작합니다.
조명이 켜졌습니다.
커튼이 열렸습니다.
에어컨 온도가 22도로 설정되었습니다.

야간 루틴을 시작합니다.
조명이 꺼졌습니다.
커튼이 닫혔습니다.
에어컨이 꺼졌습니다.
```  

## 퍼사드 패턴의 장점  

1. 단순한 인터페이스 제공: 복잡한 서브시스템을 클라이언트가 알 필요 없이 사용할 수 있다.  
2. 결합도 감소: 클라이언트와 서브시스템 간의 의존성을 줄인다.  
3. 서브시스템 변경 용이: 서브시스템의 내부 구현이 변경되어도 클라이언트 코드에 영향을 미치지 않는다.  

## 퍼사드 패턴의 단점  

1. 추가 추상화 비용: 퍼사드 클래스를 생성하는 데 시간이 추가로 소요된다.  
2. 기능 제한: 퍼사드가 제공하지 않는 서브시스템의 기능은 클라이언트가 직접 호출해야 한다.  

### 마무리  

퍼사드 패턴(Facade Pattern)은 복잡한 서브시스템을 감추고, 단순하고 직관적인 인터페이스를 제공하여 클라이언트 코드의 복잡성을 줄인다.  
특히, 모듈화된 시스템에서 퍼사드를 사용하면 코드 가독성과 유지보수성을 높일 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  
