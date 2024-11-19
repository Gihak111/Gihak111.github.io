---
layout: single
title:  "디자인 패턴 시리즈 10. 커맨드"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 10: 명령 패턴 (Command Pattern)  

명령 패턴(Command Pattern)은 요청을 객체로 변환하여 호출자(Invoker)와 수신자(Receiver)를 분리하는 디자인 패턴이다.  
호출자는 구체적으로 어떤 작업이 수행될지 알 필요가 없으며, 그저 요청을 명령 객체로 만들어서 전달할 수 있다.  
이를 통해 요청의 실행, 취소, 대기 등의 작업을 객체로 캡슐화하고 처리할 수 있게 된다.  

## 명령 패턴의 필요성  

보통 프로그램은 클라이언트(호출자)가 수신자에게 특정 작업을 요청한다.  
그런데 호출자가 수신자와 긴밀히 연결되어 있으면 유지보수가 어려워질 수 있다.  
또한 요청의 실행이나 취소를 쉽게 처리하고자 할 때, 요청 자체를 독립적인 객체로 관리하는 것이 유용하다.  

명령 패턴을 사용하면 다음과 같은 장점을 얻을 수 있다:  

1. 요청 자체를 객체로 관리할 수 있어, 다양한 요청을 큐에 저장하거나 실행을 지연하는 것이 가능하다.  
2. 요청의 취소 및 재실행이 가능해진다.  
3. 호출자와 수신자를 분리하여 코드의 유연성을 높일 수 있다.  

### 예시: 스마트 홈 시스템  

스마트 홈 시스템에서 여러 장치를 제어하는 경우, 각 장치에 대한 명령을 객체로 변환하여 호출자와 수신자를 분리할 수 있다.  
예를 들어, 전등을 켜고 끄는 작업이나, 에어컨을 켜고 끄는 작업을 명령 객체로 캡슐화하여 단일 인터페이스로 관리할 수 있다.  

## 명령 패턴의 구조  

1. Command 인터페이스: 모든 명령 객체가 구현해야 하는 공통 인터페이스로, `execute()` 메서드를 포함한다.  
2. ConcreteCommand: 구체적인 명령을 정의하며, `Command` 인터페이스를 구현한다. 이 클래스는 수신자(Receiver) 객체와 연관되며, 그 객체에서 실제 작업이 수행된다.  
3. Invoker: 명령 객체를 실행시키는 주체로, 요청을 전달하는 역할을 한다.  
4. Receiver: 실제로 작업을 수행하는 객체이다. `ConcreteCommand`는 `Receiver`와 연관되어 그 작업을 수행한다.  
5. Client: 명령 객체를 생성하고, 그것을 `Invoker`에게 전달하는 역할을 한다.  

### 구조 다이어그램  

```
Client
   └─ Invoker
        └─ Command (인터페이스)
               ├─ ConcreteCommand1
               └─ ConcreteCommand2
Receiver
   └─ 구체적인 작업 수행
```  

### 명령 패턴 동작 순서  

1. 클라이언트(Client)는 명령 객체(Command)를 생성하고, 이를 호출자(Invoker)에게 전달한다.  
2. 호출자(Invoker)는 전달받은 명령 객체에 대해 `execute()` 메서드를 호출하여, 명령을 실행한다.  
3. 명령 객체는 수신자(Receiver)에게 작업을 요청하고, 실제 작업은 수신자가 수행한다.  

## 명령 패턴 예시 (Command Pattern)  

스마트 홈 시스템에서 전등을 켜고 끄는 명령을 관리하는 예시를 통해 명령 패턴을 이해해보자.  

### Java로 명령 패턴 구현하기  

```java
// Command 인터페이스
interface Command {
    void execute();
}

// Receiver: 실제로 작업을 수행하는 전등 클래스
class Light {
    public void turnOn() {
        System.out.println("전등이 켜졌습니다.");
    }

    public void turnOff() {
        System.out.println("전등이 꺼졌습니다.");
    }
}

// ConcreteCommand: 전등을 켜는 명령
class TurnOnLightCommand implements Command {
    private Light light;

    public TurnOnLightCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.turnOn();
    }
}

// ConcreteCommand: 전등을 끄는 명령
class TurnOffLightCommand implements Command {
    private Light light;

    public TurnOffLightCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.turnOff();
    }
}

// Invoker: 명령을 실행하는 호출자
class RemoteControl {
    private Command command;

    public void setCommand(Command command) {
        this.command = command;
    }

    public void pressButton() {
        command.execute();
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        // 수신자(Receiver)
        Light light = new Light();

        // 명령 객체 생성
        Command turnOn = new TurnOnLightCommand(light);
        Command turnOff = new TurnOffLightCommand(light);

        // 호출자(Invoker)
        RemoteControl remote = new RemoteControl();

        // 전등 켜기
        remote.setCommand(turnOn);
        remote.pressButton(); // 출력: 전등이 켜졌습니다.

        // 전등 끄기
        remote.setCommand(turnOff);
        remote.pressButton(); // 출력: 전등이 꺼졌습니다.
    }
}
```  

### 코드 설명  

1. Command 인터페이스: `execute()` 메서드를 정의하여 명령 객체들이 이를 구현하도록 한다.  
2. Light (Receiver): 실제 작업을 수행하는 전등 객체로, 전등을 켜고 끄는 기능을 제공한다.  
3. TurnOnLightCommand와 TurnOffLightCommand (ConcreteCommand): 각각 전등을 켜고 끄는 명령 객체로, `execute()` 메서드를 통해 `Light` 객체의 동작을 호출한다.  
4. RemoteControl (Invoker): 명령 객체를 설정하고, 버튼을 눌러 명령을 실행하는 호출자 역할을 한다.  

### 출력 결과  

```
전등이 켜졌습니다.
전등이 꺼졌습니다.
```  

## 명령 패턴의 장점  

1. **호출자와 수신자의 분리**: 호출자는 수신자가 무엇을 어떻게 수행하는지 알 필요가 없으므로, 호출자와 수신자가 느슨하게 결합된다.
2. **확장성**: 새로운 명령을 쉽게 추가할 수 있다. 예를 들어, 에어컨을 켜고 끄는 명령을 추가하려면 `AirConditioner` 수신자와 새로운 명령 클래스만 추가하면 된다.
3. **요청의 큐 관리 및 실행 취소**: 명령 객체는 큐에 저장하거나 실행 취소(Undo) 작업을 쉽게 구현할 수 있다.

## 명령 패턴의 단점

1. **클래스의 수 증가**: 명령마다 별도의 클래스가 필요하므로, 명령의 종류가 많아질수록 클래스가 늘어날 수 있다.
2. **복잡성 증가**: 명령을 캡슐화하는 과정에서 코드의 복잡성이 다소 증가할 수 있다.

### 마무리

명령 패턴은 요청을 캡슐화하여 호출자와 수신자를 분리하고, 요청의 실행, 취소, 큐 관리 등을 유연하게 처리할 수 있는 패턴이다.  
이 패턴은 단일 명령을 독립적으로 관리해야 할 때, 또는 여러 명령을 조합하여 복합 명령을 처리해야 할 때 매우 유용하다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design/patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  