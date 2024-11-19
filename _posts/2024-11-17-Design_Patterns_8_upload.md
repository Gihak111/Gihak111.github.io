---
layout: single
title:  "디자인 패턴 시리즈 8. 상태"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 8: 상태 패턴 (State Pattern)  

상태 패턴(State Pattern)은 객체의 상태가 바뀔 때, 그에 따라 객체의 행동(behavior)도 달라지도록 하는 패턴이다.  
이를 통해 상태에 따른 조건문을 줄이고, 상태별로 행동을 개별 클래스로 분리할 수 있다.  
즉, 객체의 상태 변화에 따라 행동이 변하는 경우에 유용하게 사용된다.  

## 상태 패턴의 필요성  

객체가 여러 상태를 가지며, 상태에 따라 다르게 동작해야 하는 경우, 상태 패턴을 사용하면 상태에 따른 조건문을 객체의 내부 구조로 옮길 수 있다.  
만약 상태가 추가되거나 변경되더라도 코드 변경이 최소화되며, 유지보수가 쉬워진다.  

### 예시: 문(Door) 상태 시스템  

문을 예로 들어보자. 문은 여러 가지 상태를 가질 수 있다.  
예를 들어, 문이 열려 있거나, 닫혀 있거나, 잠겨 있는 상태가 될 수 있다.  
상태에 따라 행동이 달라지는데, 열려 있는 문은 닫을 수 있고, 닫혀 있는 문은 열 수 있다.  
상태 패턴을 사용하면 이러한 상태별 행동을 클래스로 분리할 수 있다.  

## 상태 패턴의 구조  

상태 패턴은 다음과 같은 구성 요소로 이루어진다:  

1. Context: 상태 변화를 관리하는 객체이다. 현재 상태를 알고 있으며, 상태에 따라 행동을 위임한다.  
2. State: 상태를 나타내는 인터페이스로, 특정 상태에서 수행할 행동을 정의한다.  
3. ConcreteState: 각 상태를 구체적으로 구현한 클래스들로, `State` 인터페이스를 구현하며 상태에 따른 행동을 정의한다.  

### 구조 다이어그램  

```
Context
   ├─ changeState(State)
   └─ request()

State
   └─ handle(Context)

ConcreteStateA
   └─ handle(Context)

ConcreteStateB
   └─ handle(Context)
```  

### 상태 패턴 동작 순서  

1. 상태 변경: Context 객체는 현재 상태를 알고 있으며, 특정 상태에서 발생하는 이벤트에 따라 상태를 변경할 수 있다.  
2. 행동 위임: Context 객체는 현재 상태에 따라 행동을 상태 객체에 위임한다. 상태 객체는 `handle()` 메서드를 통해 구체적인 동작을 수행한다.

## 상태 패턴 예시 (State Pattern)

이번 예시에서는 문 객체가 열리고 닫히는 상태를 구현해보겠다. 문은 세 가지 상태를 가질 수 있다: **열림(Open)**, **닫힘(Closed)**, **잠금(Locked)**.

### Java로 상태 패턴 구현하기

```java
// 상태 인터페이스 (State)
interface DoorState {
    void open(DoorContext door);
    void close(DoorContext door);
    void lock(DoorContext door);
    void unlock(DoorContext door);
}

// Context 클래스
class DoorContext {
    private DoorState currentState;

    public DoorContext() {
        currentState = new ClosedState(); // 처음에는 문이 닫혀 있는 상태
    }

    public void setState(DoorState state) {
        currentState = state;
    }

    public void open() {
        currentState.open(this);
    }

    public void close() {
        currentState.close(this);
    }

    public void lock() {
        currentState.lock(this);
    }

    public void unlock() {
        currentState.unlock(this);
    }
}

// 열림 상태 (ConcreteState)
class OpenState implements DoorState {
    @Override
    public void open(DoorContext door) {
        System.out.println("문은 이미 열려 있습니다.");
    }

    @Override
    public void close(DoorContext door) {
        System.out.println("문을 닫습니다.");
        door.setState(new ClosedState()); // 닫힌 상태로 전환
    }

    @Override
    public void lock(DoorContext door) {
        System.out.println("열린 문은 잠글 수 없습니다.");
    }

    @Override
    public void unlock(DoorContext door) {
        System.out.println("문은 이미 열려 있습니다.");
    }
}

// 닫힘 상태 (ConcreteState)
class ClosedState implements DoorState {
    @Override
    public void open(DoorContext door) {
        System.out.println("문을 엽니다.");
        door.setState(new OpenState()); // 열린 상태로 전환
    }

    @Override
    public void close(DoorContext door) {
        System.out.println("문은 이미 닫혀 있습니다.");
    }

    @Override
    public void lock(DoorContext door) {
        System.out.println("문을 잠급니다.");
        door.setState(new LockedState()); // 잠긴 상태로 전환
    }

    @Override
    public void unlock(DoorContext door) {
        System.out.println("문이 이미 닫혀 있습니다.");
    }
}

// 잠김 상태 (ConcreteState)
class LockedState implements DoorState {
    @Override
    public void open(DoorContext door) {
        System.out.println("잠긴 문은 열 수 없습니다.");
    }

    @Override
    public void close(DoorContext door) {
        System.out.println("문이 이미 닫혀 있고 잠겨 있습니다.");
    }

    @Override
    public void lock(DoorContext door) {
        System.out.println("문은 이미 잠겨 있습니다.");
    }

    @Override
    public void unlock(DoorContext door) {
        System.out.println("문을 잠금 해제합니다.");
        door.setState(new ClosedState()); // 닫힌 상태로 전환
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        DoorContext door = new DoorContext();

        door.open();  // 문을 엽니다.
        door.lock();  // 열린 문은 잠글 수 없습니다.
        door.close(); // 문을 닫습니다.
        door.lock();  // 문을 잠급니다.
        door.open();  // 잠긴 문은 열 수 없습니다.
        door.unlock(); // 문을 잠금 해제합니다.
        door.open();  // 문을 엽니다.
    }
}
```  

### 코드 설명  

1. 상태 인터페이스(State): `DoorState` 인터페이스는 문이 열리거나 닫히고 잠길 때의 행동을 정의한다.  
2. Context 클래스: `DoorContext`는 현재 문 상태를 관리하며, 상태 변경과 요청을 위임한다.  
3. 구체적인 상태(ConcreteState): `OpenState`, `ClosedState`, `LockedState`는 각기 다른 상태에서의 동작을 정의하며, 상태에 따라 문이 어떻게 행동하는지 나타낸다.  

### 출력 결과  

```
문을 엽니다.
열린 문은 잠글 수 없습니다.
문을 닫습니다.
문을 잠급니다.
잠긴 문은 열 수 없습니다.
문을 잠금 해제합니다.
문을 엽니다.
```  

## 상태 패턴의 장점  

1. 상태별 행위 분리: 상태에 따른 행동을 개별 클래스에 분리하여 복잡한 조건문을 제거할 수 있다.  
2. 유연한 상태 변경: 상태 변경이 자유롭고, 새로운 상태가 추가될 때 기존 코드에 영향을 미치지 않는다.  
3. 코드 가독성: 상태별 로직이 깔끔하게 분리되어, 가독성과 유지보수성이 향상된다.  

## 상태 패턴의 단점  

1. 클래스 증가: 상태별로 클래스를 추가해야 하기 때문에, 상태가 많을수록 클래스 개수가 늘어난다.  
2. 복잡성: 간단한 상태 전환에는 오히려 패턴이 복잡해질 수 있다. 적절한 경우에만 사용해야 한다.  


### 마무리  

상태 패턴은 객체가 상태에 따라 다르게 행동할 때 유용한 패턴이다.  
조건문을 많이 사용하는 코드에서 상태별 행동을 클래스로 분리하여, 코드의 유지보수성을 높이고 복잡도를 줄일 수 있는 방법을 제공한다.  
문과 같은 상태 변화가 빈번한 시스템에서 유용하게 적용될 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design/patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  