---
layout: single
title:  "디자인 패턴 시리즈 14. 중재자"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 14: 중재자 패턴 (Mediator Pattern)

중재자 패턴(Mediator Pattern)은 객체 간의 복잡한 통신을 캡슐화하여 객체 간의 상호작용을 중재자 객체가 담당하도록 만드는 행동 패턴이다.  
이를 통해 객체들이 직접적으로 상호작용하는 대신 중재자를 통해 통신함으로써 객체 간의 의존성을 줄이고 시스템의 유연성을 높일 수 있다.

## 중재자 패턴의 필요성

객체 지향 프로그래밍에서는 서로 연관된 여러 객체가 직접적으로 통신하면, 객체 간의 의존성이 높아지고 시스템 구조가 복잡해지는 문제가 발생할 수 있다.  
이럴 때 중재자 패턴을 사용하면 다음과 같은 이점을 얻을 수 있다:

1. 의존성 감소: 객체 간의 직접적인 의존성을 제거하고 중재자를 통해 간접적으로 상호작용함으로써 의존성을 줄인다.  
2. 유지보수 용이성: 객체 간의 의존성이 줄어들어 수정 및 확장이 용이해진다.  
3. 구조 단순화: 복잡한 상호작용 로직을 중재자 객체에 위임하여 코드의 가독성과 관리성을 향상시킨다.  

### 예시: 채팅 애플리케이션

예를 들어, 채팅 애플리케이션에서 여러 사용자가 채팅방에 참여해 메시지를 주고받는 상황을 생각해보자.  
사용자들이 서로 직접 통신하면 복잡성이 증가하지만, 중재자 패턴을 적용해 채팅방 객체를 중재자로 사용하면 이를 간단하게 구현할 수 있다.

## 중재자 패턴의 구조

1. Mediator(중재자): 객체 간의 통신을 정의하는 인터페이스 또는 추상 클래스.  
2. ConcreteMediator(구체적인 중재자): Mediator를 구현하며 객체 간의 구체적인 상호작용을 관리한다.  
3. Colleague(동료): Mediator와 상호작용하는 객체들의 공통 인터페이스.  
4. ConcreteColleague(구체적인 동료): Colleague를 구현한 구체적인 객체로, Mediator를 통해 다른 동료들과 상호작용한다.  

### 구조 다이어그램

```
    Mediator
       +
       |
ConcreteMediator
       +
       |
  Colleague
       +
       |
ConcreteColleague
```

### 중재자 패턴 동작 순서

1. ConcreteColleague는 ConcreteMediator에 요청을 전달한다.  
2. ConcreteMediator는 요청을 처리하거나, 다른 ConcreteColleague 객체와 상호작용하여 요청을 처리한다.  
3. 모든 객체 간의 통신은 ConcreteMediator를 통해 이루어진다.  

## 중재자 패턴 예시

이번 예시에서는 채팅방(Mediator)을 중재자로 사용하여 여러 사용자가 메시지를 주고받는 채팅 애플리케이션을 구현해보겠다.  

### Java로 중재자 패턴 구현하기

```java
// Mediator 인터페이스
interface ChatRoom {
    void showMessage(String user, String message);
}

// ConcreteMediator 클래스
class ChatRoomImpl implements ChatRoom {
    @Override
    public void showMessage(String user, String message) {
        System.out.println(user + ": " + message);
    }
}

// Colleague 추상 클래스
abstract class User {
    protected ChatRoom chatRoom;
    protected String name;

    public User(ChatRoom chatRoom, String name) {
        this.chatRoom = chatRoom;
        this.name = name;
    }

    public abstract void sendMessage(String message);
}

// ConcreteColleague 클래스
class ChatUser extends User {
    public ChatUser(ChatRoom chatRoom, String name) {
        super(chatRoom, name);
    }

    @Override
    public void sendMessage(String message) {
        chatRoom.showMessage(name, message);
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        ChatRoom chatRoom = new ChatRoomImpl();

        User user1 = new ChatUser(chatRoom, "Alice");
        User user2 = new ChatUser(chatRoom, "Bob");
        User user3 = new ChatUser(chatRoom, "Charlie");

        user1.sendMessage("안녕하세요!");
        user2.sendMessage("안녕하세요, Alice!");
        user3.sendMessage("모두 반갑습니다.");
    }
}
```

### 코드 설명

1. ChatRoom (Mediator): 채팅 메시지를 표시하는 인터페이스.  
2. ChatRoomImpl (ConcreteMediator): Mediator를 구현한 채팅방으로, 사용자들 간의 메시지를 중재한다.  
3. User (Colleague): Mediator와 상호작용하는 사용자 객체의 추상 클래스.  
4. ChatUser (ConcreteColleague): User를 구체화한 클래스. Mediator를 통해 메시지를 다른 사용자들에게 전달한다.  

### 출력 결과

```
Alice: 안녕하세요!
Bob: 안녕하세요, Alice!
Charlie: 모두 반갑습니다.
```  

## 중재자 패턴의 장점

1. 객체 간의 의존성 감소: 객체가 서로 직접 참조하지 않으므로, 코드가 더 유연하고 모듈화된다.  
2. 복잡성 감소: 중재자를 통해 상호작용을 캡슐화하여 코드의 가독성을 높이고, 유지보수를 용이하게 만든다.  
3. 재사용성 향상: 동료 객체와 중재자 객체를 독립적으로 변경할 수 있어 코드의 재사용성이 향상된다.  

## 중재자 패턴의 단점

1. 중재자의 복잡성 증가: 중재자에 많은 책임이 집중되면, 중재자 자체가 복잡해질 수 있다.  
2. 초기 설계 비용 증가: 객체 간의 통신을 캡슐화하려면 초기 설계 단계에서 더 많은 노력이 필요하다.

### 마무리

중재자 패턴(Mediator Pattern)은 객체 간의 상호작용을 캡슐화하여 복잡성을 줄이고, 유지보수와 확장성을 향상시키는 데 유용한 패턴이다.  
특히 많은 객체가 서로 상호작용해야 하는 상황에서 의존성을 효과적으로 줄이는 데 적합하다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  