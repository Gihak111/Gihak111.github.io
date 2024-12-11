---
layout: single
title:  "아키텍처 패턴 시리즈 7. 이벤트 기반 패턴 / 이벤트-버스 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 7: 이벤트 기반 패턴 / 이벤트-버스 패턴 (Event-Driven Pattern / Event-Bus Pattern)

이벤트 기반 패턴(Event-Driven Pattern)은 시스템 내에서 발생하는 이벤트를 중심으로 컴포넌트들이 독립적으로 상호작용할 수 있게 하는 아키텍처 패턴이다.  
이벤트-버스(Event-Bus)는 이러한 이벤트가 발생했을 때 전달될 수 있도록 이벤트를 전파하는 매개체 역할을 수행한다.  

## 이벤트 기반 패턴의 필요성

이벤트 기반 패턴은 비동기적인 상호작용이 필요한 시스템에서 주로 사용되며, 컴포넌트 간의 결합도를 낮춰 독립적인 동작이 가능하도록 해준다.  

1. 비동기 작업 처리: 실시간 데이터나 요청을 처리할 수 있는 비동기 환경이 필요할 때 유용하다.  
2. 결합도 감소: 이벤트 중심으로 작동하기 때문에 시스템의 모듈성을 높일 수 있다.  
3. 확장성: 이벤트 구독자와 발행자를 쉽게 추가하여 시스템을 확장할 수 있다.  

이벤트 기반 패턴은 채팅 애플리케이션, 주문 처리 시스템, IoT 시스템 등 다양한 분야에서 활용된다.  

### 예시: 주문 처리 시스템

주문 처리 시스템에서 주문이 들어오면 이벤트가 발생하고, 해당 이벤트를 감지한 다른 서비스가 이를 처리한다.  

## 이벤트 기반 패턴의 구조

1. Event: 특정 상황이 발생했음을 나타내는 객체로, 다른 컴포넌트에 전달될 수 있다.  
2. Event Emitter (이벤트 발행자): 이벤트를 발생시키는 컴포넌트이다.  
3. Event Listener (이벤트 리스너): 특정 이벤트에 반응하여 동작을 수행하는 컴포넌트이다.  
4. Event Bus: 이벤트를 전달하고 관리하는 중개자 역할을 한다.  

### 구조 다이어그램

```
[Event Emitter] ----> [Event Bus] ----> [Event Listener]
```

### 이벤트 기반 패턴 동작 순서

1. 이벤트 발행자가 특정 이벤트를 이벤트 버스에 전달한다.  
2. 이벤트 버스는 이를 구독하고 있는 리스너에게 이벤트를 전달한다.  
3. 이벤트 리스너는 전달받은 이벤트를 기반으로 작업을 수행한다.  

## 이벤트 기반 패턴 예시

주문 처리 시스템에서 `OrderPlacedEvent`가 발생하면 이를 감지한 리스너가 주문을 확인하고, 처리를 시작하는 시나리오를 Java로 구현할 수 있다.  

### Java로 이벤트 기반 패턴 구현하기

```java
// Event 클래스: 발생한 이벤트를 나타내는 클래스
public class Event {
    private String message;

    public Event(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
```  

```java
// EventListener 인터페이스: 특정 이벤트가 발생했을 때 수행할 작업을 정의
public interface EventListener {
    void onEvent(Event event);
}
```  

```java
// EventBus 클래스: 이벤트를 리스너에게 전달하는 중개자 역할을 하는 클래스
import java.util.ArrayList;
import java.util.List;

public class EventBus {
    private List<EventListener> listeners = new ArrayList<>();

    public void register(EventListener listener) {
        listeners.add(listener);
    }

    public void publish(Event event) {
        for (EventListener listener : listeners) {
            listener.onEvent(event);
        }
    }
}
```  

```java
// OrderService 클래스: 이벤트 발행자 역할을 하는 주문 처리 서비스 클래스
public class OrderService {
    private EventBus eventBus;

    public OrderService(EventBus eventBus) {
        this.eventBus = eventBus;
    }

    public void placeOrder(String orderDetails) {
        System.out.println("주문이 접수되었습니다: " + orderDetails);
        Event orderEvent = new Event("OrderPlacedEvent: " + orderDetails);
        eventBus.publish(orderEvent);  // 이벤트 버스를 통해 이벤트 발행
    }
}
```  

```java
// NotificationService 클래스: 주문이 접수되면 알림을 보내는 리스너 클래스
public class NotificationService implements EventListener {
    @Override
    public void onEvent(Event event) {
        System.out.println("알림: " + event.getMessage() + "에 대한 처리 시작");
    }
}
```  

```java
// Main 클래스: 주문을 처리하고 알림을 전송하는 시스템 시연
public class Main {
    public static void main(String[] args) {
        EventBus eventBus = new EventBus();

        // 리스너 등록
        NotificationService notificationService = new NotificationService();
        eventBus.register(notificationService);

        // 이벤트 발행
        OrderService orderService = new OrderService(eventBus);
        orderService.placeOrder("상품 A, 수량 2");
    }
}
```  

### 코드 설명

1. Event: 이벤트의 기본 클래스.  
2. EventListener: 이벤트 발생 시 수행할 동작을 정의하는 인터페이스.  
3. EventBus: 이벤트 발행자와 리스너 간의 중개자 역할.  
4. OrderService: 주문을 접수하고 이벤트를 발행하는 클래스.  
5. NotificationService: 이벤트 발생 시 알림을 전송하는 리스너 클래스.  

### 출력 결과

```
주문이 접수되었습니다: 상품 A, 수량 2
알림: OrderPlacedEvent: 상품 A, 수량 2에 대한 처리 시작
```  

### 이벤트 기반 패턴 활용

1. 주문 처리 시스템: 주문이 발생할 때마다 이벤트를 전파하여 필요한 처리를 수행할 수 있다.  
2. IoT 시스템: 센서의 상태 변화에 따른 이벤트를 감지하여 시스템에서 처리할 수 있다.  
3. 채팅 애플리케이션: 메시지가 수신될 때마다 이벤트를 발생시켜 다른 사용자에게 알림을 전달할 수 있다.  

## 이벤트 기반 패턴의 장점

1. 비동기 처리: 이벤트 기반으로 동작하기 때문에 비동기 작업 처리가 가능하다.  
2. 확장성: 이벤트 발행자와 리스너를 쉽게 추가하여 기능을 확장할 수 있다.  
3. 모듈성 향상: 각 컴포넌트가 독립적으로 작동하여 시스템의 모듈성이 높아진다.  

## 이벤트 기반 패턴의 단점

1. 복잡성 증가: 이벤트 흐름이 많아지면 전체 시스템의 복잡도가 높아질 수 있다.  
2. 디버깅 어려움: 이벤트의 흐름을 추적하기가 어렵고, 예측하지 못한 이벤트 발생 시 디버깅이 어려울 수 있다.  
3. 지연 문제: 이벤트 처리에 시간이 걸릴 수 있으며, 동기 처리보다 지연될 수 있다.  

### 마무리

이벤트 기반 패턴은 모듈 간 결합도를 줄이고 독립적으로 상호작용을 가능하게 하며 비동기 처리가 필요한 시스템에서 유용하다.  
특히 실시간 데이터 처리와 비동기적 메시징 시스템에서 자주 활용된다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
