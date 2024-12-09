---
layout: single
title:  "아키텍처 패턴 시리즈 5. 브로커 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 5: 브로커 패턴 (Broker Pattern)

브로커 패턴(Broker Pattern)은 시스템의 컴포넌트들이 서로 통신할 때, 브로커 역할을 하는 중간 매개체가 메시지를 전달하거나 중계하는 구조이다.  
복잡한 시스템에서 각 컴포넌트가 독립적으로 동작할 수 있도록 하고, 직접 통신이 아닌 중개를 통해 의존성을 줄일 수 있는 장점이 있다.

## 브로커 패턴의 필요성

브로커 패턴은 다양한 컴포넌트들이 서로 직접적인 연결 없이 효율적으로 통신할 때 유용하다.

1. 의존성 감소: 컴포넌트 간 결합도를 낮추어 유연성을 높인다.  
2. 확장성: 브로커를 통해 컴포넌트를 쉽게 추가하거나 제거할 수 있다.  
3. 단일화된 통신 관리: 메시지 전송, 프로세스 간 통신 등 다양한 방식의 통신을 브로커가 중개하여 관리할 수 있다.  

브로커 패턴은 특히 분산 시스템, 메시지 큐, 원격 서비스 호출 등에서 자주 사용된다.

### 예시: 원격 서비스 호출 시스템

브로커는 클라이언트와 원격 서버 간의 통신을 중개하여, 클라이언트가 원격 서버의 위치와 무관하게 서비스를 요청할 수 있도록 한다.

## 브로커 패턴의 구조  

1. Client (클라이언트): 서비스를 요청하는 주체이다.
2. Broker (브로커): 요청을 받아서 적절한 서버에 전달하고, 결과를 다시 클라이언트에게 전달한다.
3. Server (서버): 실제 요청을 처리하는 컴포넌트로, 브로커를 통해 요청을 받아 작업을 수행한다.
4. Request & Response (요청 및 응답): 클라이언트의 요청과 서버의 응답 데이터이다.  

### 구조 다이어그램

```
  Client ---> Broker ---> Server
              <---       <---
```

### 브로커 패턴 동작 순서

1. 클라이언트가 브로커에 요청을 보낸다.
2. 브로커는 요청을 적절한 서버로 전달한다.
3. 서버는 요청을 처리하고, 결과를 브로커에게 반환한다.
4. 브로커는 서버의 응답을 클라이언트에게 전달한다.  

## 브로커 패턴 예시

이번 예시에서는 클라이언트가 브로커를 통해 서버에 메시지를 요청하고, 브로커가 이를 중개하여 응답을 받는 시스템을 구현해보자.  

### Java로 브로커 패턴 구현하기

```java
// Broker 클래스: 클라이언트의 요청을 받아서 서버에 전달하고, 응답을 클라이언트에 전달
import java.util.HashMap;
import java.util.Map;

public class Broker {
    private Map<String, Server> servers = new HashMap<>();

    public void registerServer(String key, Server server) {
        servers.put(key, server);
    }

    public String processRequest(String serverKey, String request) {
        Server server = servers.get(serverKey);
        if (server != null) {
            return server.handleRequest(request);
        } else {
            return "해당 서버를 찾을 수 없습니다.";
        }
    }
}
```

```java
// Server 인터페이스: 요청을 처리하는 서버가 구현해야 하는 메서드를 정의
public interface Server {
    String handleRequest(String request);
}
```

```java
// ConcreteServer 클래스: 실제 요청을 처리하는 서버의 구현체
public class ConcreteServer implements Server {
    private String serverName;

    public ConcreteServer(String serverName) {
        this.serverName = serverName;
    }

    @Override
    public String handleRequest(String request) {
        return serverName + " 서버가 요청을 처리합니다: " + request;
    }
}
```

```java
// Main 클래스: 브로커와 서버를 설정하고, 클라이언트 요청을 처리
public class Main {
    public static void main(String[] args) {
        Broker broker = new Broker();

        Server server1 = new ConcreteServer("서버1");
        Server server2 = new ConcreteServer("서버2");

        broker.registerServer("server1", server1);
        broker.registerServer("server2", server2);

        String response1 = broker.processRequest("server1", "안녕하세요");
        String response2 = broker.processRequest("server2", "브로커 패턴 예제");

        System.out.println(response1); // 출력: 서버1 서버가 요청을 처리합니다: 안녕하세요
        System.out.println(response2); // 출력: 서버2 서버가 요청을 처리합니다: 브로커 패턴 예제
    }
}
```

### 코드 설명

1. Broker: `Broker` 클래스는 클라이언트의 요청을 받아서 적절한 서버에 전달하고, 응답을 클라이언트에 반환한다.  
2. Server: `Server` 인터페이스는 서버가 구현할 `handleRequest` 메서드를 정의한다.  
3. ConcreteServer: `ConcreteServer`는 실제 요청을 처리하는 서버의 구현체로, 요청을 받아서 응답을 생성한다.
4. Main: `Broker`에 서버들을 등록하고, 각 서버로 요청을 전달하여 응답을 처리한다.  

### 출력 결과

```
서버1 서버가 요청을 처리합니다: 안녕하세요
서버2 서버가 요청을 처리합니다: 브로커 패턴 예제
```  

### 브로커 패턴 활용

1. 메시지 큐 시스템: 클라이언트와 서버 간의 메시지를 큐에 저장하고 중개하는 시스템에서 활용된다.  
2. 분산 컴퓨팅: 원격 서버들 간의 통신을 관리하며, 서비스 요청을 중개한다.  
3. 비동기 서비스 호출: 브로커를 통해 비동기적으로 요청을 처리하여 응답을 관리할 수 있다.  

## 브로커 패턴의 장점

1. 유연한 시스템 확장: 서버를 쉽게 추가하거나 교체할 수 있어 확장성이 높다.  
2. 의존성 분리: 클라이언트와 서버가 직접 연결되지 않아 서로 독립적으로 개발 및 변경이 가능하다.  
3. 통신 관리 효율성: 브로커가 모든 통신을 관리하므로 시스템의 복잡성을 줄일 수 있다.  

## 브로커 패턴의 단점

1. 브로커 병목 현상: 브로커에 요청이 집중될 경우 병목 현상이 발생할 수 있다.  
2. 단일 장애 지점: 브로커에 장애가 발생하면 전체 시스템이 영향을 받을 수 있다.  
3. 추가 오버헤드: 중간 매개체인 브로커를 통해 통신하기 때문에 성능에 약간의 오버헤드가 발생한다.  

### 마무리

브로커 패턴(Broker Pattern)은 컴포넌트 간 통신을 효율적으로 관리할 수 있어 확장성과 유지보수성이 높은 시스템을 설계할 때 유용하다.  
컴포넌트가 독립적으로 동작하면서도 복잡한 통신을 필요로 할 때 유용하게 사용할 수 있다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
