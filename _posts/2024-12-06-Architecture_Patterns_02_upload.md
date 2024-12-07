---
layout: single
title:  "아키텍처 패턴 시리즈 2. 클라이언트-서버 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 2: 클라이언트-서버 패턴 (Client-Server Pattern)

클라이언트-서버 패턴(Client-Server Pattern)은 네트워크 아키텍처 패턴으로, 클라이언트와 서버라는 두 가지 주요 구성 요소가 서로 요청과 응답을 주고받는 구조이다.  
이 패턴은 응용 프로그램이 서버에서 데이터를 제공하고, 클라이언트가 이를 요청하여 사용하는 방식으로 설계된다.  

## 클라이언트-서버 패턴의 필요성

네트워크 기반의 소프트웨어에서 클라이언트와 서버의 역할을 분리할 때 다음과 같은 이점이 있다:  

1. 중앙 집중식 데이터 관리: 데이터와 리소스를 서버에 모아 관리하고, 보안성과 일관성을 높일 수 있다.  
2. 확장성: 서버를 독립적으로 확장할 수 있어 성능을 최적화할 수 있다.  
3. 유연한 애플리케이션 구조: 클라이언트와 서버를 분리하여 유지보수와 업데이트가 용이하다.  

클라이언트-서버 패턴은 이러한 이점을 활용하여 네트워크를 통해 데이터와 리소스를 제공하고 소비하는 시스템을 설계할 수 있게 한다.

### 예시: 웹 애플리케이션 아키텍처

웹 애플리케이션에서 클라이언트(브라우저)는 사용자 요청을 서버(백엔드)로 보내고, 서버는 해당 요청을 처리하여 응답을 반환한다.  

## 클라이언트-서버 패턴의 구조  

1. Client (클라이언트): 서버에 요청을 보내는 역할을 하며, 사용자 인터페이스와 상호작용한다.
2. Server (서버): 클라이언트의 요청을 처리하고, 데이터를 제공하는 중앙 서버 역할을 한다.
3. Network (네트워크): 클라이언트와 서버가 통신할 수 있는 네트워크 인프라를 제공한다.

### 구조 다이어그램

```
Client ↔ Network ↔ Server
```

### 클라이언트-서버 패턴 동작 순서

1. 클라이언트가 서버에 특정 요청을 전송한다.
2. 서버는 요청을 처리하고, 필요한 데이터를 조회하거나 로직을 실행한다.
3. 서버는 요청에 대한 결과를 클라이언트에 응답으로 반환한다.

## 클라이언트-서버 패턴 예시

이번 예시에서는 클라이언트가 서버에 특정 데이터를 요청하고, 서버가 이를 응답하는 방식으로 구현한다.

### Java로 클라이언트-서버 패턴 구현하기

```java
// 서버 측 코드
import java.io.*;
import java.net.*;

public class Server {
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(6789);
        System.out.println("서버가 시작되었습니다.");

        Socket connectionSocket = serverSocket.accept();
        BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
        DataOutputStream outToClient = new DataOutputStream(connectionSocket.getOutputStream());

        String clientMessage = inFromClient.readLine();
        System.out.println("클라이언트로부터 받은 메시지: " + clientMessage);

        String response = "서버에서 받은 메시지: " + clientMessage;
        outToClient.writeBytes(response + '\n');

        connectionSocket.close();
        serverSocket.close();
    }
}
```

```java
// 클라이언트 측 코드
import java.io.*;
import java.net.*;

public class Client {
    public static void main(String[] args) throws IOException {
        Socket clientSocket = new Socket("localhost", 6789);

        DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
        BufferedReader inFromServer = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));

        String message = "Hello Server!";
        outToServer.writeBytes(message + '\n');

        String response = inFromServer.readLine();
        System.out.println("서버로부터 받은 응답: " + response);

        clientSocket.close();
    }
}
```

### 코드 설명

1. Server (서버): 서버는 클라이언트의 연결 요청을 받고, 수신된 메시지를 출력한 뒤 응답을 보낸다.
2. Client (클라이언트): 클라이언트는 서버에 연결하여 메시지를 전송하고, 서버의 응답을 출력한다.
3. Network (네트워크): 클라이언트와 서버는 소켓을 통해 네트워크 상에서 통신한다.

### 출력 결과

서버 측:
```
서버가 시작되었습니다.
클라이언트로부터 받은 메시지: Hello Server!
```

클라이언트 측:
```
서버로부터 받은 응답: 서버에서 받은 메시지: Hello Server!
```

### 클라이언트-서버 패턴 활용

1. 웹 애플리케이션: 클라이언트-서버 패턴을 기반으로 브라우저와 웹 서버 간의 요청/응답 구조를 구현한다.  
2. 게임 서버: 다수의 클라이언트가 게임 서버에 접속하여 실시간으로 데이터를 교환한다.  
3. 데이터베이스 서버: 클라이언트가 데이터베이스 서버에 데이터를 저장하거나 조회할 때 사용한다.

## 클라이언트-서버 패턴의 장점

1. 데이터 관리 효율성: 데이터를 서버에 집중하여 관리할 수 있어, 보안 및 일관성을 높일 수 있다.
2. 유지보수 용이성: 클라이언트와 서버를 독립적으로 업데이트할 수 있어, 애플리케이션 유지보수가 쉬워진다.
3. 확장성: 서버 확장만으로도 성능을 개선할 수 있어 유연한 확장이 가능하다.

## 클라이언트-서버 패턴의 단점

1. 네트워크 의존성: 네트워크 상태에 따라 성능이 저하될 수 있다.
2. 서버 부하: 서버에 부하가 집중되기 때문에 트래픽이 많은 경우 성능 이슈가 발생할 수 있다.
3. 구현 복잡성: 네트워크 연결과 통신을 구현해야 하므로 시스템 설계가 복잡해질 수 있다.

### 마무리

클라이언트-서버 패턴(Client-Server Pattern)은 네트워크 환경에서 리소스와 데이터 관리를 위한 구조적 솔루션을 제공한다.  
특히, 중앙 서버를 통한 데이터 관리가 필요하거나, 다수의 클라이언트가 동일한 서버에 접근해야 하는 경우에 적합하다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
