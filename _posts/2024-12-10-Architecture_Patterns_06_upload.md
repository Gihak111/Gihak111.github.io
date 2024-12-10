---
layout: single
title:  "아키텍처 패턴 시리즈 6. 피어 투 피어 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 6: 피어 투 피어 패턴 (Peer-to-Peer Pattern)

피어 투 피어(Peer-to-Peer, P2P) 패턴은 각 노드가 서로 대등한 지위에서 통신하고 자원을 공유하는 네트워크 구조를 말한다.  
각 피어가 동시에 클라이언트와 서버의 역할을 수행하며, 중앙 서버 없이 직접적인 데이터 교환이 이루어진다.

## 피어 투 피어 패턴의 필요성

P2P 패턴은 여러 노드가 독립적으로 서로 데이터를 교환할 때 유용하며, 중앙 서버가 필요 없는 구조적 장점을 제공한다.

1. 분산 처리: 각 노드가 직접 데이터를 제공하고, 받아오며, 처리할 수 있다.  
2. 중앙 서버 의존성 제거: 서버에 대한 의존성을 줄이고 네트워크 부하를 분산할 수 있다.  
3. 확장성: 새로운 피어를 추가하는 것만으로 네트워크 확장이 가능하다.

P2P 패턴은 파일 공유 시스템, 메시징 서비스, 분산 컴퓨팅 시스템에서 널리 사용된다.  

### 예시: 파일 공유 시스템

P2P 패턴을 이용하면 파일을 저장하고 있는 피어들 간에 데이터를 직접 주고받으면서 파일을 공유할 수 있다.  

## 피어 투 피어 패턴의 구조

1. Peer (피어): 데이터를 요청하고 동시에 제공하는 노드로, 클라이언트와 서버의 역할을 모두 수행한다.  
2. Request & Response (요청 및 응답): 피어가 요청하는 데이터와 응답 데이터.  
3. P2P Network (네트워크): 각 피어가 연결되어 있는 네트워크로, 데이터를 공유하는 매개체이다.  

### 구조 다이어그램

```
   Peer <-----> Peer
      ^          |
      |          v
   Peer <-----> Peer
```

### 피어 투 피어 패턴 동작 순서

1. 피어는 네트워크 상의 다른 피어에게 필요한 데이터를 요청한다.  
2. 요청받은 피어는 자신의 데이터를 조회하여 요청한 데이터를 전달한다.  
3. 각 피어는 다른 피어에게 데이터 요청을 보낼 수도, 받을 수도 있다.  

## 피어 투 피어 패턴 예시

파일 공유 시스템에서 각 피어가 직접 데이터를 요청하고 전달하는 방식으로 P2P 패턴을 구현할 수 있다.  

### Java로 피어 투 피어 패턴 구현하기

```java
// Peer 클래스: 다른 피어에게 데이터를 요청하고 응답을 받을 수 있는 클래스
import java.util.HashMap;
import java.util.Map;

public class Peer {
    private String name;
    private Map<String, String> dataStore = new HashMap<>();

    public Peer(String name) {
        this.name = name;
    }

    public void storeData(String key, String data) {
        dataStore.put(key, data);
    }

    public String requestData(String key, Peer targetPeer) {
        return targetPeer.respondToRequest(key);
    }

    private String respondToRequest(String key) {
        return dataStore.getOrDefault(key, "데이터를 찾을 수 없습니다.");
    }

    public String getName() {
        return name;
    }
}
```

```java
// Main 클래스: 피어들이 데이터를 저장하고 요청을 처리하는 시스템을 시연
public class Main {
    public static void main(String[] args) {
        Peer peer1 = new Peer("Peer1");
        Peer peer2 = new Peer("Peer2");

        peer1.storeData("file1", "Peer1의 파일 데이터");
        peer2.storeData("file2", "Peer2의 파일 데이터");

        String response1 = peer1.requestData("file2", peer2);
        String response2 = peer2.requestData("file1", peer1);

        System.out.println(peer1.getName() + "가 요청한 데이터: " + response1); // 출력: Peer2의 파일 데이터
        System.out.println(peer2.getName() + "가 요청한 데이터: " + response2); // 출력: Peer1의 파일 데이터
    }
}
```

### 코드 설명

1. Peer: `Peer` 클래스는 데이터를 저장하고 다른 피어에게 요청을 보낼 수 있는 기능을 제공한다.  
2. storeData: 피어가 특정 데이터를 저장한다.  
3. requestData: 피어가 다른 피어에게 데이터를 요청한다.  
4. respondToRequest: 요청받은 피어가 데이터를 반환하거나, 데이터를 찾을 수 없다는 메시지를 반환한다.  

### 출력 결과

```
Peer1가 요청한 데이터: Peer2의 파일 데이터
Peer2가 요청한 데이터: Peer1의 파일 데이터
```

### 피어 투 피어 패턴 활용

1. 파일 공유 시스템: P2P 네트워크를 통해 피어들 간의 파일을 직접 공유할 수 있다.  
2. 메시징 시스템: 피어들이 서로 메시지를 주고받으며 통신할 수 있다.  
3. 분산 컴퓨팅: 피어들 간의 작업을 분산하여 처리하는 분산 컴퓨팅 환경에서 활용 가능하다.  

## 피어 투 피어 패턴의 장점

1. 확장성: 피어를 쉽게 추가할 수 있으며, 네트워크 확장이 용이하다.
2. 중앙 서버의 부재: 중앙 서버가 필요 없어, 서버 비용과 관리가 줄어든다.  
3. 효율적인 자원 분배: 피어들이 직접 데이터를 주고받으며 네트워크 부하를 분산할 수 있다.  

## 피어 투 피어 패턴의 단점

1. 데이터 관리의 어려움: 데이터가 분산되어 있어 일관성 유지와 관리가 어렵다.  
2. 신뢰성 문제: 신뢰할 수 없는 피어의 참여로 인해 데이터 무결성과 보안이 위협받을 수 있다.  
3. 복잡성: 피어들 간의 연결과 통신이 복잡해질 수 있다.  

### 마무리

피어 투 피어 패턴은 시스템 확장성과 자원 분배의 효율성을 제공하는 패턴으로, 중앙 서버가 필요 없는 네트워크 환경에서 유용하게 활용된다.  
분산된 데이터 교환이 필요할 때 적합한 구조이다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
