---
layout: single
title:  "아키텍쳐 패턴 시리즈 종류와 구분"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍쳐 패턴
우리는 디자인 패턴을 통해 로직을 객체지향적인 방법으로 효율적인 형태로 구성하는 법을 배웟다.  

## 목적
사실 디자인 패턴의 원리를 로직 단위에서 소프트웨어 시스템 단위로 올리는게 전부다.  
각각의 상호작용을 느슨하게, 유연하게 만드는게 주된 목적이다.  
스프링 부트 구조가 MVC 구조이다. 사실, 이제 올바른 구조를 만드는 것은 어렵지 않다.  
시간이 많이 흐르면서, 정형화된 구조들이 수도없이 쏟아져 나왔고, 이 구조를 구형하는것은 이제는 쉬운일이 되었다.  
이젠 누가 더 쉬운 방법으로 그 구조를 구현하느냐 싸움이다.  
더 쉬운 구조가 나오면, 그 프레임워크가 유명해진다.  
대표적으로, docs, node.js, spring 등이 있다.  
앞으로도 계속 발전된 프레임워크가 계속 나올것이며, 이를 어떻게 구워삶느냐가 관건이다.  

1. 구조화된 설계  
2. 모듈화와 재사용  
3. 느슨한 결합과 강한 응집성  
4. 유지 보수성
5. 성능 최적화  

결국, 유연하게, 느슨하게 해 주면 되는거다.  

여러가지 아키텍쳐 패턴이 있지만, 종류는 다음으로 축약된다.  

1. [계층화 패턴 (Layered Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/05/Architecture_Patterns_01_upload.html)  
2. [클라이언트-서버 패턴 (Client-Server Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/06/Architecture_Patterns_02_upload.html)  
3. [마스터-슬레이브 패턴 (Master-Slave Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/07/Architecture_Patterns_03_upload.html)  
4. [파이프-필터 패턴 (Pipe-Filter Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/08/Architecture_Patterns_04_upload.html)  
5. [브로커 패턴 (Broker Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/09/Architecture_Patterns_05_upload.html)  
6. [피어 투 피어 패턴 (Peer-to-Peer Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/10/Architecture_Patterns_06_upload.html)  
7. [이벤트-버스 패턴 (Event-Bus Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/11/Architecture_Patterns_07_upload.html)  
8. [MVC 패턴 (Model-View-Controller Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/12/Architecture_Patterns_08_upload.html)  
9. [MVP 패턴 (Model-View-Presenter Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/13/Architecture_Patterns_09_upload.html)  
10. [MVVM 패턴 (Model-View-ViewModel Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/14/Architecture_Patterns_10_upload.html)  
11. [FLUX 패턴](https://gihak111.github.io/architecture_patterns/2024/12/15/Architecture_Patterns_11_upload.html)  
12. [블랙보드 패턴 (Blackboard Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/16/Architecture_Patterns_12_upload.html)  
13. [인터프리터 패턴 (Interpreter Pattern)](https://gihak111.github.io/architecture_patterns/2024/12/17/Architecture_Patterns_13_upload.html)  
14. [마이크로서비스 아키텍처 (Microservices Architecture)](https://gihak111.github.io/architecture_patterns/2024/12/18/Architecture_Patterns_14_upload.html)  
15. [서비스 지향 아키텍처 (Service-Oriented Architecture, SOA)](https://gihak111.github.io/architecture_patterns/2024/12/19/Architecture_Patterns_15_upload.html)  
16. [헥사고날 아키텍처 (Hexagonal Architecture)](https://gihak111.github.io/architecture_patterns/2024/12/20/Architecture_Patterns_16_upload.html)  
17. [CQRS (Command Query Responsibility Segregation)](https://gihak111.github.io/architecture_patterns/2024/12/21/Architecture_Patterns_17_upload.html)  
18. 스트랭글러 패턴 (Strangler fig Pattern)  

많은 패턴들이 있지만, 위와 같은 패턴들이 대표적이다.  