---
layout: single
title:  "아키텍처 패턴 시리즈 17. CQRS (Command Query Responsibility Segregation)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 17: CQRS (Command Query Responsibility Segregation)

CQRS(Command Query Responsibility Segregation)은 명령(Command)과 조회(Query)를 분리하여 시스템 설계를 단순화하고 성능과 확장성을 높이는 아키텍처 패턴이다.  
"쓰기와 읽기의 책임 분리"라는 개념을 중심으로, 두 가지 작업의 요구사항을 독립적으로 처리한다.  

## CQRS의 필요성

다음과 같은 문제를 해결하기 위해 CQRS가 도입된다:  

1. 복잡한 비즈니스 로직: 데이터 변경(쓰기)과 데이터 조회(읽기)의 요구사항이 서로 충돌하거나 복잡성이 증가할 때.  
2. 성능 병목 현상: 동일한 데이터 모델과 데이터베이스를 쓰기와 읽기에 동시에 사용할 때 성능 저하.  
3. 확장성 제약: 쓰기와 읽기를 분리하지 않을 경우, 시스템 확장에 한계 발생.  

### 예시: 전자상거래 시스템

전자상거래 시스템에서 다음과 같은 작업이 수행된다고 가정하자:  

- 쓰기 작업: 주문 생성, 재고 업데이트.  
- 읽기 작업: 주문 내역 조회, 상품 목록 조회.  

쓰기와 읽기를 분리하면 각 작업의 요구사항에 최적화된 데이터 모델을 사용하여 성능과 확장성을 개선할 수 있다.  

## CQRS의 구조

### 주요 컴포넌트

1. 명령(Command): 데이터 변경 작업 (예: 생성, 수정, 삭제).  
2. 조회(Query): 데이터 검색 작업 (예: 조회, 필터링).  
3. 명령 모델(Command Model): 데이터 변경에 최적화된 모델.  
4. 조회 모델(Query Model): 데이터 조회에 최적화된 모델.  

### 구조 다이어그램

```
[Command Input] ---> [Command Model] ---> [Write Database]
                  \                          ^
                   \                        /
                    \                      /
                     --> [Query Model] --> [Read Database]
                              |
                        [Query Input]
```

### 동작 원리

1. 명령 모델은 데이터를 변경하며, 이를 쓰기 전용 데이터베이스에 저장.  
2. 조회 모델은 읽기 전용 데이터베이스를 사용하여 데이터를 반환.  
3. 쓰기 데이터와 읽기 데이터를 비동기적으로 동기화.  

## CQRS의 구현 예시

### 주문 관리 시스템 구현

#### 1. 명령 모델 (Command Model)

주문 생성 서비스:  

```java
// 도메인 모델: Order
public class Order {
    private String id;
    private String product;
    private int quantity;

    // 생성자 및 getter/setter
    public Order(String id, String product, int quantity) {
        this.id = id;
        this.product = product;
        this.quantity = quantity;
    }
}
```

```java
// 명령 처리 서비스
public class OrderCommandService {
    private final OrderRepository repository;

    public OrderCommandService(OrderRepository repository) {
        this.repository = repository;
    }

    public void createOrder(Order order) {
        repository.save(order);
    }
}
```

#### 2. 조회 모델 (Query Model)

**주문 조회 서비스**:

```java
// 조회 DTO
public class OrderDTO {
    private String id;
    private String product;
    private int quantity;

    public OrderDTO(String id, String product, int quantity) {
        this.id = id;
        this.product = product;
        this.quantity = quantity;
    }

    // getter만 제공
}
```

```java
// 조회 처리 서비스
public class OrderQueryService {
    private final OrderReadRepository repository;

    public OrderQueryService(OrderReadRepository repository) {
        this.repository = repository;
    }

    public List<OrderDTO> getOrders() {
        return repository.findAll();
    }
}
```

#### 3. 저장소 구현

```java
// 쓰기 전용 저장소
public interface OrderRepository {
    void save(Order order);
}
```

```java
// 읽기 전용 저장소
public interface OrderReadRepository {
    List<OrderDTO> findAll();
}
```

### CQRS 동작 흐름

1. 명령 요청: 클라이언트는 `OrderCommandService`를 호출하여 주문을 생성.  
2. 쓰기 저장소 업데이트: `OrderRepository`가 쓰기 데이터베이스를 갱신.  
3. 비동기 동기화: 쓰기 데이터베이스의 변경 사항이 읽기 데이터베이스로 복제.  
4. 조회 요청: 클라이언트는 `OrderQueryService`를 호출하여 주문 목록을 조회.  

## CQRS의 장점

1. 성능 최적화: 쓰기와 읽기를 각각 최적화할 수 있어 성능이 향상.  
2. 확장성: 쓰기와 읽기를 독립적으로 확장 가능.  
3. 복잡성 감소: 쓰기와 읽기의 요구사항을 분리하여 코드 복잡성을 낮춤.  
4. 테스트 용이성: 쓰기와 읽기를 별도로 테스트 가능.  

## CQRS의 단점

1. 구현 복잡성: 쓰기와 읽기 모델의 분리로 설계와 구현이 복잡.  
2. 데이터 동기화 문제: 쓰기 데이터와 읽기 데이터 간의 비동기 동기화로 인해 잠재적 일관성 문제 발생.  
3. 초기 비용: 두 데이터 모델과 두 데이터베이스를 유지해야 하므로 초기 설정 비용이 증가.  

### 마무리

CQRS는 성능과 확장성 향상이 필요한 시스템에서 특히 유용하다.  
다른 아키텍처 패턴이 궁금하다면 아래 글도 확인해보세요.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
