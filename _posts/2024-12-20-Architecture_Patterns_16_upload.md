---
layout: single
title:  "아키텍처 패턴 시리즈 16. 헥사고날 아키텍처 (Hexagonal Architecture)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 16: 헥사고날 아키텍처 (Hexagonal Architecture)

헥사고날 아키텍처(Hexagonal Architecture)는 애플리케이션을 도메인 로직 중심으로 설계하여 외부 의존성과의 결합을 줄이고 유연성을 확보하는 아키텍처 패턴이다.  
"포트와 어댑터(Ports and Adapters)" 아키텍처로도 알려져 있으며, 도메인 로직과 외부 요소의 명확한 분리를 강조한다.  

## 헥사고날 아키텍처의 필요성

다음과 같은 문제를 해결하기 위해 헥사고날 아키텍처가 도입된다:  

1. 강한 결합 문제: 외부 프레임워크나 기술에 종속적인 애플리케이션 설계.  
2. 테스트 어려움: 외부 의존성과 도메인 로직이 밀접하게 결합되어 테스트가 어려움.  
3. 유지보수성 저하: 특정 기술에 고정된 구현으로 인해 요구사항 변화에 따른 수정이 어려움.  

### 예시: 전자상거래 시스템

전자상거래 애플리케이션에서 다음과 같은 컴포넌트가 존재한다고 가정하자:

- 상품 관리 도메인 로직: 상품 추가, 수정, 삭제를 처리.  
- 데이터베이스: 상품 데이터를 저장.  
- REST API: 클라이언트와의 통신을 제공.  

헥사고날 아키텍처를 사용하면 도메인 로직을 중심으로 설계하여 REST API와 데이터베이스와의 결합을 최소화할 수 있다.  

## 헥사고날 아키텍처의 구조

### 주요 컴포넌트

1. 도메인 (Domain): 비즈니스 로직을 포함한 핵심 영역.  
2. 포트 (Ports): 도메인과 외부를 연결하는 인터페이스.  
    - 입력 포트: 클라이언트가 도메인을 호출하기 위한 인터페이스.  
    - 출력 포트: 도메인이 외부 시스템에 의존성을 가지기 위한 인터페이스.  
3. 어댑터 (Adapters): 포트를 구현하여 외부 시스템과의 상호작용을 처리.  

### 구조 다이어그램

```
           [Input Adapter: REST API]
                    |
[Input Port: Use Case Interface] <---> [Domain Logic] <---> [Output Port: Repository Interface]
                    |
           [Output Adapter: Database]
```  

### 동작 원리

1. 입력 어댑터는 클라이언트 요청을 수신하고 이를 도메인 로직에 전달.  
2. 도메인 로직은 입력 포트를 통해 비즈니스 로직을 실행.  
3. 출력 어댑터는 출력 포트를 통해 외부 시스템(DB, 메시징 서비스 등)과 통신.  

## 헥사고날 아키텍처 예시

### 상품 관리 애플리케이션 구현

#### 1. 도메인 (Domain)

```java
// 상품 엔티티
public class Product {
    private String id;
    private String name;
    private double price;

    // 생성자, getter 및 setter
    public Product(String id, String name, double price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }
}
```

```java
// 도메인 서비스
public class ProductService {
    private final ProductRepository repository;

    public ProductService(ProductRepository repository) {
        this.repository = repository;
    }

    public void addProduct(Product product) {
        repository.save(product);
    }

    public Product getProduct(String id) {
        return repository.findById(id);
    }
}
```

#### 2. 포트 (Ports)

```java
// 출력 포트: 저장소 인터페이스
public interface ProductRepository {
    void save(Product product);
    Product findById(String id);
}
```

#### 3. 어댑터 (Adapters)

**데이터베이스 어댑터**:

```java
import java.util.HashMap;
import java.util.Map;

// 메모리 기반 저장소 구현
public class InMemoryProductRepository implements ProductRepository {
    private final Map<String, Product> database = new HashMap<>();

    @Override
    public void save(Product product) {
        database.put(product.getId(), product);
    }

    @Override
    public Product findById(String id) {
        return database.get(id);
    }
}
```

**REST API 어댑터**:

```java
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/products")
public class ProductController {
    private final ProductService service;

    public ProductController(ProductService service) {
        this.service = service;
    }

    @PostMapping
    public void addProduct(@RequestBody Product product) {
        service.addProduct(product);
    }

    @GetMapping("/{id}")
    public Product getProduct(@PathVariable String id) {
        return service.getProduct(id);
    }
}
```

### 실행 방법

1. `InMemoryProductRepository`를 구현하여 도메인 서비스에 주입.  
2. `ProductController`를 통해 클라이언트 요청을 처리.  

## 헥사고날 아키텍처의 장점

1. 유연성: 도메인 로직은 외부 기술에 의존하지 않으므로, 기술 스택을 변경해도 영향을 최소화.  
2. 테스트 용이성: 도메인 로직은 독립적이므로 외부 의존성이 없는 상태에서 테스트 가능.  
3. 재사용성: 동일한 도메인 로직을 여러 입력 어댑터와 함께 사용 가능.  
4. 모듈화: 각 컴포넌트가 명확히 분리되어 유지보수와 확장이 쉬움.  

## 헥사고날 아키텍처의 단점

1. 초기 복잡성: 설계와 구현 초기 단계에서 구조를 세분화하는 데 많은 시간 소요.  
2. 추가 구현 비용: 포트와 어댑터로 인해 코드량이 증가.  

### 마무리

헥사고날 아키텍처는 변화와 확장에 강한 애플리케이션을 설계하는 데 유용하다.  
다른 아키텍처 패턴이 궁금하다면 아래 글도 확인해보세요.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
