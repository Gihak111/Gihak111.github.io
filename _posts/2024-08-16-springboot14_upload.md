---
layout: single
title:  "spring boot 14.스프링 부트 맥락"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 스프링 부트의 HTTP 처리 
요청이 처리되는 맥락을 어노테이션, 헤더 엔티티 등과 함꼐 알아보자.  

### 1. 클라이언트 요청
사용자가 사이트의 버튼을 클릭하면 클라이언트가 HTTP 요청을 서버로 보낸다.  
이 요청에는 URL, HTTP 메서드(GET, POST 등), 헤더, 본문 등이 포함된다.  

### 2. 스프링 부트 컨트롤러
요청이 서버에 도착하면 스프링 부트 컨트롤러가 이를 처리한다.  
컨트롤러는 주로 어노테이션을 사용하여 특정 URL 패턴과 HTTP 메서드에 매핑된다.  

```java
@RestController
@RequestMapping("/api")
public class ApiController {

    @GetMapping("/data")
    public ResponseEntity<String> getData(@RequestHeader("Authorization") String token) {
        // 서비스 레이어로 요청을 전달
        String data = dataService.fetchData(token);
        return ResponseEntity.ok(data);
    }
}
```

- `@RestController`: 이 클래스가 RESTful 웹 서비스의 컨트롤러임을 나타낸다.  
- `@RequestMapping("/api")`: 기본 URL 경로를 설정한다.  
- `@GetMapping("/data")`: GET 요청을 `/api/data` 경로에 매핑한다.  
- `@RequestHeader("Authorization") String token`: 요청 헤더에서 "Authorization" 값을 추출한다.  
하나의 특별한 겟셋이라 생각하면 편하다.  

### 3. 서비스 레이어
컨트롤러는 서비스 레이어로 요청을 전달하여 로직을 처리한다.

```java
@Service
public class DataService {

    public String fetchData(String token) {
        // 데이터베이스 또는 외부 API 호출
        return "some data";
    }
}
```

- `@Service`: 이 클래스가 서비스 레이어임을 나타낸다.

### 4. 데이터베이스와 엔티티
서비스 레이어는 데이터베이스와 상호작용한다.  
JPA를 사용하여 데이터베이스와 매핑되는 엔티티를 정의한다.  

```java
@Entity
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;

    // Getters and Setters
}
```

- `@Entity`: 이 클래스가 JPA 엔티티임을 나타낸다.  
- `@Id`: 이 필드가 기본 키임을 나타낸다.  
- `@GeneratedValue(strategy = GenerationType.IDENTITY)`: 기본 키 생성 전략을 정의한다.  

### 5. 리포지토리 레이어
리포지토리는 데이터베이스 작업을 수행한다.  

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
}
```

- `@Repository`: 이 인터페이스가 리포지토리임을 나타낸다.
- `extends JpaRepository<User, Long>`: 기본 CRUD 작업을 제공하는 JPA 리포지토리 인터페이스를 상속한다.

### 6. 요청 처리 과정
1. 클라이언트가 버튼을 클릭하여 `/api/data`에 GET 요청을 보낸다.
2. `ApiController`의 `getData` 메서드가 요청을 처리하고, "Authorization" 헤더를 추출한다.
3. `DataService`의 `fetchData` 메서드가 호출되어 데이터베이스 또는 외부 API에서 데이터를 가져온다.
4. 데이터가 `ResponseEntity`로 포장되어 클라이언트에 반환된다.

이 과정에서 어노테이션은 설정과 매핑을 단순화하고, 엔티티는 데이터베이스와 객체를 매핑하며, 헤더는 요청과 응답의 메타데이터를 전달한다.  
이를 통해 스프링 부트는 효율적으로 요청을 처리하고, 데이터를 주고받을 수 있다.