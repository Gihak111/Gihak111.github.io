---
layout: single
title:  "아키텍처 패턴 시리즈 1. 계층화 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 1: 계층화 패턴 (Layered Pattern)

계층화 패턴(Layered Pattern)은 아키텍처 패턴 중 하나로, 시스템을 여러 계층으로 나누어 각 계층이 특정 역할을 수행하도록 구조화하는 방식이다.  
이 패턴은 유지보수와 확장성을 높이는 동시에, 시스템의 복잡도를 낮추는 데에 유리하다.

## 계층화 패턴의 필요성

소프트웨어 시스템이 복잡해질수록 여러 기능이 한 곳에 집중되는 "스파게티 코드" 문제가 발생할 수 있다. 이를 방지하기 위해 계층화 패턴을 사용하여 시스템을 분리하고, 각 계층이 다음과 같은 역할을 수행하도록 설계할 수 있다:

1. **유지보수 용이성**: 각 계층은 독립적으로 수정할 수 있어 코드의 수정과 유지보수가 쉬워진다.  
2. **책임 분리**: 기능을 역할에 따라 나누어, 코드 간의 의존성을 최소화할 수 있다.  
3. **재사용성**: 특정 계층의 기능을 다른 프로젝트에서 재사용할 수 있다.  

이러한 구조적 분리는 소프트웨어 품질을 향상시키며, 계층화 패턴은 특히 대규모 애플리케이션에서 효과적이다.  

### 예시: 웹 애플리케이션 계층 구조  

일반적인 웹 애플리케이션의 경우, 클라이언트 요청이 데이터베이스에 직접 접근하는 것이 아니라, 여러 계층을 통해 흐르도록 설계된다.  

## 계층화 패턴의 구조  

1. **Presentation Layer (프레젠테이션 계층)**: 사용자 인터페이스와 상호작용하며, 사용자 요청을 받아 비즈니스 로직에 전달한다.  
2. **Business Layer (비즈니스 계층)**: 애플리케이션의 핵심 로직을 처리하고, 데이터와의 상호작용을 조율한다.  
3. **Persistence Layer (영속성 계층)**: 데이터베이스와의 직접적인 상호작용을 담당하며, 데이터를 저장하고 불러온다.  
4. **Database Layer (데이터베이스 계층)**: 실제 데이터베이스와 연결하여 데이터를 저장하고 관리하는 역할을 한다.  

### 구조 다이어그램  

```
Client → Presentation Layer → Business Layer → Persistence Layer → Database Layer
```

### 계층화 패턴 동작 순서  

1. 클라이언트는 `Presentation Layer`를 통해 요청을 보낸다.  
2. `Presentation Layer`는 요청을 `Business Layer`에 전달하여 로직을 처리한다.  
3. `Business Layer`는 필요한 경우 `Persistence Layer`에 데이터를 요청하거나 저장을 요청한다.  
4. 최종적으로 `Persistence Layer`는 `Database Layer`와 상호작용하여 데이터를 처리한다.  

## 계층화 패턴 예시

이번 예시에서는 "사용자 인증 시스템"을 계층화 패턴을 통해 구현해보자.

### Java로 계층화 패턴 구현하기

```java
// Presentation Layer
class UserController {
    private UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    public void login(String username, String password) {
        boolean success = userService.authenticate(username, password);
        if (success) {
            System.out.println("로그인 성공");
        } else {
            System.out.println("로그인 실패");
        }
    }
}

// Business Layer
class UserService {
    private UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public boolean authenticate(String username, String password) {
        User user = userRepository.findByUsername(username);
        return user != null && user.getPassword().equals(password);
    }
}

// Persistence Layer
class UserRepository {
    public User findByUsername(String username) {
        // 여기서는 단순히 예시를 위해 고정된 사용자 데이터를 반환한다.
        if ("user123".equals(username)) {
            return new User(username, "password123");
        }
        return null;
    }
}

// Database Layer
class User {
    private String username;
    private String password;

    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        UserRepository userRepository = new UserRepository();
        UserService userService = new UserService(userRepository);
        UserController userController = new UserController(userService);

        userController.login("user123", "password123");  // 로그인 성공
        userController.login("user123", "wrongPassword");  // 로그인 실패
    }
}
```

### 코드 설명  

1. **Presentation Layer**: `UserController`는 사용자로부터 요청을 받아 `UserService`로 전달한다.  
2. **Business Layer**: `UserService`는 비즈니스 로직을 처리하며, 사용자 인증 기능을 수행한다.  
3. **Persistence Layer**: `UserRepository`는 데이터베이스와 상호작용하는 역할을 수행하며, 사용자 데이터를 검색한다.  
4. **Database Layer**: `User` 클래스는 사용자 데이터를 저장하고 관리하는 역할을 한다.  
5. **Client (클라이언트)**: `Main` 클래스는 전체 계층을 조합하여 로그인 기능을 실행한다.  

### 출력 결과  

```
로그인 성공
로그인 실패
```

### 계층화 패턴 활용

1. **MVC 패턴**: 웹 애플리케이션에서 자주 사용되는 패턴으로, 프레젠테이션, 비즈니스, 데이터 계층으로 구성된다.  
2. **백엔드 시스템**: 대규모 시스템에서의 업무 처리, 데이터 접근 및 저장의 역할 분리를 위해 사용.  
3. **서비스 지향 아키텍처**: 서비스와 데이터의 계층적 분리를 통해 독립성을 강화.  

## 계층화 패턴의 장점  

1. **유지보수성**: 각 계층이 독립적으로 수정 가능하여 유지보수 용이.  
2. **모듈화**: 역할에 따른 분리를 통해 모듈화된 시스템을 구성.  
3. **유연성**: 특정 계층만 변경해도 전체 시스템에 영향을 주지 않음.  

## 계층화 패턴의 단점  

1. **복잡성 증가**: 시스템이 단순할 경우 불필요하게 복잡도가 증가할 수 있음.  
2. **성능 문제**: 모든 요청이 계층을 통과해야 하므로 성능이 저하될 가능성이 있음.  

### 마무리  

계층화 패턴(Layered Pattern)은 시스템을 구조화하여 유지보수성을 높이고 역할을 분리하는 데 유용하다.  
대규모 애플리케이션에서 각 계층의 역할을 명확히 하고, 독립적으로 관리할 수 있어 유리하다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
