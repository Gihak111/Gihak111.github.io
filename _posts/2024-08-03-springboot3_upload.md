---
layout: single
title:  "spring boot 3. 데이터베이스 액세스"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# DB 엑세스  
백엔드에서 데이터 베이스와 연동은 필수적이다.  
지금까진 ArrayList로 관리했다.  
이는 데이터의 양이 많아지면 매우 불리해진다.  
따라서 DB를 이용해 관리한다.  

스프링부트 앱에서 DB에 액세스 하려면 다음이 필요하다.  
1. 실행중인 DB - 접속 가능한 DB나, 개발하는 앱으 ㅣ내장 DB  
2. 프로그램상에서 DB 엑세스를 가능하게 해 주는 DB 드라이버  
3. 원하는 DB에 액세스하기 위한 스프링 데이터 모듈  

애플리케이션이 H2 DB와 상호작용 한다고 가정하자.  
pom.xml의 <dependencies> 위치에 다음 의존성을 추가하자.  
```html
<dependencies>
    <!-- 스프링 부트 스타터 데이터 JPA -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>

    <!-- H2 데이터 -->
    <dependency>
        <groupId>com.h2database</groupId>
        <artifactId>h2</artifactId>
        <scope>runtime</scope>
    </dependency>

    <!-- 스프링 부트 스타터 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>

```  
위의 코드를 추가하는 것으로 의존성을 추가했다.  

# @Entiy를 사용해서 접근해 보자.  
H2가 JPA 호환 DB 이므로, JPA 어노테이션을 추가해 보자.  
또한, 기존 id 맴버 변수를 DB 테이블의 ID 필드로 표시하기 위해 @Id 어노테이션도 추가해 보자.  

```java
import javax.persistence.Entity; // JPA의 Entity 어노테이션을 사용
import javax.persistence.GeneratedValue; // ID 생성 정의
import javax.persistence.GenerationType; // ID 생성 전략의 종류 정의
import javax.persistence.Id; // JPA의 ID 어노테이션을 사용

@Entity // 이 클래스가 JPA 엔티티임을 나타냄
public class Coffee {

    @Id // 이 필드가 엔티티의 식별자임을 나타냄
    @GeneratedValue(strategy = GenerationType.IDENTITY) // 자동으로 증가하는 ID를 설정
    private Long id; // 커피의 고유 ID

    private String name; // 커피의 이름
    private String origin; // 커피의 원산지
    private String roastLevel; // 커피의 로스트 레벨

    // Getter와 Setter 메서드를 추가하여 필드에 접근하고 수정할 수 있다.
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getOrigin() {
        return origin;
    }

    public void setOrigin(String origin) {
        this.origin = origin;
    }

    public String getRoastLevel() {
        return roastLevel;
    }

    public void setRoastLevel(String roastLevel) {
        this.roastLevel = roastLevel;
    }
}
```  
# JPA 리포지토리 인터페이스 구축  
엔터리를 다 만들었으니 데이터 베이스 연산을 수행해야 하므로 JPA 리포지토리 인터페이스를 만든다.  
```java
import org.springframework.data.jpa.repository.JpaRepository; // JPA 리포지토리 인터페이스를 사용

public interface CoffeeRepository extends JpaRepository<Coffee, Long> {
    // 기본 CRUD 메서드는 JpaRepository에서 제공
    Coffee findByName(String name); // 커피 이름으로 커피를 찾는 메서드이다.
}

```  

service 클래스를 만들어 비즈니스 로직을 구성한다.  
```java
import org.springframework.beans.factory.annotation.Autowired; // 의존성 주입을 위해 사용합니다.
import org.springframework.stereotype.Service; // 서비스 클래스임을 나타냅니다.

import java.util.Optional; // Null 처리를 위한 Optional 클래스입니다.

@Service // 이 클래스가 서비스 클래스임을 나타냅니다.
public class CoffeeService {

    @Autowired // CoffeeRepository의 의존성을 주입합니다.
    private CoffeeRepository coffeeRepository;

    public Coffee saveCoffee(Coffee coffee) {
        return coffeeRepository.save(coffee); // 커피 정보를 저장합니다.
    }

    public Optional<Coffee> getCoffeeById(Long id) {
        return coffeeRepository.findById(id); // ID로 커피를 조회합니다.
    }

    public Coffee getCoffeeByName(String name) {
        return coffeeRepository.findByName(name); // 이름으로 커피를 조회합니다.
    }
}

```

# 컨트롤러 클래스 구성  
컨트롤러 클래스 작성해서 HTTP 요청을 처리한다.
```java
import org.springframework.beans.factory.annotation.Autowired; // 의존성 주입을 위해 사용합니다.
import org.springframework.web.bind.annotation.*; // REST API를 위한 어노테이션들입니다.

@RestController // 이 클래스가 REST 컨트롤러임을 나타냅니다.
@RequestMapping("/coffees") // 모든 요청이 /coffees로 시작합니다.
public class CoffeeController {

    @Autowired // CoffeeService의 의존성을 주입합니다.
    private CoffeeService coffeeService;

    @PostMapping // POST 요청을 처리하는 메서드입니다.
    public Coffee createCoffee(@RequestBody Coffee coffee) {
        return coffeeService.saveCoffee(coffee); // 요청 본문에서 받은 커피 정보를 저장합니다.
    }

    @GetMapping("/{id}") // ID로 커피를 조회하는 GET 요청을 처리합니다.
    public Coffee getCoffeeById(@PathVariable Long id) {
        return coffeeService.getCoffeeById(id).orElse(null); // ID로 커피를 조회하고, 없으면 null을 반환합니다.
    }

    @GetMapping("/name/{name}") // 이름으로 커피를 조회하는 GET 요청을 처리합니다.
    public Coffee getCoffeeByName(@PathVariable String name) {
        return coffeeService.getCoffeeByName(name); // 이름으로 커피를 조회합니다.
    }
}

```

# H2 콘솔 접속  
콘솔을 통해 데이터 베이스를 직접 조회하고, SQL 쿼리를 실행할 수 있다.  
h2-console 를 url 뒤에 붙이는 것으로 접근할 수 있다.  

JDBC URL: jdbc:h2:mem:testdb  
User Name: sa  
Password: password  
이를 통해 SQL 쿼리를 실행하고, 데이터 상태를 확인할 수 있다.  

# 앱에서 실행 밑 테스트  
스프링 부트를 실행하고, ostman이나 curl을 사용해서 API를 태스트 하자.  
```java
// 커피 생성
curl -X POST http://링크 주소/coffees -H "Content-Type: application/json" -d '{"name":"Espresso","origin":"Brazil","roastLevel":"Dark"}'

// 커피 조회 ID 기준
curl http://링크 주소/coffees/1

// 커피 조회 이름 기준
curl http://링크 주소/coffees/name/Espresso

```

위 과정을 통해서 H2 데이터베이스를 사용하여 커피 정보를 저장하고 조회할 수 있다.  