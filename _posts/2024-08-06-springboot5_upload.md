---
layout: single
title:  "spring boot 5. 데이터 파고들기"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 데이터
데이터에 따라 애플리케이션의 가치는 많이 갈린다.  
복잡한 데이터를 어떻게 다루느냐에 따라서 성능이 갈리기 마련이다.  

# 스프링 데이터
주된 기능은 기본적인 데이터 저장의 특수한 속성을 유지하면서 데이터에 액세스 하는 친숙하고 일관된 스프링 기반 프로그래밍 모델을 제공하는 것이다.  
어떤 DB 엔진이나, 플렛폼을 사용하든 가능한 한 간단하고 강력하게 데이터에 액세스 하게 하는 것이다.  

# 엔티티 정의
어떤 형태로든 데이터를 다루는 거의 모든 경우, 도메인 엔티티가 존재한다.  
자바를 사용해 도메인 클래스를 생성하기 위해 맴버변수, 생성자, 접근자, 변경자 메서드 등을 사용해 생성할 수 있다.  
다음은, user 엔티티를 정의하는 User 엔티티 클래스 예쩨이다.
``` java
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity // JPA 엔티티로 지정
public class User {

    @Id // 기본 키로 지정
    @GeneratedValue(strategy = GenerationType.AUTO) // 자동으로 ID 생성
    private Long id;
    private String name;
    private String email;

    // 기본 생성자
    public User() {}

    // 매개변수가 있는 생성자
    public User(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // Getter와 Setter
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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}


```
코드를 설명하자면,  
@Entity: 이 클래스가 JPA 엔티티임을 나타낸다.  
@Id: 이 필드가 기본 키임을 나타낸다.  
@GeneratedValue(strategy = GenerationType.AUTO): 기본 키 값이 자동으로 생성됨을 나타낸다.  
기본 생성자: JPA는 엔티티 클래스에 기본 생성자가 필요하다.  
매개변수가 있는 생성자: 편의성을 위해 정의된 생성자 이다.  
Getter와 Setter: 멤버 변수에 접근하고 변경할 수 있는 메서드이다.  

UserRepository 인터페이스  
```java
import org.springframework.data.jpa.repository.JpaRepository;

// User 엔티티를 관리하기 위한 리포지토리 인터페이스
public interface UserRepository extends JpaRepository<User, Long> {
    // 커스텀 쿼리 메서드 추가 가능 (예: findByName)
    User findByName(String name);
}

```  
User 엔티티를 관리하기 위한 스프링 데이터 JPA 리포지토리 이다.  
JpaRepository<User, Long>: User 엔티티와 그 기본 키 타입(Long)을 지정하여 기본적인 CRUD 메서드를 제공  
커스텀 쿼리 메서드 findByName(String name)을 추가하여 이름으로 사용자를 검색할 수 있다.  

스프링 부트 애플리케이션 클래스  
```java
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class UserApplication {

    public static void main(String[] args) {
        SpringApplication.run(UserApplication.class, args);
    }

    @Bean
    public CommandLineRunner demo(UserRepository repository) {
        return (args) -> {
            // 데이터 저장
            repository.save(new User("Alice", "alice@example.com"));
            repository.save(new User("Bob", "bob@example.com"));

            // 데이터 조회
            User user = repository.findByName("Alice");
            System.out.println("Found user: " + user.getName() + ", email: " + user.getEmail());
        };
    }
}

```  
스프링 부트 애플리케이션을 설정하고 실행.  
@SpringBootApplication: 스프링 부트 애플리케이션의 진입점을 나타낸다.  
CommandLineRunner 빈을 사용하여 애플리케이션 시작 시 일부 예제 데이터를 저장하고 조회한다.  
위의 예제를 통해서 스프링 데이터 JPA를 사용하여 데이터베이스와 상호 작용하는 방법을 알수 있다.  


# 탬플릿 지원  
스프링 부트는 다양한 템플릿 엔진을 지원하며, 그 중 하나인 Thymeleaf를 사용하여 HTML을 템플릿으로 활용할 수 있다.   
이를 통해 HTML 파일에 동적 데이터를 삽입할 수 있다. 또한, 스프링 부트는 다양한 데이터 소스에 접근할 수 있도록 `Operations` 타입의 인터페이스를 정의하고 있으며, 이는 MongoDB, Redis, Cassandra 등 다양한 데이터베이스와 상호 작용할 수 있다.  
이 `Operations` 인터페이스들은 일종의 SPI(Service Provider Interface)로서 서비스 제공을 위한 표준화된 인터페이스를 제공합니다.  

pom.xml 파일에 Thymeleaf 의존성 추가한다.  

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-thymeleaf</artifactId>
</dependency>
```  

Thymeleaf 템플릿 파일 src/main/resources/templates/index.html  

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Home</title>
</head>
<body>
    <h1 th:text="${message}">Hello, Thymeleaf!</h1>
</body>
</html>
```  

컨트롤러 HomeController  

```java
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("message", "Welcome to Thymeleaf!");
        return "index"; // 템플릿 파일 이름 (확장자 제외)
    }
}
```  

스프링 부트의 데이터 액세스 추상화  

스프링 부트는 다양한 데이터 소스에 대해 `Operations` 타입의 인터페이스를 정의하여 데이터 액세스를 추상화 한다   
다음은 대표적인 인터페이스들 이다  

1. MongoOperations: MongoDB와 상호 작용하는 인터페이스  
2. RedisOperations: Redis와 상호 작용하는 인터페이스  
3. CassandraOperations: Cassandra와 상호 작용하는 인터페이스  

이들 인터페이스는 일종의 SPI로, 다양한 데이터 소스에 대한 공통된 접근 방법을 제공하며, 특정 데이터베이스에 종속되지 않도록 한다.  

# 템플릿의 중요성  

이러한 템플릿들은 데이터베이스와의 상호 작용을 단순화하고 반복되는 단계를 줄여줬다.  
일반적인 패턴의 데이터 액세스에서는 리포지토리 패턴이 더 나은 선택이 될 수 있다.  
리포지토리 패턴은 다음과 같은 이유로 선호된다:

1. 추상화 수준의 증가: 템플릿 기반으로 하여 데이터베이스에 대한 직접적인 접근을 피하고, 인터페이스를 통해 상호 작용하므로 코드의 유연성과 유지보수성이 향상된다.  
2. 반복 작업의 최소화: 기본적인 CRUD 작업이 반복되므로, 이를 템플릿이나 리포지토리로 캡슐화하여 코드 중복을 줄일 수 있다.  
3. 일관된 프로그래밍 모델: 스프링 데이터 리포지토리는 일관된 프로그래밍 모델을 제공하여 다양한 데이터 소스에 대해 동일한 방식으로 접근할 수 있다.  

예를 들어, 스프링 데이터 JPA를 사용하면 다음과 같이 `UserRepository` 인터페이스를 정의하여 데이터 액세스를 추상화할 수 있다:  

UserRepository 인터페이스

```java
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
    User findByName(String name);
}
```  

스프링 부트는 다양한 템플릿 엔진과 데이터 액세스 방법을 제공하여 개발자가 복잡한 데이터베이스 작업을 쉽게 수행할 수 있도록 한다.  
특히, 템플릿 기반의 접근 방식은 반복적인 작업을 줄이고 추상화를 통해 코드의 유지보수성을 높이는 데 큰 도움이 된다.  
따라서, 데이터 액세스에서 리포지토리 패턴을 사용하는 것이 더 좋은 옵션이 될 수 있다.  

# 저장소 지원  
스프링 데이터가 Repository 인터페이스를 정의하고, 이 인터페이스로부터 그 외 모든 유형의 스프링 데이터 repository 데이터가 파생된다.  
스프링 데이터의 repository는 설정보단 관습을 사용한 쿼리 뿐만 아니라 네이티브 쿼리도 지원한다. 즉, 복잡한 데이터 베이스의 상호작용을 쉽게 구축할 수 있다.  

주요 리포지토리 인터페이스에는 다음과 같은게 있다.   
1. Repository: 모든 리포지토리 인터페이스의 기본 인터페이스  
2. CrudRepository: 기본적인 CRUD(Create, Read, Update, Delete) 작업을 위한 메서드를 제공  
3. PagingAndSortingRepository: 페이징 및 정렬 기능을 추가로 제공  
4. JpaRepository: JPA를 사용한 데이터 액세스 기능을 확장  
앞서 사용했던 User 엔테테 클래스를 활용해 예제를 보자면,  
User 엔티티 클래스  
```java
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity // JPA 엔티티로 지정
public class User {

    @Id // 기본 키로 지정
    @GeneratedValue(strategy = GenerationType.AUTO) // 자동으로 ID 생성
    private Long id;
    private String name;
    private String email;

    // 기본 생성자
    public User() {}

    // 매개변수가 있는 생성자
    public User(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // Getter와 Setter
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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}

```  

UserRepository 인터페이스  
```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface UserRepository extends JpaRepository<User, Long> {
    // 관습을 사용한 쿼리 메서드
    User findByName(String name);

    // 네이티브 쿼리
    @Query(value = "SELECT * FROM User WHERE email = ?1", nativeQuery = true)
    User findByEmailNative(String email);
}

```

위에서 사용된 쿼리는 다음과 같다.  
1. 관습을 사용한 쿼리  
    메서드 이름을 통해 쿼리를 정의  
    스프링 데이터는 메서드 이름을 분석하여 적절한 쿼리를 생성  
    User findByName(String name): 이름으로 사용자를 찾는 쿼리를 자동으로 생성한다.  
2. 네러티브 쿼리  
    SQL 문을 직접 작성하여 사용  
    데이터베이스 고유의 기능을 사용할 수 있으며, 복잡한 쿼리를 직접 작성할 수 있다.  
    @Query(value = "SELECT * FROM User WHERE email = ?1", nativeQuery = true): 이메일로 사용자를 찾는 SQL 쿼리를 정의한다.  

추가적으로, 페이징 및 정렬하는 예제를 보자.  
``` java
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
    // 관습을 사용한 쿼리 메서드
    User findByName(String name);

    // 네이티브 쿼리
    @Query(value = "SELECT * FROM User WHERE email = ?1", nativeQuery = true)
    User findByEmailNative(String email);

    // 페이징 및 정렬을 위한 메서드
    Page<User> findAll(Pageable pageable);
}

```
애플리케이션 클래스  
```java
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.domain.PageRequest;

@SpringBootApplication
public class UserApplication {

    public static void main(String[] args) {
        SpringApplication.run(UserApplication.class, args);
    }

    @Bean
    public CommandLineRunner demo(UserRepository repository) {
        return (args) -> {
            // 데이터 저장
            repository.save(new User("Alice", "alice@example.com"));
            repository.save(new User("Bob", "bob@example.com"));

            // 관습을 사용한 쿼리
            User user = repository.findByName("Alice");
            System.out.println("Found user: " + user.getName() + ", email: " + user.getEmail());

            // 네이티브 쿼리
            User nativeUser = repository.findByEmailNative("bob@example.com");
            System.out.println("Found user with native query: " + nativeUser.getName() + ", email: " + nativeUser.getEmail());

            // 페이징 및 정렬
            PageRequest pageRequest = PageRequest.of(0, 1);
            repository.findAll(pageRequest).forEach(pagedUser -> {
                System.out.println("Paged user: " + pagedUser.getName() + ", email: " + pagedUser.getEmail());
            });
        };
    }
}

```

스프링 데이터는 리포지토리 패턴을 통해 데이터 액세스를 추상화하고, 설정보다 관습을 우선시하는 방식으로 쿼리를 지원하여 간단한 쿼리 작성과 복잡한 데이터베이스 상호작용을 쉽게 구축할 수 있다.  
리포지토리는 관습을 사용한 쿼리 메서드와 네이티브 쿼리뿐만 아니라 페이징 및 정렬 기능도 제공하여 다양한 데이터 액세스 요구를 충족한다.  

# @Before  
테스트 메서드가 실행되기 전에 실행될 코드를 정의할 때 사용한다.  
다양한 각도로 탐색하려면, 다용도로 사용할 도메인이 필요하다.  
@BeforeEach 어노테이션은 JUnit 5에서 각 테스트 메서드가 실행되기 전에 실행될 코드를 정의할 때 사용한다.  
다양한 각도로 테스트를 수행하려면, 다용도로 사용할 도메인 객체가 필요하다.  
다음 예제는 User 도메인 객체를 사용하여 다양한 테스트를 수행하는 방법을 보여준다.  
User 도메인 클래스  
```java
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;
    private String name;
    private String email;

    // 기본 생성자
    public User() {}

    // 매개변수가 있는 생성자
    public User(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // Getter와 Setter
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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}

```  

UserTest 클래스  
```java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class UserTest {

    private User user;

    @BeforeEach // 각 테스트 메서드 전에 실행됨
    //각 테스트 메서드가 실행되기 전에 setUp 메서드를 실행하도록 한다.
    //스트를 위한 공통 설정 작업을 수행
    public void setUp() {
        user = new User("John Doe", "john.doe@example.com");
    }

    @Test //사용자의 이름을 테스트
    public void testUserName() {
        assertEquals("John Doe", user.getName());
    }

    @Test //사용자의 이메일을 테스트
    public void testUserEmail() {
        assertEquals("john.doe@example.com", user.getEmail());
    }

    @Test //새로 생성된 사용자 객체의 ID가 기본적으로 null인지 테스트
    public void testUserIdIsNullByDefault() {
        assertNull(user.getId());
    }

    @Test //사용자 ID 설정 메서드를 테스트
    public void testSetUserId() {
        user.setId(1L);
        assertEquals(1L, user.getId());
    }

    @Test //사용자 이름 설정 메서드를 테스트
    public void testSetUserName() {
        user.setName("Jane Doe");
        assertEquals("Jane Doe", user.getName());
    }

    @Test //사용자 이메일 설정 메서드를 테스트
    public void testSetUserEmail() {
        user.setEmail("jane.doe@example.com");
        assertEquals("jane.doe@example.com", user.getEmail());
    }
}

```

User 도메인 객체를 다양한 각도로 탐색하며, 각 테스트 메서드가 실행되기 전에 공통 설정 작업을 수행하도록 한다.  
이를 통해 보다 일관되고 체계적인 테스트를 수행할 수 있다.  

# 레디스로 템플릿 기반 서비스 생성하기  
레디스는 인메모리 데이터베이스로, 서비스 내 인스턴스 간에 상태를 공유하고, 캐싱 및 서비스 간 메시지를 중계하는 용도로 많이 사용된다.  
다음은 PlannerFinder 서비스에서 얻은 정보를 저장하고 조회하는 데 레디스를 활용하는 예제이다.  
pom.xml 파일에 레디스 의존성 추가
``` java
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>

```

레디스 설정 파일 src/main/resources/application.properties  
```java
spring.redis.host=localhost
spring.redis.port=6379

```

레디스 템플릿 설정 RedisConfig 클래스  
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

@Configuration
public class RedisConfig {

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory redisConnectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(redisConnectionFactory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        return template;
    }
}

```

PlannerFinder 서비스  
``` java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class PlannerFinderService {

    private static final String PLANNER_KEY_PREFIX = "planner:";

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    // Planner 정보를 레디스에 저장
    public void savePlannerInfo(String plannerId, Object plannerInfo) {
        redisTemplate.opsForValue().set(PLANNER_KEY_PREFIX + plannerId, plannerInfo);
    }

    // Planner 정보를 레디스에서 조회
    public Object getPlannerInfo(String plannerId) {
        return redisTemplate.opsForValue().get(PLANNER_KEY_PREFIX + plannerId);
    }

    // Planner 정보를 레디스에서 삭제
    public void deletePlannerInfo(String plannerId) {
        redisTemplate.delete(PLANNER_KEY_PREFIX + plannerId);
    }
}

```

PlannerFinder 서비스 테스트  
```java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.data.redis.DataRedisTest;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.RedisTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

@SpringBootTest
public class PlannerFinderServiceTest {

    @Autowired
    private PlannerFinderService plannerFinderService;

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    @BeforeEach
    public void setUp() {
        // 테스트 전에 Redis 초기화
        redisTemplate.getConnectionFactory().getConnection().flushDb();
    }

    @Test
    public void testSaveAndGetPlannerInfo() {
        String plannerId = "123";
        String plannerInfo = "Sample Planner Info";

        plannerFinderService.savePlannerInfo(plannerId, plannerInfo);
        Object retrievedInfo = plannerFinderService.getPlannerInfo(plannerId);

        assertEquals(plannerInfo, retrievedInfo);
    }

    @Test
    public void testDeletePlannerInfo() {
        String plannerId = "123";
        String plannerInfo = "Sample Planner Info";

        plannerFinderService.savePlannerInfo(plannerId, plannerInfo);
        plannerFinderService.deletePlannerInfo(plannerId);
        Object retrievedInfo = plannerFinderService.getPlannerInfo(plannerId);

        assertNull(retrievedInfo);
    }
}

```  

레디스는 인메모리 데이터베이스로, 빠른 데이터 액세스와 상태 공유가 필요한 경우에 유용하게 사용된다.  
스프링 부트와 레디스를 함께 사용하면 간단하고 효율적으로 캐시를 관리하고 데이터를 저장할 수 있다.  

# NoSQL 도큐먼트 데이터베이스를 사용해 리포지토리 기반 서비스 만들기  
NoSQL 도큐먼트 데이터베이스는 MongoDB와 같은 데이터베이스를 사용하여 데이터를 도큐먼트 형식으로 저장하고 관리할 수 있다.  
MongoDB와 같은 NoSQL 도큐먼트 데이터베이스를 사용하여 리포지토리 기반의 서비스를 만드는 방법이다.  

pom.xml 파일에 MongoDB 의존성 추가  
```java
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-mongodb</artifactId>
</dependency>

```
도메인 클래스 정의  
```java
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "users") // MongoDB 도큐먼트로 지정하고 컬렉션 이름을 "users"로 설정
public class User {

    @Id // 기본 키로 지정
    private String id;
    private String name;
    private String email;

    // 기본 생성자
    public User() {}

    // 매개변수가 있는 생성자
    public User(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // Getter와 Setter
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}

```  
리포지토리 인터페이스 정의  
``java
import org.springframework.data.mongodb.repository.MongoRepository;

public interface UserRepository extends MongoRepository<User, String> {
    // 추가적인 쿼리 메서드를 정의할 수 있습니다.
}

```

서비스 클래스 정의  
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserService {

    private final UserRepository userRepository;

    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User getUserById(String id) {
        return userRepository.findById(id).orElseThrow(() -> new IllegalArgumentException("Invalid user Id: " + id));
    }

    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    public void saveUser(User user) {
        userRepository.save(user);
    }

    public void deleteUser(String id) {
        userRepository.deleteById(id);
    }
}

```

컨트롤러 클래스 정의
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
public class UserController {

    private final UserService userService;

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable String id) {
        User user = userService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        List<User> users = userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        userService.saveUser(user);
        return ResponseEntity.ok(user);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable String id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}

```

애플리케이션 설정
```java
spring.data.mongodb.host=localhost
spring.data.mongodb.port=27017
spring.data.mongodb.database=mydatabase

```

Spring Boot 애플리케이션 클래스  
```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

```

MongoDB 의존성을 추가하고, 도메인 클래스를 정의한 후 리포지토리 인터페이스와 서비스 클래스를 작성하여 데이터를 관리한다.  
컨트롤러 클래스를 통해 웹 요청을 처리하고, 애플리케이션 설정을 통해 MongoDB에 연결한다.  

# NoSQL 그래프 기반 데이터베이스를 사용해 리포지토리 만들기
Neo4j와 같은 NoSQL 그래프 데이터베이스를 사용하여 리포지토리 기반의 서비스를 만들수 있디.  
Neo4j는 데이터를 그래프로 표현하며, 노드와 관계를 통해 데이터의 연결성을 쉽게 다룰 수 있다.  

pom.xml 파일에 Neo4j 의존성 추가
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-neo4j</artifactId>
</dependency>

```

도메인 클래스 정의  
```java
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;

@NodeEntity // Neo4j 노드 엔티티로 지정
public class User {

    @Id // 기본 키로 지정
    @GeneratedValue // 자동으로 ID 생성
    private Long id;
    private String name;
    private String email;

    // 기본 생성자
    public User() {}

    // 매개변수가 있는 생성자
    public User(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // Getter와 Setter
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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}

```

리포지토리 인터페이스 정의  
```java
import org.springframework.data.neo4j.repository.Neo4jRepository;

public interface UserRepository extends Neo4jRepository<User, Long> {
    // 추가적인 쿼리 메서드를 정의할 수 있다.
}

```

서비스 클래스 정의  
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserService {

    private final UserRepository userRepository;

    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User getUserById(Long id) {
        return userRepository.findById(id).orElseThrow(() -> new IllegalArgumentException("Invalid user Id: " + id));
    }

    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    public void saveUser(User user) {
        userRepository.save(user);
    }

    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
}

```

컨트롤러 클래스 정의  
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
public class UserController {

    private final UserService userService;

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        User user = userService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        List<User> users = userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        userService.saveUser(user);
        return ResponseEntity.ok(user);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}

```

애플리케이션 설정  src/main/resources/application.properties에 Neo4j 연결 설정을 추가
```java
spring.data.neo4j.uri=bolt://localhost:7687
spring.data.neo4j.username=neo4j
spring.data.neo4j.password=your_password

```

Spring Boot 애플리케이션 클래스  
```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

```  
Neo4j 의존성을 추가하고, 도메인 클래스를 정의한 후 리포지토리 인터페이스와 서비스 클래스를 작성하여 데이터를 관리한다.  
컨트롤러 클래스를 통해 웹 요청을 처리하고, 애플리케이션 설정을 통해 Neo4j에 연결한다.  

위를 통해 데이터를 잘 다룰 수 있다.