---
layout: single
title:  "spring boot 10.리액티브 프로그래밍과 웹플럭스"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 리액티브 프로그래밍  
비동기 데이터 스트림을 처리하는 프로그래밍 패러다임 이다.  
정통적인 방식인 요청-응답 사이클이 아닌, 데이터 흐름과 그 변화에 반응하여 작업을 수행하는 방식이다.  
이를 통해 시스템의 확장성과 성능을 극대화할 수 있다.  
간단히 말하면, 하나의 밸류값이 변하면 그 밸류값에 반응하는 여러개의 액트가 있는거다.  
어떤 값이 변경되면 이 값을 사용하는 다른 계산 결과나 UI가 자동으로 업데이트되도록 설정 하는데에 유리하다.  

주요 개념은 다음과 같다.  
1. 데이터 스트림 Data Streams  
    지속적으로 변화하는 데이터를 스트림 형태로 처리  
2. 비동기 처리 Asynchronous Processing  
    데이터의 도착을 기다리지 않고 작업을 진행  
3. 백프레셔 Backpressure  
    데이터 생성 속도와 처리 속도의 불일치 문제를 다룸  

예제를 통해 이해하면 된다.  
pom.xml
```xml
<dependencies>
    <!-- Spring Boot Starter Webflux for Reactive Programming -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-webflux</artifactId>
    </dependency>
    <!-- Spring Boot Starter Test for testing -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>

```

ReactiveController.java  
Spring WebFlux의 @RestController를 사용하여 비동기 처리를 위한 REST API를 생성  
```java
package com.example.demo.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api")
public class ReactiveController {

    /**
     * 데이터 스트림을 생성하여 클라이언트에 지속적으로 데이터를 전송.
     * @return Flux<String> 데이터 스트림
     */
    @GetMapping("/data-stream")
    public Flux<String> getDataStream() {
        // 데이터 스트림을 생성하여 1초마다 데이터를 발행.
        return Flux.interval(Duration.ofSeconds(1))
                   .map(sequence -> "Data: " + sequence);
    }
}

```
Flux는 데이터의 스트림을 나타내며, interval 메서드를 통해 1초마다 증가하는 값을 발행한다.  
map을 사용하여 발행되는 값을 "Data: " 접두사와 함께 문자열로 변환한다.  

ReactiveService.java  
비즈니스 로직을 처리하는 서비스 클래스를 작성한다.  비동기적으로 데이터 처리와 변환을 수행한다.  
```java
package com.example.demo.service;

import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;

@Service
public class ReactiveService {

    /**
     * 비동기적으로 데이터 스트림을 처리.
     * @return Flux<String> 비동기 데이터 스트림
     */
    public Flux<String> processDataStream() {
        // 데이터를 스트림 형태로 처리하며, 데이터를 생성.
        return Flux.interval(Duration.ofMillis(500))
                   .map(sequence -> "Processed Data: " + sequence)
                   .delayElements(Duration.ofMillis(200)); // 데이터 처리 지연
    }
}

```

interval 메서드를 사용하여 500ms마다 값을 생성하고, map으로 데이터 처리 후 변환  
delayElements를 사용하여 데이터 처리 속도를 조절한다.  

BackpressureController.java  
백프레셔를 다루는 컨트롤러를 작성한다.  데이터 생성 속도와 처리 속도의 불일치를 조절  
```java
package com.example.demo.controller;

import com.example.demo.service.ReactiveService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/backpressure")
public class BackpressureController {

    private final ReactiveService reactiveService;

    public BackpressureController(ReactiveService reactiveService) {
        this.reactiveService = reactiveService;
    }

    /**
     * 백프레셔를 적용하여 데이터 스트림을 처리.
     * @return Flux<String> 데이터 스트림
     */
    @GetMapping("/data")
    public Flux<String> getProcessedDataStream() {
        // ReactiveService를 통해 데이터를 처리.
        return reactiveService.processDataStream()
                               .onBackpressureBuffer(10); // 버퍼를 사용하여 백프레셔 처리
    }
}

```

위에서의 기능은 다음과 같다.  
1. 데이터 스트림  
    ReactiveController에서 1초마다 데이터 스트림을 생성하여 클라이언트에 전송  
2. 비동기 처리  
    ReactiveService에서 데이터를 비동기적으로 생성하고 처리  
3. 백프레셔  
    BackpressureController에서 onBackpressureBuffer를 사용하여 데이터 생성 속도와 처리 속도의 불일치를 처리  

# 프로젝트 리액터  
액티브 프로그래밍을 자바에서 지원하는 라이브러리  
비동기 스트림을 처리할 수 있도록 도와준다.  
Mono와 Flux라는 두 가지 주요 타입을 제공하며, 이는 각각 단일 값 또는 0개 이상의 값을 처리할 수 있다.  
Mono: 0개 또는 1개의 요소를 포함하는 리액티브 스트림 (Publisher) 이다.  
Flux: 0개 이상의 요소를 포함할 수 있는 리액티브 스트림 (Publisher) 이다.  
리액터는 이를 통해 데이터 스트림을 생성, 변환, 필터링하고, 오류를 처리할 수 있다.  

pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>reactive-demo</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <dependencies>
        <!-- Spring Boot Starter WebFlux for reactive web applications -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>

        <!-- Spring Boot Starter Test for testing -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
            <exclusions>
                <exclusion>
                    <groupId>org.junit.vintage</groupId>
                    <artifactId>junit-vintage-engine</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>

```
ReactiveDemoApplication.java   
```java
package com.example.reactivedemo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ReactiveDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReactiveDemoApplication.class, args);
    }
}

```

ReactiveController.java   
```java
package com.example.reactivedemo.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

@RestController
public class ReactiveController {

    // 단일 값을 반환하는 예제
    @GetMapping("/mono")
    public Mono<String> getMono() {
        // Mono는 단일 값이나 빈 값을 비동기로 반환하는 리액티브 타입입니다.
        return Mono.just("Hello, Mono!")
                   .map(String::toUpperCase);  // 데이터를 변환할 수 있습니다.
    }

    // 다중 값을 반환하는 예제
    @GetMapping("/flux")
    public Flux<String> getFlux() {
        // Flux는 여러 값을 비동기로 반환하는 리액티브 타입입니다.
        List<String> data = Arrays.asList("Hello", "World", "from", "Flux");
        return Flux.fromIterable(data)
                   .delayElements(Duration.ofSeconds(1));  // 각 요소를 1초씩 지연시킵니다.
    }

    // 숫자 스트림을 반환하는 예제
    @GetMapping("/numbers")
    public Flux<Integer> getNumbers() {
        // Flux를 사용하여 1부터 5까지의 숫자를 비동기로 반환합니다.
        return Flux.range(1, 5)
                   .map(i -> i * 2)  // 각 숫자를 두 배로 변환합니다.
                   .delayElements(Duration.ofSeconds(1));  // 각 요소를 1초씩 지연시킵니다.
    }

    // 비동기 에러 처리 예제
    @GetMapping("/error")
    public Mono<String> getError() {
        // Mono를 사용하여 에러를 발생시키고 처리합니다.
        return Mono.error(new RuntimeException("Unexpected Error"))
                   .onErrorReturn("Error occurred, but it's handled!");  // 에러가 발생했을 때 기본 값을 반환합니다.
    }
}

```

ReactiveControllerTest.java   
```java
package com.example.reactivedemo;

import com.example.reactivedemo.controller.ReactiveController;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.WebFluxTest;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

@WebFluxTest(ReactiveController.class)
public class ReactiveControllerTest {

    @Autowired
    private WebTestClient webTestClient;

    @Test
    public void testMono() {
        // /mono 엔드포인트를 테스트합니다.
        webTestClient.get().uri("/mono")
                     .exchange()
                     .expectStatus().isOk()
                     .expectBody(String.class)
                     .isEqualTo("HELLO, MONO!");
    }

    @Test
    public void testFlux() {
        // /flux 엔드포인트를 테스트합니다.
        webTestClient.get().uri("/flux")
                     .exchange()
                     .expectStatus().isOk()
                     .expectBodyList(String.class)
                     .hasSize(4)
                     .contains("Hello", "World", "from", "Flux");
    }

    @Test
    public void testNumbers() {
        // /numbers 엔드포인트를 테스트합니다.
        webTestClient.get().uri("/numbers")
                     .exchange()
                     .expectStatus().isOk()
                     .expectBodyList(Integer.class)
                     .hasSize(5)
                     .contains(2, 4, 6, 8, 10);
    }

    @Test
    public void testError() {
        // /error 엔드포인트를 테스트합니다.
        webTestClient.get().uri("/error")
                     .exchange()
                     .expectStatus().isOk()
                     .expectBody(String.class)
                     .isEqualTo("Error occurred, but it's handled!");
    }
}

```  

위의 주요 매서드를 보면, 다음과 같다.  
getMono(): 단일 문자열을 대문자로 변환하여 Mono로 반환.  
getFlux(): 문자열 리스트를 Flux로 반환하며, 각 요소를 1초씩 지연시킨다.  
getNumbers(): 1부터 5까지의 숫자를 두 배로 변환하여 Flux로 반환하며, 각 요소를 1초씩 지연시킨다.  
getError(): 에러를 발생시키고 이를 기본 값으로 처리하여 반환.  

# 톰캣 vs 네티  
1. 톰캣(Tomcat)  
    전통적인 서블릿 컨테이너로, 요청-응답 기반의 동기식 처리 모델을 따른다.  
    기본적으로 리액티브 기능을 지원하지 않으며, 블로킹 I/O를 사용.  
2. 네티(Netty)  
    비동기 네트워크 애플리케이션 프레임워크로, 비동기 및 논블로킹 I/O를 지원.  
    리액티브 애플리케이션에서 네티는 데이터 스트림의 효율적인 처리를 지원.  
    스프링 웹플럭스에서는 네티를 기본 내장 서버로 사용할 수 있다.  

의존성에  
Spring Web  
Spring Reactive Web  
을 추가해서 파일을 만들고, 아래의 예제를 따라가보자.  

pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>server-comparison</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>server-comparison</name>
    <description>Comparison of Tomcat and Netty in Spring Boot</description>
    <properties>
        <java.version>11</java.version>
        <spring-boot.version>2.7.5</spring-boot.version>
    </properties>
    <dependencies>
        <!-- Spring Boot Starter Web for Tomcat -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!-- Spring Boot Starter Webflux for Netty -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>

```

ServerComparisonApplication.java  
```java
package com.example.servercomparison;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ServerComparisonApplication {

    public static void main(String[] args) {
        SpringApplication.run(ServerComparisonApplication.class, args);
    }
}

```

HelloController.java  
```java
package com.example.servercomparison.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
public class HelloController {

    @GetMapping("/hello")
    public String hello() {
        return "Hello, Tomcat!";
    }

    @GetMapping("/hello-reactive")
    public Mono<String> helloReactive() {
        return Mono.just("Hello, Netty!");
    }
}

```

application.properties  
```properties
server.port=8080
```
Spring Boot는 내장된 톰캣 서버를 사용한다.  
따로 설정하지 않아도 된다.  

네티(Netty) 사용 설정은 톰캣 의존성을 제외하고 네티를 추가해야한다.  
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
    <exclusions>
        <exclusion>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-tomcat</artifactId>
        </exclusion>
    </exclusions>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-reactor-netty</artifactId>
</dependency>

```  
위와 갔이 네티를 추가할 수 있다.  
코드를 실행하고, URL을 통해 테스트 한다.  
톰캣: http://localhost:8080/hello  
네티: http://localhost:8080/hello-reactive  

# 리액티브 데이터 엑세스  
데이터베이스와의 상호작용을 비동기적으로 처리하는 방법이다.  
전통적인 데이터 엑세스는 동기식이며, 리액티브 방식은 비동기 및 논블로킹 접근을 통해 데이터베이스 작업을 수행할 수 있다.  
### R2DBC (Reactive Relational Database Connectivity)  
액티브 데이터베이스 클라이언트 API로, SQL 데이터베이스와의 리액티브 데이터 엑세스를 지원한다.  
전통적인 JDBC와는 달리 비동기 작업을 지원한다.  
예제로, H2 데이터베이스를 사용하여 리액티브 SQL 연산을 수행하겠다.  

파일구조는 다음과 같다.  
reactive-r2dbc-example
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── reactiver2dbc
│   │   │               ├── Person.java
│   │   │               ├── PersonRepository.java
│   │   │               ├── PersonService.java
│   │   │               ├── PersonController.java
│   │   │               ├── ReactiveR2dbcExampleApplication.java
│   │   │               └── DataInitializer.java
│   │   └── resources
│   │       └── application.properties
├── pom.xml


pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>reactive-r2dbc-example</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>reactive-r2dbc-example</name>
    <description>Demo project for reactive R2DBC</description>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.0.0</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-r2dbc</artifactId>
        </dependency>
        <dependency>
            <groupId>io.r2dbc</groupId>
            <artifactId>r2dbc-h2</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>

```  

application.properties  
```properties
spring.r2dbc.url=r2dbc:h2:mem:///testdb
spring.r2dbc.username=sa
spring.r2dbc.password=password
spring.h2.console.enabled=true
```  

Person.java
Person 엔터티 클래스  
간단한 Person 엔터티 클래스를 생성  
```java
package com.example.reactiver2dbc;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

@Table("person")
public class Person {
    @Id
    private Long id;
    private String name;

    // Getters and setters
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
}

```  

PersonRepository.java
리포지토리 인터페이스  
```java
package com.example.reactiver2dbc;

import org.springframework.data.repository.reactive.ReactiveCrudRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PersonRepository extends ReactiveCrudRepository<Person, Long> {
}

```

PersonService.java
서비스 클래스
```java
package com.example.reactiver2dbc;

import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Service
public class PersonService {
    private final PersonRepository personRepository;

    public PersonService(PersonRepository personRepository) {
        this.personRepository = personRepository;
    }

    public Mono<Person> savePerson(Person person) {
        return personRepository.save(person);
    }

    public Flux<Person> getAllPersons() {
        return personRepository.findAll();
    }

    public Mono<Person> getPersonById(Long id) {
        return personRepository.findById(id);
    }

    public Mono<Void> deletePerson(Long id) {
        return personRepository.deleteById(id);
    }
}

```

PersonController.java
리액티브 REST 컨트롤러
```java
package com.example.reactiver2dbc;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/persons")
public class PersonController {
    private final PersonService personService;

    public PersonController(PersonService personService) {
        this.personService = personService;
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<Person> createPerson(@RequestBody Person person) {
        return personService.savePerson(person);
    }

    @GetMapping
    public Flux<Person> getAllPersons() {
        return personService.getAllPersons();
    }

    @GetMapping("/{id}")
    public Mono<Person> getPersonById(@PathVariable Long id) {
        return personService.getPersonById(id);
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public Mono<Void> deletePerson(@PathVariable Long id) {
        return personService.deletePerson(id);
    }
}
``` 

ReactiveR2dbcExampleApplication.java
메인 애플리케이션 클래스
```java
package com.example.reactiver2dbc;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ReactiveR2dbcExampleApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReactiveR2dbcExampleApplication.class, args);
    }
}

```  

DataInitializer.java
초기 데이터 로딩
```java
package com.example.reactiver2dbc;

import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DataInitializer {

    @Bean
    public CommandLineRunner init(PersonRepository repository) {
        return args -> {
            repository.save(new Person(null, "John Doe")).subscribe();
            repository.save(new Person(null, "Jane Doe")).subscribe();
        };
    }
}

```  

스프링 부트와 R2DBC를 사용하여 리액티브 데이터베이스 접근을 구현한다.  
위의 역할을 정리하면, 다음과 같다.  
Person.java: 데이터베이스 테이블에 매핑되는 엔터티 클래스  
PersonRepository.java: 리액티브 CRUD 리포지토리 인터페이스  
PersonService.java: 비즈니스 로직을 처리하는 서비스 클래스  
PersonController.java: RESTful API를 제공하는 컨트롤러 클래스  
ReactiveR2dbcExampleApplication.java: 스프링 부트 애플리케이션의 메인 클래스  
DataInitializer.java: 초기 데이터를 로딩하는 설정 클래스  
application.properties: 애플리케이션 설정 파일  
pom.xml: 메이븐 프로젝트 설정 파일  


# 선언형 프로그래밍  
선언형 프로그래밍은 프로그램의 동작을 '무엇을 해야 하는지' 기술하는 방식이다. 리액티브 프로그래밍에서 선언형 접근은 데이터 흐름을 선언적으로 표현하고, 그에 따라 비즈니스 로직을 정의한다.  
간단한 예제로 방법만 보자. 다음은 javascript로 만든 예제이다.   
```javascript
// RxJS 라이브러리 임포트
import { fromEvent } from 'rxjs';
import { map, filter, debounceTime, switchMap } from 'rxjs/operators';

// 입력 요소 가져오기
const searchBox = document.getElementById('search-box');

// 'input' 이벤트 스트림 생성
const keyup$ = fromEvent(searchBox, 'input');

// API 호출 함수 (예시)
const fetchSearchResults = (query) => {
  return fetch(`https://api.example.com/search?q=${query}`)
    .then(response => response.json());
};

// 선언형 방식으로 데이터 흐름 정의
const search$ = keyup$.pipe(
  debounceTime(300), // 입력 후 300ms 대기
  map(event => event.target.value), // 입력 값을 추출
  filter(query => query.length > 2), // 입력 값이 2글자 이상일 때만 진행
  switchMap(query => fetchSearchResults(query)) // 새로운 쿼리가 들어오면 이전 쿼리 취소하고 새로운 쿼리 실행
);

// 결과 구독 및 처리
search$.subscribe(results => {
  // 결과를 화면에 표시하는 로직 (예시)
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = results.map(result => `<li>${result.name}</li>`).join('');
});

```
fromEvent: searchBox 요소의 input 이벤트를 옵저버블 스트림으로 만든다.  
pipe: 연산자를 연결하여 데이터 흐름을 정의한다.  
debounceTime(300): 사용자가 입력을 멈춘 후 300ms 대기하여 너무 빈번한 호출을 방지  
map(event => event.target.value): 이벤트에서 입력 값을 추출  
filter(query => query.length > 2): 입력 값이 2글자 이상인 경우에만 다음 연산으로 진행  
switchMap(query => fetchSearchResults(query)): 입력 값이 변경될 때마다 새로운 검색 요청을 보내며, 이전 요청은 취소  
subscribe: 최종적으로 처리된 데이터를 구독하고 결과를 화면에 표시  

이 예제는 입력 이벤트를 처리하고 검색 요청을 보내는 과정을 선언형으로 정의하여, 데이터 흐름을 명확하게 나타내고 비즈니스 로직을 간결하게 작성한 것이다.  
선언형 프로그래밍의 주요 이점은 코드의 가독성과 유지보수성이 높아진다는 것이다.  


# 리액티브 Thymeleaf
Thymeleaf는 서버 사이드 템플릿 엔진으로, HTML 템플릿을 렌더링하는 데 사용된다.  
리액티브 Thymeleaf는 비동기 데이터 처리를 지원하여 템플릿에서 동적으로 데이터 스트림을 렌더링할 수 있다.  
예시를 통해 보면, 다음과 같다.  

pom.xml
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>reactive-thymeleaf</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <java.version>11</java.version>
        <spring.boot.version>2.5.4</spring.boot.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>
        <dependency>
            <groupId>org.thymeleaf.extras</groupId>
            <artifactId>thymeleaf-extras-springsecurity5</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
```  

ReactiveThymeleafApplication.java
```java
package com.example.reactivethymeleaf;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ReactiveThymeleafApplication {
    public static void main(String[] args) {
        SpringApplication.run(ReactiveThymeleafApplication.class, args);
    }
}

```  

GreetingController.java  
```java
package com.example.reactivethymeleaf.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.stream.Stream;

@Controller
public class GreetingController {

    @GetMapping("/greetings")
    public String getGreetings(Model model) {
        Flux<String> greetingFlux = Flux.fromStream(Stream.generate(() -> "Hello, Reactive Thymeleaf!"))
                .delayElements(Duration.ofSeconds(1))
                .take(10);

        model.addAttribute("greetings", greetingFlux);
        return "greetings";
    }
}

```

greetings.html  
src/main/resources/templates/greetings.html 이 위치에 있어야 한다.  
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Reactive Greetings</title>
</head>
<body>
    <h1>Greetings</h1>
    <ul>
        <li th:each="greeting : ${greetings}" th:text="${greeting}"></li>
    </ul>
</body>
</html>

```
앱을 실행하면, 리액티브 Thymeleaf를 사용하여 비동기 데이터 스트림을 렌더링 하는 것을 볼 수 있다.  
데이터는 1초 간격으로 업데이트되며, 10개의 "Hello, Reactive Thymeleaf!" 메시지가 표시된다.  


# RSocket
RSocket은 리액티브 프로그래밍을 위한 완전한 프로토콜로, 효율적인 프로세스 간 통신을 지원한다. RSocket은 다음과 같은 기능을 제공한다.  
Request-Response: 요청에 대한 응답을 받는 전통적인 통신 방식.  
Fire-and-Forget: 요청을 보내고 응답을 기다리지 않는 방식.  
Request-Stream: 요청 후, 데이터 스트림을 수신하는 방식.  
Channel: 양방향 스트림을 처리하는 방식.  
RSocket은 특히 마이크로서비스 아키텍처에서 유용하며, 낮은 지연시간과 높은 처리량을 지원한다.  
RSocket을 사용하여 Request-Response, Fire-and-Forget, Request-Stream, Channel 기능을 구현해 보자.  

```arduino
rsocket-demo
│
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── rsocketdemo
│   │   │               ├── RSocketClient.java
│   │   │               ├── RSocketConfig.java
│   │   │               ├── RSocketController.java
│   │   │               └── Application.java
│   │   └── resources
│   │       └── application.properties
│   └── test
│       └── java
│           └── com
│               └── example
│                   └── rsocketdemo
│                       └── RSocketDemoApplicationTests.java
└── pom.xml

```  

pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>rsocket-demo</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.0.0</version>
        <relativePath/> <!-- 상위 POM 파일을 찾는다 -->
    </parent>

    <properties>
        <java.version>17</java.version> <!-- 사용할 Java 버전 -->
    </properties>

    <dependencies>
        <!-- Spring Boot RSocket 스타터 의존성 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-rsocket</artifactId>
        </dependency>
        <!-- Spring Boot 기본 기능을 위한 스타터 의존성 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <!-- Spring Boot 웹 애플리케이션을 위한 스타터 의존성 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!-- 리액티브 프로그래밍을 위한 Reactor Core 의존성 -->
        <dependency>
            <groupId>io.projectreactor</groupId>
            <artifactId>reactor-core</artifactId>
        </dependency>
        <!-- 리액티브 네트워크 지원을 위한 Reactor Netty 의존성 -->
        <dependency>
            <groupId>io.projectreactor.netty</groupId>
            <artifactId>reactor-netty</artifactId>
        </dependency>
        <!-- 애플리케이션 테스트를 위한 Spring Boot 테스트 스타터 의존성 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope> <!-- 테스트 전용 -->
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Spring Boot 애플리케이션 패키징을 위한 Maven 플러그인 -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>

```

RSocketController.java  
```java
package com.example.rsocketdemo;

import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.stereotype.Controller;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.LocalTime;

@Controller
public class RSocketController {

    // Request-Response: 요청을 받고 하나의 응답을 반환.
    @MessageMapping("request-response")
    public Mono<String> requestResponse(String message) {
        return Mono.just("받은 메시지: " + message + " 시간: " + LocalTime.now());
    }

    // Fire-and-Forget: 요청을 받고 응답을 반환하지 않는다.
    @MessageMapping("fire-and-forget")
    public Mono<Void> fireAndForget(String message) {
        System.out.println("받은 메시지: " + message + " 시간: " + LocalTime.now());
        return Mono.empty(); // 응답을 보내지 않음
    }

    // Request-Stream: 요청을 받고 데이터 스트림을 응답으로 반환.
    @MessageMapping("request-stream")
    public Flux<String> requestStream(String message) {
        return Flux.interval(Duration.ofSeconds(1)) // 1초마다 새로운 항목 방출
                   .map(index -> "스트림 응답 " + index + " 메시지: " + message + " 시간: " + LocalTime.now())
                   .take(10); // 10개 항목으로 제한
    }

    // Channel: 메시지 스트림을 받고 응답 스트림을 반환.
    @MessageMapping("channel")
    public Flux<String> channel(Flux<String> messages) {
        return messages.map(message -> "처리된 메시지: " + message + " 시간: " + LocalTime.now());
    }
}

```

RSocketConfig.java  
```java
package com.example.rsocketdemo;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.rsocket.RSocketRequester;
import org.springframework.messaging.rsocket.RSocketStrategies;
import org.springframework.messaging.rsocket.annotation.support.RSocketMessageHandler;
import org.springframework.web.util.pattern.PathPatternRouteMatcher;

@Configuration
public class RSocketConfig {

    // RSocket 메시지 핸들러를 구성하고 라우트 매처와 전략을 설정.
    @Bean
    public RSocketMessageHandler messageHandler(RSocketStrategies strategies) {
        RSocketMessageHandler handler = new RSocketMessageHandler();
        handler.setRSocketStrategies(strategies);
        handler.setRouteMatcher(new PathPatternRouteMatcher()); // 라우트 매칭을 위한 설정
        return handler;
    }

    // RSocketRequester를 빌드하기 위한 빌더를 제공.
    @Bean
    public RSocketRequester.Builder rSocketRequesterBuilder(RSocketStrategies strategies) {
        return RSocketRequester.builder().rsocketStrategies(strategies);
    }
}

```

Application.java  
```java
package com.example.rsocketdemo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RSocketDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(RSocketDemoApplication.class, args); // Spring Boot 애플리케이션을 시작.
    }
}

```  

application.properties  
```properties  
# RSocket 서버가 7000번 포트에서 수신하도록 설정.
spring.rsocket.server.port=7000
```  

RSocketClient.java  
```java
package com.example.rsocketdemo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.messaging.rsocket.RSocketRequester;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;

@Component
public class RSocketClient implements CommandLineRunner {

    @Autowired
    private RSocketRequester.Builder rSocketRequesterBuilder;

    @Override
    public void run(String... args) throws Exception {
        // RSocketRequester를 빌드하고 localhost:7000에 연결합니다.
        RSocketRequester rSocketRequester = rSocketRequesterBuilder
                .connectTcp("localhost", 7000)
                .block();

        // Request-Response: 요청을 보내고 단일 응답을 기다립니다.
        rSocketRequester.route("request-response")
                        .data("안녕하세요, RSocket!")
                        .retrieveMono(String.class)
                        .doOnNext(response -> System.out.println("응답: " + response))
                        .block();

        // Fire-and-Forget: 요청을 보내고 응답을 기다리지 않습니다.
        rSocketRequester.route("fire-and-forget")
                        .data("Fire-and-Forget 메시지")
                        .send()
                        .doOnTerminate(() -> System.out.println("Fire-and-Forget 전송됨"))
                        .subscribe();

        // Request-Stream: 요청을 보내고 응답 스트림을 처리합니다.
        rSocketRequester.route("request-stream")
                        .data("스트림 요청")
                        .retrieveFlux(String.class)
                        .take(10) // 10개 응답으로 제한
                        .doOnNext(System.out::println)
                        .blockLast();

        // Channel: 메시지 스트림을 보내고 응답 스트림을 처리합니다.
        Flux<String> messages = Flux.interval(Duration.ofSeconds(1))
                                    .map(index -> "메시지 " + index)
                                    .take(10);

        rSocketRequester.route("channel")
                        .data(messages)
                        .retrieveFlux(String.class)
                        .doOnNext(System.out::println)
                        .blockLast();
    }
}

```  
위 코드를 정리하면 다음과 같다.  
1. RSocketController.java  
역할: RSocket 클라이언트 요청을 처리하는 컨트롤러  
Request-Response: 클라이언트의 요청에 대해 단일 응답을 반환  
Fire-and-Forget: 클라이언트의 요청을 받고 응답을 반환하지 않는다.  
Request-Stream: 클라이언트의 요청에 대해 데이터 스트림을 반환.  
Channel: 클라이언트가 보내는 메시지 스트림을 처리하고, 응답으로 메시지 스트림을 반환.  

2. RSocketConfig.java  
역할: RSocket 관련 설정을 구성하는 클래스.  
RSocketMessageHandler: RSocket 메시지를 처리하기 위한 핸들러를 설정.  
RSocketRequester.Builder: RSocket 클라이언트를 생성하기 위한 빌더를 제공.  

3. Application.java  
역할: Spring Boot 애플리케이션의 진입점입니다. main 메서드를 통해 애플리케이션을 실행.  

4. RSocketClient.java  
역할: RSocket 서버와의 상호작용을 테스트하기 위한 클라이언트.  
Request-Response: RSocket 서버에 요청을 보내고 응답을 출력.  
Fire-and-Forget: RSocket 서버에 요청을 보내고 응답을 기다리지 않는다.  
Request-Stream: RSocket 서버에 요청을 보내고 스트림 응답을 출력.  
Channel: 메시지 스트림을 서버로 보내고, 서버로부터의 응답 스트림을 출력.  

RSocket을 적용하면, 데이터 스트림의 실시간 처리와 양방향 통신이 가능해 진다. 이를 통해 다음과 같은 장점이 생긴다.  
효율적인 데이터 전송: 데이터 전송 지연을 최소화하고, 실시간 데이터를 빠르게 전송할 수 있다.  
비동기 통신: 요청과 응답이 비동기적으로 처리되어 성능을 향상시킨다.  

# 웹 소켓
웹 클라이언트와 서버 간에 지속적이고 양방향 통신을 가능하게 하는 프로토콜이다.  
HTTP와는 달리, 웹 소켓은 연결이 유지되는 동안 클라이언트와 서버 간에 실시간으로 데이터를 주고 받을 수 있으며,  
이 특성 덕분에 실시간 채팅 애플리케이션, 실시간 알림 시스템 등에서 유용하게 사용된다.  

```arduino
my-websocket-app/
│
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── example/
│   │   │           └── websocketdemo/
│   │   │               ├── WebSocketDemoApplication.java
│   │   │               ├── config/
│   │   │               │   └── WebSocketConfig.java
│   │   │               └── handler/
│   │   │                   └── WebSocketHandler.java
│   │   ├── resources/
│   │   │   ├── static/
│   │   │   │   └── index.html
│   │   │   └── application.properties
│   └── test/
│       └── java/
│           └── com/
│               └── example/
│                   └── websocketdemo/
│                       └── WebSocketDemoApplicationTests.java
│
├── .gitignore
├── pom.xml
└── README.md

```

pom.xml  
```xml
<dependencies>
    <!-- Spring Boot Starter Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>

    <!-- Spring Boot Starter WebSocket -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-websocket</artifactId>
    </dependency>
</dependencies>

```

WebSocketConfig.java  
웹 소켓을 구성하는 설정 클래스 웹 소켓 핸들러를 등록하고 웹 소켓 경로를 설정  
```java
package com.example.websocketdemo.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import com.example.websocketdemo.handler.WebSocketHandler;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new WebSocketHandler(), "/ws/chat").setAllowedOrigins("*");
    }
}

```

WebSocketHandler.java  
웹 소켓 메시지 송수신을 처리하는 핸들러 클래스 클라이언트의 연결 및 메시지 처리 로직을 구현  
```java
package com.example.websocketdemo.handler;

import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.handler.TextWebSocketHandler;
import org.springframework.web.socket.WebSocketMessage;

import java.io.IOException;
import java.util.concurrent.CopyOnWriteArrayList;

public class WebSocketHandler extends TextWebSocketHandler {

    private final CopyOnWriteArrayList<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws IOException {
        for (WebSocketSession s : sessions) {
            if (s.isOpen() && !s.equals(session)) {
                s.sendMessage(message);
            }
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        sessions.remove(session);
    }
}

```  

index.html  
```html
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Demo</title>
    <script>
        let socket;
        
        function init() {
            socket = new WebSocket('ws://localhost:8080/ws/chat');
            
            socket.onmessage = function(event) {
                const message = event.data;
                const messages = document.getElementById('messages');
                const messageElement = document.createElement('div');
                messageElement.textContent = message;
                messages.appendChild(messageElement);
            };

            document.getElementById('sendButton').onclick = function() {
                const messageInput = document.getElementById('messageInput');
                const message = messageInput.value;
                socket.send(message);
                messageInput.value = '';
            };
        }
    </script>
</head>
<body onload="init()">
    <h1>WebSocket Chat</h1>
    <div id="messages"></div>
    <input type="text" id="messageInput" placeholder="Enter message" />
    <button id="sendButton">Send</button>
</body>
</html>

```  
이 구조를 기반으로 스프링 부트 웹 소켓 애플리케이션을 조직적으로 관리할 수 있다.  


# 오늘 한거 복습  
오늘 한 내용들을 총합해서, PlaneFinder 예제를 만들어 보자.  
RSocket을 사용하여 PlaneFinder 앱의 비행기 위치 데이터를 실시간으로 스트리밍하는 예제이다.  

다음과 같은 파일 구조로 만들어 보자.  
```arduino
plane-finder-app/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── example/
│   │   │           └── planefinder/
│   │   │               ├── PlaneFinderApplication.java
│   │   │               ├── config/
│   │   │               │   └── RSocketConfig.java
│   │   │               ├── controller/
│   │   │               │   └── AircraftController.java
│   │   │               └── service/
│   │   │                   └── AircraftService.java
│   │   └── resources/
│   │       ├── application.properties
│   │       └── static/
│   │           └── index.html
└── pom.xml

```   
pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>planefinder-app</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>PlaneFinder App</name>
    <description>PlaneFinder app with RSocket</description>

    <properties>
        <java.version>17</java.version>
        <spring.boot.version>3.2.0</spring.boot.version>
        <rsocket.version>1.1.0</rsocket.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-rsocket</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
        <dependency>
            <groupId>io.projectreactor</groupId>
            <artifactId>reactor-core</artifactId>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>

```  

PlaneFinderApplication.java  
```java
package com.example.planefinder;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class PlaneFinderApplication {

    public static void main(String[] args) {
        SpringApplication.run(PlaneFinderApplication.class, args);
    }
}

```  

RSocketConfig.java  
```java
package com.example.planefinder.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.rsocket.RSocketRequester;
import org.springframework.messaging.rsocket.RSocketRequester.Builder;
import org.springframework.messaging.rsocket.RSocketRequester.RequesterSpec;

@Configuration
public class RSocketConfig {

    @Bean
    public RSocketRequester.Builder rSocketRequesterBuilder() {
        return RSocketRequester.builder();
    }
}

```  

AircraftService.java  
```java
package com.example.planefinder.service;

import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

@Service
public class AircraftService {

    // 비행기 위치 데이터의 스트림을 반환합니다.
    public Flux<String> getAircraftPositionStream() {
        // 실제로는 외부 API나 데이터베이스에서 비행기 위치를 가져오는 로직이 필요합니다.
        // 여기서는 예제 데이터로 대체합니다.
        return Flux.interval(Duration.ofSeconds(1))
                   .map(sequence -> "Aircraft Position: " + sequence);
    }
}

```  

AircraftController.java
```java
package com.example.planefinder.controller;

import com.example.planefinder.service.AircraftService;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.stereotype.Controller;
import reactor.core.publisher.Flux;

@Controller
public class AircraftController {

    private final AircraftService aircraftService;

    public AircraftController(AircraftService aircraftService) {
        this.aircraftService = aircraftService;
    }

    @MessageMapping("aircraft.positions")
    public Flux<String> streamAircraftPositions() {
        return aircraftService.getAircraftPositionStream();
    }
}

```  

application.properties  
```properties
spring.rsocket.server.port=7000
spring.rsocket.server.transport=tcp

```  

index.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>PlaneFinder</title>
</head>
<body>
    <h1>PlaneFinder</h1>
    <div id="positions"></div>
    <script>
        const socket = new WebSocket('ws://localhost:7000/rsocket');

        socket.onmessage = function(event) {
            const positionsDiv = document.getElementById('positions');
            positionsDiv.innerHTML += `<p>${event.data}</p>`;
        };
    </script>
</body>
</html>

```

위 코드를 설명하면, 다음과 같다.  
pom.xml: Maven 의존성을 설정합니다. 스프링 부트와 RSocket 관련 의존성 추가.  
PlaneFinderApplication.java: 스프링 부트 애플리케이션의 시작점을 정의.  
RSocketConfig.java: RSocket을 설정하는 클래스를 정의.  
AircraftService.java: 비행기 위치 데이터의 스트림을 생성하는 서비스 클래스.  
AircraftController.java: 클라이언트 요청에 대해 비행기 위치 데이터를 스트리밍하는 컨트롤러.  
application.properties: RSocket 서버의 포트와 전송 방식을 설정.  
index.html: 웹 브라우저에서 비행기 위치를 실시간으로 확인할 수 있는 HTML 파일.  


# 결론
리액티브 프로그래밍은 비동기 데이터 스트림을 효율적으로 처리하는 데 유용한 패러다임 이다.  
자바 스프링 부트와 메이븐을 사용하여 리액티브 애플리케이션을 개발할 때, 프로젝트 리액터와 R2DBC를 활용하여 비동기 데이터 엑세스와 처리 성능을 극대화할 수 있다.  
또한, RSocket을 통해 완전한 리액티브 프로세스 간 통신을 구현할 수 있다.  
이를 통해 PlaneFinder와 Aircraft Positions와 같은 애플리케이션에서 실시간 데이터 스트리밍 및 처리 성능을 크게 향상시킬 수 있다.  