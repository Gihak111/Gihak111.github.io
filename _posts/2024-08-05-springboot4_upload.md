---
layout: single
title:  "spring boot 4. 애플리케이션 설정과 검사"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 애플리케이션 코드 디버깅  

코드 디버깅은 애플리케이션 내 동작을 구축, 식별, 분리하는 한 단계에 불과하다.  
동적이고 분산된 애플리케이션이 많아지면, 다음의 단계를 실행해야 한다.  
1. 애플리케이션의 동적 설정, 재설정  
2. 현재 설정과 출처의 확인과 결정  
3. 애플리케이션 환경과 헬스 지표의 검사, 모니터링  
4. 실행중인 애플리케이션의 로깅 수준을 일시적으로 조정해 오류 원인 식별  

지금부터 스프링 부트에 내장된 설정을 유연하게 해 주는 기능들을 살펴볼꺼다.  

## 1. Spring Boot DevTools  
개발 환경에서 앱 개발을 더욱 편리하게 해 준다.  
주요 기능  
1. 자동 재시작  
    코드 변경시에 애플리케이션을 자동으로 재시작  
2. LiveReload  
    브라우저를 자동으로 새로고침하여 변경 사항을 즉시 반영  
3. 개발 전용 설정  
    개발 환경에서만 활성화되는 설정을 지원. 프로덕션 환경과 분리된 설정 관리가 가능  

Spring Initializr에서 Add dependencies...를 통해 추가하거나,  
pom.xml 파일에 DevTools 의존성을 추가할 수 있다.  
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <optional>true</optional>
</dependency>

```  

## 2. @ConfigurationProperties  

외부 설정 파일(application.properties 또는 application.yml)의 값을 Java 클래스에 매핑해 준다.  
예제를 보자면,  
application.properties  
```properties
app.name=MyApp
app.description=This is a sample application

```  
설정 클래스  
``` java
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "app")
public class AppProperties {
    private String name;
    private String description;

    // Getter와 Setter
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}

```  
사용 예시  
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AppController {

    private final AppProperties appProperties;

    @Autowired
    public AppController(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    @GetMapping("/app-info")
    public String getAppInfo() {
        return "Name: " + appProperties.getName() + ", Description: " + appProperties.getDescription();
    }
}

```  

## 3. @Value  
특정 설정 값을 개별적으로 주입받을 때 사용  

설정 파일 application.properties  
```properties
app.name=MyApp

```  
주입 코드
```java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AppNamePrinter {

    @Value("${app.name}")
    private String appName;

    public void printAppName() {
        System.out.println("Application Name: " + appName);
    }
}

```

## 4. PropertySource  
추가적인 프로퍼티 파일을 로드할 때 사용  
설정 파일 application.properties  
```properties
app.version=1.0.0

```  

설정 로드 클래스  
```java
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@PropertySource("classpath:additional.properties")
public class AdditionalConfig {
    // 추가 설정 파일에서 값을 로드
}

```  

## 5. Profile 기반 설정  
프로파일을 통해 환경별 설정을 쉽게 관리할 수 있다.  
application-dev.properties  
```properties
app.environment=development

``` 

application-prod.properties  
```properties
app.environment=production

```  

프로파일 활성화 application.properties  
```properties
spring.profiles.active=dev

```  
설정 주입  
```java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class EnvironmentPrinter {

    @Value("${app.environment}")
    private String environment;

    public void printEnvironment() {
        System.out.println("Current Environment: " + environment);
    }
}

```  

## 6. YAML 설정 파일  
application.properties 파일 외에도, application.yml을 사용할 수 있다.  
application.yml  
```yaml
app:
  name: MyApp
  description: This is a sample application
```  

설정 클래스  
```java
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "app")
public class AppProperties {
    private String name;
    private String description;

    // Getter와 Setter
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}
```  

## 7. @TestPropertySource  
테스트 환경에서 별도의 설정 파일을 사용할 수 있도록 지원  
test.properties  
```properties
app.name=TestApp
```

테스트 클래스  
```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

@SpringBootTest
@TestPropertySource(locations = "classpath:test.properties")
public class AppPropertiesTest {

    @Autowired
    private AppProperties appProperties;

    @Test
    public void testAppName() {
        assertEquals("TestApp", appProperties.getName());
    }
}

```

## 8. 잠재적 서드 파티 옵션  
데이터베이스, 메시징 시스템, 캐시 등의 서드 파티 라이브러리와 쉽게 통합할 수 있다.  
Spring Data JPA와 통합 예제.  
```java
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
}

```  

## 9. 자동 설정 리부트  
애플리케이션 실행 시 자동으로 설정된 빈과 프로퍼티들을 리포트하는 기능  
pom.xml에 의존성 추가  
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

```  

application.properties 파일에서 Actuator 엔드포인트를 활성화  
```properties
management.endpoints.web.exposure.include=*
```  

## 10. 액추에이터  
애플리케이션의 모니터링과 관리를 위한 다양한 기능을 제공  
주요 엔드포인트  
1. /actuator/health: 애플리케이션의 상태를 확인한다  
2. /actuator/metrics: 애플리케이션의 메트릭 정보를 제공한다   
3. /actuator/env: 환경 변수와 설정 정보를 제공한다   
4. /actuator/beans: 애플리케이션 컨텍스트에서 관리하는 빈 정보를 제공한다  

pom.xml 파일에 Actuator 의존성을 추가  
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

```  
application.properties 파일에서 Actuator 엔드포인트를 활성화  
```properties
management.endpoints.web.exposure.include=*

```
애플리케이션을 실행하고, 브라우저에서 다양한 Actuator 엔드포인트에 접근하여 정보를 확인할 수 있다.  
ex) http://localhost:8080/actuator/health
다양한 기능들을 통해 스프링 부트는 애플리케이션 설정을 유연하게 관리하고, 개발 및 운영 환경에서 효과적으로 애플리케이션을 관리할 수 있다.  
아래 링크에서 더 많은 내용을 볼 수 있다.  
https://docs.spring.io/spring-boot/index.html  



이 본문은 책 spring boot up & running 처음부터 제대로 배우는 스프링 부트 책의 내용을 담고 있다.  