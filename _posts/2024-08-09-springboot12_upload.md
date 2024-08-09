---
layout: single
title:  "spring boot 12.애플리케이션 보안"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 인증과 인가  
스프링 서큐리티는 인증, 인가를 우한 옵션을 HTTP 방화벽, 필터체인, IETF와 W3C 표준의관범위한 사용 등등 의 메커니즘을 결합해 애플리케이션의 보안성을 높인다.  

인증: 사실, 또는 진짜를 보여주는 행위  
    누군가가 자신이 주장하는 사람임을 증명하기  
    용자가 누구인지 확인하는 과정  
인가: 권한 부여: 더 많은 정보에 접근할 권한을 부여  
    누군가가 특정 리소스나 작업에 접근할 수 있는지 확인하기  
    인증된 사용자가 애플리케이션의 특정 리소스에 접근할 수 있는 권한을 가지고 있는지 확인하는 과정  

## 스프링 시큐리티 의존성 추가  
스프링 프레임워크의 보안 모듈로, 애플리케이션 보안을 간편하게 구현할 수 있다.  
pom.xml에 의존성을 추가하는 것으로 구현할 수 있다.  
인증과 인가를 간편하게 구현할 수 있다.  
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-oauth2-client</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-oauth2-jose</artifactId>
</dependency>

```

각각의 의존성은 다음과 같다.  
spring-boot-starter-security  
스프링 시큐리티의 기본 기능을 제공. 인증과 인가를 위한 기본 설정과 필터를 자동으로 추가해 준다.  

spring-security-oauth2-client  
OAuth2 클라이언트 기능을 제공. 이를 통해 애플리케이션이 외부 OAuth2 제공자(구글, 페이스북)를 사용하여 사용자를 인증할 수 있다.  

spring-security-oauth2-jose  
이 의존성은 JSON Web Token(JWT)과 관련된 기능을 제공. OAuth2와 함께 사용하여 토큰 기반 인증을 구현할 때 유용하다.  
필터 체인을 통해 작동. HTTP 요청이 들어오면, 이 필터 체인을 통해 요청을 검사하고 필요한 인증과 인가 절차를 수행한다.  

## 보안 설정
위 의존성을 활용해서 코드를 작성해 보자.  
SecurityConfig.java  
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

// @Configuration: 스프링 컨텍스트에 이 클래스가 설정 클래스임을 알리는 어노테이션
@Configuration
// @EnableWebSecurity: 스프링 시큐리티의 웹 보안 지원을 활성화하는 어노테이션
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    // 사용자 인증을 설정하는 메소드
    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        // 메모리 기반 인증을 설정
        auth.inMemoryAuthentication()
            // 사용자 이름이 'user'인 사용자 추가
            .withUser("user")
            // 비밀번호 'password'를 BCrypt 해시로 암호화
            .password(passwordEncoder().encode("password"))
            // 사용자의 역할을 'USER'로 설정
            .roles("USER");
    }

    // HTTP 보안을 설정하는 메소드
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            // 모든 요청은 인증을 요구하도록 설정
            .authorizeRequests()
                .anyRequest().authenticated()
                .and()
            // 기본 로그인 폼을 사용하도록 설정
            .formLogin()
                .permitAll() // 로그인 페이지는 인증 없이 접근 가능
                .and()
            // 로그아웃을 허용하도록 설정
            .logout()
                .permitAll(); // 로그아웃도 인증 없이 접근 가능
    }

    // 비밀번호를 암호화하기 위한 PasswordEncoder 빈을 생성
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}

```

위 기능으로 user와 password를 사용하여 로그인을 구현할 수 있다.  

## HTTP 방화벽  
웹 애플리케이션을 보호하기 위해 사용되는 방화벽.  
스프링 시큐리티는 기본적으로 요청을 검사하여 XSS(Cross-Site Scripting) 및 SQL 인젝션 공격을 방지한다.  
```java
import org.springframework.security.web.firewall.HttpFirewall;
import org.springframework.security.web.firewall.StrictHttpFirewall;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

// @Configuration 애너테이션은 이 클래스가 하나 이상의 @Bean 메서드를 포함하고 있음을 나타내며,
// 스프링 컨테이너에서 이 클래스를 구성 클래스로 인식하게 한다.
@Configuration
public class FirewallConfig {

    // @Bean 애너테이션은 이 메서드가 반환하는 객체가 스프링 컨테이너에 의해 관리되는 빈(Bean)으로 등록된다는 것을 나타낸다.
    @Bean
    public HttpFirewall httpFirewall() {
        // StrictHttpFirewall 클래스의 인스턴스를 생성합니다. 이는 기본적으로 엄격한 HTTP 방화벽 정책을 적용한다.
        StrictHttpFirewall firewall = new StrictHttpFirewall();
        
        // URL에서 인코딩된 슬래시('/')를 허용하도록 설정합니다. 기본적으로는 허용되지 않는다.
        firewall.setAllowUrlEncodedSlash(true);
        
        // 설정된 StrictHttpFirewall 객체를 반환하여, 스프링 컨테이너에서 빈으로 관리하도록 한다.
        return firewall;
    }
}

```
StrictHttpFirewall을 사용하여 기본적으로 엄격한 HTTP 요청 검사를 수행하지만,  
URL에 인코딩된 슬래시를 허용하도록 설정한다.  

## 보안 필터 체인  
위에서 사용한 보안 필터 체인 이다.  
애플리케이션의 HTTP 요청 및 응답을 처리하는 일련의 필터들을 체인 형태로 구성한 것이며,  
필터들은 스프링 시큐리티에서 제공하는 기능을 이용하여 요청이 애플리케이션의 컨트롤러로 전달되기 전에 다양한 보안 검사를 수행한다.  
각 필터는 특정한 보안 기능을 담당(인증, 권한 부여, 세션 관리, CSRF 등)  

보안 설정에서 사용한 코드를 재활용 해서 어떤 필더가 어떻게 구서오디어 이쓴ㄴ지 알아보자.
SecurityConfig.java  
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

// @Configuration: 스프링 컨텍스트에 이 클래스가 설정 클래스임을 알리는 어노테이션
@Configuration
// @EnableWebSecurity: 스프링 시큐리티의 웹 보안 지원을 활성화하는 어노테이션
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    // 사용자 인증을 설정하는 메소드
    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        // 메모리 기반 인증을 설정
        auth.inMemoryAuthentication()
            // 사용자 이름이 'user'인 사용자 추가
            .withUser("user")
            // 비밀번호 'password'를 BCrypt 해시로 암호화
            .password(passwordEncoder().encode("password"))
            // 사용자의 역할을 'USER'로 설정
            .roles("USER");
    }

    // HTTP 보안을 설정하는 메소드
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            // 모든 요청은 인증을 요구하도록 설정
            .authorizeRequests()
                .anyRequest().authenticated()
                .and()
            // 기본 로그인 폼을 사용하도록 설정
            .formLogin()
                .permitAll() // 로그인 페이지는 인증 없이 접근 가능
                .and()
            // 로그아웃을 허용하도록 설정
            .logout()
                .permitAll(); // 로그아웃도 인증 없이 접근 가능
    }

    // 비밀번호를 암호화하기 위한 PasswordEncoder 빈을 생성
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}

```

필터 체인의 동작 과정  
요청 초기화: SecurityContextPersistenceFilter가 요청 시작 시 SecurityContext를 복원. Spring Security의 기본 설정으로 SecurityContextPersistenceFilter는 요청의 SecurityContext를 설정. HttpSecurity 설정이 적용되기 전에 자동으로 처리   
헤더 추가: HeaderWriterFilter가 응답 헤더에 보안 관련 헤더를 추가. HttpSecurity 설정 시 자동으로 응답 헤더에 보안 관련 헤더를 추가. 기본적으로 HttpSecurity가 설정되면 활성화  
CSRF 보호: CsrfFilter가 CSRF 토큰을 생성 및 검증. HttpSecurity 설정이 기본적으로 CSRF 보호를 활성화.  별도로 CSRF 설정을 비활성화하지 않았기 때문에, 이 필터는 기본적으로 활성화  
인증 처리: UsernamePasswordAuthenticationFilter가 로그인 폼 데이터를 기반으로 사용자를 인증.  
```java
.formLogin()
    .permitAll()
```
익명 사용자 설정: AnonymousAuthenticationFilter가 인증되지 않은 사용자를 익명 사용자로 설정. HttpSecurity에서 인증되지 않은 사용자를 익명으로 설정하는 필터는 기본적으로 활성화  명 사용자 설정을 별도로 구성하지 않았기 때문에, 기본 설정이 사용됨.  
권한 검사: FilterSecurityInterceptor가 사용자의 요청에 대해 접근 권한을 검사.  
```java
.authorizeRequests()
    .anyRequest().authenticated()

```
예외 처리: ExceptionTranslationFilter가 인증 및 권한 부여 과정에서 발생하는 예외를 처리. 인증 및 권한 부여 과정에서 발생하는 예외를 처리하는 필터는 HttpSecurity의 기본 설정에 포함 예외 처리는 필터 체인에서 자동으로 처리  
로그아웃 처리: LogoutFilter가 로그아웃 요청을 처리.  
```java
.logout()
    .permitAll()

```
세션 관리: SessionManagementFilter가 세션 고정 공격을 방지. HttpSecurity 설정이 세션 관리와 관련된 기본 보안 설정을 포함. 추가적인 세션 관리 설정을 하지 않았기 때문에, 기본 설정이 사용  

요약하면,  
요청이 들어오면: 클라이언트의 HTTP 요청이 애플리케이션에 도달한다.  
필터 체인 시작: 요청은 필터 체인의 첫 번째 필터로 전달된다.  
필터 처리: 각 필터는 자신이 담당하는 보안 검사를 수행한다.  
필터 체인 종료: 필터 체인의 마지막 필터까지 요청이 처리되면, 최종적으로 컨트롤러에 요청이 전달됩니다.  
이렇게 돌아간다.  

## 요청 밑 응답 해더  
바로 위에서 본, 요청과 해더이다.  
클라이언트와 서버 간의 메타데이터를 전달하는데 사용도니다.  
여러 보안 관련 헤더를 자동으로 추가하여 보안을 강화한다.  

## 스프링 시큐리티로 폼 기반 인증 및 인가 구현  
스프링 시큐리티를 사용하여 폼 기반 인증 및 인가를 구현할 수 있다.  
사용자는 로그인 폼을 통해 인증하고, 인증된 후 특정 리소스에 접근할 수 있다.  
폼 기반 인증은 사용자 인증을 처리하는 기본적인 방법 중 하나로, 사용자가 로그인 폼을 통해 인증 정보를 제공하면 스프링 시큐리티가 이를 검증하고 사용자 세션을 생성한다.  
예시를 통해 보자.  
pom.xml
```java
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>

```

FormLoginSecurityConfig  
WebSecurityConfigurerAdapter를 확장하여 보안 구성을 정의한다.  
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;

@Configuration
@EnableWebSecurity
public class FormLoginSecurityConfig extends WebSecurityConfigurerAdapter {

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .authorizeRequests()
                .antMatchers("/login", "/resources/**").permitAll() // 로그인 페이지와 정적 리소스는 인증 없이 접근 허용
                .anyRequest().authenticated() // 나머지 요청은 인증이 필요함
                .and()
            .formLogin()
                .loginPage("/login") // 커스텀 로그인 페이지 URL
                .loginProcessingUrl("/authenticate") // 로그인 폼에서 전송될 URL
                .defaultSuccessUrl("/home", true) // 로그인 성공 시 이동할 기본 페이지
                .permitAll() // 로그인 페이지는 모든 사용자에게 접근 허용
                .and()
            .logout()
                .logoutUrl("/logout") // 로그아웃 URL
                .logoutSuccessUrl("/login?logout") // 로그아웃 후 이동할 페이지
                .permitAll(); // 로그아웃 페이지는 모든 사용자에게 접근 허용
    }

    @Bean
    @Override
    public UserDetailsService userDetailsService() {
        // 메모리 내 사용자 저장소 설정
        InMemoryUserDetailsManager manager = new InMemoryUserDetailsManager();
        manager.createUser(User.withDefaultPasswordEncoder()
                .username("user")
                .password("password")
                .roles("USER")
                .build());
        manager.createUser(User.withDefaultPasswordEncoder()
                .username("admin")
                .password("admin")
                .roles("ADMIN")
                .build());
        return manager;
    }
}

```

login.html
로그인 폼을 정의. 탬플릿에 정의한다.
```java
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Login</title>
</head>
<body>
    <h2>Login Page</h2>
    <form action="/authenticate" method="post">
        <div>
            <label for="username">Username:</label>
            <input type="text" id="username" name="username"/>
        </div>
        <div>
            <label for="password">Password:</label>
            <input type="password" id="password" name="password"/>
        </div>
        <div>
            <button type="submit">Login</button>
        </div>
    </form>
</body>
</html>

```

로그아웃은 스프링 시큐리티의 기본 로그아웃 처리를 사용하거나, 커스텀 처리를 추가할 수 있다.  
위에 만든 login 마냥 만들면 된다.  

## 인증 및 인가를 위한 OIDC와 OAuth2 구현
현대 애플리케이션에서 인증 및 인가를 구현하는 데 자주 사용되는 프로토콜이며,  
스프링 시큐리티는 이를 간편하게 통합할 수 있는 기능을 제공한다.  
외부 인증 제공자를 통해 애플리케이션의 인증 및 인가 기능을 강화하는 좋은 방법이다.  

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-oauth2-client</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-thymeleaf</artifactId>
    </dependency>
    <!-- Optional: If you need to support Thymeleaf templates -->
    <dependency>
        <groupId>org.thymeleaf.extras</groupId>
        <artifactId>thymeleaf-extras-springsecurity5</artifactId>
    </dependency>
</dependencies>

```

application.properties
```priperties
spring.security.oauth2.client.registration.google.client-id=YOUR_CLIENT_ID
spring.security.oauth2.client.registration.google.client-secret=YOUR_CLIENT_SECRET
spring.security.oauth2.client.registration.google.scope=profile,email
spring.security.oauth2.client.registration.google.redirect-uri={baseUrl}/login/oauth2/code/{registrationId}
spring.security.oauth2.client.registration.google.authorization-grant-type=authorization_code

spring.security.oauth2.client.provider.google.authorization-uri=https://accounts.google.com/o/oauth2/auth
spring.security.oauth2.client.provider.google.token-uri=https://oauth2.googleapis.com/token
spring.security.oauth2.client.provider.google.user-info-uri=https://www.googleapis.com/oauth2/v3/userinfo
spring.security.oauth2.client.provider.google.user-name-attribute=sub

```  
OAuth2와 OIDC에 대한 설정을 추가  


SecurityConfig  
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.oauth2.client.web.OAuth2LoginAuthenticationFilter;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .authorizeRequests()
                .antMatchers("/oauth2/**", "/login/**", "/public/**").permitAll() // OAuth2 관련 경로와 공개 경로 허용
                .anyRequest().authenticated() // 나머지 요청은 인증 필요
                .and()
            .oauth2Login()
                .loginPage("/login") // 커스텀 로그인 페이지
                .defaultSuccessUrl("/home", true) // 로그인 성공 시 이동할 페이지
                .failureUrl("/login?error=true") // 로그인 실패 시 이동할 페이지
                .and()
            .logout()
                .logoutSuccessUrl("/login?logout=true") // 로그아웃 후 이동할 페이지
                .permitAll();
        
        return http.build();
    }
}

```

LoginController  
```java
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class LoginController {

    @GetMapping("/login")
    public String login() {
        return "login"; // 로그인 페이지 템플릿 이름
    }

    @GetMapping("/home")
    public String home() {
        return "home"; // 인증 후 이동할 페이지 템플릿 이름
    }
}

```

login.html Thymeleaf 사용  
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Login</title>
</head>
<body>
    <h1>Login</h1>
    <a href="/oauth2/authorization/google">Login with Google</a>
</body>
</html>

```

애플리케이션을 실행한 후, /login 페이지로 이동하면 Google 로그인 버튼이 표시된다.  
이 버튼을 클릭하면 Google 인증 페이지로 리디렉션되고, 인증 후 애플리케이션으로 돌아와서 home 페이지로 이동하게 된다.  
프링 부트 애플리케이션에서 OAuth2와 OIDC를 사용하여 강력한 인증 시스템을 구현하는 기본적인 방법리며,  
실제 사용할 떄는 각 인증 제공자의 세부 설정을 참조하여 필요한 설정을 추가하고, 보안 요구 사항에 따라 추가적인 보안 조치를 고려해야 한다.  

