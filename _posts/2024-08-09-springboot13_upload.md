---
layout: single
title:  "spring boot 13.애플리케이션 배포"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 애플리케이션 프로덕션으로 진입한느 단계이다.  
스프링 부트 앱은 war과 jar을 제공하며, 보통 jar로 출력하ㄴ다.  
스프링 부트의 실행 가능한 jar을 빌드할 떄도 다양한 요구사항과 사용사례를 충독하는 배포 옵션이 많다는 거다.  

## 완전히 실행 가능한 jar  
스프링 부트 애플리케이션을 완전히 실행 가능한 JAR 파일로 패키징하는 것은 가장 일반적인 배포 방법이다.  
JAR 파일에는 애플리케이션의 모든 종속성이 포함되어 있어, 단일 파일로 배포하고 실행할 수 있다.  
pom.xml
spring-boot-maven-plugin을 사용하면 실행 가능한 JAR 파일을 생성할 수 있다.  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>myapp</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <dependencies>
        <!-- 스프링 부트 스타터 종속성 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!-- 기타 종속성 추가 -->
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
spring-boot-maven-plugin은 Maven을 사용하여 스프링 부트 애플리케이션을 빌드할 때 필요한 플러그인다.  
이 플러그인은 애플리케이션의 종속성을 포함하는 실행 가능한 JAR 파일을 생성한다.  

JAR 파일을 빌드한 후, 다음 명령어를 사용하여 실행할 수 있다.  
```bash
mvn clean package
java -jar target/myapp-1.0.0.jar

```
mvn clean package 명령어는 프로젝트를 빌드하여 target 디렉토리에 JAR 파일을 생성한다.  

## jar 확장  
실행 가능한 JAR 파일에 다양한 설정을 추가하거나 확장할 수 있다.  
이를 통해 특정 환경이나 요구 사항에 맞게 조정할 수 있다.  

예를 들어, 애플리케이션의 런타임 구성을 설정하려면 application.properties 또는 application.yml 파일을 사용하면 된다.  
application.properties
```properties
server.port=8081
spring.datasource.url=jdbc:mysql://localhost:3306/mydb

```
erver.port는 애플리케이션이 수신할 포트를 설정한다.  
spring.datasource.url은 데이터베이스 연결 URL을 설정한다.  

애플리케이션을 실행할 때 외부 설정 파일을 지정하여 환경에 따라 다르게 설정할 수 있다.  
```bash
java -jar target/myapp-1.0.0.jar --spring.config.location=file:/path/to/config/

```
--spring.config.location 옵션을 사용하여 외부 설정 파일의 경로를 지정한다.  

## 컨테이너에 스프링 부트 애플리케이션 배포하기  
스프링 부트 애플리케이션을 컨테이너화하면 애플리케이션을 다양한 환경에서 일관되게 실행할 수 있다.  
Docker를 사용하여 애플리케이션을 컨테이너로 패키징하고 배포하는 방법을 살펴보자.  

### Docker 이미지 생성  
애플리케이션을 Docker 컨테이너로 패키징하려면 Dockerfile을 작성하면 된다.  
Dockerfile 예제  
```dockerfile
# 1. OpenJDK 이미지를 기반으로 설정
FROM openjdk:17-jdk-alpine

# 2. 애플리케이션 JAR 파일을 컨테이너에 복사
COPY target/myapp-1.0.0.jar /app/myapp.jar

# 3. 컨테이너 시작 시 애플리케이션 실행
ENTRYPOINT ["java", "-jar", "/app/myapp.jar"]

```  
FROM openjdk:17-jdk-alpine: JDK 17이 포함된 알파인 리눅스 이미지를 기반으로 한다.  
COPY target/myapp-1.0.0.jar /app/myapp.jar: 애플리케이션 JAR 파일을 컨테이너의 /app 디렉토리로 복사한다.  
ENTRYPOINT ["java", "-jar", "/app/myapp.jar"]: 컨테이너가 시작될 때 JAR 파일을 실행한다.  
Docker 이미지를 빌드하고 실행하는 방법은 다음과 같다.  
```bash
docker build -t myapp .
docker run -p 8080:8080 myapp

```  
docker build -t myapp .: 현재 디렉토리의 Dockerfile을 사용하여 myapp이라는 이름의 Docker 이미지를 생성.  
docker run -p 8080:8080 myapp: 생성된 Docker 이미지를 실행하며, 컨테이너의 8080 포트를 호스트의 8080 포트에 매핑.  

## 컨테이너 이미지  
애플리케이션의 실행에 필요한 모든 것을 포함하는 파일 시스템 스냅샷  
여기에는 애플리케이션 코드, 라이브러리, 종속성, 환경 변수, 설정 파일, 그리고 애플리케이션 실행을 위한 실행 파일이 포함된다.  
이러한 이미지는 컨테이너 런타임(예: Docker)이 실행 가능한 컨테이너를 생성하는 데 사용된다.  

구성 요소  
1. 파일 시스템 레이어  
    애플리케이션 코드: 애플리케이션의 실행 파일 및 라이브러리.  
    종속성: 애플리케이션이 실행되기 위해 필요한 모든 소프트웨어 라이브러리와 패키지.  
    설정 파일: 애플리케이션의 설정 정보를 포함하는 파일.  
2. 메타데이터  
    메타데이터: 이미지의 구성 요소와 설정을 정의하는 JSON 형식의 파일로, 이미지가 어떻게 실행될지에 대한 정보를 포함.  
3. 실행 명령어  
    ENTRYPOINT: 컨테이너가 시작될 때 실행될 명령어를 정의.  
    CMD: 컨테이너 시작 시 기본적으로 실행될 명령어를 지정. ENTRYPOINT와 CMD는 함께 사용되어 컨테이너의 기본 실행 명령어를 정의.  

이미지의 장점
1. 일관성
    환경의 일관성: 동일한 컨테이너 이미지를 사용하여 개발, 테스트, 그리고 프로덕션 환경에서 일관된 실행 환경을 보장.

2. 이식성
    다양한 플랫폼 지원: 컨테이너 이미지는 다양한 운영 체제와 플랫폼에서 동일하게 실행된다. 예를 들어, 로컬 개발 환경과 클라우드 환경에서 동일한 이미지를 사용할 수 있다.  
3. 격리
    애플리케이션 격리: 컨테이너는 애플리케이션을 다른 컨테이너와 격리하여 실행하므로, 애플리케이션 간의 충돌을 방지.

4. 버전 관리
    이미지 태그: 컨테이너 이미지는 버전 태그를 사용하여 관리할 수 있다. 이를 통해 특정 버전의 애플리케이션을 쉽게 배포하고 롤백할 수 있다.

## IDE에서 컨테이너 이미지 생성 및 적용하기  
통합 개발 환경(IDE)에서 Docker 이미지를 생성하고 관리하면 개발 과정에서 편리하게 사용할 수 있다.  
IntelliJ IDEA에서 Docker 설정  
Docker 플러그인 설치: IntelliJ IDEA에서 Docker 플러그인을 설치. Settings -> Plugins에서 "Docker"를 검색하여 설치하면 됨.  
Docker 서버 설정: Docker 서버에 연결하려면 Settings -> Build, Execution, Deployment -> Docker에서 Docker 서버를 추가합니다.  

Dockerfile 생성 및 빌드  
프로젝트의 루트 디렉토리에 Dockerfile을 추가.  
IntelliJ IDEA에서 Docker 탭을 열고, + 버튼을 클릭하여 새로운 Docker 이미지 빌드를 설정.  
이미지 실행: 빌드된 이미지를 IntelliJ IDEA의 Docker 탭에서 직접 실행할 수 있다.  
IDE 내에서 Docker를 설정하면, 개발 중에도 실시간으로 컨테이너화된 애플리케이션을 테스트할 수 있습니다.  

## 스프링부트 애플리케이션 검사를 위한 유틸리티 컨테이너 이미지  
애플리케이션의 검증 및 테스트를 위해 유틸리티 컨테이너 이미지를 사용할 수 있다.  
이러한 도구들은 애플리케이션의 품질과 성능을 보장하는 데 도움을 준다.  
검사 도구에는 다음과 같은 것이 있다.  
1. Selenium: 웹 애플리케이션의 UI 테스트를 자동화  
2. JMeter: 성능 테스트를 수행하여 애플리케이션의 부하를 측정  

Docker-compose를 통한 테스트 환경 설정  
docker-compose를 사용하여 복잡한 테스트 환경을 설정할 수 있다.  
docker-compose.yml 예제
```yaml
version: '3'
services:
  web:
    image: myapp:latest
    ports:
      - "8080:8080"
  selenium:
    image: selenium/standalone-chrome
    ports:
      - "4444:4444"
  jmeter:
    image: justbrittany/jmeter
    volumes:
      - ./tests:/tests
    entrypoint: jmeter -n -t /tests/testplan.jmx -l /tests/results.jtl

```

web 서비스는 애플리케이션 컨테이너를 정의  
selenium 서비스는 Selenium을 실행하여 웹 애플리케이션의 UI 테스트를 지원  
jmeter 서비스는 JMeter를 사용하여 성능 테스트를 수행합니다. tests 디렉토리에서 테스트 계획 파일과 결과 파일을 다룬다.  

## 팩 (Pack) 
팩(Pack)은 애플리케이션을 컨테이너화하는 간단하고 표준화된 방법을 제공한다.  
Cloud Native Buildpacks를 사용하여 애플리케이션을 컨테이너 이미지로 변환할 수 있다.  

## 다이브 (Dive)
다이브(Dive)는 Docker 이미지를 분석하고 최적화하는 도구이다.  
이미지를 시각적으로 분석하고 불필요한 레이어를 식별하는 데 도움을준다.  

이 글을 마지막으로, 스프링부트는 마치겠다.  