---
layout: single
title:  "spring boot 9. GPT모델 API 활용해서 사이트 만들기"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# GPT 모델 연동  
다음과 같이 파일을 만들어 보자.  
Project: Maven  
Language: Java  
Spring Boot Version: 2.7.0 or later  
Group: com.example  
Artifact: gpt-api-demo  
Dependencies: Web, Lombok  

OpenAI API는 OpenAI의 웹사이트에서 API 키를 받아야 한다. HTTP 요청을 통해 모델과 상호작용할 수 있다.  
밑의 링크에서 자세한 니용을 볼 수 있다.  
https://platform.openai.com/docs/overview  

파일 구조는 다음과 같다.  
```scss
gpt-api-demo
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── gptapidemo
│   │   │               ├── config
│   │   │               │   └── OpenAIConfig.java
│   │   │               ├── controller
│   │   │               │   └── OpenAIController.java
│   │   │               └── GptApiDemoApplication.java
│   │   ├── resources
│   │   │   ├── static
│   │   │   │   └── index.html
│   │   │   ├── application.properties
│   │   │   └── templates (사용하지 않음, 필요한 경우 추가)
│   └── test
│       └── java
│           └── com
│               └── example
│                   └── gptapidemo
│                       └── GptApiDemoApplicationTests.java
├── .gitignore
├── mvnw
├── mvnw.cmd
├── pom.xml
└── README.md (선택 사항)

```  
GptApiDemoApplication.java  프로젝트의 진입점   
```java
package com.example.gptapidemo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class GptApiDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(GptApiDemoApplication.class, args);
    }
}

```

OpenAIConfig.java OpenAI 클라이언트 설정 파일  
```java
package com.example.gptapidemo.config;

import com.theokanning.openai.service.OpenAiService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.beans.factory.annotation.Value;

@Configuration
public class OpenAIConfig {

    @Value("${openai.api.key}")
    private String openAiApiKey;

    @Bean
    public OpenAiService openAiService() {
        return new OpenAiService(openAiApiKey);
    }
}

```

OpenAIController.java GPT-3 API 호출을 처리하는 컨트롤러  
```java
package com.example.gptapidemo.controller;

import com.theokanning.openai.completion.CompletionRequest;
import com.theokanning.openai.completion.CompletionResult;
import com.theokanning.openai.service.OpenAiService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class OpenAIController {

    private final OpenAiService openAiService;

    public OpenAIController(OpenAiService openAiService) {
        this.openAiService = openAiService;
    }

    @GetMapping("/generate")
    public String generateText(@RequestParam String prompt) {
        CompletionRequest completionRequest = CompletionRequest.builder()
                .prompt(prompt)
                .maxTokens(100)
                .build();
        
        CompletionResult completionResult = openAiService.createCompletion("text-davinci-003", completionRequest);
        return completionResult.getChoices().get(0).getText();
    }
}

```

application.properties API 키를 저장하는 설정 파일  
```properties
openai.api.key=YOUR_OPENAI_API_KEY
```

index.html 프론트엔드 HTML 파일  
```html
<!DOCTYPE html>
<html>
<head>
    <title>GPT API Demo</title>
</head>
<body>
    <h1>GPT API Demo</h1>
    <form id="gpt-form">
        <label for="prompt">Enter prompt:</label>
        <input type="text" id="prompt" name="prompt">
        <button type="submit">Generate</button>
    </form>
    <pre id="response"></pre>

    <script>
        document.getElementById('gpt-form').addEventListener('submit', function(event) {
            event.preventDefault();
            const prompt = document.getElementById('prompt').value;

            fetch(`/generate?prompt=${encodeURIComponent(prompt)}`)
                .then(response => response.text())
                .then(data => {
                    document.getElementById('response').innerText = data;
                })
                .catch(error => console.error('Error:', error));
        });
    </script>
</body>
</html>

```

pom.xml
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>gpt-api-demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>gpt-api-demo</name>
    <description>Demo project for Spring Boot and OpenAI GPT API</description>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.0</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <properties>
        <java.version>11</java.version>
    </properties>

    <dependencies>
        <!-- Spring Boot Dependencies -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        
        <!-- Lombok for reducing boilerplate code -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <scope>provided</scope>
        </dependency>

        <!-- OpenAI API Client -->
        <dependency>
            <groupId>com.theokanning.openai-gpt3-java</groupId>
            <artifactId>client</artifactId>
            <version>0.9.0</version>
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

위 코드를 통해서 openAI GPT3 API를 사용할 수 있다.