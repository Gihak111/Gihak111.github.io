---
layout: single
title:  "spring boot 6. 스프링 MVC"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 스프링 MVC  
스프링 프레임 워크의 웹 모듈이다.  
MVC(Model-View-Controller) 디자인 패턴을 기반으로 웹 애플리케이션을 쉽게 개발할 수 있도록 돕는 프레임워크로,  
효율적이고 확장 가능한 웹 애플리케이션을 만들기 위한 다양한 기능과 편리한 도구들을 제공한다.  

## 구성요소  
1. Model  
비즈니스 로직과 데이터 관리의 중심입니다.  
데이터 객체와 비즈니스 로직을 포함하며, 데이터베이스와 상호작용하거나 비즈니스 규칙을 적용합니다.  

2. View  
사용자에게 데이터를 보여주는 역할이다.  
JSP, Thymeleaf, FreeMarker 등의 템플릿 엔진을 사용해 HTML, JSON, XML 등의 형식으로 데이터를 출력한다.  

3. Controller  
사용자의 요청을 받아 적절한 모델과 뷰를 선택하여 응답을 생성하는 역할을 한다.  
사용자 요청을 처리하고, 필요한 데이터를 모델에서 가져와 뷰에 전달한다.  

## 작동 원리  
요청 처리  
사용자가 웹 애플리케이션에 요청을 보내면, DispatcherServlet이 해당 요청을 가로챈다.  

핸들러 매핑  
DispatcherServlet은 요청 URL을 기반으로 적절한 컨트롤러와 메서드를 찾기 위해 핸들러 매핑을 사용한다.  

컨트롤러 실행  
핸들러 매핑을 통해 찾은 컨트롤러 메서드가 실행된다.  
이 메서드는 비즈니스 로직을 수행하고, 필요한 데이터를 모델에 담아 반환한다.  

뷰 리졸버  
컨트롤러가 반환한 모델과 뷰 이름을 기반으로 뷰 리졸버가 적절한 뷰를 찾는다.  

응답 생성  
뷰 리졸버가 선택한 뷰 템플릿이 모델 데이터를 사용해 최종 HTML 페이지를 생성한다.  
DispatcherServlet이 이 페이지를 사용자에게 응답으로 보낸다.  

## 주요 어노테이션  
1. @Controller: 클래스가 컨트롤러 역할을 한다는 것을 명시한다.  
2. @RequestMapping: 특정 URL 요청을 처리할 메서드를 정의한다.  
3. @GetMapping, @PostMapping, @PutMapping, @DeleteMapping: HTTP 메서드별로 요청을 처리할 메서드를 정의한다.  
4. @ModelAttribute: 모델 데이터를 뷰에 전달하기 위해 사용된다.  
5. @ResponseBody: 메서드의 반환값을 HTTP 응답 본문으로 직접 변환한다.  

어디서 많이 봤던 것들이다 ㅇㅇ  
간단한 블로그 애플리케이션을 예시로 만들어 보면서 다시 한번 봐 보자.  
전에 비해 보이는 것들이 엄청나게 많아졌을 것이다.  

다음은 만들 앱의 자료구조이다.  
```arduino
spring-mvc-blog  
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── blog
│   │   │               ├── controller
│   │   │               │   └── BlogController.java
│   │   │               ├── model
│   │   │               │   └── Post.java
│   │   │               ├── repository
│   │   │               │   └── PostRepository.java
│   │   │               ├── service
│   │   │               │   └── PostService.java
│   │   │               └── BlogApplication.java
│   │   ├── resources
│   │   │   ├── application.properties
│   │   │   └── templates
│   │   │       ├── index.html
│   │   │       ├── post.html
│   │   │       └── new.html
│   └── test
│       └── java
│           └── com
│               └── example
│                   └── blog
│                       └── BlogApplicationTests.java
└── pom.xml

```  

pom.xml  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>blog</artifactId>
    <version>1.0-SNAPSHOT</version>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.3</version>
    </parent>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
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

BlogApplication.java  
```java
package com.example.blog;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BlogApplication {
    public static void main(String[] args) {
        SpringApplication.run(BlogApplication.class, args);
    }
}

```  

Post.java Modle  
```java
package com.example.blog.model;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity
public class Post {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;
    private String title;
    private String content;

    // Getters and setters
}

```  

PostRepository.java Repository  
```java
package com.example.blog.repository;

import com.example.blog.model.Post;
import org.springframework.data.repository.CrudRepository;

public interface PostRepository extends CrudRepository<Post, Long> {
}

```  

PostService.java Service  
```java
package com.example.blog.service;

import com.example.blog.model.Post;
import com.example.blog.repository.PostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PostService {
    @Autowired
    private PostRepository postRepository;

    public List<Post> findAll() {
        return (List<Post>) postRepository.findAll();
    }

    public Post findById(Long id) {
        return postRepository.findById(id).orElse(null);
    }

    public Post save(Post post) {
        return postRepository.save(post);
    }
}

```  

BlogController.java Controller  
```java
package com.example.blog.controller;

import com.example.blog.model.Post;
import com.example.blog.service.PostService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
public class BlogController {
    @Autowired
    private PostService postService;

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("posts", postService.findAll());
        return "index";
    }

    @GetMapping("/posts/{id}")
    public String viewPost(@PathVariable Long id, Model model) {
        Post post = postService.findById(id);
        model.addAttribute("post", post);
        return "post";
    }

    @GetMapping("/posts/new")
    public String newPostForm(Model model) {
        model.addAttribute("post", new Post());
        return "new";
    }

    @PostMapping("/posts")
    public String savePost(@ModelAttribute Post post) {
        postService.save(post);
        return "redirect:/";
    }
}

```  

index.html View  
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Blog</title>
</head>
<body>
<h1>Blog Posts</h1>
<ul>
    <li th:each="post : ${posts}">
        <a th:href="@{'/posts/' + ${post.id}}" th:text="${post.title}">Post Title</a>
    </li>
</ul>
<a href="/posts/new">New Post</a>
</body>
</html>

```  

post.html View  
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Blog</title>
</head>
<body>
<h1 th:text="${post.title}">Post Title</h1>
<p th:text="${post.content}">Post Content</p>
<a href="/">Back to Home</a>
</body>
</html>

```  

new.html View  
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>New Post</title>
</head>
<body>
<h1>New Post</h1>
<form action="#" th:action="@{/posts}" th:object="${post}" method="post">
    <div>
        <label for="title">Title</label>
        <input type="text" id="title" th:field="*{title}" />
    </div>
    <div>
        <label for="content">Content</label>
        <textarea id="content" th:field="*{content}"></textarea>
    </div>
    <div>
        <button type="submit">Save</button>
    </div>
</form>
<a href="/">Back to Home</a>
</body>
</html>

```  

BlogApplication.java를 실행하면, 스프링 부트가 실행된다.  
스프링 MVC를 사용하여 간단한 블로그 애플리케이션을 만드는 전체 과정을 보기 위한 예시 이다.  
데이터베이스는 H2 인메모리 데이터베이스를 사용하므로 추가 설정 없이 테스트할 수 있다.  