---
layout: single
title:  "spring boot 1. 스프링부트 시작 원리 이해"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  
# spring boot  
스프링 부트의 핵심 기능은 3가지다.  
1. 의존성 관리 간소화  
2. 배포 간소화  
3. 자동 설정.  
직접 코딩하면서 체감해 보자.  

메이븐, 자바로 진행할 것이며, Intellij IDEA로 진행한다.  
https://www.jetbrains.com/ko-kr/idea/  

먼저, 스프링 부트 파일을 받기 위해, 다음 링크에 접속한다.  
https://start.spring.io/  
위 링크에서 스프링 부트 패키지를 받을 수 있다.  
일단, 아무런 Dependencies를 추가하지 말고 바로 다운 해 보자. 다음과 같은 자료 구조를 볼 수 있다.   
```arduino
my-spring-boot-project
│
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── myproject
│   │   │               ├── MySpringBootApplication.java
│   │   │               └── controller
│   │   │                   └── MyController.java
│   │   ├── resources
│   │   │   ├── application.properties
│   │   │   └── static
│   │   │       └── (static resources like HTML, CSS, JavaScript)
│   │   └── webapp
│   │       └── WEB-INF
│   │           └── (JSP files if you are using JSP)
│   ├── test
│   │   └── java
│   │       └── com
│   │           └── example
│   │               └── myproject
│   │                   └── MySpringBootApplicationTests.java
│
├── .gitignore
├── mvnw
├── mvnw.cmd
├── pom.xml (for Maven projects)
├── build.gradle (for Gradle projects)
└── README.md
```  

위의 자료구조에서 각 스크립트들의 역할을 정리하면 다음과 같다.  
1. MySpringBootApplication.java  
    스프링부트의 진입점이다. 여기서 스프링 부트의 모든것이 시작된다.  
    @SpringBootApplication 어노테이션이 붙은 클래스를 통해 스프링부트 애플리캐이션이 시작된다.  
    나중에 이 코드를 실행해서 앱이 실행되고, bean 파일이 초기화 된다.  
2. MyController.java  
    웹 요청을 처리하는 컨트롤러 클래스.  
    @RestController 또는 @Controller 어노테이션이 사용된다.  
    클라이언트의 HTTP 요청을 처리하고, 적절한 응답을 반환한다.  
3. application.properties  
    애플리케이션의 설정파일.  
    데이터베이스 설정, 포트 번호, 로깅 설정 등 여러 설정을 정의한다.  
    앱이 시작될 때, 자동으로 로드 되어 설정값을 적용한다. 이걸로 동작 조정도 가능하다.  
4. static  
    정적 리소스를 저장한다.  
    웹 체이지 css 파일 같은거나 자바슼 같은거 여기에 저장한다.  
5. test  
    테스트 클래스를 정의.  
    테스트 프레임워크 같은거 뭐 jUnit 같은거 사용해서 단위 테스트, 및 통합 할 수 있다.  
    @Test 어노테이션 사용해서 테스트 메서드 정의하고 기능 잘 돌아가는지 확인할 수 있다.  
    근데 웬만하면 건들일 없다 ㅇㅇ 디버깅 할때가 봄  
6. pom.xml  
    존성 및 빌드 구성을 정의하는 파일  
    의존성 자동으로 다운해서 관리한다.  
    빌드하면 알아서 생길꺼임 아마도?  
    ```cmd  
        mvn clean install  
    ```
    이 코드로 프로젝트 빌드 할 수 있다.  

거 뭐냐 메이든에 자바면 이정도 알면 기본적인 것들 할 수 있다.  

# REST API
이제, REST API 에 대해 보자.  
일반적인 프로그램은, 백앤드로 뒤에서 기능을 돌리고, 프론트 앤드로 사용자에게 기능을 보여준다.  
우리가 코딩을 할때, 각 기능과 함수마다의 연결을 느슨히 하고 유연성 있게 코딩하는 이유는 변화에 적응하기 쉽고 관리하기 쉽기 때문일 거다.  
이처럼, 백앤드도 각각의 기능을 분리하여 따로 관리하는 것으로 더 편하게 할 수 있다.  
서비스를 분리하고 따로 관리한다.  
이런 구조를 쉽게 적용할 수 있는게 스프링부트 이다.  
API는 개발자가 만든 사양이자 인터페이스 이다.  
REST API는 웹에서 자원을 간단한 작업으로 쉽게 관리할 수 있게 해준다.  
자원은 식별되는 데이터이며, 상태를 가지고 이를 HTTP 를 통해 주고받는다.  
HTTP 메서드는 다음이 있다.  

1. GET: 자원을 조회.  
2. POST: 새로운 자원을 생성. 
3. PUT: 기존 자원을 업데이트. 
4. DELETE: 자원을 삭제.  
5. PATCH : 매서드 리소스 생성.  

HTTP 상태코드는 다음과 같다.  

1. 200: 성공  
2. 201: 생성됨  
3. 400: 잘못된 요청  
4. 404: 자원을 찾을 수 없음  
5. 500: 서버 오류  

# 예제 만들어 보기
이제, 이니셜 라이저와 함께 메서드를 사용해 보자.  
먼저, 밑의 도메인에 들어가서 Maven, JAVA를 선택한다.  
https://start.spring.io/  

ADD DEPENDENCIES를 눌러서 spring web과 Spring Boot DevTools를 추가한다.  
Project Metadata의 내용을 다음과 같이 변경한다.  
Group: com.example  
Artifact: fileUpload  
Name: fileUpload  
Package Name: com.example.fileupload  
Packaging: jar  

 zip 파일을 다운 받고 inteliJ로 켜보자.  
이름을 demo에서 변경하지 않았으면 이런 주소에 DemoApplication.java 파일이 있을 것이다.  
이제, 사진을 업로드 하고, 이를 보여주는 기능을 구현해 보자.  

zip 파일을 해제하고 인텔리제이로 실행한다.  
밑의 코드를 각각의 파일에 집어넣어 보자.  
없는 파일은 직접 만들면 된다.  
자료구조는 다음과 같다.  
```arduino  
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── fileupload
│   │   │               ├── FileUploadApplication.java
│   │   │               └── FileUploadController.java
│   │   └── resources
│   │       └── INDEX.html
│   │       └── application.properties
│   └── test
│       └── java
│           └── com
│               └── example
│                   └── fileupload
│                       └── FileUploadApplicationTests.java
└── pom.xml
```  

```java  
// FileUploadApplication.java
package com.example.fileUpload; //자바 패키지 선언

import org.springframework.boot.SpringApplication; //이거 시작 파일로 사용한다는 뭐 그런뜻이다. ㅇㅇ
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication // Spring Boot 애플리케이션의 시작점임을 나타냄 내부적으로 여러 애노테이션들을 포함하여, 스프링 부트의 자동 설정 활성화
public class FileUploadApplication {

	public static void main(String[] args) {
		// Spring Boot 애플리케이션을 실행
		SpringApplication.run(FileUploadApplication.class, args);
	}
}
```  

```java
// FileUploadController.java
package com.example.fileUpload; // 패키지 선언: 이 클래스는 'fileupload' 패키지에 속합니다.

import org.springframework.beans.factory.annotation.Value; // Spring의 @Value 어노테이션을 사용하여 프로퍼티 값을 주입받기 위한 클래스
import org.springframework.http.HttpHeaders; // HTTP 응답 헤더를 설정하기 위한 클래스
import org.springframework.http.HttpStatus; // HTTP 상태 코드를 정의하는 클래스
import org.springframework.http.MediaType; // HTTP 미디어 타입을 정의하는 클래스 (사용되지 않음)
import org.springframework.http.ResponseEntity; // HTTP 응답을 반환하기 위한 클래스, 상태 코드, 헤더, 바디를 포함할 수 있음
import org.springframework.util.StringUtils; // 문자열 유틸리티 클래스, 파일 이름 정리 등에서 사용
import org.springframework.web.bind.annotation.*; // Spring MVC의 RESTful 컨트롤러 관련 어노테이션들을 포함하는 패키지
import org.springframework.web.multipart.MultipartFile; // 파일 업로드를 처리하는 인터페이스
import org.springframework.web.servlet.support.ServletUriComponentsBuilder; // 현재 컨텍스트 경로를 기준으로 URI를 생성하는 유틸리티 클래스

import java.io.File; // 파일 작업을 위한 클래스
import java.io.IOException; // 입출력 예외 처리를 위한 클래스
import java.nio.file.Files; // 파일 읽기 및 쓰기를 위한 클래스
import java.nio.file.Path; // 파일 시스템의 경로를 나타내는 클래스
import java.nio.file.Paths; // 경로 객체를 생성하는 클래스
import java.util.UUID; // 고유 식별자를 생성하기 위한 클래스

@RestController // 이 클래스가 RESTful 웹 서비스의 컨트롤러임을 나타내며, HTTP 요청을 처리하고 JSON, XML 등의 형태로 응답을 반환합니다.
public class FileUploadController {

    @Value("${upload.dir}") // application.properties 파일에서 'upload.dir' 속성 값을 읽어와서 'uploadDir' 필드에 주입합니다.
    private String uploadDir;

    // 파일 업로드 처리 메소드
    @PostMapping("/upload") // HTTP POST 요청을 처리하는 메소드
    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) { // 업로드된 파일이 비어 있는지 확인
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Please select a file to upload"); // 파일이 비어 있을 경우 400 Bad Request 응답 반환
        }

        try {
            // 파일 이름을 UUID로 변경하여 고유하게 만듭니다. 기존 파일 이름을 클린업하여 안전한 이름으로 만듭니다.
            String fileName = UUID.randomUUID() + "-" + StringUtils.cleanPath(file.getOriginalFilename());
            Path path = Paths.get(uploadDir + File.separator + fileName); // 파일을 저장할 경로를 생성
            Files.copy(file.getInputStream(), path); // 파일을 지정된 경로에 저장

            // 파일 다운로드 URI를 생성
            String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                    .path("/files/")
                    .path(fileName)
                    .toUriString();

            // 파일 다운로드 URI를 응답으로 반환
            return ResponseEntity.ok(fileDownloadUri);

        } catch (IOException ex) { // 파일 저장 중 예외가 발생할 경우
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Could not store the file. Error: " + ex.getMessage()); // 500 Internal Server Error 응답 반환
        }
    }

    // 업로드된 파일을 보여주는 메소드
    @GetMapping("/files/{fileName:.+}") // HTTP GET 요청을 처리하는 메소드
    public ResponseEntity<byte[]> getFile(@PathVariable String fileName) {
        try {
            Path path = Paths.get(uploadDir + File.separator + fileName); // 파일 경로를 생성
            byte[] fileBytes = Files.readAllBytes(path); // 파일을 바이트 배열로 읽어옴
            HttpHeaders headers = new HttpHeaders(); // HTTP 헤더 객체 생성
            headers.add(HttpHeaders.CONTENT_TYPE, Files.probeContentType(path)); // 파일의 MIME 타입을 설정 (브라우저가 파일을 올바르게 표시하도록 함)
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + fileName + "\""); // 파일을 브라우저에서 인라인으로 표시하도록 설정
            return new ResponseEntity<>(fileBytes, headers, HttpStatus.OK); // 바이트 배열과 헤더를 응답으로 반환
        } catch (IOException ex) { // 파일 읽기 중 예외가 발생할 경우
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null); // 404 Not Found 응답 반환
        }
    }

    // 파일 다운로드 처리 메소드
    @GetMapping("/download/{fileName:.+}") // HTTP GET 요청을 처리하는 메소드
    public ResponseEntity<byte[]> downloadFile(@PathVariable String fileName) {
        try {
            Path path = Paths.get(uploadDir + File.separator + fileName); // 파일 경로를 생성
            byte[] fileBytes = Files.readAllBytes(path); // 파일을 바이트 배열로 읽어옴
            HttpHeaders headers = new HttpHeaders(); // HTTP 헤더 객체 생성
            headers.add(HttpHeaders.CONTENT_TYPE, Files.probeContentType(path)); // 파일의 MIME 타입을 설정 (다운로드 시 파일 형식을 올바르게 처리하도록 함)
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + fileName + "\""); // 파일 다운로드로 설정
            return new ResponseEntity<>(fileBytes, headers, HttpStatus.OK); // 바이트 배열과 헤더를 응답으로 반환
        } catch (IOException ex) { // 파일 읽기 중 예외가 발생할 경우
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null); // 404 Not Found 응답 반환
        }
    }
}

```  

```java
#application.properties
# 업로드 디렉토리 설정
upload.dir=uploads
```

```java
//FileUploadApplicationTests.java
// 이건 안해도 실행에 문제 없다.
// test는 디버깅에 이용된다 생각해라
package com.example.fileupload; //자바 패키지 선언

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest // Spring Boot 테스트를 위한 애노테이션
class FileUploadApplicationTests {

    @Test
    void contextLoads() {
        // 컨텍스트 로딩 테스트
    }
}
```  

```html
<!DOCTYPE html> <!-- 문서가 HTML5 문서임을 선언합니다. -->
<html lang="en"> <!-- HTML 문서의 언어를 영어로 설정합니다. -->
<head>
    <meta charset="UTF-8"> <!-- 문자 인코딩을 UTF-8로 설정합니다. -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> <!-- 모바일 장치에서 페이지의 크기와 비율을 설정합니다. -->
    <title>File Upload</title> <!-- 브라우저 탭에 표시될 제목을 설정합니다. -->
    <style>
        body {
            font-family: Arial, sans-serif; /* 페이지의 기본 글꼴을 Arial로 설정합니다. */
            margin: 20px; /* 페이지의 모든 방향에 20px의 여백을 추가합니다. */
        }
        #imagePreview {
            margin-top: 20px; /* 이미지 미리보기 섹션의 상단에 20px의 여백을 추가합니다. */
        }
        img {
            max-width: 100%; /* 이미지의 최대 너비를 부모 요소의 너비로 설정합니다. */
            height: auto; /* 이미지의 높이를 자동으로 조절하여 비율을 유지합니다. */
        }
    </style>
</head>
<body>
<h1>File Upload and Download</h1> <!-- 페이지의 제목을 표시합니다. -->

<!-- 파일 업로드 폼 -->
<form id="uploadForm" enctype="multipart/form-data"> <!-- 파일 업로드를 위한 폼을 생성합니다. enctype="multipart/form-data"는 파일 업로드를 허용합니다. -->
    <input type="file" id="fileInput" name="file" required> <!-- 사용자가 파일을 선택할 수 있는 입력 필드입니다. 'required' 속성은 파일 선택을 필수로 만듭니다. -->
    <button type="submit">Upload</button> <!-- 폼을 제출하는 버튼입니다. -->
</form>

<!-- 이미지 미리보기 및 다운로드 버튼 -->
<div id="imagePreview">
    <img id="previewImage" src="" alt="Image Preview" style="display: none;"> <!-- 이미지 미리보기를 위한 img 요소입니다. 초기에는 보이지 않습니다. -->
    <br>
    <a id="downloadLink" href="#" download style="display: none;">Download</a> <!-- 다운로드 링크를 생성합니다. 초기에는 보이지 않습니다. -->
</div>

<script>
    // 폼 제출 이벤트를 처리하는 스크립트
    document.getElementById('uploadForm').addEventListener('submit', function(event) {
        event.preventDefault(); // 폼의 기본 제출 동작을 방지합니다.

        var formData = new FormData(); // 폼 데이터를 담을 FormData 객체를 생성합니다.
        var fileInput = document.getElementById('fileInput'); // 파일 입력 필드를 선택합니다.
        formData.append('file', fileInput.files[0]); // 선택된 파일을 FormData 객체에 추가합니다.

        fetch('/upload', {
            method: 'POST', // HTTP POST 요청을 사용하여 파일을 업로드합니다.
            body: formData // FormData 객체를 요청 본문으로 설정합니다.
        })
            .then(response => response.text()) // 서버 응답을 텍스트로 변환합니다.
            .then(fileDownloadUri => {
                // 서버에서 제공한 URI를 사용하여 이미지 미리보기 및 다운로드 링크를 설정합니다.
                var previewImage = document.getElementById('previewImage'); // 이미지 미리보기 요소를 선택합니다.
                var downloadLink = document.getElementById('downloadLink'); // 다운로드 링크 요소를 선택합니다.

                // 이미지 미리보기 설정
                previewImage.src = fileDownloadUri; // 이미지 소스 URL을 설정합니다.
                previewImage.style.display = 'block'; // 이미지를 표시합니다.

                // 다운로드 링크 설정
                downloadLink.href = fileDownloadUri.replace("/files/", "/download/"); // 다운로드 링크 URL을 설정합니다. "/files/"를 "/download/"로 교체합니다.
                downloadLink.style.display = 'block'; // 다운로드 링크를 표시합니다.
                downloadLink.textContent = 'Download'; // 다운로드 링크의 텍스트를 설정합니다.
            })
            .catch(error => console.error('Error uploading file:', error)); // 파일 업로드 중 오류가 발생하면 콘솔에 오류 메시지를 출력합니다.
    });
</script>
</body>
</html>

```  

위의 코드를 인텔리제이에서 실행해보고, 다음 로컬로 들어가 보자.  
따로 로컬 주소를 설정하지 않으면 웬만하면 이 주소로 들어가 진다.  
http://localhost:8080/upload  
접속해서 잘 돌아가는지 확인해 보자.  
파일이 잘 올라가 지지만, 잘 표시되지 않고, 다운역시 잘 되지 않는것이 보인다.
원인은, 파일을 읽고 코드 내부에도 돌리는 과정에 json으로 변환되기 때문이다.
암튼 대충 이런 구조로 돌아간다.  
위의 자료를 통해서 간단히 스프링 부트의 구조와 작동 원리를 이해할 수 잇다.
결국 하나의 패키지다. 그 이상 이하도 아니다.
