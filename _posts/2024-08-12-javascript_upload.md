---
layout: single
title:  "자바 스크립트"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 자바스크립트  
리액트 네이티브를 할떄 주로 사용하는 언어다.  
자바와 내용을 비교해 가며 빠르게 알아보자.  

## 자바 스크립트 기본   

### 1. 자바스크립트로 할 수 있는 일  
- **자바스크립트**  
  - **클라이언트 측 웹 개발**: 웹 페이지의 동적 콘텐츠와 사용자 인터페이스 제어.  
  - **서버 측 개발**: Node.js를 통해 서버 구축, RESTful API 구현.  
  - **모바일 앱 개발**: React Native와 같은 프레임워크를 사용하여 크로스 플랫폼 모바일 앱 개발.  
  - **데스크탑 앱 개발**: Electron을 사용하여 데스크탑 애플리케이션 개발.  
  - **게임 개발**: Phaser와 같은 프레임워크를 사용하여 브라우저 기반 게임 개발.  
  
- **자바**  
  - **서버 측 애플리케이션**: 엔터프라이즈 수준의 백엔드 시스템 개발.  
  - **모바일 앱 개발**: 안드로이드 앱 개발.  
  - **데스크탑 애플리케이션**: JavaFX, Swing을 사용한 데스크탑 애플리케이션 개발.  
  - **대규모 엔터프라이즈 시스템**: 금융, 보험, 제조업 등의 대규모 시스템 개발.  

### 2. 기본용어, 출력, 기본 자료형, 변수, 복합대입연산자, 증감 연산자, 자료형 검사, undefined 자료형, 강제 자료형 변환, 자동 자료형 변환, 일치 연산자  
- **자바스크립트**  
  - **기본 자료형**: `string`, `number`, `boolean`, `undefined`, `null`, `symbol`, `object`  
  - **출력**: `console.log()`  
  - **변수 선언**: `var`, `let`, `const`  
  - **복합대입연산자**: `+=`, `-=`, `*=`, `/=`  
  - **증감 연산자**: `++`, `--`  
  - **자료형 검사**: `typeof`  
  - **강제 자료형 변환**: `String()`, `Number()`, `Boolean()`  
  - **자동 자료형 변환**: 덜 엄격  
  - **일치 연산자**: `===` (엄격한 비교)  

- **자바**  
  - **기본 자료형**: `int`, `float`, `double`, `boolean`, `char`, `byte`, `short`, `long`  
  - **출력**: `System.out.println()`  
  - **변수 선언**: `int`, `double`, `boolean` 등 기본 자료형 타입으로 선언  
  - **복합대입연산자**: `+=`, `-=`, `*=`, `/=`  
  - **증감 연산자**: `++`, `--`  
  - **자료형 검사**: `instanceof`  
  - **강제 자료형 변환**: `(int)`, `(double)`  
  - **자동 자료형 변환**: 더 엄격  
  - **일치 연산자**: `==` (기본 자료형), `equals()` (객체)  

### 3. 조건문 (if, if else, switch 등), 조건 연산자  
- **자바스크립트**  
  - **조건문**: `if`, `else if`, `else`, `switch`  
  - **조건 연산자**: `? :`  

- **자바**  
  - **조건문**: `if`, `else if`, `else`, `switch`  
  - **조건 연산자**: `? :`  

### 4. 반복문 (배열 활용하는 반복문, 중첩 반복문 포함), break, continue, 스코프, 호이스팅  
- **자바스크립트**  
  - **반복문**: `for`, `for...of`, `for...in`, `while`, `do...while`  
  - **배열 반복**: `forEach`, `map`, `filter`, `reduce`  
  - **중첩 반복문**: `for` 내에 `for`, `while` 내에 `for` 등  
  - **break, continue**: 반복문 제어  
  - **스코프**: 함수 스코프, 블록 스코프 (ES6 이후 `let`, `const`)  
  - **호이스팅**: 변수 선언이 스코프의 최상위로 끌어올려짐 (`var`)  

- **자바**  
  - **반복문**: `for`, `enhanced for`, `while`, `do...while`  
  - **배열 반복**: `for`, `for-each`  
  - **중첩 반복문**: `for` 내에 `for`, `while` 내에 `for` 등  
  - **break, continue**: 반복문 제어  
  - **스코프**: 블록 스코프  
  - **호이스팅**: 없음  

### 5. 함수 (익명 함수, 선언적 함수, 화살표 함수), 기본 형태, 활용 형태, 변수 초기화, 콜백 함수, 표준 내장 함수, 익명 함수와 선언적 함수 생성 순서, 일반 함수와 화살표 함수의 차이  
- **자바스크립트**  
  - **함수 선언**: `function`  
  - **익명 함수**: `function() {}`  
  - **화살표 함수**: `() => {}`  
  - **콜백 함수**: 함수의 인수로 전달되는 함수  
  - **표준 내장 함수**: `Array.prototype.forEach()`, `String.prototype.includes()`  
  - **함수의 활용**: 고차 함수, 비동기 처리 (Promise, async/await)  

- **자바**  
  - **함수(메소드) 선언**: `public void methodName() {}`  
  - **익명 클래스/메소드**: `new Runnable() { public void run() {} }`  
  - **람다식**: `(parameters) -> expression`  
  - **콜백 함수**: 인터페이스를 사용한 콜백  
  - **표준 내장 함수**: `Collections.sort()`  
  - **함수의 활용**: 스트림 API, 병렬 처리  

### 6. 객체 (객체 기본, 객체와 반복문, 속성과 메소드, 클래스)  
- **자바스크립트**  
  - **객체 생성**: `{}`, `new Object()`  
  - **속성 접근**: `obj.property`, `obj['property']`  
  - **메소드**: 객체의 속성으로 정의된 함수  
  - **클래스**: `class MyClass {}` (ES6 이후)  

- **자바**  
  - **객체 생성**: `new ClassName()`  
  - **속성 접근**: `obj.property`  
  - **메소드**: 클래스의 멤버 함수  
  - **클래스**: `class MyClass {}`  

### 7. 표준 내장 객체 (내장 객체 기반, 기본 자료형과 객체 자료형의 차이, Number 객체, String 객체, Date 객체, Array 객체, 콜백 함수와 함께 사용하는 메소드, Lodash 라이브러리, JSON 객체)  
- **자바스크립트**  
  - **내장 객체**: `Number`, `String`, `Date`, `Array`, `Object`  
  - **기본 자료형과 객체 자료형**: `number` vs `Number`, `string` vs `String`  
  - **Lodash 라이브러리**: 유틸리티 함수 모음 (예: `_.map()`, `_.reduce()`)  
  - **JSON 객체**: `JSON.parse()`, `JSON.stringify()`  

- **자바**  
  - **내장 클래스**: `Integer`, `String`, `Date`, `ArrayList`  
  - **기본 자료형과 객체 자료형**: `int` vs `Integer`, `double` vs `Double`  
  - **라이브러리**: Apache Commons, Google Guava 등 유틸리티 라이브러리  
  - **JSON 객체**: `JSONObject`, `JSONArray`  

### 8. 예외 처리 (예외와 기본 예외 처리, 고급 예외 처리, 예외 객체, 예외 강제 발생)  
- **자바스크립트**  
  - **예외 처리**: `try`, `catch`, `finally`, `throw`  
  - **예외 객체**: `Error`, `TypeError`, `RangeError`  
  - **비동기 예외 처리**: `async/await`와 `try/catch`  

- **자바**  
  - **예외 처리**: `try`, `catch`, `finally`, `throw`  
  - **예외 객체**: `Exception`, `RuntimeException`, `IOException`  
  - **예외 강제 발생**: `throw new Exception("Error message")`  

### 예제 코드  
자바 스크립트 예제  
```javascript
// 자바스크립트 예제
// 1. 자바스크립트로 할 수 있는 일
console.log("Hello, World!");

// 2. 기본 자료형, 변수, 연산자


let num = 5; // 변수 선언
let str = "Hello"; // 문자열
let isTrue = true; // 불리언
console.log(typeof num); // 자료형 검사

// 3. 조건문
if (num > 3) {
  console.log("Num is greater than 3");
} else {
  console.log("Num is not greater than 3");
}

switch (num) {
  case 1:
    console.log("Num is 1");
    break;
  case 5:
    console.log("Num is 5");
    break;
  default:
    console.log("Num is not 1 or 5");
}

// 4. 반복문
for (let i = 0; i < 3; i++) {
  console.log(i);
}

let arr = [1, 2, 3];
arr.forEach(item => console.log(item));

// 5. 함수
function greet(name) {
  return `Hello, ${name}`;
}
console.log(greet("World"));

let add = (a, b) => a + b;
console.log(add(2, 3));

// 6. 객체
let person = {
  name: "Alice",
  age: 25,
  greet: function() {
    console.log("Hello!");
  }
};
console.log(person.name);
person.greet();

// 7. 표준 내장 객체
let date = new Date();
console.log(date.toDateString());

let jsonString = JSON.stringify({ name: "Alice", age: 25 });
console.log(jsonString);

// 8. 예외 처리
try {
  throw new Error("Something went wrong");
} catch (e) {
  console.log(e.message);
}
```
자바 예제  
```java
// 자바 예제
// 1. 자바로 할 수 있는 일
public class Main {
  public static void main(String[] args) {
    System.out.println("Hello, World!");

    // 2. 기본 자료형, 변수, 연산자
    int num = 5; // 변수 선언
    String str = "Hello"; // 문자열
    boolean isTrue = true; // 불리언
    System.out.println(((Object)num).getClass().getSimpleName()); // 자료형 검사

    // 3. 조건문
    if (num > 3) {
      System.out.println("Num is greater than 3");
    } else {
      System.out.println("Num is not greater than 3");
    }

    switch (num) {
      case 1:
        System.out.println("Num is 1");
        break;
      case 5:
        System.out.println("Num is 5");
        break;
      default:
        System.out.println("Num is not 1 or 5");
    }

    // 4. 반복문
    for (int i = 0; i < 3; i++) {
      System.out.println(i);
    }

    int[] arr = {1, 2, 3};
    for (int item : arr) {
      System.out.println(item);
    }

    // 5. 함수
    System.out.println(greet("World"));

    // 6. 객체
    Person person = new Person("Alice", 25);
    System.out.println(person.getName());
    person.greet();

    // 7. 표준 내장 객체
    java.util.Date date = new java.util.Date();
    System.out.println(date.toString());

    // 8. 예외 처리
    try {
      throw new Exception("Something went wrong");
    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

  public static String greet(String name) {
    return "Hello, " + name;
  }
}

class Person {
  private String name;
  private int age;

  public Person(String name, int age) {
    this.name = name;
    this.age = age;
  }

  public String getName() {
    return name;
  }

  public void greet() {
    System.out.println("Hello!");
  }
}
```  
두 언어를 비교해서 보면, 한쪽 언어를 잘 알고 있을 때 반대쪽 언어를 쉽고 빠르게 흡수 할 수 있다.  
이어서 달려보자.  

## 서버 자바 스클비트  

Node.js는 서버 사이드 자바스크립트 실행 환경으로, 비동기 이벤트 기반 아키텍처를 제공하여 고성능 웹 서버를 구현할 수 있다.  
스프링 부트와는 다른 철학을 가지고 있지만, 웹 서버 개발과 관련된 많은 공통 기능을 제공한다.  

### 1. Node.js의 기본 전역변수, 모듈 및 기능  
#### 1.1 전역 변수  
- **`__dirname`**: 현재 모듈의 디렉터리 이름을 나타낸다.  
- **`__filename`**: 현재 모듈의 파일 이름을 나타낸다.  

```javascript
// __dirname과 __filename은 Node.js에서 제공하는 전역 변수이다.

// __dirname: 현재 모듈의 디렉터리 이름
console.log(__dirname);  // 현재 파일이 위치한 디렉터리의 절대 경로를 출력한다.

// __filename: 현재 모듈의 파일 이름
console.log(__filename); // 현재 파일의 절대 경로를 출력한다.

```  

#### 1.2 `process` 객체의 속성과 이벤트  
- **속성**  
  - **`process.env`**: 환경 변수에 접근할 수 있다.  
  - **`process.argv`**: 명령줄 인수를 배열로 제공한다.  
- **이벤트**  
  - **`beforeExit`**: 이벤트 루프가 끝날 때 발생한다.  
  - **`exit`**: 프로세스가 종료될 때 발생한다.  

```javascript
// process 객체는 Node.js 프로세스에 대한 정보를 제공하고, 제어할 수 있게 해준다.

// process.env: 시스템 환경 변수
console.log(process.env);  // 시스템의 환경 변수를 출력한다.

// process.argv: 명령줄 인수 배열
console.log(process.argv); // Node.js 실행 시 전달된 명령줄 인수들을 배열로 출력한다.

// beforeExit 이벤트: 이벤트 루프가 모두 비워지고, 프로세스가 종료되기 전에 발생한다.
process.on('beforeExit', (code) => {
  console.log('Process beforeExit event with code:', code);
});

// exit 이벤트: 프로세스가 완전히 종료될 때 발생한다.
process.on('exit', (code) => {
  console.log('Process exit event with code:', code);
});

```  

#### 1.3 `os` 모듈  
- 시스템의 운영 체제 정보를 제공한다.  

```javascript
// os 모듈은 운영 체제에 대한 정보를 제공하는 모듈
const os = require('os');

// os.platform(): 운영 체제의 플랫폼을 반환
console.log(os.platform()); // 예: 'darwin', 'win32', 'linux'

// os.cpus(): CPU 코어에 대한 정보를 배열로 반환
console.log(os.cpus());     // 각 CPU 코어의 정보가 담긴 객체 배열을 출력

```  

#### 1.4 `url` 모듈  
- URL 문자열을 구문 분석하고 조작할 수 있다.  

```javascript
// url 모듈은 URL 문자열을 구문 분석하고 조작하는 기능을 제공한다.
const url = require('url');

// 새로운 URL 객체를 생성한다.
const myURL = new URL('https://example.org:8000/pathname/?search=test#hash');

// myURL.hostname: 호스트 이름을 반환한다.
console.log(myURL.hostname); // 'example.org' 출력

// myURL.pathname: 경로 이름을 반환한다.
console.log(myURL.pathname); // '/pathname/' 출력

```  

#### 1.5 `fs` (File System) 모듈  
- 파일 시스템과 상호 작용할 수 있다.  

```javascript
// fs 모듈은 파일 시스템과 상호 작용하는 기능을 제공
const fs = require('fs');

// fs.readFile(): 파일을 비동기적으로 읽어들인다.
fs.readFile('example.txt', 'utf8', (err, data) => {
  if (err) throw err; // 오류가 발생하면 예외를 던진다.
  console.log(data); // 파일의 내용을 출력한다.
});

```  

#### 1.6 노드 패키지 매니저 (npm)  
- Node.js의 패키지 매니저로, 패키지 설치와 관리 기능을 제공.  

```bash
# npm (Node Package Manager)은 Node.js의 패키지 매니저로, 패키지를 설치하고 관리할 수 있다.

# express 모듈을 설치
npm install express  # 이 명령어는 express 패키지를 현재 프로젝트에 설치

```  

#### 1.7 `request` 모듈  
- HTTP 요청을 쉽게 수행할 수 있게 도와준다.  

```javascript
// request 모듈은 HTTP 요청을 쉽게 수행할 수 있게 해준다.
const request = require('request');

// GET 요청을 보내고, 콜백 함수를 통해 응답을 처리
request('http://www.google.com', (error, response, body) => {
  if (error) throw error; // 요청 중 오류가 발생하면 예외를 던진다.
  console.log('body:', body); // 응답 본문을 출력한다.
});

```  

#### 1.8 `cheerio` 모듈  
- 서버 측에서 jQuery와 같은 방식으로 HTML을 조작할 수 있다.  

```javascript
// cheerio 모듈은 서버 측에서 jQuery와 유사한 방식으로 HTML을 조작할 수 있게 해준다.
const cheerio = require('cheerio');

// HTML 문자열을 로드
const html = '<h2 class="title">Hello world</h2>';
const $ = cheerio.load(html); // cheerio 객체를 생성

// 선택자를 사용하여 HTML 요소의 텍스트를 추출.
console.log($('.title').text()); // 'Hello world' 출력

```  

#### 1.9 `async` 모듈  
- 비동기 작업을 쉽게 관리할 수 있는 도구를 제공.  

```javascript
// async 모듈은 비동기 작업을 쉽게 관리할 수 있는 도구를 제공
const async = require('async');

// async.series(): 여러 비동기 함수를 순차적으로 실행
async.series([
  function(callback) {
    setTimeout(() => {
      console.log('Task 1'); // 첫 번째 작업
      callback(null, 'one'); // 콜백을 호출하여 다음 작업으로 넘어
    }, 200); // 200ms 대기
  },
  function(callback) {
    setTimeout(() => {
      console.log('Task 2'); // 두 번째 작업
      callback(null, 'two'); // 콜백을 호출하여 완료를 알린다
    }, 100); // 100ms 대기
  }
], (err, results) => {
  if (err) throw err; // 작업 중 오류가 발생하면 예외를 던진다.
  console.log(results); // ['one', 'two'] 출력 - 각 작업의 결과를 배열로 반환합
});

```  

### 예제 코드  
```javascript
// Node.js 기본 예제
const fs = require('fs');
const os = require('os');
const url = require('url');
const request = require('request');
const cheerio = require('cheerio');
const async = require('async');

console.log(__dirname); // 현재 디렉터리 출력
console.log(__filename); // 현재 파일명 출력
console.log(process.env); // 환경 변수 출력
console.log(process.argv); // 명령줄 인수 출력

process.on('beforeExit', (code) => {
  console.log('Process beforeExit event with code:', code); // 프로세스가 종료되기 직전에 실행
});

process.on('exit', (code) => {
  console.log('Process exit event with code:', code); // 프로세스가 종료될 때 실행
});

console.log(os.platform()); // 현재 운영체제 플랫폼 출력
console.log(os.cpus()); // CPU 정보 출력

const myURL = new URL('https://example.org:8000/pathname/?search=test#hash');
console.log(myURL.hostname); // 'example.org'
console.log(myURL.pathname); // '/pathname/'

fs.readFile('example.txt', 'utf8', (err, data) => {
  if (err) throw err; // 파일 읽기 중 오류가 발생하면 예외를 던짐
  console.log(data); // 파일 내용 출력
});

request('http://www.google.com', (error, response, body) => {
  if (error) throw error; // 요청 중 오류가 발생하면 예외를 던짐
  console.log('body:', body); // 응답 본문 출력
});

const html = '<h2 class="title">Hello world</h2>';
const $ = cheerio.load(html); // cheerio를 사용하여 HTML을 로드
console.log($('.title').text()); // 'Hello world' 출력

async.series([
  function(callback) {
    setTimeout(() => {
      console.log('Task 1'); // 첫 번째 작업
      callback(null, 'one'); // 콜백 호출
    }, 200); // 200ms 대기
  },
  function(callback) {
    setTimeout(() => {
      console.log('Task 2'); // 두 번째 작업
      callback(null, 'two'); // 콜백 호출
    }, 100); // 100ms 대기
  }
], (err, results) => {
  if (err) throw err; // 작업 중 오류가 발생하면 예외를 던짐
  console.log(results); // ['one', 'two'] 출력
});

```  

### 2. Express 모듈을 사용한 웹 서버 개발  
#### 2.1 웹 요청과 응답   
- **`express` 모듈**: Node.js 웹 애플리케이션 프레임워크 이다  

```javascript
// Express는 Node.js에서 웹 애플리케이션을 쉽게 구축할 수 있도록 도와주는 프레임워크.
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성

// 기본 경로('/')로 GET 요청이 들어오면 'Hello World!'를 응답으로 보낸다.
app.get('/', (req, res) => {
  res.send('Hello World!'); // 클라이언트에 응답을 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력
});

```

#### 2.2 서버 생성과 실행  
- **서버 생성**: `express()` 함수를 사용하여 애플리케이션 인스턴스를 만든다.  
- **서버 실행**: `app.listen()` 메서드를 사용하여 서버를 실행.  

```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성한다.

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```

#### 2.3 페이지 라우팅  
- **라우팅**: URL 경로와 HTTP 메서드에 따라 요청을 처리한다.  

```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성한다.

// GET 요청으로 /users 경로에 접근하면 'Users Page'를 응답한다.
app.get('/users', (req, res) => {
  res.send('Users Page'); // 클라이언트에 응답을 보낸다.
});

// POST 요청으로 /users 경로에 접근하면 'Create User'를 응답한다.
app.post('/users', (req, res) => {
  res.send('Create User'); // 클라이언트에 응답을 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```  

#### 2.4 요청 메시지와 응답 메시지  
- **요청 객체**: `req`로 표현되며, 클라이언트로부터의 요청 정보를 포함한다.  
- **응답 객체**: `res`로 표현되며, 서버가 클라이언트에게 응답을 보낼 때 사용한다.  

```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성한다.

// 경로 매개변수를 사용하여 사용자 ID를 받아온다.
app.get('/users/:id', (req, res) => {
  const userId = req.params.id; // 요청 경로에서 사용자 ID를 추출한다.
  res.send(`User ID: ${userId}`); // 추출한 사용자 ID를 응답으로 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```

#### 2.5 미들웨어  
- **미들웨어**: 요청과 응답 사이에 실행되는 함수로, 여러 작업을 수행할 수 있다.  

```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성한다.

// 모든 요청에 대해 미들웨어를 실행한다.
app.use((req, res, next) => {
  console.log('Request URL:', req.originalUrl); // 요청 URL을 로그에 출력한다.
  next(); // 다음 미들웨어 또는 라우터로 제어를 넘긴다.
});

// 기본 경로('/')로 GET 요청이 들어오면 'Hello World!'를 응답으로 보낸다.
app.get('/', (req, res) => {
  res.send('Hello World!'); // 클라이언트에 응답을 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```

#### 2.6 RESTful 웹 서비스  
- **RESTful API**: REST 아키텍처 스타일을 따르는 웹 서비스입니다.  

```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성();

// /api/products 경로로 GET 요청이 들어오면 제품 목록을 JSON 형태로 응답한다.
app.get('/api/products', (req, res) => {
  res.json([{ id: 1, name: 'Product A' }, { id: 2, name: 'Product B' }]); // JSON 형식으로 응답을 보낸다.
});

// /api/products 경로로 POST 요청이 들어오면 새 제품을 생성하고, 생성된 제품을 JSON 형태로 응답한다.
app.post('/api/products', (req, res) => {
  res.status(201).json({ id: 3, name: 'Product C' }); // 상태 코드 201(Created)과 함께 JSON 형식으로 응답을 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```

### 예제 코드  
```javascript
const express = require('express'); // express 모듈을 가져온다.
const app = express(); // express 애플리케이션 객체를 생성한다.

// 모든 요청에 대해 미들웨어를 실행한다.
app.use((req, res, next) => {
  console.log('Request URL:', req.originalUrl); // 요청 URL을 로그에 출력한다.
  next(); // 다음 미들웨어 또는 라우터로 제어를 넘긴다.
});

// 기본 경로('/')로 GET 요청이 들어오면 'Hello World!'를 응답으로 보낸다.
app.get('/', (req, res) => {
  res.send('Hello World!'); // 클라이언트에 응답을 보낸다.
});

// GET 요청으로 /users 경로에 접근하면 'Users Page'를 응답한다.
app.get('/users', (req, res) => {
  res.send('Users Page'); // 클라이언트에 응답을 보낸다.
});

// POST 요청으로 /users 경로에 접근하면 'Create User'를 응답한다.
app.post('/users', (req, res) => {
  res.send('Create User'); // 클라이언트에 응답을 보낸다.
});

// 경로 매개변수를 사용하여 사용자 ID를 받아온다.
app.get('/users/:id', (req, res) => {
  const userId = req.params.id; // 요청 경로에서 사용자 ID를 추출한다.
  res.send(`User ID: ${userId}`); // 추출한 사용자 ID를 응답으로 보낸다.
});

// /api/products 경로로 GET 요청이 들어오면 제품 목록을 JSON 형태로 응답한다.
app.get('/api/products', (req, res) => {
  res.json([{ id: 1, name: 'Product A' }, { id: 2, name: 'Product B' }]); // JSON 형식으로 응답을 보낸다.
});

// /api/products 경로로 POST 요청이 들어오면 새 제품을 생성하고, 생성된 제품을 JSON 형태로 응답한다.
app.post('/api/products', (req, res) => {
  res.status(201).json({ id: 3, name: 'Product C' }); // 상태 코드 201(Created)과 함께 JSON 형식으로 응답을 보낸다.
});

// 서버를 포트 3000에서 실행한다.
app.listen(3000, () => {
  console.log('Server is running on port 3000'); // 서버가 실행 중임을 알리는 메시지를 출력한다.
});

```

위 예제를 통해서 서버에서 구동하는 자바스크립트에 대해 대략적으로 알 수 있다.  

## 클라이언트 자바스크립트  
자바스크립트는 웹 브라우저와 상호작용할 수 있는 강력한 기능을 제공하며, 다양한 객체 모델을 통해 이를 구현한다.  
각 목차별로 내용을 설명하고, 예제 코드를 통해 구체적으로 사용하는 방법을 알아보자.  

### 1. 웹 브라우저에서의 자바스크립트  

#### 1.1 브라우저 객체 모델 (BOM)  
브라우저 객체 모델(BOM)은 웹 브라우저와 상호작용하기 위한 객체를 제공한다.  
주요 객체는 `window`, `document`, `navigator`, `screen`, `location`, `history` 등이 있습니다.  

#### 1.2 `window` 객체  
- **window 객체**: 브라우저 창을 나타내며, 모든 전역 자바스크립트 객체와 함수는 `window` 객체의 속성이다.  

```javascript
// window.alert(): 브라우저 경고 창을 표시한다.
window.alert("Hello, World!");

// window.innerWidth와 window.innerHeight: 브라우저 창의 너비와 높이를 반환한다.
console.log(`Window width: ${window.innerWidth}, height: ${window.innerHeight}`);
```  

#### 1.3 `screen` 객체  
- **screen 객체**: 사용자의 화면에 대한 정보를 제공한다.  

```javascript
// screen.width와 screen.height: 화면의 너비와 높이를 반환한다.
console.log(`Screen width: ${screen.width}, height: ${screen.height}`);
```  

#### 1.4 `location` 객체와 `history` 객체  
- **location 객체**: 현재 문서의 URL 정보를 제공한다.  
- **history 객체**: 브라우저의 세션 기록을 관리한다.  

```javascript
// location.href: 현재 페이지의 URL을 반환한다.
console.log(location.href);

// location.assign(): 새로운 페이지로 이동한다.
location.assign("https://www.example.com");

// history.back(): 이전 페이지로 이동한다.
history.back();

// history.forward(): 다음 페이지로 이동한다.
history.forward();
```  

#### 1.5 `navigator` 객체  
- **navigator 객체**: 브라우저와 사용자에 대한 정보를 제공한다.  

```javascript
// navigator.userAgent: 사용자가 사용 중인 브라우저의 정보를 반환한다.
console.log(navigator.userAgent);

// navigator.language: 브라우저의 언어 설정을 반환한다.
console.log(navigator.language);
```  

### 종합 예제  

```html
<!DOCTYPE html>
<html>
<head>
  <title>Browser Objects Example</title>
  <script> // 여기부터 자바슼
    // 윈도우 객체 예제
    window.onload = function() {
      alert("Page is loaded!"); // 페이지가 로드되면 경고 창을 표시한다.
      console.log(`Window width: ${window.innerWidth}, height: ${window.innerHeight}`); // 창의 너비와 높이 출력

      // 화면 객체 예제
      console.log(`Screen width: ${screen.width}, height: ${screen.height}`); // 화면의 너비와 높이 출력

      // 위치 객체 예제
      console.log(`Current URL: ${location.href}`); // 현재 페이지의 URL 출력

      // 네비게이터 객체 예제
      console.log(`User Agent: ${navigator.userAgent}`); // 브라우저 정보 출력
      console.log(`Browser Language: ${navigator.language}`); // 브라우저 언어 출력
    };
  </script>
</head>
<body>
  <h1>Browser Objects Example</h1>
  <button onclick="location.assign('https://www.example.com')">Go to Example.com</button> <!-- 버튼 클릭 시 새로운 페이지로 이동 -->
  <button onclick="history.back()">Go Back</button> <!-- 버튼 클릭 시 이전 페이지로 이동 -->
  <button onclick="history.forward()">Go Forward</button> <!-- 버튼 클릭 시 다음 페이지로 이동 -->
</body>
</html>
```  

### 2. 문서 객체 모델 (DOM)  

#### 2.1 문서 객체 모델 관련 용어  
- **DOM (Document Object Model)**: 웹 페이지를 트리 구조로 표현하여, 자바스크립트를 통해 HTML 요소를 조작할 수 있게 한다.  

#### 2.2 웹 페이지 생성 순서  
1. **HTML 파싱**: HTML 문서가 브라우저에 의해 파싱된다.  
2. **DOM 트리 생성**: 파싱된 HTML을 기반으로 DOM 트리가 생성된다.  
3. **CSSOM 생성**: CSS가 파싱되어 CSSOM(CSS Object Model)이 생성된다.  
4. **렌더 트리 생성**: DOM과 CSSOM을 결합하여 렌더 트리가 생성된다.  
5. **레이아웃**: 렌더 트리에 따라 각 요소의 레이아웃이 계산된다.  
6. **페인팅**: 요소들이 화면에 그려진다.  

#### 2.3 문서 객체 선택  
- **하나의 문서 객체 선택**: `document.querySelector()`  
- **여러 개의 문서 객체 선택**: `document.querySelectorAll()`  

```javascript
// 하나의 문서 객체 선택
const header = document.querySelector('h1'); // 첫 번째 h1 요소를 선택한다.
console.log(header.innerText); // 선택한 요소의 텍스트를 출력한다.

// 여러 개의 문서 객체 선택
const items = document.querySelectorAll('li'); // 모든 li 요소를 선택한다.
items.forEach(item => {
  console.log(item.innerText); // 각 요소의 텍스트를 출력한다.
});
```  

#### 2.4 문서 객체 조작  

##### 문자 조작  

```javascript
// 텍스트 내용 변경
const header = document.querySelector('h1');
header.innerText = "New Header Text"; // h1 요소의 텍스트를 변경한다.
```

##### 스타일 조작  

```javascript
// 스타일 변경
header.style.color = 'blue'; // h1 요소의 텍스트 색상을 파란색으로 변경한다.
header.style.fontSize = '2em'; // h1 요소의 텍스트 크기를 2em으로 변경한다.
```  

##### 속성 조작  

```javascript
// 속성 변경
const link = document.querySelector('a');
link.setAttribute('href', 'https://www.new-url.com'); // a 요소의 href 속성을 변경한다.
```  

#### 2.5 이벤트  

##### 이벤트 관련 용어 정리  
- **이벤트**: 사용자나 브라우저가 수행하는 행동 (예: 클릭, 스크롤, 키보드 입력 등).  
- **이벤트 리스너**: 이벤트가 발생했을 때 실행되는 함수.  

##### 인라인 이벤트 모델  

```html
<button onclick="alert('Button clicked!')">Click Me</button> <!-- 클릭 시 경고 창을 표시한다. -->
```  

##### 고전 이벤트 모델  

```javascript
const button = document.querySelector('button');
button.onclick = function() {
  alert('Button clicked!'); // 클릭 시 경고 창을 표시한다.
};
```  

##### 이벤트 객체  

```javascript
// 이벤트 객체 사용
button.addEventListener('click', function(event) {
  console.log(event.type); // 이벤트 타입 출력 (예: 'click')
  console.log(event.target); // 이벤트가 발생한 요소 출력
});
```  

##### 기본 동작 제거  

```javascript
const link = document.querySelector('a');
link.addEventListener('click', function(event) {
  event.preventDefault(); // 링크의 기본 동작(페이지 이동)을 막는다.
  console.log('Link clicked!'); // 메시지를 출력한다.
});
```  

### 2번 챕터의 종합 예제  

```html
<!DOCTYPE html>
<html>
<head>
  <title>DOM Manipulation Example</title>
  <style>
    .highlight {
      color: red; /* 텍스트 색상을 빨간색으로 설정 */
      font-weight: bold; /* 텍스트를 굵게 설정 */
    }
  </style>
  <script>
    window.onload = function() {
      // 문서 객체 선택 예제
      const header = document.querySelector('h1'); // 첫 번째 h1 요소를 선택한다.
      console.log(header.innerText); // 선택한 요소의 텍스트를 출력한다.

      // 문서 객체 조작 예제
      header.innerText = "New Header Text"; // h1 요소의 텍스트를 변경한다.
      header.style.color = 'blue'; // h1 요소의 텍스트 색상을 파란색으로 변경힌다.
      header.style.fontSize = '2em'; // h1 요소의 텍스트 크기를 2em으로 변경힌다.

      const link = document.querySelector('a');
      link.setAttribute('href', 'https://www.new-url.com'); // a 요소의 href 속성을 변경힌다.

      // 이벤트 예제
      const button = document.querySelector('button');
      button.onclick = function() {
        alert('Button clicked!'); // 클릭 시 경고 창을 표시힌다.
      };

      button.addEventListener('click', function(event) {
        console.log(event.type); // 이벤트 타입 출력 (예: 'click')
        console.log(event.target); // 이벤트가 발생한 요소 출력
      });

      link.addEventListener('click', function(event) {
        event.preventDefault(); // 링크의 기본 동작(페이지 이동)을 막는다.
        console.log('Link clicked!'); // 메시지를 출력한다.
      });
    };
  </script>
</head>
<body>
  <h1>DOM Manipulation Example</h1>
  <a

 href="https://www.example.com">Go to Example.com</a> <!-- 링크의 href 속성은 스크립트에서 변경한다. -->
  <button>Click Me</button> <!-- 버튼 클릭 시 이벤트가 발생한다. -->
</body>
</html>
```  

### 3. jQuery  

#### 3.1 사용 준비  
- **jQuery 사용 준비**: jQuery를 사용하기 위해서는 jQuery 라이브러리를 포함해야 한다.  

```html
<!-- jQuery 라이브러리를 포함한다. -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
```  

#### 3.2 객체  
- **jQuery 객체**: jQuery로 선택한 요소는 jQuery 객체가 된다.  

```javascript
$(document).ready(function() {
  // 문서가 준비되면 실행됩니다.
  const $header = $('h1'); // jQuery 객체로 h1 요소를 선택한다.
  console.log($header.text()); // jQuery 객체의 텍스트를 출력한다.
});
```  

#### 3.3 문서 객체 선택  
- **문서 객체 선택**: jQuery에서는 CSS 선택자를 사용하여 문서 객체를 선택한다.  

```javascript
$(document).ready(function() {
  const $items = $('li'); // 모든 li 요소를 선택한다.
  $items.each(function(index, item) {
    console.log($(item).text()); // 각 li 요소의 텍스트를 출력한다.
  });
});
```  

#### 3.4 문서 객체 개별 조작  
- **개별 조작**: 선택된 요소 중 특정 요소만 조작할 수 있다.  

```javascript
$(document).ready(function() {
  const $firstItem = $('li').first(); // 첫 번째 li 요소를 선택한다.
  $firstItem.text('First Item'); // 선택한 요소의 텍스트를 변경한다.
});
```  

#### 3.5 문서 객체 조작  

##### 글자 조작  

```javascript
$(document).ready(function() {
  const $header = $('h1');
  $header.text('New Header Text'); // h1 요소의 텍스트를 변경한다.
});
```  

##### 스타일 조작  

```javascript
$(document).ready(function() {
  const $header = $('h1');
  $header.css('color', 'blue'); // h1 요소의 텍스트 색상을 파란색으로 변경한다.
  $header.css('fontSize', '2em'); // h1 요소의 텍스트 크기를 2em으로 변경한다.
});
```  

##### 속성 조작  

```javascript
$(document).ready(function() {
  const $link = $('a');
  $link.attr('href', 'https://www.new-url.com'); // a 요소의 href 속성을 변경한다.
});
```  

#### 3.6 문서 객체 생성  

```javascript
$(document).ready(function() {
  const $newItem = $('<li>New Item</li>'); // 새로운 li 요소를 생성.
  $('ul').append($newItem); // 생성한 요소를 ul 요소의 끝에 추가.
});
```  

#### 3.7 이벤트  

##### 이벤트 직접 연결  

```javascript
$(document).ready(function() {
  $('button').click(function() {
    alert('Button clicked!'); // 버튼 클릭 시 경고 창을 표시한다.
  });
});
```  

##### 이벤트 간접 연결  

```javascript
$(document).ready(function() {
  $('ul').on('click', 'li', function() {
    alert('List item clicked!'); // 동적으로 추가된 li 요소 클릭 시 경고 창을 표시하ㅏㄴ다.
  });
});
```  

##### 이벤트 제거  

```javascript
$(document).ready(function() {
  const $button = $('button');
  const clickHandler = function() {
    alert('Button clicked!'); // 버튼 클릭 시 경고 창을 표시한다.
  };

  $button.click(clickHandler); // 클릭 이벤ㅌ ㅡ 핸들러를 추가한다.
  $button.off('click', clickHandler); // 클릭 이벤트 핸들러를 제거한다.
});
```  

#### 3.8 애니메이션  

```javascript
$(document).ready(function() {
  $('button').click(function() {
    $('h1').fadeOut(1000).fadeIn(1000); // h1 요소를 1초 동안 사라지게 한 후, 다시 1초 동안 나타나게 한다.
  });
});
```  

### 종합 예제  

```html
<!DOCTYPE html>
<html>
<head>
  <title>jQuery Example</title>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script>
    $(document).ready(function() {
      // 문서 객체 선택 예제
      const $header = $('h1');
      console.log($header.text()); // h1 요소의 텍스트를 출력한다.

      // 문서 객체 조작 예제
      $header.text('New Header Text'); // h1 요소의 텍스트를 변경.
      $header.css('color', 'blue'); // h1 요소의 텍스트 색상을 파란색으로 변경.
      $header.css('fontSize', '2em'); // h1 요소의 텍스트 크기를 2em으로 변경.

      const $link = $('a');
      $link.attr('href', 'https://www.new-url.com'); // a 요소의 href 속성을 변경.

      const $newItem = $('<li>New Item</li>'); // 새로운 li 요소를 생성합.
      $('ul').append($newItem); // 생성한 요소를 ul 요소의 끝에 추가.

      // 이벤트 예제
      $('button').click(function() {
        alert('Button clicked!'); // 버튼 클릭 시 경고 창을 표시.
      });

      $('ul').on('click', 'li', function() {
        alert('List item clicked!'); // 동적으로 추가된 li 요소 클릭 시 경고 창을 표시.
      });

      $('button').click(function() {
        $('h1').fadeOut(1000).fadeIn(1000); // h1 요소를 1초 동안 사라지게 한 후, 다시 1초 동안 나타나게 한다.
      });
    });
  </script>
</head>
<body>
  <h1>jQuery Example</h1>
  <a href="https://www.example.com">Go to Example.com</a> <!-- 링크의 href 속성은 스크립트에서 변경된다. -->
  <ul>
    <li>Item 1</li>
    <li>Item 2</li>
  </ul>
  <button>Click Me</button> <!-- 버튼 클릭 시 이벤트가 발생한다. -->
</body>
</html>
```  

### 4. 프로젝트: 이미지 슬라이더  

#### 4.1 이미지 슬라이더 개요  
이미지 슬라이더는 여러 이미지를 슬라이드 형태로 보여주는 UI 요소 이다.  
특정 시간마다 이밎비가 바꿔어야 하고,  
숫자 버튼을 누르면 해당 위치로 이동한다.  

#### 4.2 이미지 슬라이더 구성  
이미지 슬라이더는 HTML, CSS, JavaScript로 구성 된다.  

#### 4.3 기본 모드 구성  

```html
<!DOCTYPE html> 
<html> 
<head> 
    <title>이미지 슬라이더</title> 
    <script src="jquery-3.6.0.min.js"></script>
    <script> 
    $(document).ready(function () { 
        // 변수를 선언한다 
        const width = 460; 
        const height = 300; 
        let current = 0; 
        
        // 함수를 선언한다 
        const moveTo = function () {
            $('.images').animate({ 
                left: -current * width 
            }, 1000);
        };

        // 슬라이더 내부의 이미지 개수를 확인한다 
        const imageLength = $('.slider').find('.image').length;

        // 슬라이더 버튼 추가  
        for (let i = 0; i < imageLength; i++) {
            $('<button></button>') 
            .attr('data-position', i) 
            .text(i) 
            .click(function () { 
                current = $(this).attr('data-position');
                moveTo();
            })
            .insertBefore('.slider'); 
        }

        // 슬라이더 스타일 설정  
        $('.slider').css({ 
            position: 'relative', 
            width: width, 
            height: height, 
            overflow: 'hidden'
        });
        $('.images').css({
            position: 'absolute', 
            width: width * imageLength,
        });
        $('.image').css({ 
            margin: 0, 
            padding: 0,
            width: width,
            height: height, 
            display: 'block', 
            float: 'left' 
        });
        
        // 3초마다 슬라이더를 이동시킨다.  
        setInterval(function() {
            current = (current + 1) % imageLength; 
            moveTo();
        }, 3000);
    });
    </script> 
</head> 
<body> 
    <div class="slider"> 
        <div class="images"> 
            <img class="image" src="https://via.placeholder.com/460x300?text=image.0" /> 
            <img class="image" src="https://via.placeholder.com/460x300?text=image_1" /> 
            <img class="image" src="https://via.placeholder.com/460x300?text=image_2" /> 
            <img class="image" src="https://via.placeholder.com/460x300?text=image_3" /> 
            <div class="image"> 
                <h1>이미지가 아닌 것</h1> 
                <p>Lorem ipsum dolor sit amet.</p> 
            </div> 
        </div> 
    </div> 
</body> 
</html>

```
위 코드로 구성할 수 있다.  
내용을 보자면, 

### 주석 설명

1. **`const width = 460;`**: 슬라이더의 너비를 설정.  
2. **`const height = 300;`**: 슬라이더의 높이를 설정.  
3. **`let current = 0;`**: 현재 표시되고 있는 이미지의 인덱스를 저장합니다. `let`으로 수정하여 값을 변경할 수 있도록 한다.  
4. **`const moveTo = function () { ... }`**: 슬라이더를 이동시키는 함수입니다. 현재 이미지 인덱스를 기반으로 슬라이더를 애니메이션 한다.  
5. **`$('.images').animate({ left: -current * width }, 1000);`**: 이미지 슬라이더의 위치를 애니메이션으로 변경한다. `left` 속성을 통해 현재 이미지에 맞게 이동한다.  
6. **`const imageLength = $('.slider').find('.image').length;`**: 슬라이더에 포함된 이미지의 총 개수를 계산한다.
7. **`$('<button></button>') ... .insertBefore('.slider');`**: 슬라이더 버튼을 생성하여 각 이미지에 대한 버튼을 추가한다. 이 버튼을 클릭하면 슬라이더가 해당 이미지로 이동한다.
8. **`$('.slider').css({ ... });`**: 슬라이더의 스타일을 설정한다. `position: 'relative'`, `width`, `height`, `overflow` 속성을 설정한다.
9. **`$('.images').css({ ... });`**: 모든 이미지를 포함하는 컨테이너의 스타일을 설정한다. `position: 'absolute'`와 `width` 속성을 사용하여 슬라이더가 수평으로 배치되도록 한다.
10. **`$('.image').css({ ... });`**: 개별 이미지의 스타일을 설정한다. `display: 'block'`과 `float: 'left'` 속성으로 이미지를 수평으로 나열한다.
11. **`setInterval(function() { ... }, 3000);`**: 3초마다 슬라이더를 자동으로 다음 이미지로 이동시키는 타이머를 설정한다.

위 코드는 플러그인을 사용하여 더욱 간단하게 구성할 수 있다.  
위의 코드를 플러그인으로 변경하고, 그 코드를 html에서 사용해 보자.  

플러그인으로 변경한 javascript코드
```javascript
(function($) {
    // jQuery 플러그인 정의
    $.fn.imageSlider = function(options) {
        // 기본 설정을 정의하고, 사용자 설정으로 덮어씌움
        const settings = $.extend({
            width: 460, // 슬라이더의 너비
            height: 300, // 슬라이더의 높이
            animationDuration: 1000, // 애니메이션 지속 시간 (밀리초)
            interval: 3000 // 슬라이드 자동 전환 간격 (밀리초)
        }, options);

        // 각 슬라이더에 대해 초기화 함수 실행
        return this.each(function() {
            const $slider = $(this); // 현재 슬라이더 요소를 jQuery 객체로 선택
            const $images = $slider.find('.images'); // 슬라이더 내의 이미지 컨테이너 선택
            const $imageItems = $images.find('.image'); // 이미지 항목들 선택
            const imageLength = $imageItems.length; // 이미지의 개수
            let current = 0; // 현재 표시 중인 이미지의 인덱스

            // 이미지 슬라이더 이동 함수 정의
            const moveTo = function() {
                $images.animate({ 
                    left: -current * settings.width // 현재 인덱스에 따라 왼쪽으로 이동
                }, settings.animationDuration); // 애니메이션 지속 시간 설정
            };

            // 슬라이더 버튼 추가
            for (let i = 0; i < imageLength; i++) {
                $('<button></button>') // 새 버튼 요소 생성
                    .attr('data-position', i) // 버튼에 위치 정보 저장
                    .text(i) // 버튼에 인덱스 번호 텍스트 설정
                    .click(function() { // 버튼 클릭 이벤트 핸들러
                        current = $(this).attr('data-position'); // 클릭한 버튼의 위치를 현재 위치로 설정
                        moveTo(); // 슬라이더 이동
                    })
                    .insertBefore($slider); // 버튼을 슬라이더 앞에 삽입
            }

            // 슬라이더 스타일 설정
            $slider.css({
                position: 'relative', // 슬라이더를 상대적으로 위치
                width: settings.width, // 슬라이더의 너비 설정
                height: settings.height, // 슬라이더의 높이 설정
                overflow: 'hidden' // 넘치는 부분을 숨김
            });
            $images.css({
                position: 'absolute', // 이미지 컨테이너를 절대적으로 위치
                width: settings.width * imageLength // 이미지 컨테이너의 총 너비 설정
            });
            $imageItems.css({
                margin: 0, // 이미지의 외부 여백 제거
                padding: 0, // 이미지의 내부 여백 제거
                width: settings.width, // 이미지의 너비 설정
                height: settings.height, // 이미지의 높이 설정
                display: 'block', // 이미지를 블록 요소로 설정
                float: 'left' // 이미지를 왼쪽으로 정렬
            });

            // 자동 슬라이드 설정
            setInterval(function() {
                current = (current + 1) % imageLength; // 현재 인덱스를 증가시키고, 마지막 인덱스 이후에는 처음으로 돌아감
                moveTo(); // 슬라이더 이동
            }, settings.interval); // 설정된 간격마다 반복
        });
    };
})(jQuery); // jQuery를 플러그인 내에서 사용할 수 있게 즉시 실행 함수로 감싸기

```

위 킆퍼그인을 사용하는 html
```html
<!DOCTYPE html>
<html>
<head>
    <title>이미지 슬라이더</title>
    <script src="jquery-3.6.0.min.js"></script> <!-- jQuery 라이브러리 포함 -->
    <script src="slider-plugin.js"></script> <!-- 플러그인 코드가 저장된 파일 포함 -->
</head>
<body>
    <div class="slider"> <!-- 슬라이더 컨테이너 -->
        <div class="images"> <!-- 이미지 컨테이너 -->
            <img class="image" src="https://via.placeholder.com/460x300?text=image.0" /> <!-- 첫 번째 이미지 -->
            <img class="image" src="https://via.placeholder.com/460x300?text=image_1" /> <!-- 두 번째 이미지 -->
            <img class="image" src="https://via.placeholder.com/460x300?text=image_2" /> <!-- 세 번째 이미지 -->
            <img class="image" src="https://via.placeholder.com/460x300?text=image_3" /> <!-- 네 번째 이미지 -->
            <div class="image"> <!-- 이미지가 아닌 콘텐츠 -->
                <h1>이미지가 아닌 것</h1> <!-- 제목 -->
                <p>Lorem ipsum dolor sit amet.</p> <!-- 내용 -->
            </div>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            $('.slider').imageSlider({ // 슬라이더 플러그인 초기화
                width: 460, // 슬라이더 너비
                height: 300, // 슬라이더 높이
                animationDuration: 1000, // 애니메이션 지속 시간
                interval: 3000 // 자동 전환 간격
            });
        });
    </script>
</body>
</html>

```
위 방법을 통해서 재사용성을 높일수 있다.  

이 내용을 토대로 React Native를 해보자  