---
layout: single
title:  "아키텍처 패턴 시리즈 14. 마이크로서비스 아키텍처 (Microservices Architecture)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 14: 마이크로서비스 아키텍처 (Microservices Architecture)

마이크로서비스 아키텍처는 애플리케이션을 독립적으로 배포 및 관리할 수 있는 작은 서비스들로 나누는 아키텍처 패턴이다.  
각 서비스는 특정 도메인에 집중하여 독립적으로 개발, 배포, 확장할 수 있다.  

## 마이크로서비스 아키텍처의 필요성

다음과 같은 문제를 해결하기 위해 마이크로서비스 아키텍처가 사용된다:

1. 모놀리식 아키텍처의 비효율성: 하나의 큰 코드베이스로 구성된 시스템은 배포 및 확장이 어려움.  
2. 빠른 배포 요구: 독립적인 배포를 통해 특정 서비스만 업데이트하거나 수정 가능.  
3. 복잡한 비즈니스 요구: 비즈니스 요구사항이 다양하고 확장 가능성이 클 때 유리.  

### 예시: 전자 상거래 애플리케이션

전자 상거래 플랫폼은 일반적으로 다음과 같은 기능이 필요하다:  

```
- 사용자 관리
- 상품 카탈로그
- 주문 처리
- 결제
```

마이크로서비스 아키텍처에서는 각각의 기능을 독립적인 서비스로 분리하여 관리한다.  

## 마이크로서비스 아키텍처의 구조

### 주요 컴포넌트

1. 서비스: 특정 기능을 담당하는 독립적인 모듈.  
2. API 게이트웨이: 클라이언트와 서비스 간의 중재자 역할.  
3. 서비스 디스커버리: 동적으로 서비스 위치를 찾기 위한 시스템.  
4. 데이터베이스: 각 서비스는 자신만의 데이터베이스를 가지며 데이터 독립성을 유지.  
5. 메시징 시스템: 서비스 간 비동기 통신을 처리.  

### 구조 다이어그램

```
[Client]
   |
[API Gateway]
   |
[Service A] [Service B] [Service C]
   |          |           |
[DB A]      [DB B]      [DB C]
```

### 동작 원리

1. 클라이언트가 API 게이트웨이를 통해 요청을 보냄.  
2. API 게이트웨이는 요청을 적절한 서비스로 전달.  
3. 각 서비스는 독립적으로 요청을 처리하고, 필요시 다른 서비스와 통신.  
4. 데이터는 각 서비스의 독립적인 데이터베이스에 저장.  

## 마이크로서비스 아키텍처 예시

### Node.js 기반 마이크로서비스 구현

#### 1. 사용자 서비스 (User Service)

```javascript
const express = require('express');
const app = express();
app.use(express.json());

// 사용자 데이터
const users = [{ id: 1, name: 'Alice' }, { id: 2, name: 'Bob' }];

// 사용자 목록 반환
app.get('/users', (req, res) => {
    res.json(users);
});

// 사용자 추가
app.post('/users', (req, res) => {
    const user = { id: users.length + 1, name: req.body.name };
    users.push(user);
    res.status(201).json(user);
});

app.listen(3001, () => {
    console.log('User Service running on port 3001');
});
```

#### 2. 주문 서비스 (Order Service)

```javascript
const express = require('express');
const app = express();
app.use(express.json());

// 주문 데이터
const orders = [{ id: 1, user: 1, product: 'Book' }];

// 주문 목록 반환
app.get('/orders', (req, res) => {
    res.json(orders);
});

// 주문 추가
app.post('/orders', (req, res) => {
    const order = { id: orders.length + 1, user: req.body.user, product: req.body.product };
    orders.push(order);
    res.status(201).json(order);
});

app.listen(3002, () => {
    console.log('Order Service running on port 3002');
});
```

#### 3. API 게이트웨이

```javascript
const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// 사용자 서비스로 요청 전달
app.get('/users', async (req, res) => {
    const response = await axios.get('http://localhost:3001/users');
    res.json(response.data);
});

// 주문 서비스로 요청 전달
app.get('/orders', async (req, res) => {
    const response = await axios.get('http://localhost:3002/orders');
    res.json(response.data);
});

app.listen(3000, () => {
    console.log('API Gateway running on port 3000');
});
```

### 실행 방법

1. 사용자 서비스, 주문 서비스, API 게이트웨이를 각각 실행한다.
2. 클라이언트는 API 게이트웨이(`http://localhost:3000`)를 통해 요청을 보낸다.

## 마이크로서비스 아키텍처의 장점

1. 독립적인 배포: 각 서비스는 독립적으로 배포 및 확장 가능.  
2. 높은 유지보수성: 서비스 단위로 코드베이스를 분리하여 관리 용이.  
3. 확장성: 필요에 따라 특정 서비스만 수평 확장 가능.  
4. 팀 분리: 각 팀이 특정 서비스에만 집중하여 개발 가능.  

## 마이크로서비스 아키텍처의 단점

1. 복잡한 관리: 서비스 수가 많아질수록 관리와 모니터링이 어려움.  
2. 통신 오버헤드: 서비스 간 통신 비용이 증가할 수 있음.  
3. 데이터 일관성 문제: 서비스별 데이터베이스로 인해 데이터 동기화가 어려움.  
4. 배포 복잡성: 다수의 서비스 배포 과정이 복잡해질 수 있음.  

### 마무리

마이크로서비스 아키텍처는 대규모 시스템에서 확장성과 독립성을 극대화하는 데 적합하다.  
그러나, 설계와 관리가 복잡하므로 신중한 접근이 필요하다.  


아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
