---
layout: single
title:  "SQL, No.SQL 정리"
categories: "SQL"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# SQL 및 NoSQL 데이터베이스 정리  

## 목차

1. [SQL 기본 개념](#sql-기본-개념)
2. [SQL 데이터 정의 언어 (DDL)](#sql-데이터-정의-언어-ddl)
3. [SQL 데이터 조작 언어 (DML)](#sql-데이터-조작-언어-dml)
4. [SQL 고급 기능](#sql-고급-기능)
5. [NoSQL 기본 개념](#nosql-기본-개념)
6. [NoSQL 데이터 모델](#nosql-데이터-모델)
7. [Spring Boot와 SQL 연동](#spring-boot와-sql-연동)
8. [Spring Boot와 MySQL 연동](#spring-boot와-mysql-연동)
9. [결론](#결론)

---

## 1. SQL 기본 개념
### 1.1 SQL이란?

SQL(Structured Query Language)은 관계형 데이터베이스에서 데이터를 정의하고 조작하는 데 사용되는 표준 언어이다.  
데이터베이스에서 데이터를 조회, 삽입, 업데이트 및 삭제하는 기능을 제공한다.  

### 1.2 관계형 데이터베이스
관계형 데이터베이스는 데이터를 테이블 형태로 저장한다.  
각 테이블은 행과 열로 구성되며, 행은 레코드, 열은 속성을 나타냅니다. 테이블 간에는 관계가 설정될 수 있다.  

#### 1.2.1 테이블 구조

테이블은 다음과 같은 구조를 가진다:  

- **테이블명**: 데이터가 저장되는 기본 단위  
- **열 (컬럼)**: 데이터의 속성 또는 특성  
- **행 (레코드)**: 데이터의 개별 인스턴스  

예를 들어, 직원 정보를 저장하는 `employees` 테이블은 다음과 같은 구조를 가질 수 있다:  

| employee_id | first_name | last_name | email                | hire_date  |
|-------------|------------|-----------|----------------------|------------|
| 1           | John       | Doe       | john.doe@example.com | 2022-01-15 |
| 2           | Jane       | Smith     | jane.smith@example.com | 2023-05-22 |

SQL 데이터베이스는 데이터를 표 형식으로 저장하며, 각 표는 행과 열로 구성된다.  
주요 SQL 데이터베이스 시스템으로는 MySQL, PostgreSQL, Oracle, Microsoft SQL Server 등이 있다.  
### 1.3 SQL의 주요 명령어

SQL은 다양한 명령어를 제공하여 데이터베이스를 관리합니다. 주요 명령어는 다음과 같다:  

- **SELECT**: 데이터 조회
- **INSERT**: 데이터 삽입
- **UPDATE**: 데이터 업데이트
- **DELETE**: 데이터 삭제


## 2. SQL 데이터 정의 언어 (DDL)

### 2.1 테이블 생성

테이블을 생성하려면 `CREATE TABLE` 명령어를 사용합니다. 예를 들어, `employees` 테이블을 생성하는 SQL 문은 다음과 같다:  

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    hire_date DATE
);
```

- **employee_id**: 직원의 고유 식별자  
- **first_name**: 직원의 이름  
- **last_name**: 직원의 성  
- **email**: 직원의 이메일 주소  
- **hire_date**: 직원의 채용 날짜  

### 2.2 테이블 수정

테이블 구조를 수정하려면 `ALTER TABLE` 명령어를 사용합니다. 예를 들어, `employees` 테이블에 `phone_number` 열을 추가하려면 다음과 같이 한다:  

```sql
ALTER TABLE employees
ADD phone_number VARCHAR(15);
```
이렇게 할 수도 있다.  
```sql
ALTER TABLE employees ADD COLUMN salary DECIMAL(10, 2);
```

### 2.3 테이블 삭제

테이블을 삭제하려면 `DROP TABLE` 명령어를 사용합니다. 다음 SQL 문은 `employees` 테이블을 삭제한다:  

```sql
DROP TABLE employees;
```



## 3. SQL 데이터 조작 언어 (DML)

### 3.1 데이터 삽입  

데이터를 삽입하려면 `INSERT INTO` 명령어를 사용합니다. 예를 들어, `employees` 테이블에 새 직원을 추가하려면 다음과 같은 SQL 문을 사용한다:  

```sql
INSERT INTO employees (employee_id, first_name, last_name, email, hire_date)
VALUES (1, 'John', 'Doe', 'john.doe@example.com', '2022-01-15');
```
새 데이터를 체이블에 추가한다면 이런 식이다.  
```sql
INSERT INTO employees (id, name, age, department, salary)
VALUES (1, 'John Doe', 30, 'Engineering', 60000.00);
```

### 3.2 데이터 조회

데이터를 조회하려면 `SELECT` 명령어를 사용합니다. 예를 들어, `employees` 테이블에서 모든 직원을 조회하려면 다음과 같은 SQL 문을 사용합니다:

```sql
SELECT * FROM employees;
```

특정 열만 조회하려면 다음과 같이 쿼리를 수정할 수 있습니다:

```sql
SELECT first_name, email FROM employees;
```

### 3.3 데이터 업데이트

데이터를 업데이트하려면 `UPDATE` 명령어를 사용합니다. 예를 들어, `employees` 테이블에서 특정 직원의 이메일 주소를 업데이트하려면 다음과 같은 SQL 문을 사용합니다:

```sql
UPDATE employees
SET email = 'john.newemail@example.com'
WHERE employee_id = 1;
```

### 3.4 데이터 삭제

데이터를 삭제하려면 `DELETE FROM` 명령어를 사용합니다. 예를 들어, `employees` 테이블에서 특정 직원을 삭제하려면 다음과 같은 SQL 문을 사용합니다:

```sql
DELETE FROM employees
WHERE employee_id = 1;
```

---

위 내용을 간단 명료하게 정리하면, 다음과 같다.  
### 데이터 조작 언어 (DML)
- **INSERT INTO**: 새로운 데이터를 테이블에 추가한다.

  ```sql
  INSERT INTO employees (id, name, age, department, salary)
  VALUES (1, 'John Doe', 30, 'Engineering', 60000.00);
  ```

- **UPDATE**: 기존 데이터를 수정한다.

  ```sql
  UPDATE employees
  SET salary = 65000.00
  WHERE id = 1;
  ```

- **DELETE**: 데이터를 삭제한다.

  ```sql
  DELETE FROM employees
  WHERE id = 1;
  ```

- **SELECT**: 데이터를 조회한다.

  ```sql
  SELECT * FROM employees;
  ```

추가적인 내용을 더 보자면,  
### 데이터 제어 언어 (DCL)
- **GRANT**: 사용자에게 권한을 부여한다.

  ```sql
  GRANT SELECT, INSERT ON employees TO user1;
  ```

- **REVOKE**: 사용자 권한을 철회한다.

  ```sql
  REVOKE INSERT ON employees FROM user1;
  ```

## 4. SQL 고급 기능

### 4.1 JOIN

`JOIN` 명령어를 사용하여 여러 테이블의 데이터를 결합할 수 있다.  
주요 `JOIN` 유형은 다음과 같다:  

- **INNER JOIN**: 두 테이블에서 일치하는 행만 반환합니다.
- **LEFT JOIN**: 왼쪽 테이블의 모든 행과 오른쪽 테이블의 일치하는 행을 반환합니다.
- **RIGHT JOIN**: 오른쪽 테이블의 모든 행과 왼쪽 테이블의 일치하는 행을 반환합니다.
- **FULL JOIN**: 두 테이블의 모든 행을 반환합니다.

예시를 통해 보자면,  
#### 조인 (JOIN)

- **INNER JOIN**: 두 테이블에서 일치하는 행만 조회한다.

  ```sql
  SELECT employees.name, departments.department_name
  FROM employees
  INNER JOIN departments
  ON employees.department = departments.department_id;
  ```

- **LEFT JOIN**: 왼쪽 테이블의 모든 행과 오른쪽 테이블의 일치하는 행을 조회한다.

  ```sql
  SELECT employees.name, departments.department_name
  FROM employees
  LEFT JOIN departments
  ON employees.department = departments.department_id;
  ```

- **RIGHT JOIN**: 오른쪽 테이블의 모든 행과 왼쪽 테이블의 일치하는 행을 조회한다.

  ```sql
  SELECT employees.name, departments.department_name
  FROM employees
  RIGHT JOIN departments
  ON employees.department = departments.department_id;
  ```

- **FULL OUTER JOIN**: 두 테이블의 모든 행을 조회하며, 일치하지 않는 경우 NULL을 반환한다.

  ```sql
  SELECT employees.name, departments.department_name
  FROM employees
  FULL OUTER JOIN departments
  ON employees.department = departments.department_id;
  ```
이런 구성이다.  

예를 들어, `employees` 테이블과 `departments` 테이블을 결합하려면 다음과 같은 SQL 문을 사용할 수 있다:  

```sql
SELECT e.first_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;
```

추가적으로 더 볼수 있는 것은 서브쿼리가 있다.  
#### 서브쿼리 (Subquery)
- **단일 행 서브쿼리**: 하나의 값만 반환하는 서브쿼리 이다.  

  ```sql
  SELECT name
  FROM employees
  WHERE age = (SELECT MAX(age) FROM employees);
  ```

- **다중 행 서브쿼리**: 여러 값을 반환하는 서브쿼리 이다.  

  ```sql
  SELECT name
  FROM employees
  WHERE department IN (SELECT department_id FROM departments WHERE department_name = 'Engineering');
  ```

- **상관 서브쿼리**: 외부 쿼리와 연관된 서브쿼리 이다.  

  ```sql
  SELECT name
  FROM employees e1
  WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e1.department = e2.department);
  ```

### 4.2 집계 함수

집계 함수는 데이터를 요약하는 데 사용된다.  
주요 집계 함수는 다음과 같다:  

- **COUNT**: 행의 수를 계산합니다.
- **SUM**: 열의 합계를 계산합니다.
- **AVG**: 열의 평균 값을 계산합니다.
- **MAX**: 열의 최대 값을 계산합니다.
- **MIN**: 열의 최소 값을 계산합니다.

예시를 통해 보자.  
#### 집계 함수 (Aggregate Functions)

- **COUNT**: 행의 수를 센다.

  ```sql
  SELECT COUNT(*) FROM employees;
  ```

- **SUM**: 값의 합계를 계산한다.

  ```sql
  SELECT SUM(salary) FROM employees;
  ```

- **AVG**: 값의 평균을 계산한다.

  ```sql
  SELECT AVG(salary) FROM employees;
  ```

- **MIN**: 최소 값을 찾는다.

  ```sql
  SELECT MIN(salary) FROM employees;
  ```

- **MAX**: 최대 값을 찾는다.

  ```sql
  SELECT MAX(salary) FROM employees;
  ```

예를 들어, `employees` 테이블에서 직원 수를 계산하려면 다음과 같은 SQL 문을 사용한다면:  

```sql
SELECT COUNT(*) FROM employees;
```

### 4.3 서브쿼리

서브쿼리는 다른 쿼리의 결과를 사용하는 쿼리이다. 
서브쿼리는 `SELECT`, `INSERT`, `UPDATE`, `DELETE` 문에서 사용될 수 있다.

예를 들어, `employees` 테이블에서 가장 높은 급여를 받는 직원을 조회하려면 다음과 같은 SQL 문을 사용할 수 있다:

```sql
SELECT * FROM employees
WHERE salary = (SELECT MAX(salary) FROM employees);
```

### 4.4 인덱스
 데이터베이스 테이블의 검색 성능을 향상시키기 위한 데이터 구조이며, 인덱스를 사용하면 검색 속도가 크게 향상될 수 있다.  
- **인덱스 생성**:  

  ```sql
  CREATE INDEX idx_salary
  ON employees(salary);
  ```

- **인덱스 삭제**:  

  ```sql
  DROP INDEX idx_salary;
  ```

인덱스는 데이터베이스 테이블의 검색 성능을 향상시키는 데 사용된다.  
인덱스를 생성하려면 `CREATE INDEX` 명령어를 사용한다.  
예를 들어, `employees` 테이블의 `last_name` 열에 인덱스를 추가하려면 다음과 같은 SQL 문을 사용하면 된다:  

```sql
CREATE INDEX idx_last_name
ON employees (last_name);
```
### 4.5 트랜잭션
데이터베이스 작업의 일련의 연산을 하나의 작업 단위로 묶는 것이다.  
트랜잭션은 ACID 속성(원자성, 일관성, 고립성, 지속성)을 보장한다.  

- **트랜잭션 시작**:

  ```sql
  START TRANSACTION;
  ```

- **트랜잭션 커밋**:

  ```sql
  COMMIT;
  ```

- **트랜잭션 롤백**:

  ```sql
  ROLLBACK;
  ```


## 5. NoSQL 기본 개념

### 5.1 NoSQL이란?

NoSQL(Not Only SQL) 데이터베이스는 비관계형 데이터베이스로, 다양한 데이터 모델을 지원한다.  
NoSQL 데이터베이스는 대규모 데이터 저장과 높은 성능을 제공하며, 수평 확장이 용이하다.  

### 5.2 NoSQL 데이터 모델

NoSQL 데이터베이스는 여러 가지 데이터 모델을 지원한다.  
주요 데이터 모델은 다음과 같다:  

- **문서 지향 데이터베이스**: JSON과 유사한 문서 형태로 데이터를 저장한다. 예: MongoDB
- **열 기반 데이터베이스**: 데이터를 열 단위로 저장한다. 예: Apache Cassandra
- **키-값 저장소**: 데이터를 키와 값의 쌍으로 저장한다. 예: Redis
- **그래프 기반 데이터베이스**: 데이터를 그래프 형태로 저장한다. 예: Neo4j


## 6. NoSQL 데이터 모델
oSQL(Not Only SQL) 데이터베이스는 비관계형 데이터베이스 시스템으로, 구조화되지 않은 데이터를 저장하고 처리하는 데 적합하다.  
NoSQL 데이터베이스는 다음과 같은 네 가지 주요 유형으로 나눌 수 있다: 문서 기반, 열 기반, 키-값 저장소, 그래프 기반.  

- **MongoDB**: 문서 기반 데이터베이스로, JSON과 유사한 BSON 포맷을 사용한다.  
- **Cassandra**: 열 기반 데이터베이스로, 대규모 데이터 처리에 적합하다.  
- **Redis**: 키-값 저장소로, 메모리 기반 데이터베이스 이다.  
- **Neo4j**: 그래프 기반 데이터베이스로, 그래프 구조의 데이터를 효율적으로 처리한다.  
### 6.1 문서 지향 데이터베이스

문서 지향 데이터베이스는 JSON과 유사한 문서 형태로 데이터를 저장합니다. 문서 지향 데이터베이스의 주요 특징은 다음과 같다:  

- **스키마 유연성**: 각 문서가 서로 다른 구조를 가질 수 있다.  
- **중첩된 데이터**: 문서 내에 중첩된 데이터를 포함할 수 있다.  

예를 들어, MongoDB에서 문서를 삽입하려면 다음과 같은 명령어를 사용할 수 있다:  

```javascript
db.employees.insertOne({
    employee_id: 1,
    first_name: "John",
    last_name: "Doe",
    email: "john.doe@example.com",
    hire_date: new Date("2022-01-15")
});
```

### 6.2 문서 기반 데이터베이스
문서 기반 데이터베이스는 데이터를 문서 형태로 저장한다. 각 문서는 키-값 쌍으로 구성되며, JSON 형식으로 저장된다.  

- **MongoDB 예제**:

  ```json
  {
      "_id": "1

      "name": "Alice",
      "age": 29,
      "address": {
          "street": "123 Elm Street",
          "city": "Springfield"
      }
  }
  ```

- **문서 삽입**:

  ```javascript
  db.employees.insertOne({
      "_id": "1",
      "name": "Alice",
      "age": 29,
      "address": {
          "street": "123 Elm Street",
          "city": "Springfield"
      }
  });
  ```

- **문서 조회**:

  ```javascript
  db.employees.find({ "name": "Alice" });
  ```

- **문서 업데이트**:

  ```javascript
  db.employees.updateOne(
      { "name": "Alice" },
      { $set: { "age": 30 } }
  );
  ```

- **문서 삭제**:

  ```javascript
  db.employees.deleteOne({ "name": "Alice" });
  ```

### 6.3 열 기반 데이터베이스

열 기반 데이터베이스는 데이터를 열 단위로 저장한다. 이는 대규모 데이터 분석과 쿼리 성능을 향상시키는 데 유리하다.  

- **Cassandra 예제**:

  ```cql
  CREATE TABLE employees (
      id UUID PRIMARY KEY,
      name TEXT,
      age INT,
      department TEXT
  );
  ```

- **열 삽입**:

  ```cql
  INSERT INTO employees (id, name, age, department)
  VALUES (uuid(), 'Bob', 35, 'Marketing');
  ```

- **열 조회**:

  ```cql
  SELECT * FROM employees WHERE name = 'Bob';
  ```

- **열 업데이트**:

  ```cql
  UPDATE employees
  SET age = 36
  WHERE name = 'Bob';
  ```

- **열 삭제**:

  ```cql
  DELETE FROM employees
  WHERE name = 'Bob';
  ```

### 6.4 키-값 저장소

키-값 저장소는 간단한 데이터 모델을 가지고 있으며, 키와 값을 쌍으로 저장한다. 매우 빠른 조회 성능을 제공한다.  

- **Redis 예제**:

  ```redis
  SET user:1000 "Alice"
  ```

- **키 조회**:

  ```redis
  GET user:1000
  ```

- **키 업데이트**:

  ```redis
  SET user:1000 "Bob"
  ```

- **키 삭제**:

  ```redis
  DEL user:1000
  ```

#### 6.5 그래프 기반 데이터베이스

그래프 기반 데이터베이스는 데이터와 관계를 그래프 형태로 저장한다. 복잡한 관계를 모델링하고 쿼리하는 데 유리하다.  

- **Neo4j 예제**:

  ```cypher
  CREATE (a:Person {name: 'Alice', age: 29})
  CREATE (b:Person {name: 'Bob', age: 35})
  CREATE (a)-[:KNOWS]->(b)
  ```

- **노드 조회**:

  ```cypher
  MATCH (n:Person) WHERE n.name = 'Alice' RETURN n
  ```

- **관계 조회**:

  ```cypher
  MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
  ```

- **노드 업데이트**:

  ```cypher
  MATCH (n:Person {name: 'Alice'}) SET n.age = 30
  ```

- **노드 삭제**:

  ```cypher
  MATCH (n:Person {name: 'Alice'}) DELETE n
  ```


### 6.6 NoSQL의 특징과 사용 사례
#### 특징

- **스케일 아웃**: NoSQL 데이터베이스는 수평 확장이 용이하여, 대량의 데이터를 처리하는 데 유리합니다.
- **유연한 데이터 모델**: 다양한 형태의 데이터를 저장할 수 있으며, 스키마가 유연합니다.
- **고성능**: 읽기 및 쓰기 성능이 우수하여 실시간 데이터 처리가 가능합니다.

#### 사용 사례

- **소셜 미디어**: 사용자 프로필, 친구 관계, 게시물 등을 저장하고 처리하는 데 적합합니다.
- **추천 시스템**: 사용자 행동 데이터를 분석하고 개인화된 추천을 제공하는 데 유리합니다.
- **IoT**: 센서 데이터, 로그 데이터를 수집하고 분석하는 데 유용합니다.


## 7. Spring Boot와 SQL 연동

### 7.1 Spring Boot 소개

Spring Boot는 Spring 프레임워크의 확장으로, 빠르고 간편하게 스프링 애플리케이션을 개발할 수 있도록 도와준다. Spring Boot는 자동 설정, 내장 서버, 운영 상태 점검 등의 기능을 제공한다.

### 7.2 Spring Boot 프로젝트 생성

Spring Boot 프로젝트를 생성하려면 Spring Initializr를 사용할 수 있다. 다음 단계를 따라 프로젝트를 생성하자:  

1. [Spring Initializr](https://start.spring.io/)에 접속하자.  
2. 프로젝트 메타데이터를 입력하자.  
   - **Project**: Maven Project
   - **Language**: Java
   - **Spring Boot**: 2.7.5 (예: 최신 버전 선택)
   - **Group**: com.example
   - **Artifact**: myapplication
   - **Dependencies**: Spring Web, Spring Data JPA, MySQL Driver
3. **Generate** 버튼을 클릭하여 프로젝트를 다운로드한다.  
4. 다운로드한 ZIP 파일을 압축 해제하고, IDE에서 프로젝트를 연다.  

### 7.3 application.properties 설정

`src/main/resources/application.properties` 파일에 데이터베이스 연결 정보를 설정한다:  

```properties
# MySQL 데이터베이스 설정
spring.datasource.url=jdbc:mysql://localhost:3306/mydatabase
spring.datasource.username=myuser
spring.datasource.password=mypassword
spring.jpa.database-platform=org.hibernate.dialect.MySQL5Dialect
```

### 7.4 엔티티 클래스 생성

JPA(Entity) 클래스를 생성하여 데이터베이스 테이블과 매핑한다.  
예를 들어, `Employee` 엔티티 클래스를 생성하려면 다음과 같이 작성한다:  

```java
package com.example.myapplication.model;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.util.Date;

@Entity
@Table(name = "employees")
public class Employee {

    @Id
    private int employeeId;
    private String firstName;
    private String lastName;
    private String email;
    private Date hireDate;

    // Getters and Setters
}
```

### 7.5 리포지토리 인터페이스 생성

`EmployeeRepository` 인터페이스를 생성하여 데이터베이스 작업을 수행한다:  

```java
package com.example.myapplication.repository;

import com.example.myapplication.model.Employee;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EmployeeRepository extends JpaRepository<Employee, Integer> {
}
```

### 7.6 서비스 클래스 생성

서비스 클래스에서 비즈니스 로직을 구현한다:  

```java
package com.example.myapplication.service;

import com.example.myapplication.model.Employee;
import com.example.myapplication.repository.EmployeeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EmployeeService {

    @Autowired
    private EmployeeRepository employeeRepository;

    public List<Employee> getAllEmployees() {
        return employeeRepository.findAll();
    }

    public Employee getEmployeeById(int id) {
        return employeeRepository.findById(id).orElse(null);
    }

    public Employee saveEmployee(Employee employee) {
        return employeeRepository.save(employee);
    }

    public void deleteEmployee(int id) {
        employeeRepository.deleteById(id);
    }
}
```

### 7.7 컨트롤러 클래스 생성

RESTful API를 제공하는 컨트롤러 클래스를 생성한다:  

```java
package com.example.myapplication.controller;

import com.example.myapplication.model.Employee;
import com.example.myapplication.service.EmployeeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/employees")
public class EmployeeController {

    @Autowired
    private EmployeeService employeeService;

    @GetMapping
    public List<Employee> getAllEmployees() {
        return employeeService.getAllEmployees();
    }

    @GetMapping("/{id}")
    public Employee getEmployeeById(@PathVariable int id) {
        return employeeService.getEmployeeById(id);
    }

    @PostMapping
    public Employee createEmployee(@RequestBody Employee employee) {
        return employeeService.saveEmployee(employee);
    }

    @PutMapping("/{id}")
    public Employee updateEmployee(@PathVariable int id, @RequestBody Employee employee) {
        employee.setEmployeeId(id);
        return employeeService.saveEmployee(employee);
    }

    @DeleteMapping("/{id}")
    public void deleteEmployee(@PathVariable int id) {
        employeeService.deleteEmployee(id);
    }
}
```

---

## 8. Spring Boot와 MySQL 연동

### 8.1 MySQL 설치

MySQL을 설치하려면 [MySQL 공식 웹사이트](https://dev.mysql.com/downloads/)에서 설치 파일을 다운로드하고 설치하면 된다.  
설치 후, MySQL 서버를 시작하자.  

### 8.2 데이터베이스 및 사용자 생성

MySQL 데이터베이스와 사용자를 생성하려면 MySQL 명령줄 클라이언트 또는 MySQL Workbench를 사용한다.  
다음 SQL 명령어를 사용하여 데이터베이스와 사용자를 생성한다:  

```sql
CREATE DATABASE mydatabase;
CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'mypassword';
GRANT ALL PRIVILEGES ON mydatabase.* TO 'myuser'@'localhost';
FLUSH PRIVILEGES;
```

### 8.3 데이터베이스 연결 테스트  

Spring Boot 애플리케이션을 실행하여 데이터베이스 연결을 테스트한다.  
애플리케이션이 성공적으로 실행되면, 데이터베이스와의 연결이 성공한 것이다.  

### 8.4 CRUD 테스트

Postman 또는 cURL을 사용하여 RESTful API를 테스트한다.  
예를 들어, 다음과 같은 요청을 사용하여 직원 데이터를 조회, 추가, 수정 및 삭제할 수 있다:  

- **GET /employees**: 모든 직원 조회
- **GET /employees/{id}**: 특정 직원 조회
- **POST /employees**: 직원 추가
- **PUT /employees/{id}**: 직원 수정
- **DELETE /employees/{id}**: 직원 삭제

---

위 과정을 따르면, SQL, NOSQL에 대해 알 수 있다.  