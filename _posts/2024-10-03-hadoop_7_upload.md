---
layout: single
title:  "하둡 강좌 7편 HiveQL"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 7편: **HiveQL 심화 및 데이터 분석**  
이번 강좌에서는 **HiveQL**을 더 깊이 있게 다루며, 복잡한 데이터 분석을 위한 다양한 기능을 학습하겠다.  
HiveQL은 SQL과 유사한 쿼리 언어로, 하둡의 분산 저장소에서 대규모 데이터를 쉽게 처리할 수 있다.  
기본적인 쿼리에서 나아가 조인, 서브쿼리, 집계 함수 등을 이용한 고급 데이터 분석 방법을 설명하겠다.  

## 1. HiveQL의 기본 개념 복습  
이전 강좌에서 Hive에서 데이터를 쿼리하는 기본적인 방법을 배웠다.  
Hive 테이블을 생성하고 데이터를 로드한 후, 기본적인 `SELECT` 문으로 데이터를 조회하는 방법을 익혔다.  

### 1.1 HiveQL과 SQL의 차이  
- **Schema on Read**: Hive는 데이터를 로드할 때 스키마를 적용한다. 데이터 파일은 읽기 전까지는 구조화되지 않는다.  
- **Lazy Execution**: Hive 쿼리는 즉시 실행되지 않으며, 쿼리 실행 계획이 세워진 후에야 실행된다.  
- **MapReduce 작업 변환**: HiveQL 쿼리는 백엔드에서 **MapReduce** 작업으로 변환되어 실행된다.  

## 2. 조인(Join) 사용하기  
Hive에서도 SQL과 유사하게 테이블 간의 **조인(Join)**을 사용할 수 있다.  
조인은 두 개 이상의 테이블에서 데이터를 결합할 때 사용한다. Hive에서 지원하는 조인의 종류는 다음과 같다:  
- **INNER JOIN**: 두 테이블에서 공통된 데이터를 결합.  
- **LEFT OUTER JOIN**: 왼쪽 테이블의 모든 데이터와, 오른쪽 테이블에서 일치하는 데이터를 결합.  
- **RIGHT OUTER JOIN**: 오른쪽 테이블의 모든 데이터와, 왼쪽 테이블에서 일치하는 데이터를 결합.  
- **FULL OUTER JOIN**: 양쪽 테이블의 모든 데이터를 결합.  

### 2.1 INNER JOIN 예제  
`employee`와 `department` 테이블이 있다고 가정하자.  
두 테이블을 조인하여 직원이 속한 부서를 가져오는 쿼리를 실행해보자.  
```sql
SELECT e.name, d.dept_name
FROM employee e
JOIN department d
ON e.dept_id = d.dept_id;
```  

### 2.2 LEFT OUTER JOIN 예제  
부서가 없는 직원을 포함하여 모든 직원의 정보를 가져오려면 **LEFT OUTER JOIN**을 사용한다.  
```sql
SELECT e.name, d.dept_name
FROM employee e
LEFT OUTER JOIN department d
ON e.dept_id = d.dept_id;
```  

## 3. 서브쿼리(Subquery) 사용하기  
**서브쿼리**는 쿼리 안에 포함된 또 다른 쿼리로, 복잡한 데이터 분석에서 매우 유용하다.  
Hive에서 서브쿼리를 사용하여 데이터 필터링이나 집계 결과를 활용할 수 있다.  

### 3.1 서브쿼리 예제  
다음 예제에서는 평균 급여보다 높은 급여를 받는 직원들을 조회한다.  
```sql
SELECT name, salary
FROM employee
WHERE salary > (SELECT AVG(salary) FROM employee);
```  

## 4. 집계 함수(Aggregate Function)  
**집계 함수**는 데이터의 통계적 요약을 할 때 사용한다.  
Hive는 다양한 집계 함수를 지원하며, 대표적인 집계 함수는 다음과 같다:  
- **COUNT**: 행의 수를 반환.  
- **SUM**: 숫자 값의 합을 반환.  
- **AVG**: 숫자 값의 평균을 반환.  
- **MAX**: 최대값을 반환.  
- **MIN**: 최소값을 반환.  

### 4.1 COUNT 예제  
직원의 수를 계산하려면 `COUNT` 함수를 사용한다.  
```sql
SELECT COUNT(*) FROM employee;
```  

### 4.2 AVG 예제  
직원의 평균 급여를 계산하는 쿼리.  
```sql
SELECT AVG(salary) FROM employee;
```  

### 4.3 GROUP BY와 함께 사용하기  
집계 함수는 **GROUP BY**와 함께 사용하여 데이터를 그룹화할 수 있다.  
예를 들어, 부서별로 직원의 평균 급여를 계산할 수 있다.  
```sql
SELECT dept_id, AVG(salary)
FROM employee
GROUP BY dept_id;
```  

## 5. Hive에서 복잡한 데이터 처리  
HiveQL은 기본적인 데이터 조회뿐만 아니라 복잡한 데이터 처리 작업도 지원한다.  
이를 통해 데이터 필터링, 정렬, 페이징 등의 다양한 작업을 수행할 수 있다.  

### 5.1 데이터 필터링  
**WHERE** 절을 사용하여 특정 조건에 맞는 데이터를 필터링할 수 있다.  
예를 들어, 급여가 5000 이상인 직원들을 조회한다.  
```sql
SELECT name, salary
FROM employee
WHERE salary >= 5000;
```  

### 5.2 데이터 정렬  
**ORDER BY**를 사용하여 데이터를 정렬할 수 있다.  
예를 들어, 급여를 기준으로 오름차순으로 정렬하는 쿼리다.  
```sql
SELECT name, salary
FROM employee
ORDER BY salary ASC;
```  

### 5.3 페이징 처리  
Hive는 SQL의 **LIMIT** 절을 지원하여 결과에서 원하는 만큼의 행을 제한할 수 있다.  
```sql
SELECT name, salary
FROM employee
ORDER BY salary DESC
LIMIT 10;
```  

## 6. 복잡한 데이터 처리 예제  
HiveQL을 활용하여 대규모 데이터를 분석하는 복잡한 예제를 살펴보자.  
예를 들어, 다음 쿼리는 각 부서에서 급여가 가장 높은 직원의 정보를 반환한다.  
```sql
SELECT name, salary, dept_id
FROM employee e
WHERE salary = (SELECT MAX(salary) FROM employee WHERE dept_id = e.dept_id);
```  
이 쿼리는 서브쿼리와 집계 함수를 사용하여 부서별로 최대 급여를 받는 직원의 정보를 가져온다.  

## 7. 하둡 클러스터에서 Hive 최적화  
대규모 데이터를 처리할 때는 성능 최적화가 중요하다.  
Hive는 기본적으로 MapReduce 작업으로 변환되어 실행되므로, 쿼리 성능을 최적화하려면 몇 가지 전략을 고려해야 한다.  

### 7.1 파티셔닝(Partitioning)  
**파티셔닝**은 데이터를 특정 기준에 따라 분할하여 성능을 향상시킨다.  
파티셔닝된 테이블은 특정 파티션만 읽기 때문에 전체 데이터를 스캔하지 않아도 된다.  
```sql
CREATE TABLE employee_partitioned (
    id INT,
    name STRING,
    salary FLOAT
)
PARTITIONED BY (dept_id INT);
```  

### 7.2 버킷팅(Bucketing)  
**버킷팅**은 파티션 내에서 데이터를 더 세분화하는 방법이다.  
버킷팅은 데이터를 해시 기반으로 나누어 성능을 높인다.  
```sql
CREATE TABLE employee_bucketed (
    id INT,
    name STRING,
    salary FLOAT
)
CLUSTERED BY (id) INTO 4 BUCKETS;
```

### 7.3 데이터 압축  
Hive는 데이터를 압축하여 저장하고 처리 성능을 향상시킬 수 있다.  
Hive는 여러 가지 압축 코덱을 지원한다.  
```sql
SET hive.exec.compress.output=true;
SET mapred.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;
```

---

## 8. 마무리  
이번 강좌에서는 HiveQL의 심화 내용을 다루며, 고급 쿼리와 데이터 분석 방법을 학습했다. HiveQL은 하둡에서 대규모 데이터를 SQL과 유사한 방식으로 처리할 수 있어 매우 강력하다. 다음 강좌에서는 Hive에서 사용자 정의 함수(UDF)를 사용하여 복잡한 데이터 처리를 자동화하는 방법을 다루겠다.  