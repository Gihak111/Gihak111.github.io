---
layout: single
title:  "하둡 강좌 8편 Hive UDF"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 8편: **Hive 사용자 정의 함수(UDF) 활용**  
이번 강좌에서는 **사용자 정의 함수=**를 활용하여 **Hive**에서 복잡한 데이터 처리 작업을 자동화하는 방법을 배워보겠다.  
Hive는 기본적인 SQL 함수 외에도 자신만의 함수를 정의하여 복잡한 비즈니스 로직이나 데이터 변환 작업을 처리할 수 있다.  

## 1. Hive UDF 개념  
Hive에서 제공하는 기본 함수는 한정적일 수 있으므로, 사용자가 직접 정의한 함수를 사용할 수 있는 기능을 제공한다.  
UDF는 단일 값 변환을 위해 사용되며, UDF 외에도 **UDAF**(사용자 정의 집계 함수), **UDTF**(사용자 정의 테이블 생성 함수) 등도 있다.  

### 1.1 UDF의 종류  
- **UDF**: 입력값 하나에 대해 하나의 값을 반환.  
- **UDAF**: 집계 함수로 다수의 행을 입력받아 하나의 결과를 반환.  
- **UDTF**: 하나의 입력값을 여러 개의 행으로 변환.  

## 2. 간단한 UDF 작성  
Hive에서 UDF를 작성하기 위해서는 자바 코드를 작성하고 이를 Hive에 등록하는 과정이 필요하다.  
간단한 UDF를 작성하는 예제를 통해 이를 설명하겠다.  

### 2.1 Maven 프로젝트 생성  
먼저, 자바 기반으로 UDF를 작성할 Maven 프로젝트를 생성하자.  
아래는 기본적인 `pom.xml` 파일 구조이다.  
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>hive-udf-example</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
        <!-- Hive 의존성 추가 -->
        <dependency>
            <groupId>org.apache.hive</groupId>
            <artifactId>hive-exec</artifactId>
            <version>3.1.2</version>
        </dependency>
    </dependencies>
</project>
```  

### 2.2 간단한 UDF 작성  
이제 간단한 문자열을 처리하는 UDF를 작성해보자.  
여기서는 문자열을 대문자로 변환하는 UDF를 작성해보겠다.  
```java
package com.example.hive.udf;

import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.io.Text;

public class UppercaseUDF extends UDF {

    // 입력값을 대문자로 변환하는 UDF 함수
    public Text evaluate(Text input) {
        if (input == null) {
            return null;
        }
        return new Text(input.toString().toUpperCase());
    }
}
```  

### 2.3 UDF 컴파일 및 Jar 파일 생성  
위 코드를 작성한 후 프로젝트를 컴파일하고 Jar 파일을 생성하자.  
Maven을 이용해 다음 명령어로 Jar 파일을 생성할 수 있다.  
```bash
mvn clean package
```  

## 3. Hive에서 UDF 등록 및 사용  
이제 생성한 UDF를 Hive에 등록하고 사용하는 방법을 살펴보자.  

### 3.1 UDF 등록  
Hive에 사용자 정의 함수를 등록하려면 먼저 생성한 Jar 파일을 Hive에서 사용할 수 있도록 경로를 설정하고 등록해야 한다.  
```sql
ADD JAR /path/to/hive-udf-example-1.0-SNAPSHOT.jar;
CREATE TEMPORARY FUNCTION to_uppercase AS 'com.example.hive.udf.UppercaseUDF';
```  
위의 명령어에서 **`to_uppercase`**는 Hive에서 사용할 함수 이름이고, **`com.example.hive.udf.UppercaseUDF`**는 UDF 클래스의 경로이다.  

### 3.2 UDF 사용  
이제 Hive 쿼리에서 직접 이 UDF를 사용할 수 있다.  
```sql
SELECT to_uppercase(name) FROM employee;
```  
위 쿼리는 **employee** 테이블의 **name** 열을 대문자로 변환하여 반환한다.  

## 4. 복잡한 UDF 작성  
이제 조금 더 복잡한 UDF를 작성해보자.  
예를 들어, 문자열이 주어졌을 때 문자열의 길이를 계산하는 UDF를 작성해보겠다.  
```java
package com.example.hive.udf;

import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

public class StringLengthUDF extends UDF {

    // 입력 문자열의 길이를 반환하는 UDF 함수
    public IntWritable evaluate(Text input) {
        if (input == null) {
            return new IntWritable(0);
        }
        return new IntWritable(input.toString().length());
    }
}
```  
위와 같은 방법으로 이 UDF를 Hive에 등록하고 사용할 수 있다.  
```sql
CREATE TEMPORARY FUNCTION string_length AS 'com.example.hive.udf.StringLengthUDF';

SELECT name, string_length(name) FROM employee;
```  

## 5. UDAF (사용자 정의 집계 함수)  
Hive에서 집계 함수는 여러 행을 입력받아 하나의 결과를 반환하는 작업을 수행한다.  
사용자 정의 집계 함수를 작성하는 방법을 살펴보자.  

### 5.1 UDAF 구조  
집계 함수는 기본 UDF보다 복잡한 구조를 가진다.  
각 단계에서 데이터를 수집하고 최종 결과를 반환하는 방식으로 동작한다.  
```java
package com.example.hive.udaf;

import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;

public class SumUDAF extends UDAF {

    public static class SumEvaluator implements UDAFEvaluator {

        private int sum;

        // 초기화
        public void init() {
            sum = 0;
        }

        // 각 값 처리
        public boolean iterate(IntWritable value) {
            if (value != null) {
                sum += value.get();
            }
            return true;
        }

        // 부분 결과 반환
        public IntWritable terminatePartial() {
            return new IntWritable(sum);
        }

        // 부분 결과 병합
        public boolean merge(IntWritable other) {
            if (other != null) {
                sum += other.get();
            }
            return true;
        }

        // 최종 결과 반환
        public IntWritable terminate() {
            return new IntWritable(sum);
        }
    }
}
```  

### 5.2 UDAF 등록 및 사용  
집계 함수를 Hive에 등록하고 사용하는 방법도 UDF와 유사하다.  
```sql
ADD JAR /path/to/hive-udf-example-1.0-SNAPSHOT.jar;
CREATE TEMPORARY FUNCTION custom_sum AS 'com.example.hive.udaf.SumUDAF';

SELECT custom_sum(salary) FROM employee;
```  
이 UDAF는 **salary** 열의 값을 모두 더한 값을 반환한다.  

## 6. 사용자 정의 테이블 생성 함수(UDTF)  
**UDTF**는 하나의 입력을 여러 개의 행으로 변환할 때 사용된다.  
UDTF는 테이블 형식의 데이터를 반환하므로, 복잡한 데이터 변환 작업에 적합하다.  

### 6.1 간단한 UDTF 예제  
예를 들어, 문자열을 공백으로 나누어 여러 행으로 반환하는 UDTF를 작성할 수 있다.  
```java
package com.example.hive.udtf;

import org.apache.hadoop.hive.ql.exec.UDTF;
import org.apache.hadoop.io.Text;

public class SplitUDTF extends UDTF {

    public void process(Text[] input) {
        if (input == null || input.length == 0) {
            return;
        }
        String[] parts = input[0].toString().split(" ");
        for (String part : parts) {
            forward(new Object[]{new Text(part)});
        }
    }

    public void close() {}
}
```  
이 함수를 Hive에 등록하고 사용해보자.  
```sql
CREATE TEMPORARY FUNCTION split_string AS 'com.example.hive.udtf.SplitUDTF';

SELECT split_string(name) FROM employee;
```  

---

## 7. 마무리  
이번 강좌에서는 **Hive UDF**를 활용한 복잡한 데이터 처리 방법을 학습했다. 기본적인 UDF부터 집계 함수(UDAF), 테이블 함수(UDTF)까지 다양한 사용자 정의 함수를 작성하고 활용하는 방법을 배웠다. Hive는 이러한 사용자 정의 함수들을 통해 더욱 유연하고 강력한 데이터 처리 환경을 제공한다. 다음 강좌에서는 하둡에서 데이터 파티셔닝과 분산 파일 시스템을 최적화하는 방법을 다루겠다.  