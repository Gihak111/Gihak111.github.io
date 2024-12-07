---
layout: single
title:  "아키텍처 패턴 시리즈 3. 마스터-슬레이브 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 3: 마스터-슬레이브 패턴 (Master-Slave Pattern)

마스터-슬레이브 패턴(Master-Slave Pattern)은 분산 시스템 아키텍처에서 자주 사용되는 패턴으로, '마스터'와 '슬레이브'의 역할을 분리하여 병렬 처리를 통해 작업을 효율적으로 분담하는 구조이다.  
주로 데이터 일관성을 유지하면서 작업을 분산할 때, 시스템 성능과 확장성을 높이기 위해 사용된다.

## 마스터-슬레이브 패턴의 필요성

다수의 데이터 연산을 병렬로 처리하거나, 작업의 일관성과 신뢰성을 유지하면서 분산 처리가 필요할 때 마스터-슬레이브 패턴은 다음과 같은 이점을 제공한다:

1. 작업 분산: 작업을 여러 슬레이브에게 나누어 주어 병렬 처리 성능을 높인다.  
2. 데이터 일관성: 마스터가 데이터를 관리하고 동기화하여 데이터의 일관성을 보장한다.  
3. 확장성: 슬레이브를 추가하여 성능을 쉽게 확장할 수 있다.  

이 패턴은 대규모 분산 시스템에서 효율적이고 안정적인 작업 수행을 가능하게 한다.

### 예시: 데이터베이스 복제 시스템

데이터베이스 복제 시스템에서 마스터는 데이터를 읽고 쓸 수 있는 권한을 가지며, 슬레이브는 읽기 전용으로 마스터의 데이터를 복제하여 보관한다.  

## 마스터-슬레이브 패턴의 구조  

1. Master (마스터): 전체 작업을 조정하고 슬레이브에게 작업을 분배하며, 결과를 모아 최종적으로 처리한다.
2. Slave (슬레이브): 마스터가 분배한 작업을 수행하며, 처리된 결과를 마스터에게 전달한다.
3. Network (네트워크): 마스터와 슬레이브 간의 통신을 담당하며, 데이터와 명령이 전달된다.

### 구조 다이어그램

```
         ┌────────┐      ┌──────────┐
         │ Master │ ---> │  Slave 1 │
         └────────┘      └──────────┘
               │              │
               │ ---> ┌──────────┐
               │      │  Slave 2 │
               │      └──────────┘
               │              │
               │ ---> ┌──────────┐
               │      │  Slave n │
               │      └──────────┘
               ▼
```

### 마스터-슬레이브 패턴 동작 순서

1. 마스터는 전체 작업을 수집하고, 각 슬레이브에 세부 작업을 분배한다.
2. 슬레이브는 할당된 작업을 수행하고, 처리 결과를 마스터에게 전달한다.
3. 마스터는 모든 슬레이브로부터 받은 결과를 모아 최종 처리를 수행한다.

## 마스터-슬레이브 패턴 예시

이번 예시에서는 마스터가 숫자 배열을 슬레이브들에게 분배하고, 슬레이브가 각각 부분 합을 계산하여 마스터에 결과를 반환하는 방식으로 구현한다.

### Java로 마스터-슬레이브 패턴 구현하기

```java
// Master 클래스: 슬레이브들에게 작업을 분배하고 결과를 수집
import java.util.concurrent.*;

public class Master {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(3);
        int[] numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        Future<Integer> slave1Result = executor.submit(new Slave(numbers, 0, 4));
        Future<Integer> slave2Result = executor.submit(new Slave(numbers, 5, 9));

        int total = slave1Result.get() + slave2Result.get();
        System.out.println("전체 합계: " + total);

        executor.shutdown();
    }
}
```

```java
// Slave 클래스: 할당된 부분 합을 계산하여 마스터에게 반환
import java.util.concurrent.Callable;

public class Slave implements Callable<Integer> {
    private int[] numbers;
    private int start, end;

    public Slave(int[] numbers, int start, int end) {
        this.numbers = numbers;
        this.start = start;
        this.end = end;
    }

    @Override
    public Integer call() {
        int sum = 0;
        for (int i = start; i <= end; i++) {
            sum += numbers[i];
        }
        System.out.println("부분 합계 (" + start + " ~ " + end + "): " + sum);
        return sum;
    }
}
```

### 코드 설명

1. Master (마스터): 마스터는 숫자 배열을 슬레이브들에게 할당하고, 결과를 수집하여 최종 합계를 계산한다.
2. Slave (슬레이브): 각 슬레이브는 할당된 숫자 배열의 부분 합을 계산하여 마스터에 반환한다.
3. Network (네트워크): 마스터와 슬레이브는 자바 스레드 풀을 통해 병렬로 통신 및 연산을 수행한다.

### 출력 결과

```
부분 합계 (0 ~ 4): 15
부분 합계 (5 ~ 9): 40
전체 합계: 55
```

### 마스터-슬레이브 패턴 활용

1. 데이터베이스 복제: 마스터가 데이터를 관리하고, 여러 슬레이브 서버가 이를 복제하여 읽기 작업을 수행한다.  
2. 병렬 처리 시스템: 마스터가 작업을 나누고, 슬레이브들이 이를 병렬로 처리하여 처리 속도를 높인다.  
3. 디지털 신호 처리: 마스터가 신호를 분석하고, 슬레이브들이 세부 작업을 수행한다.  

## 마스터-슬레이브 패턴의 장점

1. 병렬 처리 효율성: 작업을 슬레이브에 분배하여 병렬로 처리할 수 있어 성능이 개선된다.
2. 데이터 일관성: 마스터가 데이터 동기화를 담당하므로 데이터의 일관성을 유지할 수 있다.
3. 확장성: 슬레이브를 추가하여 성능을 확장하기 쉬우며, 시스템이 유연하게 확장 가능하다.  

## 마스터-슬레이브 패턴의 단점

1. 마스터 부하: 마스터에 작업 분배와 결과 수집이 집중되어 성능이 저하될 수 있다.
2. 단일 실패 지점: 마스터가 실패하면 전체 시스템이 작동을 멈출 수 있어 높은 안정성이 요구된다.
3. 복잡성: 슬레이브 간 동기화와 마스터의 분배 로직을 관리해야 하므로 시스템이 복잡해질 수 있다.  

### 마무리

마스터-슬레이브 패턴(Master-Slave Pattern)은 데이터 일관성을 유지하면서 분산 처리를 통해 효율적인 작업 처리를 지원하는 아키텍처이다.  
특히 대규모 데이터 처리 및 복제 시스템에서 매우 유용하다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  


