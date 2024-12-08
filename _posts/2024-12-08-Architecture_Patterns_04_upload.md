---
layout: single
title:  "아키텍처 패턴 시리즈 4. 파이프-필터 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 4: 파이프-필터 패턴 (Pipe-Filter Pattern)

파이프-필터 패턴(Pipe-Filter Pattern)은 데이터를 단계별로 처리할 수 있도록 각 작업을 필터로 나누고, 이들 필터 간 데이터를 전달하기 위해 파이프를 사용하는 구조이다.  
데이터의 순차적 처리가 필요한 시스템에서 주로 사용되며, 필터 단위로 작업을 나누어 독립적으로 처리할 수 있는 장점이 있다.

## 파이프-필터 패턴의 필요성

파이프-필터 패턴은 복잡한 데이터 처리를 단계별로 나누어 처리하고, 각 단계의 결과를 순차적으로 다음 단계로 넘길 때 유용하다.  

1. 단계별 처리: 데이터를 여러 필터를 거쳐 순차적으로 처리할 수 있다.  
2. 유연한 설계: 각 필터가 독립적으로 동작하므로, 필터를 교체하거나 추가하기 쉽다.  
3. 병렬 처리 가능: 필터가 독립적이므로, 병렬 처리를 통해 성능을 향상시킬 수 있다.  

파이프-필터 패턴은 특히 데이터 스트림, 이미지 프로세싱, 데이터 변환 등 다양한 분야에서 사용된다.

### 예시: 이미지 처리 파이프라인

이미지 처리 시스템에서 필터는 밝기 조정, 색상 보정, 필터 적용 등의 작업을 수행하며, 파이프를 통해 데이터를 전달한다.

## 파이프-필터 패턴의 구조  

1. Filter (필터): 데이터의 특정 작업을 수행하는 개별 단위이다.
2. Pipe (파이프): 필터 간의 데이터 전달을 담당하며, 필터의 출력을 다음 필터로 전달한다.
3. Data Stream (데이터 스트림): 필터를 거쳐 순차적으로 처리되는 데이터 흐름이다.  

### 구조 다이어그램

```
  Input Data ---> [ Filter 1 ] ---> [ Filter 2 ] ---> [ Filter n ] ---> Output Data
```

### 파이프-필터 패턴 동작 순서

1. 입력 데이터가 첫 번째 필터로 전달된다.
2. 각 필터는 특정 작업을 수행하고, 결과를 다음 필터로 넘긴다.
3. 모든 필터를 거친 데이터가 최종 출력으로 반환된다.  

## 파이프-필터 패턴 예시

이번 예시에서는 문자열 데이터를 필터링하는 파이프-필터 패턴을 구현해보자. 각 필터는 문자열을 변환하는 역할을 하며, 파이프를 통해 순차적으로 데이터를 전달한다.  

### Java로 파이프-필터 패턴 구현하기

```java
// Pipeline 클래스: 필터를 연결하고 데이터 스트림을 처리
import java.util.ArrayList;
import java.util.List;

public class Pipeline {
    private List<Filter> filters = new ArrayList<>();

    public void addFilter(Filter filter) {
        filters.add(filter);
    }

    public String execute(String input) {
        String result = input;
        for (Filter filter : filters) {
            result = filter.process(result);
        }
        return result;
    }
}
```

```java
// Filter 인터페이스: 필터가 구현해야 하는 프로세스 메서드를 정의
public interface Filter {
    String process(String input);
}
```

```java
// UppercaseFilter 클래스: 문자열을 대문자로 변환
public class UppercaseFilter implements Filter {
    @Override
    public String process(String input) {
        return input.toUpperCase();
    }
}
```

```java
// ExclamationFilter 클래스: 문자열 끝에 느낌표를 추가
public class ExclamationFilter implements Filter {
    @Override
    public String process(String input) {
        return input + "!";
    }
}
```

```java
// Main 클래스: 파이프라인을 구성하고 필터를 실행
public class Main {
    public static void main(String[] args) {
        Pipeline pipeline = new Pipeline();
        pipeline.addFilter(new UppercaseFilter());
        pipeline.addFilter(new ExclamationFilter());

        String input = "hello";
        String output = pipeline.execute(input);

        System.out.println("최종 결과: " + output); // 출력: "HELLO!"
    }
}
```

### 코드 설명

1. Pipeline: `Pipeline` 클래스는 여러 필터를 추가하고 순차적으로 실행할 수 있는 기능을 제공한다.
2. Filter: `Filter` 인터페이스는 각 필터가 구현할 `process` 메서드를 정의한다.
3. UppercaseFilter & ExclamationFilter: 각각의 필터는 문자열을 대문자로 변환하거나 느낌표를 추가하는 작업을 수행한다.
4. Main: `Pipeline`에 필터들을 추가하고, `execute` 메서드를 통해 입력 문자열을 변환하여 출력한다.  

### 출력 결과

```
최종 결과: HELLO!
```

### 파이프-필터 패턴 활용

1. 데이터 스트림 처리: 데이터 스트림을 필터를 통해 순차적으로 처리하여 다양한 변환을 수행할 수 있다.  
2. 이미지 처리: 이미지 처리 파이프라인에서 여러 필터를 통해 이미지를 조정하고 처리한다.  
3. 파일 처리: 로그 파일이나 데이터를 필터링하여 가공하고 분석할 때 활용된다.  

## 파이프-필터 패턴의 장점

1. 유연성: 필터를 추가하거나 교체하기 쉬워 다양한 요구 사항에 유연하게 대응할 수 있다.
2. 모듈성: 각 필터가 독립적이므로, 코드의 재사용성과 유지보수성이 높아진다.
3. 병렬 처리 가능성: 필터가 독립적이기 때문에 병렬 처리를 통해 성능을 향상시킬 수 있다.  

## 파이프-필터 패턴의 단점

1. 성능 오버헤드: 필터 간 데이터 전달 시 오버헤드가 발생할 수 있다.
2. 에러 처리 복잡성: 각 필터가 독립적으로 동작하므로 에러 처리와 데이터 검증이 복잡해질 수 있다.
3. 데이터 포맷 요구사항: 필터 간 데이터 포맷이 맞지 않으면 문제가 발생할 수 있으므로, 데이터 포맷 변환이 필요하다.  

### 마무리

파이프-필터 패턴(Pipe-Filter Pattern)은 데이터를 단계별로 처리할 수 있어 복잡한 데이터 처리나 변환 작업에 유용하다.  
필터를 독립적으로 동작하게 하여 유지보수성과 유연성을 높일 수 있다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
