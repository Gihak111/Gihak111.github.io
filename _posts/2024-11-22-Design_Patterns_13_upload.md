---
layout: single
title:  "디자인 패턴 시리즈 13. 반복자"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 13: 반복자 패턴 (Iterator Pattern)

반복자 패턴(Iterator Pattern)은 컬렉션(Collection)의 내부 표현을 노출하지 않고도, 컬렉션의 요소들에 하나씩 접근할 수 있도록 해주는 행위(Behavioral) 패턴이다.  
이를 통해 반복 작업을 수행하는 코드를 캡슐화하여, 컬렉션 객체와 별개로 요소를 순회하는 방법을 제공한다.  

## 반복자 패턴의 필요성

컬렉션 데이터를 순회해야 하는 경우, 이를 직접 구현하면 컬렉션 구조가 변경될 때마다 관련 코드가 수정되어야 하는 문제가 발생할 수 있다.  
반복자 패턴은 이러한 문제를 해결하기 위해 다음과 같은 이점을 제공한다:

1. 내부 구조 캡슐화: 컬렉션의 내부 구현을 감추고도 요소를 순회할 수 있다.
2. 단일 책임 원칙(SRP): 컬렉션의 데이터 저장과 순회 방식을 분리하여, 각 책임을 단순화한다.
3. 호환성 유지: 컬렉션의 구현에 의존하지 않으므로, 다른 종류의 컬렉션에서도 동일한 방식으로 요소를 순회할 수 있다.

### 예시: 다양한 데이터 구조의 순회

예를 들어, 리스트(List), 집합(Set), 스택(Stack) 등 여러 형태의 데이터 구조를 처리하는 프로그램에서 반복자 패턴을 사용하면,  
컬렉션의 종류와 상관없이 동일한 방식으로 데이터 구조를 순회할 수 있다.

## 반복자 패턴의 구조

1. Iterator (반복자 인터페이스): 순회에 필요한 메서드를 정의한다.  
2. ConcreteIterator (구체적인 반복자): Iterator 인터페이스를 구현하여, 실제 데이터 구조의 요소를 순회하는 방법을 제공한다.  
3. Aggregate (컬렉션 인터페이스): 컬렉션을 나타내며, 반복자를 반환하는 메서드를 정의한다.  
4. ConcreteAggregate (구체적인 컬렉션): Aggregate 인터페이스를 구현하여, 반복자를 생성하고 데이터 구조를 관리한다.

### 구조 다이어그램

```
  Aggregate                   Iterator
      +                          +
      |                          |
ConcreteAggregate         ConcreteIterator
      +                          +
      +--------------------------+
               Collection
```

### 반복자 패턴 동작 순서

1. Aggregate는 Iterator 객체를 생성하여 반환한다.  
2. Iterator는 컬렉션 요소에 순차적으로 접근할 수 있는 메서드를 제공한다.  
3. 클라이언트는 Iterator를 사용하여 컬렉션의 내부 구현을 알지 못해도, 요소를 순회할 수 있다.  

## 반복자 패턴 예시 (Iterator Pattern)

다음 예시에서는 간단한 숫자 리스트를 순회하는 반복자 패턴을 Java로 구현해 보겠다.

### Java로 반복자 패턴 구현하기

```java
// Iterator 인터페이스
interface Iterator {
    boolean hasNext();
    int next();
}

// ConcreteIterator 클래스
class NumberIterator implements Iterator {
    private int[] numbers;
    private int position = 0;

    public NumberIterator(int[] numbers) {
        this.numbers = numbers;
    }

    @Override
    public boolean hasNext() {
        return position < numbers.length;
    }

    @Override
    public int next() {
        if (!hasNext()) {
            throw new IllegalStateException("더 이상 요소가 없습니다.");
        }
        return numbers[position++];
    }
}

// Aggregate 인터페이스
interface Collection {
    Iterator createIterator();
}

// ConcreteAggregate 클래스
class NumberCollection implements Collection {
    private int[] numbers;

    public NumberCollection(int[] numbers) {
        this.numbers = numbers;
    }

    @Override
    public Iterator createIterator() {
        return new NumberIterator(numbers);
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};
        Collection numberCollection = new NumberCollection(numbers);
        Iterator iterator = numberCollection.createIterator();

        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }
}
```

### 코드 설명

1. Iterator: 컬렉션 요소를 순회하기 위한 메서드인 `hasNext()`와 `next()`를 정의한다.  
2. NumberIterator: `Iterator`를 구현하여, 숫자 배열의 각 요소를 순회하는 기능을 제공한다.  
3. Collection: 반복자를 반환하는 `createIterator()` 메서드를 정의한다.  
4. NumberCollection: `Collection`을 구현하며, 구체적인 데이터 구조와 반복자를 관리한다.  

### 출력 결과

```
1
2
3
4
5
```

## 반복자 패턴의 장점

1. 컬렉션 구조 변경의 영향 최소화: 컬렉션의 내부 구현이 변경되어도, 클라이언트 코드는 영향을 받지 않는다.  
2. 단일 책임 원칙 준수: 데이터 저장과 순회 논리를 분리하여, 코드의 가독성과 유지보수성을 향상시킨다.  
3. 일관된 순회 인터페이스: 다양한 데이터 구조에서 동일한 순회 방식을 사용할 수 있다.  

## 반복자 패턴의 단점

1. 성능 저하 가능성: 요소 접근 방식이 간접적이므로, 직접 접근보다 성능이 떨어질 수 있다.  
2. 복잡성 증가: 단순한 데이터 구조에서는 불필요하게 복잡한 구조를 도입할 수 있다.  

### 마무리

반복자 패턴(Iterator Pattern)은 컬렉션 요소를 순회하는 코드를 캡슐화하여, 데이터 구조와 독립적인 순회 방식을 제공한다.  
이 패턴을 활용하면 데이터 구조와 순회 방식을 분리하여 유연성을 높일 수 있으며, 다양한 데이터 구조를 일관되게 처리할 수 있다.

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)
