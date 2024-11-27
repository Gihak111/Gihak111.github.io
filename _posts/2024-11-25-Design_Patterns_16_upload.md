---
layout: single
title:  "디자인 패턴 시리즈 16. 비지터"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 16: 비지터 패턴 (Visitor Pattern)

비지터 패턴(Visitor Pattern)은 객체 구조를 변경하지 않고 새로운 동작(기능)을 추가할 수 있도록 설계된 행동 패턴이다.  
이 패턴은 요소(Element)와 방문자(Visitor)의 협력을 통해 복잡한 연산을 객체 외부에서 처리할 수 있도록 한다.  

## 비지터 패턴의 필요성

어떤 객체 구조가 고정되어 있고, 그 위에 다양한 연산을 반복적으로 수행해야 하는 경우를 생각해보자.  
예를 들어, 파일 시스템 내 파일과 폴더 구조를 탐색하며 크기 계산, 백업, 삭제 등의 작업을 해야 하는 경우다.  

비지터 패턴을 사용하면:

1. 새로운 기능 추가가 용이: 기존 클래스 구조를 변경하지 않고 새로운 동작을 추가할 수 있다.  
2. 객체-연산 분리: 객체의 데이터와 연산을 분리하여 각자의 책임을 명확히 한다.  

## 비지터 패턴의 구조

1. Visitor (방문자): 각 요소에 적용될 연산을 정의.  
2. ConcreteVisitor (구체적 방문자): 각 요소에 대해 실제로 수행될 연산 구현.  
3. Element (요소): 방문자를 받아들이는 인터페이스를 제공.  
4. ConcreteElement (구체적 요소): 데이터를 가지고 있고 방문자를 수용하여 연산을 실행.  

### 구조 다이어그램

```
  +-----------------+       +-------------------+
  |     Visitor     |<------|    Element        |
  +-----------------+       +-------------------+
  | visitA(ElementA)|       | accept(Visitor)   |
  | visitB(ElementB)|-----> +-------------------+
  +-----------------+       |ConcreteElementA/B |
        ^                     (구체적 데이터)  
        |
  +-----------------+
  | ConcreteVisitor |
  +-----------------+
```  

## 비지터 패턴 예시

### 상황: 쇼핑 카트 시스템
사용자는 여러 상품(Item)을 담을 수 있는 쇼핑 카트를 사용한다. 각 상품의 총합 계산과 같은 연산은 쇼핑 카트에서 이루어진다.  
새로운 요구사항으로 "배송 비용 계산"과 같은 기능이 추가될 경우, 비지터 패턴을 활용하여 유연하게 대응할 수 있다.  


### Java로 비지터 패턴 구현  

```java
// Visitor 인터페이스
interface Visitor {
    void visit(Book book);
    void visit(Electronic electronic);
}

// Element 인터페이스
interface Item {
    void accept(Visitor visitor);
    double getPrice();
}

// ConcreteElement 1: Book
class Book implements Item {
    private double price;

    public Book(double price) {
        this.price = price;
    }

    @Override
    public double getPrice() {
        return price;
    }

    @Override
    public void accept(Visitor visitor) {
        visitor.visit(this);
    }
}

// ConcreteElement 2: Electronic
class Electronic implements Item {
    private double price;

    public Electronic(double price) {
        this.price = price;
    }

    @Override
    public double getPrice() {
        return price;
    }

    @Override
    public void accept(Visitor visitor) {
        visitor.visit(this);
    }
}

// ConcreteVisitor: PriceCalculator
class PriceCalculator implements Visitor {
    private double totalPrice = 0.0;

    @Override
    public void visit(Book book) {
        totalPrice += book.getPrice();
    }

    @Override
    public void visit(Electronic electronic) {
        totalPrice += electronic.getPrice();
    }

    public double getTotalPrice() {
        return totalPrice;
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        List<Item> shoppingCart = List.of(
            new Book(12.99),
            new Electronic(299.99),
            new Book(5.99)
        );

        PriceCalculator calculator = new PriceCalculator();

        for (Item item : shoppingCart) {
            item.accept(calculator); // 방문자에 위임
        }

        System.out.println("총합: $" + calculator.getTotalPrice());
    }
}
```  


### 출력 결과

```
총합: $318.97
```  


## 비지터 패턴의 장점

1. 새로운 연산 추가 용이: 기존 클래스(Element)를 수정하지 않고 Visitor에 새 연산을 추가 가능.  
2. 캡슐화 유지: 연산을 Element 외부(Visitor)에서 정의하여 클래스의 복잡성을 줄인다.  
3. 다양한 응용: 동일한 객체 구조에 대해 여러 연산을 독립적으로 정의할 수 있다.  


## 비지터 패턴의 단점

1. 객체 추가 어려움: 새로운 Element(요소)를 추가하려면 모든 Visitor를 수정해야 한다.  
2. 복잡성 증가: Element와 Visitor 간 상호작용 설계가 복잡해질 수 있다.  


## 마무리

비지터 패턴은 객체의 내부 구조를 수정하지 않고도 새로운 연산을 추가해야 하는 경우 매우 유용하다.  
다만, 요소(Element)의 변경이 자주 일어나는 경우에는 적합하지 않을 수 있으므로 사용 시 신중하게 고려해야 한다.  

다음 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  
