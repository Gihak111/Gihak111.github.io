---
layout: single
title:  "디자인 패턴 시리즈 19. 데코레이터"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 19: 데코레이터 패턴 (Decorator Pattern)

데코레이터 패턴(Decorator Pattern)은 구조적 디자인 패턴으로, 기존 객체의 기능을 확장하거나 수정할 때, 상속 대신 객체를 감싸는 방식으로 새로운 동작을 추가한다.  
이를 통해 기존 코드 변경 없이 기능을 동적으로 추가할 수 있다.  

## 데코레이터 패턴의 필요성  

소프트웨어 개발에서 클래스의 동작을 확장하려면 보통 상속을 사용하지만, 다음과 같은 문제점이 발생할 수 있다:  

1. 클래스 폭발: 각 기능 조합을 위해 수많은 하위 클래스를 생성해야 한다.  
2. 유연성 부족: 런타임 중 동적으로 기능을 추가하거나 제거하기 어렵다.  

데코레이터 패턴은 이러한 문제를 해결하기 위해 다음과 같은 이점을 제공한다:  

1. 동적 확장: 객체의 기능을 동적으로 추가하거나 제거할 수 있다.  
2. 유지보수 용이성: 기존 클래스는 수정하지 않고 확장이 가능하다.  

### 예시: 커피 주문 시스템  

커피 주문 시스템에서 기본 커피에 다양한 옵션(우유, 시럽 등)을 추가할 수 있다.  
데코레이터 패턴을 사용하면 기본 커피 클래스를 변경하지 않고 옵션을 동적으로 추가할 수 있다.  

## 데코레이터 패턴의 구조  

1. Component(구성 요소): 동적으로 기능을 추가할 객체에 대한 공통 인터페이스를 정의한다.  
2. ConcreteComponent(구체 구성 요소): 기본 동작을 구현하는 클래스.  
3. Decorator(데코레이터): Component를 확장하기 위한 추상 클래스나 인터페이스.  
4. ConcreteDecorator(구체 데코레이터): 추가 기능을 구현하는 클래스.  

### 구조 다이어그램  

```
Component
    ↑
ConcreteComponent ←──── Decorator
                            ↑
                     ConcreteDecorator
```  

### 데코레이터 패턴 동작 순서  

1. 클라이언트는 기본 객체(ConcreteComponent) 또는 데코레이터(Decorator)를 통해 작업을 요청한다.  
2. 데코레이터는 요청을 가로채고, 자신의 추가 동작을 수행한 뒤, 구성 요소에 요청을 전달한다.  

## 데코레이터 패턴 예시

이번 예시에서는 "커피와 추가 옵션"을 데코레이터 패턴으로 구현해보겠다.

### Java로 데코레이터 패턴 구현하기
 
```java
// Component 인터페이스
interface Coffee {
    String getDescription();
    double getCost();
}

// ConcreteComponent 클래스
class BasicCoffee implements Coffee {
    @Override
    public String getDescription() {
        return "기본 커피";
    }

    @Override
    public double getCost() {
        return 2.0; // 기본 커피 가격
    }
}

// Decorator 추상 클래스
abstract class CoffeeDecorator implements Coffee {
    protected Coffee decoratedCoffee;

    public CoffeeDecorator(Coffee coffee) {
        this.decoratedCoffee = coffee;
    }

    @Override
    public String getDescription() {
        return decoratedCoffee.getDescription();
    }

    @Override
    public double getCost() {
        return decoratedCoffee.getCost();
    }
}

// ConcreteDecorator 클래스
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }

    @Override
    public String getDescription() {
        return super.getDescription() + ", 우유";
    }

    @Override
    public double getCost() {
        return super.getCost() + 0.5; // 우유 추가 가격
    }
}

class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }

    @Override
    public String getDescription() {
        return super.getDescription() + ", 설탕";
    }

    @Override
    public double getCost() {
        return super.getCost() + 0.2; // 설탕 추가 가격
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        Coffee coffee = new BasicCoffee();
        System.out.println(coffee.getDescription() + " 가격: $" + coffee.getCost());

        // 우유 추가
        coffee = new MilkDecorator(coffee);
        System.out.println(coffee.getDescription() + " 가격: $" + coffee.getCost());

        // 설탕 추가
        coffee = new SugarDecorator(coffee);
        System.out.println(coffee.getDescription() + " 가격: $" + coffee.getCost());
    }
}
```

### 코드 설명

1. Coffee (Component): 커피 객체의 공통 인터페이스를 정의한다.  
2. BasicCoffee (ConcreteComponent): 기본 커피 클래스로, 기본 기능을 구현한다.  
3. CoffeeDecorator (Decorator): 추가 기능을 구현할 수 있도록 Coffee 인터페이스를 확장한다.  
4. MilkDecorator, SugarDecorator (ConcreteDecorator): 우유와 설탕을 추가하는 동작을 구현한다.  
5. Main (Client): Coffee 객체를 생성하고, 동적으로 데코레이터를 추가한다.  

### 출력 결과  

```
기본 커피 가격: $2.0
기본 커피, 우유 가격: $2.5
기본 커피, 우유, 설탕 가격: $2.7
```  

## 데코레이터 패턴의 장점  

1. 동적 기능 확장: 객체의 기능을 런타임 중 동적으로 추가할 수 있다.  
2. 단일 책임 원칙 준수: 추가 기능을 별도 클래스로 분리하여 코드 재사용성을 높인다.  
3. 기존 클래스 수정 불필요: 기존 코드 변경 없이 기능 확장이 가능하다.  

## 데코레이터 패턴의 단점  

1. 객체 생성의 복잡성 증가: 많은 데코레이터가 중첩되면 복잡성이 커질 수 있다.  
2. 디버깅 어려움: 중첩된 데코레이터로 인해 디버깅이 어려울 수 있다.  

### 마무리

데코레이터 패턴(Decorator Pattern)은 상속보다 유연한 대안으로, 객체의 동작을 확장하고 수정할 때 매우 유용하다.  
특히, 기능 조합이 다양한 시스템에서 코드를 단순화하고 유지보수를 용이하게 만든다.nn

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  