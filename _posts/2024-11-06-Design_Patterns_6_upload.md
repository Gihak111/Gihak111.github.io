---
layout: single
title:  "디자인 패턴 시리즈 5. 프로토타입 패턴"
categories: "Design Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 5: 프로토타입 (Prototype Pattern)  
**프로토타입 패턴**은 **객체의 복사본**을 쉽게 생성할 수 있도록 하는 패턴이다.  
이 패턴은 기존 객체를 복제하여 새로운 객체를 만들기 때문에, 객체를 직접 생성하는 비용이 큰 경우 유용하게 사용할 수 있다.  

## 프로토타입 패턴의 작동 방식  
프로토타입 패턴은 객체의 필드 값을 복제하여 새로운 객체를 만든다.  
이때, 객체에 private 필드가 있더라도 해당 필드를 포함한 전체 상태를 복사할 수 있다.  
이 복제는 직접적인 `new` 연산자를 통해 객체를 생성하는 것과는 다르다.  
새 객체의 모든 필드를 원본 객체에서 복사해 오기 때문에, 생성 시마다 고정된 초기 상태를 가질 수 있다.  

그러나 일부 필드가 private로 선언되어 있거나, 다른 객체와의 강한 의존 관계가 있는 경우 복제가 제한될 수 있다.  
이러한 문제를 해결하기 위해 공통 **인터페이스**를 도입한다.  
이 인터페이스를 통해 객체가 복제될 수 있도록 하여 객체 간 결합을 느슨하게 만들어준다.  
즉, 원본 객체를 인터페이스로 복제하여 인터페이스의 메서드를 통해 필드 접근 문제를 해결하는 방식이다.  

## 프로토타입 패턴의 필요성  
시스템 내에서 동일한 형태의 객체가 반복적으로 사용되어야 할 경우, 또는 생성 비용이 높은 객체를 효율적으로 관리하고자 할 때 프로토타입 패턴은 매우 유용하다.  
이 패턴을 사용하면 기본 상태가 동일한 객체를 여러 개 생성할 수 있으면서, 각 객체가 서로 독립적으로 유지된다.  

## 프로토타입 패턴의 구성 요소  
프로토타입 패턴은 다음과 같은 구성 요소로 이루어진다:  
1. **프로토타입 인터페이스(Prototype)**: 복제 기능을 정의하는 인터페이스.  
2. **구체적인 프로토타입(Concrete Prototype)**: 복제할 객체의 원형을 제공하는 클래스.  
3. **클라이언트(Client)**: 복제된 객체를 사용하는 역할을 하는 부분.  

### 구조 다이어그램  
```
Client
   └─ Prototype
        ├─ ConcretePrototypeA
        ├─ ConcretePrototypeB
```

## 프로토타입 패턴 예시  
이번 예시는 **서적(Book)**을 복제할 수 있는 도서 관리 프로그램을 예로 들어 보겠다.  
각 서적에는 고유한 정보가 있으며, 서적을 복제함으로써 같은 형태의 다른 서적을 여러 개 생성할 수 있다. 

### Java로 프로토타입 패턴 구현하기  
```java
// 프로토타입 인터페이스
interface Prototype {
    Prototype clone();
}

// 구체적인 프로토타입: 서적
class Book implements Prototype {
    private String title;
    private String author;

    public Book(String title, String author) {
        this.title = title;
        this.author = author;
    }

    // 복제 메서드
    @Override
    public Prototype clone() {
        return new Book(this.title, this.author);
    }

    public void displayInfo() {
        System.out.println("제목: " + title + ", 저자: " + author);
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        Book originalBook = new Book("디자인 패턴", "에릭 감마");
        originalBook.displayInfo();

        // 서적 복제
        Book clonedBook = (Book) originalBook.clone();
        clonedBook.displayInfo();
    }
}
```

### 코드 설명  
1. **프로토타입 인터페이스(Prototype)**: `Prototype` 인터페이스는 `clone()` 메서드를 정의하여 객체 복제 기능을 제공한다.  
2. **구체적인 프로토타입(Concrete Prototype)**: `Book` 클래스는 `Prototype` 인터페이스를 구현하며, `clone()` 메서드를 통해 자신을 복제한다. 복제된 객체는 `title`과 `author` 필드를 그대로 복사한다.  
3. **클라이언트(Client)**: `Main` 클래스에서는 원본 서적을 생성한 후 이를 복제하여 동일한 정보를 가진 새로운 서적 객체를 만든다.  

### 출력 결과  
```
제목: 디자인 패턴, 저자: 에릭 감마
제목: 디자인 패턴, 저자: 에릭 감마
```  

## 프로토타입 패턴의 장점  
1. **객체 생성 비용 절감**: 복제를 통해 기존 객체를 재사용하므로, 객체 생성 비용을 줄일 수 있다.  
2. **유연성**: 기존 객체의 구조를 그대로 유지한 상태에서 새로운 객체를 만들 수 있으므로, 다양한 형태의 객체 생성을 간편하게 처리할 수 있다. 
3. **복잡한 구조 관리**: 복제된 객체는 원본과 독립적이므로, 서로 다른 형태의 객체를 관리할 때 유리하다.  

## 프로토타입 패턴의 단점  
1. **객체 간 결합도 증가**: 잘못 구현할 경우, 원본 객체에 대한 종속성이 높아질 수 있다.  
2. **클론 메서드의 구현 어려움**: 복제된 객체의 상태가 다를 수 있어, 복잡한 객체 구조에서는 `clone()` 메서드를 구현하기 까다롭다.  

---

### 마무리  
**프로토타입 패턴(Prototype Pattern)**은 시스템 내에서 객체의 복제를 통해 효율성을 극대화하고자 할 때 유용한 패턴이다.  
특히 생성 비용이 큰 객체나 상태가 복잡한 객체의 복제를 손쉽게 관리할 수 있어, 다양한 상황에서 활용된다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design/patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  