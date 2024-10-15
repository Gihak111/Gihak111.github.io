---
layout: single
title:  "디자인 패턴 시리즈 2. 팩토리"
categories: "Design Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 2: 팩토리 패턴 (Factory Pattern)

**팩토리 패턴**은 객체 생성에 대한 책임을 별도의 클래스나 메서드로 분리하는 디자인 패턴이다. 이를 통해 객체 생성 로직이 클라이언트 코드와 분리되며, 코드의 재사용성과 유연성을 높일 수 있다. 클라이언트는 생성될 객체의 구체적인 클래스에 대한 정보를 몰라도 되고, 그 대신 **팩토리**에서 객체를 제공받는다.

## 팩토리 패턴의 필요성

객체를 생성하는 코드가 여러 곳에 흩어져 있으면, 객체 생성 방식이 변경될 때 모든 관련된 코드를 수정해야 하는 문제가 발생한다. 팩토리 패턴은 이 문제를 해결하기 위해 **객체 생성 로직을 한 곳에 집중**시키고, **유연한 객체 생성을 지원**하는 데 목적이 있다.

### 팩토리 패턴의 유형

1. **단순 팩토리(Simple Factory)**: 하나의 클래스에서 다양한 객체를 생성하는 방식으로, 가장 간단한 형태의 팩토리 패턴이다.
2. **팩토리 메서드(Factory Method)**: 객체 생성을 서브클래스에서 담당하게 하여, 서브클래스가 어떤 객체를 생성할지를 결정하게 하는 패턴이다.
3. **추상 팩토리(Abstract Factory)**: 관련 객체들의 그룹을 생성할 수 있는 인터페이스를 제공하는 패턴이다. 다양한 제품군을 묶어서 생성해야 할 때 유용하다.

## 단순 팩토리 패턴 예시 (Simple Factory)

단순 팩토리는 **한 클래스**에서 다양한 객체를 생성하고 반환하는 패턴이다. 자주 쓰이는 방식 중 하나로, 클래스 내부에서 객체 생성 로직을 관리한다.

### Java로 단순 팩토리 패턴 구현하기

아래는 자동차(Car)를 생성하는 단순 팩토리 패턴의 Java 구현이다. `Car`라는 부모 클래스를 상속받는 다양한 `Sedan`, `SUV` 등의 구체적인 차종을 생성한다.

```java
// 제품 클래스
abstract class Car {
    public abstract void drive();
}

// 구체적인 제품 클래스들
class Sedan extends Car {
    @Override
    public void drive() {
        System.out.println("세단을 운전 중입니다.");
    }
}

class SUV extends Car {
    @Override
    public void drive() {
        System.out.println("SUV를 운전 중입니다.");
    }
}

// 팩토리 클래스
class CarFactory {
    // 입력에 따라 구체적인 제품을 생성하는 팩토리 메서드
    public static Car createCar(String type) {
        if (type.equalsIgnoreCase("Sedan")) {
            return new Sedan();
        } else if (type.equalsIgnoreCase("SUV")) {
            return new SUV();
        } else {
            throw new IllegalArgumentException("알 수 없는 차종입니다.");
        }
    }
}

// 메인 클래스
public class Main {
    public static void main(String[] args) {
        // 팩토리 메서드를 사용하여 객체 생성
        Car sedan = CarFactory.createCar("Sedan");
        sedan.drive();  // "세단을 운전 중입니다." 출력

        Car suv = CarFactory.createCar("SUV");
        suv.drive();  // "SUV를 운전 중입니다." 출력
    }
}
```

### 코드 설명
- **`Car` 클래스**: 공통 인터페이스 또는 추상 클래스 역할을 하는 기본 클래스이다.
- **`Sedan`과 `SUV` 클래스**: `Car`를 상속받아 구체적인 차종을 구현한 클래스이다.
- **`CarFactory` 클래스**: `createCar()`라는 정적 메서드를 통해 클라이언트가 원하는 차종의 객체를 반환한다.
- 클라이언트는 `CarFactory`의 `createCar()` 메서드만 호출하여 객체를 생성하고, 구체적인 객체 생성 로직을 몰라도 된다.

### 출력 결과
```
세단을 운전 중입니다.
SUV를 운전 중입니다.
```

### 장점
1. **유연한 객체 생성**: 객체 생성 코드를 팩토리에 위임함으로써, 클라이언트는 객체 생성 방법에 대한 구체적인 지식이 없어도 된다.
2. **코드의 재사용성 증가**: 팩토리에서 객체를 생성하므로, 객체 생성 로직이 중복되지 않고 재사용 가능하다.
3. **확장성 용이**: 새로운 차종을 추가할 때 팩토리 메서드만 수정하면 된다. 기존 클라이언트 코드를 수정할 필요가 없다.

### 단점
1. **단일 클래스에 의존**: 모든 객체 생성 로직이 하나의 팩토리 클래스에 집중되기 때문에, 이 클래스가 복잡해질 수 있다.

## 팩토리 메서드 패턴 (Factory Method Pattern)

팩토리 메서드 패턴은 팩토리 로직을 **서브클래스에 위임**하는 패턴이다. 구체적으로 어떤 객체를 생성할지는 서브클래스가 결정하며, 객체 생성을 위한 인터페이스만 제공된다.

### Java로 팩토리 메서드 패턴 구현하기

팩토리 메서드 패턴은 클래스의 확장을 통해 더 유연한 객체 생성을 지원한다.

```java
// 제품 인터페이스
interface Product {
    void use();
}

// 구체적인 제품 클래스들
class ConcreteProductA implements Product {
    @Override
    public void use() {
        System.out.println("제품 A를 사용 중입니다.");
    }
}

class ConcreteProductB implements Product {
    @Override
    public void use() {
        System.out.println("제품 B를 사용 중입니다.");
    }
}

// 팩토리 클래스
abstract class ProductFactory {
    // 팩토리 메서드 (서브클래스에서 구현)
    public abstract Product createProduct();

    // 생성된 제품을 사용하는 로직
    public void useProduct() {
        Product product = createProduct();
        product.use();
    }
}

// 구체적인 팩토리 클래스들
class ProductFactoryA extends ProductFactory {
    @Override
    public Product createProduct() {
        return new ConcreteProductA();
    }
}

class ProductFactoryB extends ProductFactory {
    @Override
    public Product createProduct() {
        return new ConcreteProductB();
    }
}

// 메인 클래스
public class Main {
    public static void main(String[] args) {
        ProductFactory factoryA = new ProductFactoryA();
        factoryA.useProduct();  // "제품 A를 사용 중입니다." 출력

        ProductFactory factoryB = new ProductFactoryB();
        factoryB.useProduct();  // "제품 B를 사용 중입니다." 출력
    }
}
```

### 코드 설명
- **`Product` 인터페이스**: 제품 객체들이 따를 공통 인터페이스이다.
- **`ConcreteProductA`, `ConcreteProductB`**: 각각의 구체적인 제품 클래스이다.
- **`ProductFactory` 클래스**: 객체 생성 책임을 서브클래스에 위임하는 추상 팩토리 클래스이다.
- **`ProductFactoryA`, `ProductFactoryB`**: 각각 특정 제품을 생성하는 구체적인 팩토리 클래스이다.

### 출력 결과
```
제품 A를 사용 중입니다.
제품 B를 사용 중입니다.
```

### 장점
1. **확장성**: 새로운 제품을 추가하려면 서브클래스를 확장하여 새로운 팩토리를 만들면 된다.
2. **클래스 구조 명확화**: 객체 생성 책임이 명확하게 서브클래스로 분리된다.

### 단점
1. **복잡성 증가**: 팩토리와 제품 클래스의 계층 구조가 복잡해질 수 있다.

## 추상 팩토리 패턴 (Abstract Factory Pattern)

추상 팩토리 패턴은 **관련된 객체들의 그룹**을 생성하는 데 사용된다. 이 패턴은 여러 제품군을 묶어서 생성해야 할 때 유용하며, 구체적인 클래스 대신 **팩토리 인터페이스**를 통해 제품을 생성한다.

---

### 마무리

**팩토리 패턴**은 객체 생성 로직을 클라이언트 코드에서 분리하고, 다양한 객체 생성 방식을 제공하여 코드의 확장성과 유지보수성을 향상시킨다. 이번 글에서는 **단순 팩토리**, **팩토리 메서드**, 그리고 **추상 팩토리** 패턴을 소개하였다. 다음 글에서는 **추상 팩토리 패턴**에 대해 자세히 살펴보겠다.
