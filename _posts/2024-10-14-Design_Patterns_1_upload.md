---
layout: single
title:  "디자인 패턴 시리즈 1. 싱글턴"
categories: "Design Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 1: 싱글턴 패턴 (Singleton Pattern)

**디자인 패턴**은 소프트웨어 설계에서 자주 발생하는 문제를 해결하기 위한 **재사용 가능한 솔루션**이다. 각각의 패턴은 특정 문제 상황에서 최적화된 구조를 제공하여 유지보수성과 확장성을 높이는 데 기여한다. 이번 글에서는 가장 널리 알려진 패턴 중 하나인 **싱글턴 패턴(Singleton Pattern)**을 다룬다.

## 싱글턴 패턴이란?

**싱글턴 패턴**은 클래스의 인스턴스를 **오직 하나만** 생성하도록 제한하는 디자인 패턴이다. 이는 애플리케이션 전역에서 동일한 인스턴스를 공유해야 할 때 유용하다. 예를 들어, 데이터베이스 연결 객체나 설정 파일 관리 객체는 하나만 존재해야 하는 경우가 많다. 이러한 상황에서 싱글턴 패턴을 사용하면 **중복 인스턴스 생성**을 방지할 수 있다.

### 주요 특징
1. **오직 하나의 인스턴스**: 클래스의 인스턴스는 프로그램 내에서 단 하나만 존재한다.
2. **전역 접근**: 싱글턴 인스턴스는 어디서나 접근 가능하다.
3. **지연 초기화**: 필요할 때까지 인스턴스를 생성하지 않는다. 이를 **Lazy Initialization(지연 초기화)**라고 한다.

## 왜 싱글턴 패턴을 사용하는가?

싱글턴 패턴은 **전역 상태**를 관리할 필요가 있을 때 적합하다. 인스턴스가 하나만 있어야 하는 상황을 고려해보자:
- **데이터베이스 연결 관리**: 여러 개의 데이터베이스 연결 인스턴스를 만드는 것은 비효율적일 수 있다.
- **설정 관리 객체**: 애플리케이션 전반에서 동일한 설정 파일을 공유해야 할 때 인스턴스가 여러 개일 경우 문제가 발생할 수 있다.
- **로그 파일 관리**: 로그 기록을 여러 곳에서 남기지만, 모든 로그가 하나의 파일에 저장되도록 해야 할 때 싱글턴 패턴을 사용하면 편리하다.

## Java에서 싱글턴 패턴 구현하기

Java에서 싱글턴 패턴을 구현하는 방법에는 여러 가지가 있다. 그중 기본적인 두 가지 방법을 소개하겠다.

### 1. 고전적인 싱글턴 패턴 (Lazy Initialization)

다음은 Java에서 싱글턴 패턴을 구현하는 가장 전통적인 방법이다.

```java
public class Singleton {
    // Singleton 인스턴스를 저장하는 private static 변수
    private static Singleton instance;

    // private 생성자: 외부에서 인스턴스를 생성하지 못하게 막음
    private Singleton() {
        System.out.println("새로운 인스턴스 생성");
    }

    // 인스턴스를 얻는 public static 메서드
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();  // 인스턴스가 없다면 생성
        }
        return instance;
    }
}

public class Main {
    public static void main(String[] args) {
        Singleton singleton1 = Singleton.getInstance();
        Singleton singleton2 = Singleton.getInstance();

        // 두 객체가 동일한지 확인
        System.out.println(singleton1 == singleton2);  // true 출력
    }
}
```

### 코드 설명
- **`instance` 변수**: `static`으로 선언된 변수로, 클래스 내 유일한 인스턴스를 저장한다.
- **`private` 생성자**: 외부에서 직접 객체를 생성하지 못하게 `private`으로 선언되었다.
- **`getInstance()` 메서드**: 외부에서 인스턴스를 가져오는 메서드이다. `instance`가 `null`일 때만 새로운 객체를 생성하고, 이후에는 동일한 객체를 반환한다.

### 출력 결과
```
새로운 인스턴스 생성
true
```

`singleton1`과 `singleton2`는 동일한 인스턴스임을 확인할 수 있다. 두 객체는 같은 메모리 주소를 참조하며, 중복된 인스턴스 생성이 방지된다.

### 2. 스레드 안전한 싱글턴 패턴 (Thread-Safe Singleton)

싱글턴 패턴을 멀티스레드 환경에서 안전하게 만들려면 **동기화(synchronization)**를 사용해야 한다. 아래는 스레드 안전한 싱글턴 패턴의 구현이다.

```java
public class Singleton {
    // Singleton 인스턴스를 저장하는 private static 변수
    private static Singleton instance;

    // private 생성자: 외부에서 인스턴스를 생성하지 못하게 막음
    private Singleton() {
        System.out.println("새로운 인스턴스 생성");
    }

    // 스레드 안전한 인스턴스 생성 메서드
    public static synchronized Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();  // 인스턴스가 없다면 생성
        }
        return instance;
    }
}

public class Main {
    public static void main(String[] args) {
        Singleton singleton1 = Singleton.getInstance();
        Singleton singleton2 = Singleton.getInstance();

        // 두 객체가 동일한지 확인
        System.out.println(singleton1 == singleton2);  // true 출력
    }
}
```

### 코드 설명
- **`synchronized` 키워드**: `getInstance()` 메서드에 추가된 `synchronized`는 멀티스레드 환경에서 동시 접근을 막아준다. 여러 스레드가 동시에 인스턴스를 생성하려고 할 때, 단 하나의 스레드만 인스턴스를 생성할 수 있게 한다.

### 출력 결과
```
새로운 인스턴스 생성
true
```

멀티스레드 환경에서도 안전하게 동작하며, 두 객체는 동일한 인스턴스임을 확인할 수 있다.

### 3. 이른 초기화(Eager Initialization)

이 방법은 클래스가 로드될 때 인스턴스를 미리 생성해 두는 방식이다. 이 방식은 단순하지만, 인스턴스를 사용하지 않더라도 미리 생성된다는 단점이 있다.

```java
public class Singleton {
    // 이른 초기화 방식으로 인스턴스를 미리 생성
    private static final Singleton instance = new Singleton();

    // private 생성자
    private Singleton() {
        System.out.println("이른 초기화로 인스턴스 생성");
    }

    // 인스턴스를 반환하는 메서드
    public static Singleton getInstance() {
        return instance;
    }
}

public class Main {
    public static void main(String[] args) {
        Singleton singleton1 = Singleton.getInstance();
        Singleton singleton2 = Singleton.getInstance();

        // 두 객체가 동일한지 확인
        System.out.println(singleton1 == singleton2);  // true 출력
    }
}
```

### 출력 결과
```
이른 초기화로 인스턴스 생성
true
```

### 싱글턴 패턴의 한계

1. **테스트 어려움**: 전역 인스턴스를 사용하면 객체의 상태가 여러 곳에서 공유되기 때문에, 객체의 상태를 변경하는 테스트 시 문제가 발생할 수 있다.
2. **의존성 주입 문제**: 다른 객체들이 싱글턴에 의존하게 되면, 이러한 의존성 주입이 복잡해질 수 있다.
3. **멀티스레드 환경**: 멀티스레드 환경에서는 인스턴스가 두 번 이상 생성될 수 있는 문제가 생길 수 있다. 이러한 문제를 해결하기 위해 스레드 안전성을 고려해야 한다.

### 마무리

**싱글턴 패턴**은 클래스의 인스턴스를 하나로 제한하고, 애플리케이션 전체에서 동일한 인스턴스를 공유할 때 유용하다. Java에서는 다양한 방식으로 싱글턴 패턴을 구현할 수 있으며, 상황에 맞게 선택할 수 있다. 그러나 패턴의 남용은 테스트나 확장성을 어렵게 할 수 있으므로, 필요할 때 신중하게 적용하는 것이 중요하다.

다음 글에서는 **팩토리 패턴(Factory Pattern)**을 다루어, 객체 생성과 관련된 패턴을 더 깊이 알아보겠다.