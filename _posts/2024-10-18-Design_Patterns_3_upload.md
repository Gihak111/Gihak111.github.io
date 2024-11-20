---
layout: single
title:  "디자인 패턴 시리즈 3. 추상 팩토리"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


# 디자인 패턴 시리즈 3: 추상 팩토리 패턴 (Abstract Factory Pattern)  

**추상 팩토리 패턴**은 객체 생성의 확장된 방식으로, 관련된 객체들을 그룹으로 묶어 생성하는 역할을 한다.  
**팩토리 메서드 패턴**이 개별 객체의 생성 책임을 서브클래스에 위임하는 것과 달리, 추상 팩토리 패턴은 **제품군(Product family)**을 묶어서 생성한다.  

## 추상 팩토리 패턴의 필요성

어떤 시스템에서 **서로 연관된 객체**들이 다수 존재할 때, 이 객체들을 생성하는 방식이 복잡해질 수 있다.  
또한, 각 제품군 간의 의존성을 유지하면서도 시스템이 확장될 때마다 코드 변경을 최소화하고 싶을 때 추상 팩토리 패턴이 유용하다.  

이 패턴은 다양한 제품군이 필요하거나 제품들이 함께 생성되어야 할 때 사용한다.  
예를 들어, **GUI 라이브러리**를 구현할 때 윈도우나 Mac용 버튼, 체크박스 등을 제품군으로 묶어 생성하는 상황에서 적용된다.  

## 추상 팩토리 패턴의 구조

추상 팩토리 패턴은 다음과 같은 구성 요소로 이루어진다:  

1. **추상 팩토리(Abstract Factory)**: 제품군을 생성하는 인터페이스.  
2. **구체적인 팩토리(Concrete Factory)**: 추상 팩토리를 상속하여 제품군을 실제로 생성하는 클래스.  
3. **추상 제품(Abstract Product)**: 제품군의 인터페이스를 정의한 추상 클래스.  
4. **구체적인 제품(Concrete Product)**: 추상 제품을 상속받아 구체적인 제품을 구현한 클래스.  

### 구조 다이어그램

```
AbstractFactory
   ├─ ConcreteFactoryA
   ├─ ConcreteFactoryB
AbstractProduct1       AbstractProduct2
   ├─ ConcreteProductA1   ├─ ConcreteProductA2
   ├─ ConcreteProductB1   └─ ConcreteProductB2
```  

## 추상 팩토리 패턴 예시 (Abstract Factory Pattern)

이번 예시는 서로 다른 테마의 UI 요소(Button과 Checkbox)를 생성하는 **GUI 프로그램**을 개발한다고 가정하자.  
여기서 각 테마는 서로 다른 제품군을 나타낸다.  

### Java로 추상 팩토리 패턴 구현하기

```java
// 추상 제품 인터페이스들
interface Button {
    void click();
}

interface Checkbox {
    void toggle();
}

// 구체적인 제품 클래스들 (윈도우 테마)
class WindowsButton implements Button {
    @Override
    public void click() {
        System.out.println("윈도우 스타일 버튼 클릭");
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public void toggle() {
        System.out.println("윈도우 스타일 체크박스 토글");
    }
}

// 구체적인 제품 클래스들 (Mac 테마)
class MacButton implements Button {
    @Override
    public void click() {
        System.out.println("Mac 스타일 버튼 클릭");
    }
}

class MacCheckbox implements Checkbox {
    @Override
    public void toggle() {
        System.out.println("Mac 스타일 체크박스 토글");
    }
}

// 추상 팩토리 인터페이스
interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// 구체적인 팩토리 클래스들 (윈도우와 Mac에 맞는 UI 요소 생성)
class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

class MacFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

// 클라이언트 코드
public class Main {
    private Button button;
    private Checkbox checkbox;

    // 클라이언트는 추상 팩토리만 사용하여 제품을 생성한다.
    public Main(GUIFactory factory) {
        button = factory.createButton();
        checkbox = factory.createCheckbox();
    }

    public void render() {
        button.click();
        checkbox.toggle();
    }

    public static void main(String[] args) {
        // Windows 테마를 선택
        GUIFactory factory = new WindowsFactory();
        Main app = new Main(factory);
        app.render();  // "윈도우 스타일 버튼 클릭" 및 "윈도우 스타일 체크박스 토글" 출력

        // Mac 테마를 선택
        factory = new MacFactory();
        app = new Main(factory);
        app.render();  // "Mac 스타일 버튼 클릭" 및 "Mac 스타일 체크박스 토글" 출력
    }
}
```

### 코드 설명

1. **추상 제품 인터페이스**: `Button`과 `Checkbox`는 제품군의 추상화된 인터페이스이다.  
2. **구체적인 제품 클래스**: `WindowsButton`, `MacButton`과 같은 클래스는 각 플랫폼(윈도우, Mac)에 맞는 구체적인 구현을 제공한다.  
3. **추상 팩토리 인터페이스**: `GUIFactory`는 버튼과 체크박스를 생성할 수 있는 인터페이스를 제공한다.  
4. **구체적인 팩토리 클래스**: `WindowsFactory`와 `MacFactory`는 윈도우와 Mac 테마에 맞는 UI 요소를 생성하는 팩토리 클래스이다.  
5. **클라이언트**: 클라이언트는 구체적인 구현을 알 필요 없이, 팩토리를 통해 제품을 생성하고 사용할 수 있다.  

### 출력 결과

```
윈도우 스타일 버튼 클릭  
윈도우 스타일 체크박스 토글  
Mac 스타일 버튼 클릭  
Mac 스타일 체크박스 토글  
```

### 장점

1. **확장성**: 새로운 제품군을 추가할 때, 기존 클라이언트 코드를 수정할 필요 없이 새로운 팩토리 클래스만 추가하면 된다.  
2. **일관성 유지**: 각 제품군의 객체들이 함께 일관성 있게 생성된다.  

### 단점

1. **복잡성 증가**: 시스템이 확장될수록 팩토리와 제품 클래스가 늘어나서 복잡해질 수 있다.  
2. **추가적인 코드 요구**: 새로운 제품군이나 변형이 추가될 때마다 새로운 팩토리와 제품 클래스를 작성해야 한다.  

---

### 마무리

**추상 팩토리 패턴**은 시스템에서 여러 관련된 객체들을 생성해야 할 때 유용하다.  
각 제품군을 팩토리로 묶어 관리함으로써 객체 생성의 복잡성을 줄일 수 있고, 새로운 제품군을 쉽게 추가할 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  