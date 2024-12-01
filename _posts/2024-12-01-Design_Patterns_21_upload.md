---
layout: single
title:  "디자인 패턴 시리즈 21. 플라이웨이트"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 21: 플라이웨이트 패턴 (Flyweight Pattern)

플라이웨이트 패턴(Flyweight Pattern)은 구조적 디자인 패턴으로, 공유를 통해 객체를 효율적으로 관리하여 메모리 사용량을 줄이는 데 도움을 준다.  
이는 동일하거나 유사한 객체를 다수 생성할 때, 공통된 상태를 공유하도록 설계하여 성능을 최적화하는 데 유용하다.  

## 플라이웨이트 패턴의 필요성

소프트웨어에서 동일하거나 유사한 데이터를 가진 객체가 많아질 경우 다음과 같은 문제가 발생할 수 있다:

1. 메모리 낭비: 불필요한 객체가 다수 생성된다.  
2. 성능 저하: 객체 생성과 관리에 많은 리소스가 사용된다.  

플라이웨이트 패턴은 공유 가능한 상태를 분리하고, 이를 재사용하여 다음과 같은 이점을 제공한다:  

1. 메모리 사용 감소: 객체 수를 줄여 메모리 효율을 높인다.  
2. 성능 향상: 객체 생성과 소멸에 필요한 비용을 절감한다.  

### 예시: 그래픽 편집기의 텍스트 렌더링

그래픽 편집기에서 폰트, 크기, 색상 등이 동일한 문자를 렌더링할 때, 플라이웨이트 패턴을 사용하면 효율적으로 메모리를 관리할 수 있다.  

## 플라이웨이트 패턴의 구조

1. Flyweight (플라이웨이트): 공유되는 객체의 인터페이스를 정의한다.  
2. ConcreteFlyweight (구체적인 플라이웨이트): 공유 가능한 객체를 구현한다.  
3. FlyweightFactory (플라이웨이트 팩토리): 공유 객체를 생성하고 관리한다.  
4. Client (클라이언트): 플라이웨이트 객체를 사용하며, 공유 상태와 비공유 상태를 관리한다.  

### 구조 다이어그램

```
Client → FlyweightFactory → Flyweight
                    ↘ ConcreteFlyweight
```

### 플라이웨이트 패턴 동작 순서

1. 클라이언트는 `FlyweightFactory`를 통해 플라이웨이트 객체를 요청한다.  
2. 팩토리는 기존 객체를 반환하거나 새 객체를 생성하여 반환한다.  
3. 클라이언트는 객체의 공유 상태와 비공유 상태를 조합하여 작업을 수행한다.  

## 플라이웨이트 패턴 예시

이번 예시에서는 "텍스트 렌더링 시스템"을 플라이웨이트 패턴으로 구현한다.  

### Java로 플라이웨이트 패턴 구현하기

```java
// Flyweight 인터페이스
interface TextStyle {
    void applyStyle(String text);
}

// ConcreteFlyweight 클래스
class ConcreteTextStyle implements TextStyle {
    private String font;
    private int size;
    private String color;

    public ConcreteTextStyle(String font, int size, String color) {
        this.font = font;
        this.size = size;
        this.color = color;
    }

    @Override
    public void applyStyle(String text) {
        System.out.println("텍스트: \"" + text + "\" | 폰트: " + font + " | 크기: " + size + " | 색상: " + color);
    }
}

// FlyweightFactory 클래스
class TextStyleFactory {
    private Map<String, TextStyle> styles = new HashMap<>();

    public TextStyle getTextStyle(String font, int size, String color) {
        String key = font + size + color;
        if (!styles.containsKey(key)) {
            styles.put(key, new ConcreteTextStyle(font, size, color));
            System.out.println("새로운 스타일 생성: " + key);
        } else {
            System.out.println("기존 스타일 재사용: " + key);
        }
        return styles.get(key);
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        TextStyleFactory styleFactory = new TextStyleFactory();

        // 동일한 스타일 공유
        TextStyle style1 = styleFactory.getTextStyle("Arial", 12, "Black");
        style1.applyStyle("안녕하세요!");

        TextStyle style2 = styleFactory.getTextStyle("Arial", 12, "Black");
        style2.applyStyle("플라이웨이트 패턴");

        // 다른 스타일 생성
        TextStyle style3 = styleFactory.getTextStyle("Times New Roman", 14, "Blue");
        style3.applyStyle("디자인 패턴 예제입니다.");
    }
}
```

### 코드 설명

1. Flyweight (플라이웨이트): `TextStyle` 인터페이스는 스타일의 공통 동작을 정의한다.  
2. ConcreteFlyweight (구체적인 플라이웨이트): `ConcreteTextStyle` 클래스는 공유 가능한 스타일 객체를 구현한다.  
3. FlyweightFactory (플라이웨이트 팩토리): `TextStyleFactory` 클래스는 스타일 객체를 생성하고 관리한다.  
4. Client (클라이언트): `Main` 클래스는 팩토리를 통해 객체를 요청하고 사용한다.  

### 출력 결과

```
새로운 스타일 생성: Arial12Black
텍스트: "안녕하세요!" | 폰트: Arial | 크기: 12 | 색상: Black
기존 스타일 재사용: Arial12Black
텍스트: "플라이웨이트 패턴" | 폰트: Arial | 크기: 12 | 색상: Black
새로운 스타일 생성: Times New Roman14Blue
텍스트: "디자인 패턴 예제입니다." | 폰트: Times New Roman | 크기: 14 | 색상: Blue
```

## 플라이웨이트 패턴의 장점

1. 메모리 사용 최적화: 공유 가능한 객체를 재사용하여 메모리 낭비를 줄인다.  
2. 객체 관리 간소화: 공유 상태와 비공유 상태를 명확히 분리하여 관리한다.  

## 플라이웨이트 패턴의 단점

1. 복잡한 코드 구조: 공유 상태와 비공유 상태를 분리하고 관리해야 하므로 코드가 복잡해질 수 있다.  
2. 객체 간 의존성 증가: 공유 객체를 잘못 수정하면 여러 클라이언트에 영향을 미칠 수 있다.  

### 마무리

플라이웨이트 패턴(Flyweight Pattern)은 반복적으로 사용되는 객체를 효율적으로 관리하는 데 적합하다.  
특히, 대규모 시스템에서 동일하거나 유사한 객체가 많이 생성될 때 메모리와 성능 최적화에 큰 도움을 줄 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  