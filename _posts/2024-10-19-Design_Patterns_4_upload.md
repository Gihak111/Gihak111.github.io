---
layout: single
title:  "디자인 패턴 시리즈 4. 빌더"
categories: "Design Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 4: 빌더 패턴 (Builder Pattern)

**빌더 패턴**은 복잡한 객체를 생성하는 과정에서 **단계별로 세밀하게 설정**할 수 있도록 도와주는 생성 패턴이다.  
객체의 생성 로직이 복잡할 때, 모든 매개변수를 한 번에 생성자에서 처리하는 대신, **단계적으로** 각 속성을 설정하면서 객체를 생성할 수 있게 한다.  

## 빌더 패턴의 필요성

빌더 패턴은 **매개변수의 수가 많거나**, 특정 속성들이 선택적일 때 유용하다.  
복잡한 객체를 생성하는데 생성자에 너무 많은 매개변수를 전달하는 방식은 코드의 가독성을 떨어뜨리고 유지보수가 어려워질 수 있다.  
빌더 패턴은 이 문제를 해결하면서, 객체가 **불변성(immutability)**을 유지할 수 있도록 한다.  

예를 들어, **컴퓨터**와 같은 복잡한 객체를 생각해보자.  
컴퓨터는 다양한 구성 요소(CPU, RAM, 저장 장치 등)가 존재하고, 각 요소의 설정이 복잡할 수 있다.  
빌더 패턴을 사용하면, 이러한 복잡한 객체를 **단계별로 설정**하며 만들 수 있다.  

## 빌더 패턴의 구조  

빌더 패턴은 다음과 같은 구성 요소로 이루어진다:  

1. **제품(Product)**: 최종적으로 생성되는 객체.  
2. **빌더 인터페이스(Builder)**: 제품을 구성하는 각 단계의 메서드를 정의한 인터페이스.  
3. **구체적인 빌더(Concrete Builder)**: 빌더 인터페이스를 구현하고, 실제로 객체를 구성하는 클래스.  
4. **감독자(Director)**: 빌더를 이용하여 객체 생성 과정을 관리하는 클래스.  
5. **클라이언트(Client)**: 빌더를 사용하여 객체를 생성하는 코드.  

### 구조 다이어그램  

```
Director
   └─ Builder
         ├─ ConcreteBuilder
Product
```  

## 빌더 패턴 예시 (Builder Pattern)  

이번 예시는 **컴퓨터(Computer)** 객체를 생성하는 예시를 통해 빌더 패턴을 설명하겠다.  
컴퓨터는 다양한 부품을 가지고 있으며, CPU, RAM, 저장 장치 등의 구성 요소가 있다.  

### Java로 빌더 패턴 구현하기  

```java
// Product 클래스
class Computer {
    private String CPU;
    private String RAM;
    private String storage;
    private String graphicsCard;

    // Computer 객체의 빌더를 위한 정적 내부 클래스
    public static class Builder {
        private String CPU;
        private String RAM;
        private String storage;
        private String graphicsCard;

        public Builder(String CPU, String RAM) {
            this.CPU = CPU;  // 필수 속성
            this.RAM = RAM;  // 필수 속성
        }

        public Builder setStorage(String storage) {
            this.storage = storage;  // 선택적 속성
            return this;
        }

        public Builder setGraphicsCard(String graphicsCard) {
            this.graphicsCard = graphicsCard;  // 선택적 속성
            return this;
        }

        // 최종적으로 Computer 객체 생성
        public Computer build() {
            Computer computer = new Computer();
            computer.CPU = this.CPU;
            computer.RAM = this.RAM;
            computer.storage = this.storage;
            computer.graphicsCard = this.graphicsCard;
            return computer;
        }
    }

    @Override
    public String toString() {
        return "Computer [CPU=" + CPU + ", RAM=" + RAM + ", storage=" + storage + ", graphicsCard=" + graphicsCard + "]";
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        // 필수 속성만 설정된 컴퓨터 생성
        Computer basicComputer = new Computer.Builder("Intel i7", "16GB")
                .build();
        System.out.println(basicComputer);

        // 모든 속성이 설정된 컴퓨터 생성
        Computer gamingComputer = new Computer.Builder("AMD Ryzen 9", "32GB")
                .setStorage("1TB SSD")
                .setGraphicsCard("NVIDIA RTX 3080")
                .build();
        System.out.println(gamingComputer);
    }
}
```  

### 코드 설명  

1. **Product 클래스**: `Computer` 클래스는 최종적으로 생성될 객체이다. 컴퓨터의 필수적인 구성 요소는 CPU와 RAM이고, 선택적으로 저장 장치와 그래픽카드를 추가할 수 있다.  
2. **Builder 클래스**: `Computer.Builder`는 `Computer` 객체를 단계적으로 구성할 수 있는 내부 정적 클래스이다. 필수 속성(CPU, RAM)은 생성자에서 설정하고, 선택적 속성(storage, graphicsCard)은 메서드 체인을 통해 설정할 수 있다.  
3. **build() 메서드**: 모든 속성이 설정된 후, `build()` 메서드를 통해 최종 `Computer` 객체를 반환한다.  
4. **클라이언트**: `Main` 클래스에서 빌더 패턴을 사용하여 서로 다른 구성을 가진 `Computer` 객체를 생성할 수 있다. 첫 번째 컴퓨터는 필수 속성만 가지고 있고, 두 번째 컴퓨터는 선택적 속성까지 모두 설정된 고사양 컴퓨터이다.  

### 출력 결과  

```
Computer [CPU=Intel i7, RAM=16GB, storage=null, graphicsCard=null]
Computer [CPU=AMD Ryzen 9, RAM=32GB, storage=1TB SSD, graphicsCard=NVIDIA RTX 3080]
```  

### 장점  

1. **가독성**: 객체 생성 시 빌더 메서드를 통해 단계별로 속성을 설정할 수 있어 가독성이 좋아진다.  
2. **유연성**: 선택적 속성을 유연하게 설정할 수 있으며, 생성자의 인자 수가 많아질 때 이를 깔끔하게 해결할 수 있다.  
3. **불변성**: 객체를 생성 후에는 불변 상태로 유지할 수 있다.  

### 단점  

1. **복잡성 증가**: 클래스가 복잡해질 수 있으며, 특히 객체가 여러 단계로 이루어져 있을 때 빌더 클래스가 길어질 수 있다.  

---

### 마무리

**빌더 패턴**은 복잡한 객체를 단계적으로 구성해야 할 때 유용하다.  
특히, 매개변수의 수가 많거나 객체가 복잡한 구조를 가질 때 가독성을 높이고 유지보수성을 개선할 수 있다.  
