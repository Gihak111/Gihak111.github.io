---
layout: single
title:  "디자인 패턴 시리즈 12. 브리지"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 12: 브리지 패턴 (Bridge Pattern)  

브리지 패턴Bridge Pattern은 기능(Abstraction)과 구현(Implementation)을 분리하여 독립적으로 확장 가능하도록 하는 구조 패턴이다.  
이를 통해 상속의 복잡성을 피하고, 기능과 구현을 독립적으로 변경할 수 있어 확장성과 유연성이 매우 높아진다.  
주로 플랫폼 간의 차이를 극복하거나, 클래스 계층의 복잡성을 줄이기 위해 사용된다.  

## 브리지 패턴의 필요성  

객체 지향 프로그래밍에서 상속은 기능을 재사용하거나 확장하는데 매우 유용하지만, 때로는 클래스 계층 구조가 복잡해지고 유지보수가 어려워지는 문제가 발생할 수 있다.  
특히 기능의 확장과 구현의 확장이 동시에 필요할 때, 기능과 구현이 결합된 구조는 매우 복잡해진다.  
이런 문제를 해결하기 위해 브리지 패턴을 사용하면 다음과 같은 이점을 얻을 수 있다:  

1. 기능과 구현의 분리: 기능 계층과 구현 계층을 각각 분리하여, 서로 독립적으로 확장할 수 있다.  
2. 코드 복잡성 감소: 상속을 줄이고 클래스 계층 구조를 단순화할 수 있다.  
3. 확장성 향상: 새로운 기능이나 구현을 추가할 때, 서로의 영향을 받지 않고 독립적으로 확장할 수 있다.  

### 예시: 다양한 형태의 디바이스 제어  

예를 들어, TV와 라디오와 같은 다양한 디바이스를 원격으로 제어하는 기능을 개발할 때, 디바이스마다 다른 제어 방식이 필요할 수 있다.  
TV와 라디오의 제어 방식은 서로 다르지만, "제어한다"는 기능 자체는 동일하다. 이때 브리지 패턴을 사용하면, 제어하는 기능과 각 디바이스에 대한 구현을 분리하여 쉽게 확장 가능하게 만들 수 있다.  

## 브리지 패턴의 구조  

1. Abstraction(추상화 클래스): 기능의 인터페이스 또는 추상 클래스. 구현에 해당하는 객체를 참조하고, 고수준의 기능을 제공한다.  
2. RefinedAbstraction(구체적인 추상화 클래스): Abstraction을 구체화한 클래스. 추가적인 기능을 제공한다.  
3. Implementor(구현자): 기능을 실제로 구현하는 인터페이스 또는 추상 클래스.  
4. ConcreteImplementor(구체적인 구현자): Implementor를 구현한 구체적인 클래스. 각 기능에 맞는 실제 구현을 제공한다.  

### 구조 다이어그램  

```
    Abstraction               Implementor
        +                         +
        |                         |
  RefinedAbstraction       ConcreteImplementor
        +                         +
        |                         |
        +-------------------------+
               Bridge
```  

### 브리지 패턴 동작 순서  

1. Abstraction은 Implementor를 참조하며, 고수준의 기능을 정의한다.  
2. RefinedAbstraction은 Abstraction을 상속받아 구체적인 기능을 구현한다.  
3. ConcreteImplementor는 실제로 구현된 기능을 제공한다.  
4. Abstraction과 Implementor가 서로 분리되어 있으므로, 각각을 독립적으로 확장할 수 있다.  

## 브리지 패턴 예시 (Bridge Pattern)  

이번 예시에서는 TV와 라디오를 제어하는 원격 컨트롤(Abstraction)을 만들고, 각 디바이스에 맞게 동작을 구현하는 구체적인 구현자(ConcreteImplementor)를 생성해 보겠다.  

### Java로 브리지 패턴 구현하기  

```java
// Implementor 인터페이스
interface Device {
    void turnOn();
    void turnOff();
    void setVolume(int percent);
}

// ConcreteImplementor 클래스들
class TV implements Device {
    private int volume = 0;

    @Override
    public void turnOn() {
        System.out.println("TV 켜기");
    }

    @Override
    public void turnOff() {
        System.out.println("TV 끄기");
    }

    @Override
    public void setVolume(int percent) {
        this.volume = percent;
        System.out.println("TV 볼륨 설정: " + volume + "%");
    }
}

class Radio implements Device {
    private int volume = 0;

    @Override
    public void turnOn() {
        System.out.println("라디오 켜기");
    }

    @Override
    public void turnOff() {
        System.out.println("라디오 끄기");
    }

    @Override
    public void setVolume(int percent) {
        this.volume = percent;
        System.out.println("라디오 볼륨 설정: " + volume + "%");
    }
}

// Abstraction 클래스
abstract class RemoteControl {
    protected Device device;

    public RemoteControl(Device device) {
        this.device = device;
    }

    abstract void togglePower();
    abstract void volumeUp();
    abstract void volumeDown();
}

// RefinedAbstraction 클래스
class AdvancedRemoteControl extends RemoteControl {
    public AdvancedRemoteControl(Device device) {
        super(device);
    }

    @Override
    void togglePower() {
        System.out.println("전원 스위치 누름");
        device.turnOn();
    }

    @Override
    void volumeUp() {
        System.out.println("볼륨 업 버튼 누름");
        device.setVolume(50); // 볼륨을 50으로 설정
    }

    @Override
    void volumeDown() {
        System.out.println("볼륨 다운 버튼 누름");
        device.setVolume(10); // 볼륨을 10으로 설정
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        Device tv = new TV();
        RemoteControl tvRemote = new AdvancedRemoteControl(tv);

        tvRemote.togglePower();
        tvRemote.volumeUp();
        tvRemote.volumeDown();

        Device radio = new Radio();
        RemoteControl radioRemote = new AdvancedRemoteControl(radio);

        radioRemote.togglePower();
        radioRemote.volumeUp();
        radioRemote.volumeDown();
    }
}
```  

### 코드 설명

1. Device (Implementor 인터페이스): TV나 라디오와 같은 장치들의 공통 인터페이스로, 장치들의 켜기, 끄기, 볼륨 설정 기능을 정의한다.  
2. TV, Radio (ConcreteImplementor): Implementor 인터페이스를 구현하여 각 장치에 맞는 구체적인 기능을 구현한다.  
3. RemoteControl (Abstraction 클래스): Device 객체와 상호작용하는 추상화 클래스로, 원격 컨트롤의 기본 기능을 정의한다.  
4. AdvancedRemoteControl (RefinedAbstraction): RemoteControl을 구체화한 클래스. 전원 전환과 볼륨 제어 기능을 제공한다.  

### 출력 결과  

```
전원 스위치 누름
TV 켜기
볼륨 업 버튼 누름
TV 볼륨 설정: 50%
볼륨 다운 버튼 누름
TV 볼륨 설정: 10%
전원 스위치 누름
라디오 켜기
볼륨 업 버튼 누름
라디오 볼륨 설정: 50%
볼륨 다운 버튼 누름
라디오 볼륨 설정: 10%
```  

## 브리지 패턴의 장점  

1. 기능과 구현의 분리: 기능(Abstraction)과 구현(Implementor)을 분리하여 각각 독립적으로 확장할 수 있다.  
2. 확장성 및 유연성 향상: 새로운 기능이나 구현을 추가할 때 서로의 영향을 받지 않으므로, 시스템의 유연성과 확장성이 향상된다.  
3. 코드 중복 감소: 여러 기능과 구현이 조합될 때, 상속을 남용하는 대신 조합을 통해 코드 중복을 줄일 수 있다.  

## 브리지 패턴의 단점  

1. 구조가 복잡해질 수 있음: 기능과 구현이 분리되므로 코드 구조가 다소 복잡해질 수 있다.  
2. 초기 설계 비용 증가: 기능과 구현을 분리하기 위해 초기 설계가 더 복잡하고 비용이 증가할 수 있다.  

### 마무리  

브리지 패턴(Bridge Pattern)은 기능과 구현을 분리하여 각각 독립적으로 확장 가능하도록 만드는 패턴이다.  
이를 통해 상속의 문제점을 피하면서 기능을 확장할 수 있으며, 플랫폼 간의 차이를 극복하거나 기능 확장이 필요한 상황에서 유용하게 활용할 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  