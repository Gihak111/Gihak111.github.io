---
layout: single
title:  "디자인 패턴 시리즈 5. 전략"
categories: "Design Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 5: 전략 패턴 (Strategy Pattern)  
**전략 패턴**은 여러 알고리즘을 하나의 인터페이스로 추상화하여, **런타임에 알고리즘을 교체**할 수 있도록 해주는 패턴이다.  
특정 작업을 수행하는 여러 가지 방법이 있을 때, 각각의 방법을 개별 클래스로 분리하여 유연하게 사용하거나 변경할 수 있다.  

## 전략 패턴의 필요성  
여러 가지 방식으로 수행할 수 있는 작업이 있을 때, 모든 방식을 한 클래스에 다 넣으면 코드가 **복잡**해지고 **유지보수**가 어려워진다.  
또한, 알고리즘을 변경해야 할 때마다 코드를 수정해야 하는 문제가 발생할 수 있다. **전략 패턴**은 이런 문제를 해결해준다.  

예를 들어, 게임 캐릭터가 여러 가지 **공격 방식**을 가지고 있을 수 있다.  
캐릭터가 공격할 때마다 특정 공격 방법을 선택할 수 있도록 구현하고 싶다면, 전략 패턴을 사용하여 각 공격 방법을 별도의 클래스로 분리하고, 캐릭터가 공격 방식을 동적으로 바꾸도록 할 수 있다.  

## 전략 패턴의 구조  
전략 패턴은 다음과 같은 구성 요소로 이루어진다:  
1. **전략 인터페이스(Strategy)**: 알고리즘을 정의하는 인터페이스.  
2. **구체적인 전략(Concrete Strategy)**: 다양한 알고리즘을 구현한 클래스들.  
3. **컨텍스트(Context)**: 전략을 사용하는 클래스. 알고리즘을 실행할 때 전략을 호출하여 사용한다.  
4. **클라이언트(Client)**: 전략을 설정하고 컨텍스트에서 이를 사용하는 부분.  

### 구조 다이어그램  
```
Context
   └─ Strategy
        ├─ ConcreteStrategyA
        ├─ ConcreteStrategyB
```  

## 전략 패턴 예시 (Strategy Pattern)  
이번 예시는 **캐릭터(Character)**가 여러 가지 공격 방법을 가지고 있는 게임을 구현한다고 가정하자.  
캐릭터는 검으로 공격할 수도 있고, 활로 공격할 수도 있으며, 마법을 사용할 수도 있다.  
이런 다양한 공격 방법을 전략 패턴을 통해 **동적으로 선택**할 수 있도록 구현할 것이다.  

### Java로 전략 패턴 구현하기  
```java
// 전략 인터페이스
interface AttackStrategy {
    void attack();
}

// 구체적인 전략: 검 공격
class SwordAttack implements AttackStrategy {
    @Override
    public void attack() {
        System.out.println("검으로 공격합니다!");
    }
}

// 구체적인 전략: 활 공격
class BowAttack implements AttackStrategy {
    @Override
    public void attack() {
        System.out.println("활로 공격합니다!");
    }
}

// 구체적인 전략: 마법 공격
class MagicAttack implements AttackStrategy {
    @Override
    public void attack() {
        System.out.println("마법으로 공격합니다!");
    }
}

// 컨텍스트: 캐릭터 클래스
class Character {
    private AttackStrategy attackStrategy;

    // 전략을 설정하는 메서드
    public void setAttackStrategy(AttackStrategy attackStrategy) {
        this.attackStrategy = attackStrategy;
    }

    // 공격 메서드: 설정된 전략을 사용하여 공격
    public void performAttack() {
        if (attackStrategy != null) {
            attackStrategy.attack();
        } else {
            System.out.println("공격 방법이 설정되지 않았습니다.");
        }
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        Character character = new Character();

        // 검으로 공격
        character.setAttackStrategy(new SwordAttack());
        character.performAttack();

        // 활로 공격
        character.setAttackStrategy(new BowAttack());
        character.performAttack();

        // 마법으로 공격
        character.setAttackStrategy(new MagicAttack());
        character.performAttack();
    }
}
```  

### 코드 설명  
1. **전략 인터페이스(Strategy)**: `AttackStrategy` 인터페이스는 `attack()` 메서드를 정의한다. 각 공격 방법은 이 인터페이스를 구현하게 된다.  
2. **구체적인 전략(Concrete Strategy)**: `SwordAttack`, `BowAttack`, `MagicAttack` 클래스는 각각 검, 활, 마법으로 공격하는 방법을 구현한 클래스들이다.  
3. **컨텍스트(Context)**: `Character` 클래스는 전략을 설정하고 실행하는 역할을 한다. 공격 방식은 `setAttackStrategy()` 메서드를 통해 동적으로 설정되며, `performAttack()` 메서드를 호출하면 설정된 전략에 따라 공격이 수행된다.  
4. **클라이언트(Client)**: `Main` 클래스에서 캐릭터 객체를 생성한 후, 여러 가지 공격 방법을 설정하고 이를 실행한다.  

### 출력 결과  
```
검으로 공격합니다!
활로 공격합니다!
마법으로 공격합니다!
```  

## 전략 패턴의 장점  
1. **유연성**: 알고리즘을 동적으로 선택하거나 변경할 수 있어 매우 유연하다.  
2. **유지보수성**: 새로운 알고리즘을 추가할 때 기존 코드를 수정하지 않고, 새로운 전략 클래스를 추가하는 방식으로 확장할 수 있다.  
3. **단일 책임 원칙(SRP)**: 각 알고리즘은 별도의 클래스에 정의되므로, 각 클래스는 하나의 책임만 가지게 된다.  

## 전략 패턴의 단점  
1. **클래스 수 증가**: 각 알고리즘을 별도의 클래스로 분리하기 때문에 클래스 수가 많아질 수 있다.  
2. **복잡성**: 작은 프로젝트에서는 오히려 과도한 복잡성을 초래할 수 있다.  

---

### 마무리  
**전략 패턴(Strategy Pattern)**은 다양한 알고리즘을 동적으로 적용해야 하는 경우에 유용한 패턴이다.  
특히 게임과 같은 시나리오에서 캐릭터의 다양한 공격 방식을 구현할 때, 이 패턴을 사용하면 코드의 유연성과 확장성을 높일 수 있다.  

다음 글에서는 **옵저버 패턴(Observer Pattern)**을 다룰 예정이다.  
옵저버 패턴은 **상태 변화에 따라 자동으로 알림을 받는 패턴**으로, 주로 이벤트 기반 시스템에서 많이 사용된다.  
