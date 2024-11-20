---
layout: single
title:  "디자인 패턴 시리즈 7. 옵저버"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 7: 옵저버 패턴 (Observer Pattern)  
옵저버 패턴(Observer Pattern)은 주체(Subject)와 옵저버(Observer)의 관계에서, 주체의 상태 변화가 있을 때 옵저버들에게 자동으로 이를 통지하는 패턴이다.  
여러 객체가 특정 객체의 상태 변화를 감지할 수 있게 하며, 그 변화를 실시간으로 전달받아 적절히 대응할 수 있게 한다.  

## 옵저버 패턴의 필요성  
애플리케이션 내에서 여러 객체가 특정 객체의 상태에 의존할 때, 의존 관계를 직접적으로 연결하지 않고도 상태 변화를 통지받는 방법이 필요하다.  
이를 위해 옵저버 패턴은 상태 변화를 자동으로 반영하여 의존 객체들에 통지할 수 있다.  
이 패턴을 사용하면, 주체와 옵저버 간의 결합도가 낮아져서 유연하고 확장성 있는 구조를 만들 수 있다.  

### 예시: 뉴스 발행 시스템

뉴스 발행 시스템에서 새로운 뉴스가 발생하면, 뉴스 구독자(옵저버)들이 이를 실시간으로 통보받고 각기 다른 행동을 할 수 있다.   
예를 들어, 한 구독자는 뉴스 알림을 이메일로 받을 수 있고, 다른 구독자는 모바일 앱 알림으로 받을 수 있다.  

## 옵저버 패턴의 구조  
옵저버 패턴은 다음과 같은 구성 요소로 이루어진다:  

1. 주체(Subject): 상태를 관리하며, 상태 변화를 옵저버들에게 통지한다.  
2. 옵저버(Observer): 주체의 상태 변화를 통지받아 적절히 반응하는 객체들이다.  
3. 구체적인 주체(Concrete Subject): 주체의 상태 변화를 구현하고, 옵저버들에게 알린다.  
4. 구체적인 옵저버(Concrete Observer): 주체의 변화를 통지받아 특정 동작을 실행한다.  

### 구조 다이어그램  

```
Subject
   ├─ addObserver(Observer)
   ├─ removeObserver(Observer)
   └─ notifyObservers()

Observer
   └─ update()
```  

### 옵저버 패턴 동작 순서  

1. **옵저버 등록**: 주체(Subject)에 옵저버(Observer)들을 등록한다.  
2. **상태 변경**: 주체의 상태가 변경되면, 등록된 모든 옵저버들에게 상태 변화를 알린다.  
3. **옵저버 업데이트**: 옵저버들은 주체로부터 통지받은 상태 변화를 반영하여 적절한 작업을 수행한다.  

## 옵저버 패턴 예시 (Observer Pattern)  

이번 예시에서는 **뉴스 발행 시스템**을 구현해보자.  
뉴스가 발행되면, 다양한 구독자들이 이를 통지받아 각기 다른 방식으로 알림을 받는 구조를 구현해보겠다.  

### Java로 옵저버 패턴 구현하기  

```java
// 주체 인터페이스 (Subject)
interface NewsPublisher {
    void addObserver(NewsSubscriber observer);
    void removeObserver(NewsSubscriber observer);
    void notifyObservers();
}

// 옵저버 인터페이스 (Observer)
interface NewsSubscriber {
    void update(String news);
}

// 구체적인 주체 클래스 (Concrete Subject)
class NewsAgency implements NewsPublisher {
    private List<NewsSubscriber> subscribers = new ArrayList<>();
    private String news;

    @Override
    public void addObserver(NewsSubscriber observer) {
        subscribers.add(observer);
    }

    @Override
    public void removeObserver(NewsSubscriber observer) {
        subscribers.remove(observer);
    }

    @Override
    public void notifyObservers() {
        for (NewsSubscriber subscriber : subscribers) {
            subscriber.update(news);
        }
    }

    public void setNews(String news) {
        this.news = news;
        notifyObservers();
    }
}

// 구체적인 옵저버 클래스 (Concrete Observer)
class EmailSubscriber implements NewsSubscriber {
    private String name;

    public EmailSubscriber(String name) {
        this.name = name;
    }

    @Override
    public void update(String news) {
        System.out.println(name + "에게 이메일로 전달된 뉴스: " + news);
    }
}

class MobileSubscriber implements NewsSubscriber {
    private String name;

    public MobileSubscriber(String name) {
        this.name = name;
    }

    @Override
    public void update(String news) {
        System.out.println(name + "에게 모바일 알림으로 전달된 뉴스: " + news);
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        NewsAgency newsAgency = new NewsAgency();

        NewsSubscriber emailSubscriber = new EmailSubscriber("김재성");
        NewsSubscriber mobileSubscriber = new MobileSubscriber("강민준");

        // 구독자 등록
        newsAgency.addObserver(emailSubscriber);
        newsAgency.addObserver(mobileSubscriber);

        // 뉴스 발행 및 알림 전송
        newsAgency.setNews("디자인 패턴 강의가 새로 올라왔습니다!");
        newsAgency.setNews("전략 패턴에 대한 글이 공개되었습니다!");
    }
}
```

### 코드 설명  

1. 주체 인터페이스(Subject): `NewsPublisher` 인터페이스는 옵저버를 등록(`addObserver()`), 제거(`removeObserver()`), 알림(`notifyObservers()`) 기능을 제공한다.  
2. 옵저버 인터페이스(Observer): `NewsSubscriber` 인터페이스는 `update()` 메서드를 통해 상태 변화를 통지받는다.  
3. 구체적인 주체(Concrete Subject): `NewsAgency` 클래스는 뉴스 발행 시스템을 구현하며, 상태가 변경될 때마다 등록된 옵저버들에게 새로운 뉴스를 통지한다.  
4. 구체적인 옵저버(Concrete Observer): `EmailSubscriber`, `MobileSubscriber`는 각기 다른 방식으로 뉴스를 통지받고, 이를 출력한다.  

### 출력 결과

```
김재성에게 이메일로 전달된 뉴스: 디자인 패턴 강의가 새로 올라왔습니다!
강민준에게 모바일 알림으로 전달된 뉴스: 디자인 패턴 강의가 새로 올라왔습니다!
김재성에게 이메일로 전달된 뉴스: 전략 패턴에 대한 글이 공개되었습니다!
강민준에게 모바일 알림으로 전달된 뉴스: 전략 패턴에 대한 글이 공개되었습니다!
```  

## 옵저버 패턴의 장점  

1. 느슨한 결합: 주체와 옵저버 간의 의존성이 낮아지며, 서로의 변경 사항에 큰 영향을 주지 않는다.  
2. 유연성: 주체와 옵저버가 독립적으로 확장 가능하다. 새로운 옵저버를 추가할 때 기존 코드를 수정하지 않고 등록만 하면 된다.  
3. 실시간 업데이트: 주체의 상태 변화가 있을 때, 즉시 옵저버들이 통지받아 즉각적인 반응이 가능하다.  

## 옵저버 패턴의 단점

1. 성능 이슈: 옵저버가 많아질수록 주체가 모든 옵저버들에게 일일이 통지해야 하므로 성능이 저하될 수 있다.  
2. 예측 불가: 옵저버들이 어떻게 반응할지 예측하기 어려워 복잡한 로직에서는 디버깅이 힘들 수 있다.  

### 마무리  
옵저버 패턴(Observer Pattern)은 이벤트 기반 시스템에서 매우 유용한 패턴으로, 여러 객체가 하나의 객체 상태 변화를 감지하고 동적으로 반응해야 할 때 유용하다.  
뉴스 발행 시스템처럼 상태 변화를 실시간으로 전달해야 하는 곳에서 자주 사용되며, 확장성과 유연성을 보장하는 중요한 디자인 패턴 중 하나다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  