---
layout: single
title:  "디자인 패턴 시리즈 11. 책임 연쇄"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 10: 책임 연쇄 패턴 (Chain of Responsibility Pattern)  

책임 연쇄 패턴은 여러 객체들이 순차적으로 요청을 처리할 기회를 가지는 패턴이다.  
각 객체는 처리할 수 있는 범위에 따라 요청을 처리할 수 있고, 처리하지 못하면 다음 객체로 요청을 넘겨 처리하도록 한다.  
이를 통해 객체 간의 결합도를 낮추고, 유연한 요청 처리 구조를 제공할 수 있다.  

## 책임 연쇄 패턴의 필요성  

보통 프로그램은 요청을 처리할 객체를 명확히 지정해야 하지만, 특정 객체에 요청을 고정시키면 유연성이 떨어진다.  
예를 들어, 다양한 조건에 따라 여러 객체가 처리할 수 있는 상황이 있을 때, 요청을 처리할 객체를 고정시키면 코드가 복잡해지고 유지보수가 어려워질 수 있다.  

책임 연쇄 패턴을 사용하면 다음과 같은 장점을 얻을 수 있다:  

1. 요청을 여러 객체들이 처리할 기회를 가질 수 있으며, 각 객체는 자신이 처리할 수 있는지 판단하여 처리할지, 다음 객체에게 넘길지를 결정한다.  
2. 객체 간의 결합도를 낮추고, 요청 처리 과정에서 발생할 수 있는 변경을 쉽게 대응할 수 있다.  
3. 유연한 요청 처리 흐름을 설계할 수 있다.  

### 예시: 고객 지원 시스템  

고객이 회사에 문의할 때, 문의 내용에 따라 다양한 부서에서 처리될 수 있다.  
예를 들어, 일반적인 문의는 고객 상담 부서에서 처리하지만, 기술적인 문제는 기술 지원팀에서 처리하고, 재무 관련 문제는 재무팀에서 처리해야 한다.  
이런 경우 책임 연쇄 패턴을 사용하면 각 부서가 요청을 처리할 수 있도록 연쇄적으로 연결하여, 각 부서가 적절히 문의를 처리하도록 만들 수 있다.  

## 책임 연쇄 패턴의 구조  

1. Handler 인터페이스: 모든 처리 객체가 구현해야 하는 인터페이스로, `handleRequest()` 메서드를 정의한다. 각 객체는 다음 처리 객체를 가리키는 참조를 갖고 있다.  
2. ConcreteHandler: 요청을 처리하는 구체적인 처리 객체들이다. 자신이 요청을 처리할 수 있으면 처리하고, 처리하지 못하면 다음 객체로 요청을 넘긴다.  
3. Client: 요청을 처리할 객체 체인을 설정하고, 요청을 보내는 역할을 한다.  

### 구조 다이어그램  

```
Client
   └─ Handler (인터페이스)
         ├─ ConcreteHandler1
         ├─ ConcreteHandler2
         └─ ConcreteHandler3
               └─ 처리할 객체가 없을 때 요청이 종료됨
```  

### 책임 연쇄 패턴 동작 순서  

1. *라이언트(Client)는 요청을 처리할 첫 번째 핸들러 객체에게 요청을 보낸다.  
2. 핸들러(Handler)는 자신이 요청을 처리할 수 있는지 확인한 후, 처리할 수 있으면 처리하고, 그렇지 않으면 다음 핸들러로 요청을 넘긴다.  
3. 요청이 체인의 끝까지 전달되거나, *간에서 처리되면 요청이 종료된다.  

## 책임 연쇄 패턴 예시 (Chain of Responsibility Pattern)  

고객 문의를 처리하는 시스템을 예시로 책임 연쇄 패턴을 적용해 보겠다.  
각 부서(일반 문의, 기술 지원, 재무팀)가 연쇄적으로 연결되어, 각 부서가 처리할 수 있는 범위 내에서 문의를 처리하거나, 다음 부서로 넘기게 된다.  

### Java로 책임 연쇄 패턴 구현하기  

```java
// Handler 인터페이스
abstract class SupportHandler {
    protected SupportHandler nextHandler;

    public void setNextHandler(SupportHandler nextHandler) {
        this.nextHandler = nextHandler;
    }

    public abstract void handleRequest(String request);
}

// ConcreteHandler: 일반 문의 처리
class GeneralSupportHandler extends SupportHandler {
    @Override
    public void handleRequest(String request) {
        if (request.equals("일반 문의")) {
            System.out.println("일반 문의 처리: 고객 상담 부서에서 처리합니다.");
        } else if (nextHandler != null) {
            nextHandler.handleRequest(request);
        }
    }
}

// ConcreteHandler: 기술 지원 처리
class TechnicalSupportHandler extends SupportHandler {
    @Override
    public void handleRequest(String request) {
        if (request.equals("기술 지원")) {
            System.out.println("기술 지원 처리: 기술 지원팀에서 처리합니다.");
        } else if (nextHandler != null) {
            nextHandler.handleRequest(request);
        }
    }
}

// ConcreteHandler: 재무 문제 처리
class FinancialSupportHandler extends SupportHandler {
    @Override
    public void handleRequest(String request) {
        if (request.equals("재무 문제")) {
            System.out.println("재무 문제 처리: 재무팀에서 처리합니다.");
        } else if (nextHandler != null) {
            nextHandler.handleRequest(request);
        }
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        // 핸들러 체인 설정
        SupportHandler generalSupport = new GeneralSupportHandler();
        SupportHandler technicalSupport = new TechnicalSupportHandler();
        SupportHandler financialSupport = new FinancialSupportHandler();

        generalSupport.setNextHandler(technicalSupport);
        technicalSupport.setNextHandler(financialSupport);

        // 요청 처리
        generalSupport.handleRequest("일반 문의"); // 출력: 일반 문의 처리: 고객 상담 부서에서 처리합니다.
        generalSupport.handleRequest("기술 지원"); // 출력: 기술 지원 처리: 기술 지원팀에서 처리합니다.
        generalSupport.handleRequest("재무 문제"); // 출력: 재무 문제 처리: 재무팀에서 처리합니다.
    }
}
```  

### 코드 설명  

1. SupportHandler (Handler 인터페이스): 요청을 처리하거나, 처리하지 못하면 다음 핸들러로 요청을 넘긴다. `setNextHandler()` 메서드를 통해 체인을 설정하고, `handleRequest()` 메서드에서 요청을 처리한다.  
2. GeneralSupportHandler, TechnicalSupportHandler, FinancialSupportHandler (ConcreteHandler): 각각 일반 문의, 기술 지원, 재무 문제를 처리하는 구체적인 핸들러들이다.  
3. Main (Client): 핸들러 체인을 설정하고, 요청을 보내어 처리하는 클라이언트 코드이다.  

### 출력 결과  

```
일반 문의 처리: 고객 상담 부서에서 처리합니다.
기술 지원 처리: 기술 지원팀에서 처리합니다.
재무 문제 처리: 재무팀에서 처리합니다.
```  

## 책임 연쇄 패턴의 장점  

1. 객체 간의 결합도 감소: 클라이언트는 어떤 객체가 요청을 처리할지 알 필요가 없으며, 요청을 처리할 객체들이 자동으로 결정된다.
2. 유연한 요청 처리: 요청의 처리 순서와 흐름을 쉽게 변경하거나 확장할 수 있다. 새로운 요청 처리 객체를 추가하는 것도 매우 용이하다.  
3. 동적 체인 구성: 체인의 구성을 동적으로 변경할 수 있어 다양한 요청 처리 흐름을 유연하게 설계할 수 있다.  

## 책임 연쇄 패턴의 단점  

1. 요청의 처리 지연: 체인에서 여러 객체를 거치며 요청이 처리되기 때문에, 처리 속도가 느려질 수 있다.  
2. 체인이 너무 길어질 수 있음: 체인이 너무 길어지면 처리할 객체들이 많아져 비효율적일 수 있다.  



### 마무리  

책임 연쇄 패턴은 여러 객체들이 요청을 처리할 기회를 가지는 패턴으로, 객체 간의 결합도를 줄이고 요청 처리 과정을 유연하게 만들 수 있다.  
각 객체는 자신이 처리할 수 있는 요청만 처리하고, 그렇지 않으면 다음 객체로 넘기며, 클라이언트는 어느 객체가 요청을 처리할지 알 필요 없이 체인만 설정하면 된다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design/patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  