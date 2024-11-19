---
layout: single
title:  "디자인 패턴 시리즈 9. 탬플릿"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 9: 템플릿 메서드 패턴 (Template Method Pattern)  

템플릿 메서드 패턴은 어떤 작업을 처리하는 일련의 과정을 상위 클래스에서 정의하고, 그 과정의 일부를 하위 클래스에서 구체화하는 패턴이다.  
즉, 기본적인 작업 흐름은 상위 클래스에서 제공하면서도, 세부적인 부분은 하위 클래스에서 재정의할 수 있게 한다.  
이로써 코드 재사용성이 높아지고, 동일한 처리 과정을 가진 다양한 구현이 가능해진다.  

## 템플릿 메서드 패턴의 필요성  

템플릿 메서드 패턴은코드 중복을 줄이고, 알고리즘의 구조를 재사용해야 할 때 유용하다.  
비슷한 절차를 따르는 여러 클래스가 있다면, 공통된 로직은 상위 클래스에 두고, 차별화된 부분만 하위 클래스에서 구현함으로써 유지보수성을 높일 수 있다.  

### 예시: 보고서 작성 시스템  

보고서를 작성하는 과정은 기본적으로 비슷할 수 있지만, 각 보고서의 형식이나 내용은 달라질 수 있다.  
예를 들어, PDF 보고서와 HTML 보고서는 작성 흐름은 동일하나, 최종 출력 형식이 다를 것이다.  
이처럼 공통의 처리 흐름을 유지하면서 구체적인 부분만 다르게 구현해야 할 때 템플릿 메서드 패턴을 사용할 수 있다.  

## 템플릿 메서드 패턴의 구조  

1. AbstractClass: 템플릿 메서드를 정의하는 상위 클래스이다. 템플릿 메서드는 알고리즘의 골격을 제공하며, 일부는 추상 메서드로 하위 클래스에서 구현한다.  
2. ConcreteClass: AbstractClass에서 정의된 템플릿 메서드를 구체화하여, 각기 다른 세부 사항을 구현하는 하위 클래스들이다.  

### 구조 다이어그램  

```
AbstractClass
   ├─ templateMethod()
   ├─ primitiveOperation1()
   └─ primitiveOperation2()

ConcreteClass1
   └─ primitiveOperation1()
   └─ primitiveOperation2()

ConcreteClass2
   └─ primitiveOperation1()
   └─ primitiveOperation2()
```  

### 템플릿 메서드 패턴 동작 순서  

1. 상위 클래스에서 기본 알고리즘 정의: 상위 클래스는 기본 처리 절차(알고리즘)를 정의하고, 그 과정에서 필요한 일부 메서드를 추상화한다.  
2. 하위 클래스에서 구체적인 행동 구현: 하위 클래스는 추상 메서드를 구체적으로 구현하여, 각 클래스마다 다른 세부적인 처리 로직을 제공한다.  

## 템플릿 메서드 패턴 예시 (Template Method Pattern)  

이번 예시에서는 보고서 작성 시스템을 구현해보겠다.  
보고서를 작성하는 과정은 동일하지만, PDF 형식의 보고서와 HTML 형식의 보고서가 있을 수 있다.  
템플릿 메서드 패턴을 사용하여 보고서 작성 과정은 동일하게 유지하면서도, 출력 형식에 따라 각기 다른 보고서를 생성해보겠다.  

### Java로 템플릿 메서드 패턴 구현하기  

```java
// 추상 클래스 (AbstractClass)
abstract class ReportTemplate {
    // 템플릿 메서드: 보고서 작성 과정
    public final void generateReport() {
        gatherData();
        analyzeData();
        formatReport();
        printReport();
    }

    // 데이터 수집: 모든 보고서에 공통
    private void gatherData() {
        System.out.println("데이터를 수집합니다.");
    }

    // 데이터 분석: 모든 보고서에 공통
    private void analyzeData() {
        System.out.println("데이터를 분석합니다.");
    }

    // 보고서 형식 지정: 하위 클래스에서 구현
    protected abstract void formatReport();

    // 보고서 출력: 하위 클래스에서 구현
    protected abstract void printReport();
}

// PDF 보고서 (ConcreteClass)
class PDFReport extends ReportTemplate {
    @Override
    protected void formatReport() {
        System.out.println("PDF 형식으로 보고서를 작성합니다.");
    }

    @Override
    protected void printReport() {
        System.out.println("PDF 보고서를 출력합니다.");
    }
}

// HTML 보고서 (ConcreteClass)
class HTMLReport extends ReportTemplate {
    @Override
    protected void formatReport() {
        System.out.println("HTML 형식으로 보고서를 작성합니다.");
    }

    @Override
    protected void printReport() {
        System.out.println("HTML 보고서를 출력합니다.");
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        ReportTemplate pdfReport = new PDFReport();
        pdfReport.generateReport(); // PDF 보고서 작성

        System.out.println();

        ReportTemplate htmlReport = new HTMLReport();
        htmlReport.generateReport(); // HTML 보고서 작성
    }
}
```  

### 코드 설명  

1. ReportTemplate (AbstractClass): `generateReport()` 메서드는 보고서를 작성하는 기본 알고리즘을 제공한다. 데이터 수집과 분석은 공통 로직이므로 상위 클래스에서 구현하고, `formatReport()`와 `printReport()`는 각기 다른 보고서 형식에 맞게 하위 클래스에서 구현하도록 했다.  
2. PDFReport (ConcreteClass): PDF 형식의 보고서를 생성하며, `formatReport()`와 `printReport()` 메서드를 PDF에 맞게 구현했다.  
3. HTMLReport (ConcreteClass): HTML 형식의 보고서를 생성하며, HTML에 맞는 형식과 출력 로직을 구현했다.  

### 출력 결과  

```
데이터를 수집합니다.
데이터를 분석합니다.
PDF 형식으로 보고서를 작성합니다.
PDF 보고서를 출력합니다.

데이터를 수집합니다.
데이터를 분석합니다.
HTML 형식으로 보고서를 작성합니다.
HTML 보고서를 출력합니다.
```  

## 템플릿 메서드 패턴의 장점  

1. 코드 재사용성: 공통적인 알고리즘 흐름을 상위 클래스에서 정의하고, 중복 코드를 줄일 수 있다.  
2. 유연성: 하위 클래스에서 세부 사항을 구현함으로써 유연한 구조를 제공한다. 하위 클래스는 상위 클래스의 템플릿 메서드를 변경하지 않고, 그 과정에서 필요한 메서드만 재정의한다.  
3. 알고리즘의 분리: 기본적인 처리 흐름과 세부적인 구현을 분리하여, 변경이 용이하고 가독성이 높아진다.  

## 템플릿 메서드 패턴의 단점  

1. 상속을 통한 구현 의존성: 템플릿 메서드 패턴은 상속을 기반으로 하기 때문에, 상속에 의한 의존성이 생길 수 있다.  
2. 추상 클래스 설계의 어려움: 상위 클래스에서 정의해야 할 알고리즘의 골격과, 하위 클래스에서 구현해야 할 부분을 명확하게 구분해야 하므로 설계가 까다로울 수 있다.  

### 마무리  

템플릿 메서드 패턴은 동일한 처리 절차를 가진 다양한 구현을 제공하고자 할 때 유용한 패턴이다.  
알고리즘의 기본 흐름을 상위 클래스에서 제공하고, 하위 클래스는 그 흐름을 따르면서 구체적인 동작만을 재정의함으로써 코드의 유연성과 재사용성을 높일 수 있다.  
특히, 공통적인 로직을 많이 사용하는 시스템에서 효과적으로 사용할 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design/patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  