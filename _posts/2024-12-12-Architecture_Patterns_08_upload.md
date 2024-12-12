---
layout: single
title:  "아키텍처 패턴 시리즈 8. MVC 패턴 (Model-View-Controller Pattern)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 8: MVC 패턴 (Model-View-Controller Pattern)

MVC 패턴(Model-View-Controller Pattern)은 애플리케이션을 모델(Model), 뷰(View), 컨트롤러(Controller) 세 가지 주요 컴포넌트로 분리하여 유저 인터페이스와 애플리케이션 로직을 분리하는 아키텍처 패턴이다.  
MVC 패턴은 특히 웹 애플리케이션과 데스크톱 애플리케이션에서 주로 사용된다.  

## MVC 패턴의 필요성  

MVC 패턴은 대규모 애플리케이션에서 복잡한 유저 인터페이스와 비즈니스 로직을 효율적으로 관리하고 모듈화를 가능하게 한다.  

1. 유저 인터페이스와 로직 분리: UI와 로직을 분리하여 코드의 가독성과 유지 보수성을 높인다.  
2. 코드 재사용성 증가: 모델과 컨트롤러 로직의 재사용이 용이해진다.  
3. 테스트 용이성: UI와 로직이 분리되므로 테스트가 쉬워진다.  

MVC 패턴은 데이터 기반 애플리케이션, 웹 애플리케이션, 모바일 앱 등에서 활용되며, 특히 UI와 비즈니스 로직의 변경 사항이 자주 발생하는 경우에 유리하다.  

### 예시: 온라인 상점 애플리케이션

온라인 상점에서 상품의 데이터를 관리하고 이를 사용자에게 보여주며, 사용자가 특정 상품을 클릭하면 상세 정보를 불러오는 구조에서 MVC 패턴이 사용될 수 있다.  

## MVC 패턴의 구조

1. Model (모델): 데이터와 관련된 비즈니스 로직을 처리한다.  
2. View (뷰): 사용자에게 데이터를 시각적으로 보여주는 역할을 한다.  
3. Controller (컨트롤러): 사용자 요청을 받고, 적절한 모델을 호출하고, 뷰에 데이터를 전달한다.  

### 구조 다이어그램

```
[View] <------> [Controller] <------> [Model]
```  

### MVC 패턴 동작 순서

1. 사용자가 View를 통해 요청을 보낸다.  
2. Controller가 요청을 받아 적절한 Model을 호출하여 데이터를 가져온다.  
3. Controller는 데이터를 View에 전달하여 사용자에게 보여준다.  

## MVC 패턴 예시

온라인 상점 애플리케이션에서 상품의 데이터를 관리하고 보여주는 예제를 Java로 구현할 수 있다.  

### Java로 MVC 패턴 구현하기

```java
// Product 모델 클래스: 상품 데이터를 관리하는 클래스
public class Product {
    private String name;
    private double price;

    public Product(String name, double price) {
        this.name = name;
        this.price = price;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }
}
```

```java
// ProductView 클래스: 사용자에게 데이터를 시각적으로 보여주는 역할
public class ProductView {
    public void displayProductDetails(String productName, double productPrice) {
        System.out.println("상품명: " + productName);
        System.out.println("가격: $" + productPrice);
    }
}
```

```java
// ProductController 클래스: 사용자의 요청을 처리하고, 모델과 뷰를 연결하는 역할
public class ProductController {
    private Product model;
    private ProductView view;

    public ProductController(Product model, ProductView view) {
        this.model = model;
        this.view = view;
    }

    public void setProductName(String name) {
        model = new Product(name, model.getPrice());
    }

    public void setProductPrice(double price) {
        model = new Product(model.getName(), price);
    }

    public void updateView() {
        view.displayProductDetails(model.getName(), model.getPrice());
    }
}
```

```java
// Main 클래스: MVC 패턴을 사용하여 상품 데이터를 설정하고 출력하는 예시
public class Main {
    public static void main(String[] args) {
        // 초기 데이터 설정
        Product product = new Product("노트북", 1200.00);
        ProductView view = new ProductView();

        // Controller 생성
        ProductController controller = new ProductController(product, view);

        // 초기 상품 정보 출력
        controller.updateView();

        // 상품 정보 업데이트
        controller.setProductName("게이밍 노트북");
        controller.setProductPrice(1500.00);

        // 업데이트된 상품 정보 출력
        controller.updateView();
    }
}
```

### 코드 설명

1. Product (Model): 상품의 이름과 가격을 관리하는 모델 클래스.  
2. ProductView (View): 상품의 데이터를 화면에 출력하는 뷰 클래스.  
3. ProductController (Controller): 모델과 뷰를 연결하고, 데이터를 업데이트하며 뷰에 데이터를 전달한다.  

### 출력 결과

```
상품명: 노트북
가격: $1200.0

상품명: 게이밍 노트북
가격: $1500.0
```

### MVC 패턴 활용

1. 온라인 상점: 상품 데이터와 같은 정보를 관리하고 사용자에게 보여주는 다양한 애플리케이션에서 사용 가능하다.  
2. 대시보드 애플리케이션: 사용자에게 실시간 데이터를 보여주는 애플리케이션에서 사용 가능하다.  
3. 모바일 애플리케이션: 다양한 비즈니스 로직과 UI가 결합된 모바일 앱에서 활용 가능하다.  

## MVC 패턴의 장점

1. 유지 보수성: 모델, 뷰, 컨트롤러가 분리되어 있어 유지보수가 용이하다.  
2. 유연성: 특정 컴포넌트(View 또는 Model)만 교체 가능하여 유연성이 높다.  
3. 테스트 용이성: 비즈니스 로직(Model)과 UI(View)가 분리되어 있어 단위 테스트가 용이하다.  

## MVC 패턴의 단점

1. 복잡도 증가: 소규모 애플리케이션에 적용하면 과도한 구조가 될 수 있다.  
2. 이벤트 흐름 관리 어려움: Controller가 여러 모델과 뷰를 관리하는 경우 이벤트 흐름이 복잡해질 수 있다.  
3. 의존성 증가: Controller가 모델과 뷰에 의존하기 때문에 의존성 관리가 필요하다.  

### 마무리

MVC 패턴은 다양한 UI와 비즈니스 로직을 갖춘 애플리케이션에서 유용하게 쓰이며, 코드의 유지 보수성과 확장성을 높여준다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
