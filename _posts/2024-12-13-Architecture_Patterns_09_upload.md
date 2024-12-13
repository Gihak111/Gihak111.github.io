---
layout: single
title:  "아키텍처 패턴 시리즈 9. MVP 패턴 (Model-View-Presenter Pattern)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 9: MVP 패턴 (Model-View-Presenter Pattern)

MVP 패턴(Model-View-Presenter Pattern)은 애플리케이션을 모델(Model), 뷰(View), 프리젠터(Presenter) 세 가지 주요 컴포넌트로 분리하여 유저 인터페이스와 비즈니스 로직을 분리하는 아키텍처 패턴이다.  
특히 모바일 애플리케이션과 데스크톱 애플리케이션에서 자주 사용된다.  

## MVP 패턴의 필요성

MVP 패턴은 UI와 비즈니스 로직의 결합을 줄여 코드의 재사용성과 유지보수성을 높인다.  
MVC 패턴의 단점을 보완한 구조로, 컨트롤러 대신 프리젠터가 UI 논리를 담당한다.  

1. UI와 비즈니스 로직 분리: 뷰와 비즈니스 로직의 분리로 가독성과 테스트 용이성이 향상된다.  
2. 테스트 가능성 향상: 프리젠터는 뷰와 독립적이기 때문에 단위 테스트가 쉽다.  
3. 유연성 증가: 뷰와 프리젠터의 의존성을 최소화하여 재사용 가능성이 높아진다.  

MVP 패턴은 특히 비즈니스 로직이 복잡하고 다양한 뷰를 지원해야 하는 애플리케이션에서 효과적이다.  

### 예시: 메모 애플리케이션

사용자가 메모를 작성하고 저장하거나 삭제하는 애플리케이션에서 MVP 패턴을 적용할 수 있다.  

## MVP 패턴의 구조

1. Model (모델): 데이터와 비즈니스 로직을 처리한다.  
2. View (뷰): 사용자와 상호작용하며 데이터를 시각적으로 보여준다.  
3. Presenter (프리젠터): 모델과 뷰 사이의 중개자 역할을 하며 UI 논리를 담당한다.  

### 구조 다이어그램

```
[View] <--> [Presenter] <--> [Model]
```

### MVP 패턴 동작 순서

1. 사용자가 View를 통해 요청을 보낸다.  
2. Presenter가 요청을 받아 적절한 Model을 호출하고 데이터를 처리한다.  
3. Presenter가 데이터를 View에 전달하여 사용자에게 보여준다.  

## MVP 패턴 예시

메모 애플리케이션에서 메모를 관리하고 보여주는 예제를 Java로 구현할 수 있다.  

### Java로 MVP 패턴 구현하기

```java
// Note 모델 클래스: 메모 데이터를 관리하는 클래스
public class Note {
    private String content;

    public Note() {}

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }
}
```

```java
// NoteView 인터페이스: 사용자에게 메모 데이터를 보여주는 뷰 역할
public interface NoteView {
    void showNoteContent(String content);
    void showSaveSuccessMessage();
    void showError(String errorMessage);
}
```

```java
// NotePresenter 클래스: 모델과 뷰를 연결하는 프리젠터 역할
public class NotePresenter {
    private Note model;
    private NoteView view;

    public NotePresenter(Note model, NoteView view) {
        this.model = model;
        this.view = view;
    }

    public void loadNote() {
        String content = model.getContent();
        if (content != null && !content.isEmpty()) {
            view.showNoteContent(content);
        } else {
            view.showError("메모 내용이 비어 있습니다.");
        }
    }

    public void saveNoteContent(String content) {
        if (content == null || content.trim().isEmpty()) {
            view.showError("메모 내용을 입력해야 합니다.");
        } else {
            model.setContent(content);
            view.showSaveSuccessMessage();
        }
    }
}
```

```java
// ConsoleNoteView 클래스: 콘솔 기반으로 사용자에게 데이터를 출력하는 뷰 구현
public class ConsoleNoteView implements NoteView {
    @Override
    public void showNoteContent(String content) {
        System.out.println("메모 내용: " + content);
    }

    @Override
    public void showSaveSuccessMessage() {
        System.out.println("메모가 성공적으로 저장되었습니다!");
    }

    @Override
    public void showError(String errorMessage) {
        System.out.println("오류: " + errorMessage);
    }
}
```

```java
// Main 클래스: MVP 패턴을 사용하여 메모 데이터를 관리하는 예시
public class Main {
    public static void main(String[] args) {
        // 초기 데이터 설정
        Note model = new Note();
        NoteView view = new ConsoleNoteView();
        NotePresenter presenter = new NotePresenter(model, view);

        // 메모 저장 및 출력
        presenter.saveNoteContent("MVP 패턴 학습하기");
        presenter.loadNote();

        // 빈 메모 저장 시도
        presenter.saveNoteContent("");
    }
}
```

### 출력 결과

```
메모가 성공적으로 저장되었습니다!
메모 내용: MVP 패턴 학습하기
오류: 메모 내용을 입력해야 합니다.
```

### 코드 설명

1. Note (Model): 메모 데이터를 관리한다.  
2. NoteView (View): 메모 내용을 사용자에게 보여준다.  
3. NotePresenter (Presenter): 사용자의 요청을 처리하고 모델과 뷰를 연결한다.  

### MVP 패턴 활용

1. 모바일 애플리케이션: 안드로이드와 같은 플랫폼에서 UI 논리를 쉽게 테스트하고 관리할 수 있다.  
2. 데스크톱 애플리케이션: 사용자 인터페이스가 복잡한 애플리케이션에서 사용 가능하다.  
3. 데이터 시각화 애플리케이션: 데이터 처리와 시각화 로직을 분리하여 재사용성과 유지보수성을 높인다.  

## MVP 패턴의 장점

1. 테스트 용이성: 프리젠터와 뷰가 분리되어 있어 단위 테스트가 쉽다.  
2. 모듈화: 뷰, 프리젠터, 모델이 분리되어 코드의 재사용성과 유지보수성이 높아진다.  
3. 유연성: 뷰 또는 프리젠터를 쉽게 교체할 수 있다.  

## MVP 패턴의 단점

1. 복잡도 증가: 간단한 애플리케이션에서는 불필요하게 복잡해질 수 있다.  
2. 의존성 증가: 프리젠터와 뷰 간의 상호작용이 많아 의존성 관리가 필요하다.  
3. 코드 중복: 여러 뷰에서 비슷한 로직을 사용할 경우 프리젠터 코드 중복이 발생할 수 있다.  

### 마무리

MVP 패턴은 UI와 비즈니스 로직이 분리된 아키텍처로, 유지보수성과 테스트 용이성이 중요한 프로젝트에 적합하다.   

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
