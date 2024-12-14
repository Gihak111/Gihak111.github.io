---
layout: single
title:  "아키텍처 패턴 시리즈 10. MVVM 패턴 (Model-View-ViewModel Pattern)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 10: MVVM 패턴 (Model-View-ViewModel Pattern)

MVVM 패턴(Model-View-ViewModel Pattern)은 애플리케이션의 UI 로직을 모델(Model), 뷰(View), 뷰모델(ViewModel)로 분리하여 코드의 재사용성과 테스트 용이성을 향상시키는 아키텍처 패턴이다.  
특히 데이터 바인딩(Data Binding)을 지원하는 프레임워크에서 효과적이다.  

## MVVM 패턴의 필요성

MVVM 패턴은 대규모 애플리케이션에서 UI와 비즈니스 로직의 결합을 줄이고 코드의 가독성과 유지보수성을 높인다.  
특히 데이터 바인딩을 통해 뷰(View)와 뷰모델(ViewModel)의 상호작용을 자동화할 수 있다.  

1. UI와 데이터 로직 분리: 비즈니스 로직과 UI 로직을 명확히 구분한다.  
2. 데이터 바인딩 활용: 뷰와 뷰모델 간의 데이터 동기화를 자동화한다.  
3. 재사용성 증가: 모델과 뷰모델이 독립적이므로 여러 뷰에서 재사용 가능하다.  

MVVM 패턴은 리액트(React), 앵귤러(Angular), WPF(Windows Presentation Foundation), 안드로이드 같은 프레임워크에서 흔히 사용된다.  

### 예시: 할 일 관리 애플리케이션

사용자가 할 일을 추가하거나 삭제하는 애플리케이션에서 MVVM 패턴을 활용할 수 있다.  

## MVVM 패턴의 구조

1. Model (모델): 데이터 및 비즈니스 로직을 처리한다.  
2. View (뷰): 사용자와 상호작용하며 데이터를 시각적으로 보여준다.  
3. ViewModel (뷰모델): 뷰와 모델을 연결하며 UI 로직과 상태를 관리한다.  

### 구조 다이어그램

```
[View] <--> [ViewModel] <--> [Model]
```

### MVVM 패턴 동작 순서

1. 사용자가 View를 통해 액션을 트리거한다.  
2. ViewModel이 사용자 요청을 처리하여 Model에 작업을 요청한다.  
3. Model이 데이터를 반환하면 ViewModel이 이를 View에 전달한다.  
4. 데이터 바인딩을 통해 View는 UI를 자동으로 업데이트한다.  

## MVVM 패턴 예시

할 일 관리 애플리케이션에서 할 일 목록을 표시하고 관리하는 기능을 구현해 보자.  

### Java로 MVVM 패턴 구현하기

```java
// Task 모델 클래스: 할 일 데이터를 관리하는 클래스
public class Task {
    private String title;

    public Task(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }
}
```

```java
// TaskViewModel 클래스: 할 일 목록과 사용자 액션을 처리하는 뷰모델 역할
import java.util.ArrayList;
import java.util.List;

public class TaskViewModel {
    private List<Task> tasks = new ArrayList<>();

    public List<Task> getTasks() {
        return tasks;
    }

    public void addTask(String title) {
        if (title == null || title.trim().isEmpty()) {
            System.out.println("오류: 할 일 제목을 입력해야 합니다.");
            return;
        }
        tasks.add(new Task(title));
        System.out.println("할 일이 추가되었습니다: " + title);
    }

    public void removeTask(String title) {
        tasks.removeIf(task -> task.getTitle().equals(title));
        System.out.println("할 일이 삭제되었습니다: " + title);
    }
}
```

```java
// TaskView 클래스: 사용자와 상호작용하며 데이터를 출력하는 뷰 역할
public class TaskView {
    private TaskViewModel viewModel;

    public TaskView(TaskViewModel viewModel) {
        this.viewModel = viewModel;
    }

    public void displayTasks() {
        List<Task> tasks = viewModel.getTasks();
        System.out.println("할 일 목록:");
        for (Task task : tasks) {
            System.out.println("- " + task.getTitle());
        }
    }
}
```

```java
// Main 클래스: MVVM 패턴으로 할 일을 관리하는 예시
public class Main {
    public static void main(String[] args) {
        TaskViewModel viewModel = new TaskViewModel();
        TaskView view = new TaskView(viewModel);

        // 할 일 추가
        viewModel.addTask("MVVM 패턴 학습");
        viewModel.addTask("Java 프로젝트 완료");
        view.displayTasks();

        // 할 일 삭제
        viewModel.removeTask("Java 프로젝트 완료");
        view.displayTasks();
    }
}
```

### 출력 결과

```
할 일이 추가되었습니다: MVVM 패턴 학습
할 일이 추가되었습니다: Java 프로젝트 완료
할 일 목록:
- MVVM 패턴 학습
- Java 프로젝트 완료
할 일이 삭제되었습니다: Java 프로젝트 완료
할 일 목록:
- MVVM 패턴 학습
```

### 코드 설명

1. Task (Model): 할 일 데이터를 관리한다.  
2. TaskViewModel (ViewModel): 사용자 요청을 처리하고 모델과 뷰 간의 데이터 로직을 연결한다.  
3. TaskView (View): 데이터를 사용자에게 보여준다.  

### MVVM 패턴 활용

1. 모바일 애플리케이션: 안드로이드에서 데이터 바인딩을 통해 효율적인 UI 업데이트를 구현할 수 있다.  
2. 데스크톱 애플리케이션: WPF와 같은 플랫폼에서 MVVM을 활용해 UI와 비즈니스 로직을 분리한다.  
3. 웹 애플리케이션: 리액트와 같은 라이브러리에서 MVVM 구조를 활용할 수 있다.  

## MVVM 패턴의 장점

1. 테스트 용이성: 뷰모델과 모델이 독립적이어서 단위 테스트가 쉽다.  
2. 코드 재사용성: 뷰모델과 모델이 여러 뷰에서 재사용 가능하다.  
3. 유지보수성: 데이터와 UI 로직이 분리되어 유지보수가 용이하다.  
4. 데이터 바인딩 지원: 뷰와 뷰모델 간의 데이터 동기화가 자동으로 이루어진다.  

## MVVM 패턴의 단점

1. 초기 복잡도 증가: 데이터 바인딩과 구조 분리가 간단한 프로젝트에서는 불필요할 수 있다.  
2. 의존성: 데이터 바인딩 라이브러리에 의존하는 경우가 많다.  
3. 학습 곡선: 초보 개발자에게는 뷰모델의 역할과 데이터 바인딩의 동작 원리가 복잡하게 느껴질 수 있다.  

### 마무리

MVVM 패턴은 데이터 바인딩을 적극 활용하여 UI와 비즈니스 로직을 효과적으로 분리하는 패턴이다.  
효율적인 아키텍처 설계를 위해 MVVM 패턴을 고려해보자.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
