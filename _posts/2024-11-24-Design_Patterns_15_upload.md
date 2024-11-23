---
layout: single
title:  "디자인 패턴 시리즈 15. 메멘토"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 15: 메멘토 패턴 (Memento Pattern)

메멘토 패턴(Memento Pattern)은 객체의 상태를 저장하고 필요할 때 해당 상태를 복원할 수 있도록 하는 행동 패턴이다.  
주로 "실행 취소(Undo)" 기능을 구현하는 데 사용된다.  

이 패턴은 캡슐화를 유지하면서도 객체의 내부 상태를 외부에 노출하지 않고 상태를 저장하고 복원할 수 있는 구조를 제공한다.  

## 메멘토 패턴의 필요성

어떤 객체가 변경되었을 때, 이전 상태로 되돌려야 하는 상황이 종종 발생한다.  
예를 들어, 텍스트 편집기에서 "Ctrl + Z"를 누르면 이전 작업 상태로 되돌리는 기능을 생각할 수 있다.  

메멘토 패턴을 사용하면 다음과 같은 장점을 얻을 수 있다:

1. 이전 상태 복원: 객체의 이전 상태를 저장하여 필요 시 복원할 수 있다.  
2. 캡슐화 유지: 객체의 내부 구조를 외부에 노출하지 않고 상태를 저장한다.  
3. 변경 추적: 상태 변경 이력을 관리하기 쉽다.  

### 예시: 텍스트 편집기

텍스트 편집기에서 사용자가 작업 중인 내용을 "저장"하거나 "실행 취소"할 수 있는 상황을 생각해보자.  
이전 상태를 저장하고 되돌리는 메멘토 패턴을 적용하여 구현할 수 있다.

## 메멘토 패턴의 구조

1. Memento(메멘토): 객체의 상태를 저장하는 역할을 담당.  
2. Originator(기원자): 저장하거나 복원할 상태를 가지고 있는 객체.  
3. Caretaker(보관자): 메멘토를 관리하며, 저장된 상태를 요청받으면 기원자에게 전달한다.  

### 구조 다이어그램

```
  +---------------+       +---------------+       +---------------+
  |   Originator  |<----->|   Memento     |<----->|   Caretaker   |
  +---------------+       +---------------+       +---------------+
  |  saveState()  |       |  getState()   |       |  addMemento() |
  | restoreState()|       |  setState()   |       | getMemento()  |
  +---------------+       +---------------+       +---------------+
```  

### 메멘토 패턴 동작 순서

1. Originator 객체는 자신의 상태를 Memento 객체에 저장한다.  
2. Caretaker는 Memento 객체를 저장하거나 요청받으면 전달한다.  
3. Originator는 Memento로부터 이전 상태를 복원한다.  

## 메멘토 패턴 예시

이번 예시에서는 텍스트 편집기의 "실행 취소(Undo)" 기능을 구현해보겠다.  

### Java로 메멘토 패턴 구현하기

```java
// Memento 클래스
class Memento {
    private final String state;

    public Memento(String state) {
        this.state = state;
    }

    public String getState() {
        return state;
    }
}

// Originator 클래스
class TextEditor {
    private String content;

    public void setContent(String content) {
        this.content = content;
    }

    public String getContent() {
        return content;
    }

    public Memento save() {
        return new Memento(content);
    }

    public void restore(Memento memento) {
        this.content = memento.getState();
    }
}

// Caretaker 클래스
class History {
    private final Stack<Memento> mementos = new Stack<>();

    public void save(Memento memento) {
        mementos.push(memento);
    }

    public Memento undo() {
        if (!mementos.isEmpty()) {
            return mementos.pop();
        }
        return null;
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        TextEditor editor = new TextEditor();
        History history = new History();

        editor.setContent("안녕하세요.");
        history.save(editor.save()); // 상태 저장
        System.out.println("현재 상태: " + editor.getContent());

        editor.setContent("안녕하세요, 세계!");
        history.save(editor.save()); // 상태 저장
        System.out.println("현재 상태: " + editor.getContent());

        editor.restore(history.undo()); // 상태 복원
        System.out.println("복원된 상태: " + editor.getContent());

        editor.restore(history.undo()); // 상태 복원
        System.out.println("복원된 상태: " + editor.getContent());
    }
}
```

### 출력 결과

```
현재 상태: 안녕하세요.
현재 상태: 안녕하세요, 세계!
복원된 상태: 안녕하세요.
복원된 상태: 안녕하세요.
```  

### 코드 설명

1. Memento: 객체의 상태를 저장한다.  
2. TextEditor (Originator): 저장하거나 복원할 상태를 가진 클래스.  
3. History (Caretaker): Memento 객체를 관리하며, 실행 취소 시 상태를 제공한다.  

## 메멘토 패턴의 장점

1. 캡슐화 유지: Originator의 내부 상태를 외부에 노출하지 않고 안전하게 상태를 저장하고 복원한다.  
2. 유지보수 용이성: 상태 변경 이력을 관리하여 디버깅 및 기능 구현이 용이하다.  
3. 다양한 응용 가능성: 실행 취소, 작업 이력 저장 등 다양한 기능에 활용할 수 있다.  

## 메멘토 패턴의 단점  

1. 메모리 사용 증가: 상태를 저장할 때마다 메멘토 객체가 생성되므로 메모리를 많이 사용할 수 있다.  
2. 복잡도 증가: 상태를 저장하고 복원하는 로직을 설계해야 하므로 코드가 복잡해질 수 있다.  

### 마무리

메멘토 패턴(Memento Pattern)은 객체의 상태를 안전하게 저장하고 복원할 수 있는 강력한 도구다.  
실행 취소 기능이나 변경 이력 관리와 같은 기능을 구현할 때 특히 유용하다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  