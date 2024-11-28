---
layout: single
title:  "디자인 패턴 시리즈 18. 복합체"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 18: 복합체 패턴 (Composite Pattern)

복합체 패턴(Composite Pattern)은 구조적 디자인 패턴으로, 객체를 트리 구조로 구성하여 개별 객체와 객체 그룹을 동일하게 취급할 수 있도록 한다.  
이를 통해 클라이언트는 단일 객체와 복합 객체를 구별하지 않고 다룰 수 있다.  

## 복합체 패턴의 필요성

소프트웨어 개발에서 개별 요소와 복합 요소를 처리하는 로직이 각각 다르다면 코드 중복과 복잡성이 증가한다.  
복합체 패턴을 사용하면 다음과 같은 이점을 얻을 수 있다:  

1. 객체 트리 구조 표현: 계층적인 데이터를 구조적으로 표현할 수 있다.  
2.일관성 제공: 단일 객체와 복합 객체를 동일하게 처리할 수 있다.    
3. 유지보수성 향상: 구조 변경 시 영향을 최소화한다.  

### 예시: 파일 시스템  

파일 시스템에서 파일과 폴더는 모두 `열기`, `복사`, `삭제` 같은 공통 동작을 가진다.  
복합체 패턴을 사용하면 파일과 폴더를 동일한 인터페이스로 처리할 수 있다.  

## 복합체 패턴의 구조

1. Component(구성 요소): 개별 객체와 복합 객체에 공통되는 인터페이스를 정의한다.  
2. Leaf(잎): 단일 객체로, 실제 작업을 수행하는 클래스.  
3. Composite(복합체): 다른 Leaf와 Composite 객체를 포함하는 컨테이너 역할을 하는 클래스.  

### 구조 다이어그램  

```
Component
    ↑
  Composite ←──── Leaf
``` 

### 복합체 패턴 동작 순서   

1. 클라이언트는 Component를 통해 작업을 요청한다.  
2. Leaf는 요청을 직접 처리한다.  
3. Composite는 요청을 자식 객체(Leaf 또는 Composite)로 전달한다.  

## 복합체 패턴 예시  

이번 예시에서는 "파일과 폴더의 트리 구조"를 복합체 패턴으로 구현해보겠다.  

### Java로 복합체 패턴 구현하기  

```java
// Component 인터페이스
interface FileSystemComponent {
    void display();
}

// Leaf 클래스
class File implements FileSystemComponent {
    private String name;

    public File(String name) {
        this.name = name;
    }

    @Override
    public void display() {
        System.out.println("파일: " + name);
    }
}

// Composite 클래스
class Folder implements FileSystemComponent {
    private String name;
    private List<FileSystemComponent> components = new ArrayList<>();

    public Folder(String name) {
        this.name = name;
    }

    public void add(FileSystemComponent component) {
        components.add(component);
    }

    public void remove(FileSystemComponent component) {
        components.remove(component);
    }

    @Override
    public void display() {
        System.out.println("폴더: " + name);
        for (FileSystemComponent component : components) {
            component.display();
        }
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        FileSystemComponent file1 = new File("문서1.txt");
        FileSystemComponent file2 = new File("문서2.txt");
        FileSystemComponent file3 = new File("이미지.png");

        Folder folder1 = new Folder("문서 폴더");
        folder1.add(file1);
        folder1.add(file2);

        Folder rootFolder = new Folder("루트 폴더");
        rootFolder.add(folder1);
        rootFolder.add(file3);

        // 전체 파일 시스템 출력
        rootFolder.display();
    }
}
```  

### 코드 설명

1. FileSystemComponent (Component): 파일과 폴더가 공통적으로 구현해야 할 인터페이스.  
2. File (Leaf): 단일 파일 객체로, 실제 파일 작업을 수행한다.  
3. Folder (Composite): 파일과 폴더를 포함하는 컨테이너 역할을 하는 클래스.  
4. Main (Client): 파일과 폴더를 동일한 방식으로 처리한다.  

### 출력 결과

```
폴더: 루트 폴더
폴더: 문서 폴더
파일: 문서1.txt
파일: 문서2.txt
파일: 이미지.png
```  

## 복합체 패턴의 장점  

1. 단순화된 클라이언트 코드: 개별 객체와 복합 객체를 동일하게 처리할 수 있다.  
2. 유연성: 객체 추가 및 삭제가 쉽다.  
3. 트리 구조 표현: 복잡한 계층 구조를 명확하게 표현할 수 있다.  

## 복합체 패턴의 단점  

1. 클래스 설계 복잡성 증가: 클래스 계층이 많아질 수 있다.  
2. 단일 책임 원칙 위반 가능성: Composite 클래스가 너무 많은 책임을 가질 수 있다.  

### 마무리   

복합체 패턴(Composite Pattern)은 개별 객체와 복합 객체를 동일하게 처리할 수 있는 강력한 구조적 패턴이다.  
특히, 계층적 데이터를 다룰 때 유용하며, 클라이언트 코드를 단순화할 수 있다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)   