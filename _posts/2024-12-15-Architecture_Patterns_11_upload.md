---
layout: single
title:  "아키텍처 패턴 시리즈 11. FLUX 패턴"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 11: FLUX 패턴

FLUX 패턴은 단방향 데이터 흐름을 중심으로 애플리케이션 상태를 관리하기 위해 페이스북에서 설계된 아키텍처 패턴이다.  
특히 React 애플리케이션에서 사용되며, 데이터를 효과적으로 처리하고 UI를 업데이트하는 데 중점을 둔다.  

## FLUX 패턴의 필요성

대규모 애플리케이션에서는 상태 관리와 데이터 흐름이 복잡해질 수 있다.  
FLUX는 단방향 데이터 흐름을 통해 이를 해결한다.  

1. 상태 일관성 유지: 데이터 흐름이 단방향으로 고정되어 상태 관리가 명확하다.  
2. 중앙 집중식 상태 관리: 상태를 중앙에서 관리해 여러 컴포넌트 간 데이터 동기화 문제를 해결한다.  
3. 데이터 변경 추적 용이: 데이터 변경의 출발점을 쉽게 파악할 수 있다.  

### 예시: Todo 애플리케이션

사용자가 할 일을 추가, 수정, 삭제할 수 있는 React 기반 애플리케이션을 FLUX 패턴으로 구현할 수 있다.  

## FLUX 패턴의 구조

FLUX 패턴은 다음 네 가지 주요 컴포넌트로 구성된다.  

1. Action (액션): 상태 변경 요청을 나타내는 객체이다.  
2. Dispatcher (디스패처): 액션을 스토어로 전달하는 중앙 허브 역할을 한다.  
3. Store (스토어): 상태(state)를 관리하고 변경 사항을 뷰(View)에 전달한다.
4. View (뷰): 사용자와 상호작용하며 스토어의 상태를 시각적으로 표현한다.  

### 구조 다이어그램

```
[Action] -> [Dispatcher] -> [Store] -> [View]
                    ^---------------------|
```  

### FLUX 패턴 동작 순서

1. 사용자가 View를 통해 액션(예: 버튼 클릭)을 트리거한다.  
2. Action 객체가 생성되어 Dispatcher로 전달된다.  
3. Dispatcher가 액션을 Store로 전달한다.  
4. Store는 상태를 변경하고 View에 변경 사항을 알린다.  
5. View는 상태를 반영하여 UI를 업데이트한다.  

## FLUX 패턴 예시

React 애플리케이션에서 FLUX 패턴을 적용해 Todo 관리 기능을 구현해 보자.  

### FLUX 컴포넌트 구현

#### Action (액션)

```javascript
// actions/TodoActions.js
const TodoActions = {
    addTodo: function (title) {
        return {
            type: "ADD_TODO",
            payload: title,
        };
    },
    removeTodo: function (id) {
        return {
            type: "REMOVE_TODO",
            payload: id,
        };
    },
};

export default TodoActions;
```

#### Dispatcher (디스패처)

```javascript
// dispatcher/AppDispatcher.js
import { Dispatcher } from "flux";

const AppDispatcher = new Dispatcher();

export default AppDispatcher;
```

#### Store (스토어)

```javascript
// stores/TodoStore.js
import { EventEmitter } from "events";
import AppDispatcher from "../dispatcher/AppDispatcher";

const TodoStore = Object.assign({}, EventEmitter.prototype, {
    todos: [],

    getTodos: function () {
        return this.todos;
    },

    addTodo: function (title) {
        const id = Date.now();
        this.todos.push({ id, title });
        this.emit("change");
    },

    removeTodo: function (id) {
        this.todos = this.todos.filter((todo) => todo.id !== id);
        this.emit("change");
    },
});

// Dispatcher에 스토어 등록
AppDispatcher.register(function (action) {
    switch (action.type) {
        case "ADD_TODO":
            TodoStore.addTodo(action.payload);
            break;
        case "REMOVE_TODO":
            TodoStore.removeTodo(action.payload);
            break;
        default:
            break;
    }
});

export default TodoStore;
```

#### View (뷰)

```javascript
// components/TodoApp.js
import React, { useState, useEffect } from "react";
import TodoActions from "../actions/TodoActions";
import TodoStore from "../stores/TodoStore";

const TodoApp = () => {
    const [todos, setTodos] = useState([]);

    // 스토어의 변경 사항을 감지
    useEffect(() => {
        const updateTodos = () => setTodos(TodoStore.getTodos());
        TodoStore.on("change", updateTodos);

        return () => {
            TodoStore.removeListener("change", updateTodos);
        };
    }, []);

    const handleAddTodo = () => {
        const title = prompt("할 일을 입력하세요:");
        if (title) {
            AppDispatcher.dispatch(TodoActions.addTodo(title));
        }
    };

    const handleRemoveTodo = (id) => {
        AppDispatcher.dispatch(TodoActions.removeTodo(id));
    };

    return (
        <div>
            <h1>Todo List</h1>
            <button onClick={handleAddTodo}>할 일 추가</button>
            <ul>
                {todos.map((todo) => (
                    <li key={todo.id}>
                        {todo.title}{" "}
                        <button onClick={() => handleRemoveTodo(todo.id)}>삭제</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default TodoApp;
```

### 출력 결과

- 사용자는 "할 일을 추가" 버튼을 눌러 새로운 할 일을 추가할 수 있다.  
- 할 일을 삭제하려면 "삭제" 버튼을 누르면 된다.  

### 코드 설명

1. Action: 상태 변경 요청을 정의한다.  
2. Dispatcher: Action을 Store로 전달한다.  
3. Store: 상태를 관리하며 변경 사항을 View에 알린다.  
4. View: 사용자와 상호작용하며 UI를 업데이트한다.  

## FLUX 패턴의 장점

1. 단방향 데이터 흐름: 데이터 흐름이 단방향으로 고정되어 상태 관리가 명확하다.  
2. 상태의 중앙 집중화: 모든 상태 변경이 Store를 통해 이루어진다.  
3. 유연성: 다양한 애플리케이션에서 활용할 수 있다.  

## FLUX 패턴의 단점

1. 코드 복잡성 증가: Dispatcher, Store, Action 간 상호작용이 복잡할 수 있다.  
2. 러닝 커브: 패턴의 개념을 익히는 데 시간이 필요하다.  
3. 반복 작업: 작은 프로젝트에서는 불필요하게 느껴질 수 있다.  

### 마무리

FLUX 패턴은 단방향 데이터 흐름을 중심으로 대규모 애플리케이션에서 상태 관리를 명확히 하는 강력한 패턴이다.  
React와 같은 라이브러리에서 상태 관리를 고민 중이라면 FLUX 패턴을 적용해 보자.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
