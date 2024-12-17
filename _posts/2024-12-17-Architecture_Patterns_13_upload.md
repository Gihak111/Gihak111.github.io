---
layout: single
title:  "아키텍처 패턴 시리즈 13. 인터프리터 패턴 (Interpreter Pattern)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 13: 인터프리터 패턴 (Interpreter Pattern)

인터프리터 패턴은 문법이나 언어의 규칙을 정의하고 해석하는 데 사용되는 아키텍처 패턴이다.  
주로 간단한 언어 처리, 명령어 실행, 스크립트 해석 등에 사용되며, 각 표현식(Expression)을 객체로 표현하고 해석(Interpretation) 작업을 수행한다.  

## 인터프리터 패턴의 필요성

인터프리터 패턴은 다음과 같은 상황에서 유용하다:  

1. 고정된 문법 분석: 고정된 문법의 간단한 언어를 해석해야 할 때 적합하다.  
2. 반복적인 해석 작업: 명령어 또는 언어를 반복적으로 해석하고 실행해야 하는 경우.  
3. 확장 가능성 요구: 새로운 표현식을 추가하거나 변경할 가능성이 높은 경우.  

### 예시: 간단한 계산기

문자열 형태의 수식(`"5 + 3 - 2"`)을 해석하여 결과를 계산하는 계산기를 생각해 보자.  
인터프리터 패턴은 이를 효과적으로 처리할 수 있는 구조를 제공한다.  

## 인터프리터 패턴의 구조

인터프리터 패턴은 다음과 같은 주요 컴포넌트로 구성된다:  

1. AbstractExpression (추상 표현식): 표현식 해석을 위한 공통 인터페이스를 정의한다.  
2. TerminalExpression (터미널 표현식): 문법의 말단 요소를 처리한다.  
3. NonTerminalExpression (비터미널 표현식): 문법 규칙을 처리하며, 다른 표현식을 포함한다.  
4. Context (문맥): 해석을 위한 전역 정보를 저장한다.  
5. Client (클라이언트): 해석 과정을 시작한다.  

### 구조 다이어그램

```
[Client]
   |
[Context]
   |
[AbstractExpression]
   |
   |-- [TerminalExpression]
   |-- [NonTerminalExpression]
```

### 인터프리터 패턴 동작 순서

1. 문맥 정의: 해석에 필요한 데이터를 `Context`에 저장한다.  
2. 문법 표현: 수식이나 문법을 `AbstractExpression`의 구현체로 표현한다.  
3. 해석 수행: 각 표현식의 `interpret()` 메서드를 호출하여 해석 작업을 수행한다.  

## 인터프리터 패턴 예시

아래는 간단한 계산기를 구현한 Python 코드이다.  


#### 1. 추상 표현식 (AbstractExpression)

```python
class Expression:
    def interpret(self, context):
        raise NotImplementedError("Subclasses must implement interpret method")
```

#### 2. 터미널 표현식 (TerminalExpression)

```python
class NumberExpression(Expression):
    def __init__(self, number):
        self.number = number

    def interpret(self, context):
        return self.number
```

#### 3. 비터미널 표현식 (NonTerminalExpression)

```python
class AddExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self, context):
        return self.left.interpret(context) + self.right.interpret(context)
```

```python
class SubtractExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self, context):
        return self.left.interpret(context) - self.right.interpret(context)
```

#### 4. 문맥 (Context)

```python
class Context:
    def __init__(self):
        self.data = {}
```

#### 5. 클라이언트 (Client)

```python
# 수식: 5 + 3 - 2
context = Context()

# 표현식 구성
expr1 = NumberExpression(5)
expr2 = NumberExpression(3)
expr3 = NumberExpression(2)

add_expr = AddExpression(expr1, expr2)
subtract_expr = SubtractExpression(add_expr, expr3)

# 결과 해석
result = subtract_expr.interpret(context)
print("결과:", result)  # 출력: 결과: 6
```

### 코드 설명

1. NumberExpression: 숫자를 표현하는 터미널 표현식이다.  
2. AddExpression/SubtractExpression: 더하기와 빼기를 수행하는 비터미널 표현식이다.  
3. Context: 현재 예시에서는 데이터를 저장하지 않지만, 복잡한 해석에서는 추가적인 정보 저장에 활용된다.  
4. Client: 전체 해석 프로세스를 시작하고 결과를 얻는다.  

## 인터프리터 패턴의 장점

1. 문법의 명시적 표현: 문법 규칙을 명확하게 정의할 수 있다.  
2. 확장 용이성: 새로운 표현식을 추가하기 쉽다.  
3. 재사용성 증가: 각 표현식을 독립적으로 설계하여 다양한 해석 작업에 활용할 수 있다.  

## 인터프리터 패턴의 단점

1. 복잡성 증가: 문법이 복잡할수록 많은 클래스를 생성해야 하므로 코드 복잡도가 높아진다.  
2. 성능 문제: 대규모 데이터나 복잡한 해석 작업에서는 성능 저하가 발생할 수 있다.  
3. 제한된 적용 범위: 고정된 문법이나 작은 규모의 작업에서만 효과적이다.  

### 마무리

인터프리터 패턴은 고정된 문법을 해석하고 처리하는 데 효과적인 도구를 제공한다.  
다양한 언어 처리나 스크립트 실행 작업에서 활용 가능하며, 새로운 문법 확장에 용이하다.  
다른 아키텍처 패턴이 궁금하다면 아래 글도 확인해보세요.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
