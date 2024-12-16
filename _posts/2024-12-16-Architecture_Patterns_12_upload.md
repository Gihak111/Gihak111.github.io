---
layout: single
title:  "아키텍처 패턴 시리즈 12. 블랙보드 패턴 (Blackboard Pattern)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 12: 블랙보드 패턴 (Blackboard Pattern)

블랙보드 패턴은 다양한 독립 모듈(컴포넌트)이 공유 데이터 저장소(블랙보드)를 통해 협력하여 문제를 해결하는 아키텍처 패턴이다.  
주로 복잡한 문제를 해결하기 위해 인공지능(AI), 음성 인식, 패턴 인식 등과 같은 분야에서 사용된다.  

## 블랙보드 패턴의 필요성

블랙보드 패턴은 다음과 같은 상황에서 유용하다:  

1. 복잡한 문제 해결: 정형화되지 않은 문제를 해결할 때 사용된다.  
2. 다양한 전문가 시스템 통합: 서로 다른 방식으로 작동하는 모듈을 통합할 수 있다.  
3. 점진적 해결: 문제를 한 번에 해결하지 않고, 작은 단위로 접근하여 점진적으로 최종 솔루션에 도달한다.  

### 예시: 음성 인식 시스템

음성 인식 시스템에서는 음성 데이터를 분석하기 위해 여러 모듈이 단계별로 작업을 수행한다.  
블랙보드 패턴은 이 과정에서 각 모듈 간의 데이터 공유와 협업을 지원한다.  

## 블랙보드 패턴의 구조

블랙보드 패턴은 다음 세 가지 주요 컴포넌트로 구성된다:

1. Blackboard (블랙보드): 문제 해결에 필요한 데이터를 저장하는 공유 저장소.  
2. Knowledge Source (지식 소스): 특정 작업을 수행하는 독립적 모듈.  
3. Control Component (제어 컴포넌트): 블랙보드를 모니터링하고, 적절한 지식 소스를 호출하여 문제 해결을 진행.  

### 구조 다이어그램

```
[Knowledge Source 1]       [Knowledge Source 2]
           \                       /
            --------[Blackboard]--------
                        |
               [Control Component]
```

### 블랙보드 패턴 동작 순서

1. 초기화: 블랙보드에 초기 데이터를 입력한다.  
2. 감시 및 호출: 제어 컴포넌트가 블랙보드를 감시하고, 필요에 따라 지식 소스를 호출한다.  
3. 작업 수행: 지식 소스는 블랙보드의 데이터를 읽고 작업을 수행한 뒤 결과를 다시 블랙보드에 기록한다.  
4. 반복: 문제 해결이 완료될 때까지 이 과정을 반복한다.  

## 블랙보드 패턴 예시

아래는 블랙보드 패턴을 활용해 간단한 데이터 분석 작업을 구현한 Python 코드이다.  

### 블랙보드 컴포넌트 구현

#### 1. 블랙보드

```python
class Blackboard:
    def __init__(self):
        self.data = {}  # 공유 데이터 저장소

    def read(self, key):
        return self.data.get(key)

    def write(self, key, value):
        self.data[key] = value

    def display(self):
        print("Blackboard Data:", self.data)
```

#### 2. 지식 소스

```python
class KnowledgeSource:
    def analyze(self, blackboard):
        raise NotImplementedError("Subclass must implement analyze method")
```

```python
class DataNormalizer(KnowledgeSource):
    def analyze(self, blackboard):
        raw_data = blackboard.read("raw_data")
        if raw_data:
            normalized_data = [x / max(raw_data) for x in raw_data]
            blackboard.write("normalized_data", normalized_data)
            print("DataNormalizer: 데이터 정규화 완료")
```

```python
class DataAnalyzer(KnowledgeSource):
    def analyze(self, blackboard):
        normalized_data = blackboard.read("normalized_data")
        if normalized_data:
            analysis_result = sum(normalized_data) / len(normalized_data)
            blackboard.write("analysis_result", analysis_result)
            print("DataAnalyzer: 데이터 분석 완료")
```

#### 3. 제어 컴포넌트

```python
class ControlComponent:
    def __init__(self, blackboard, knowledge_sources):
        self.blackboard = blackboard
        self.knowledge_sources = knowledge_sources

    def run(self):
        for ks in self.knowledge_sources:
            ks.analyze(self.blackboard)
        self.blackboard.display()
```

### 실행

```python
# 초기 데이터
blackboard = Blackboard()
blackboard.write("raw_data", [10, 20, 30, 40, 50])

# 지식 소스 설정
knowledge_sources = [DataNormalizer(), DataAnalyzer()]

# 제어 컴포넌트 실행
controller = ControlComponent(blackboard, knowledge_sources)
controller.run()
```

### 출력 결과

```
DataNormalizer: 데이터 정규화 완료
DataAnalyzer: 데이터 분석 완료
Blackboard Data: {'raw_data': [10, 20, 30, 40, 50], 'normalized_data': [0.2, 0.4, 0.6, 0.8, 1.0], 'analysis_result': 0.6}
```

### 코드 설명

1. Blackboard: 공유 데이터를 저장하고 관리한다.  
2. KnowledgeSource: 데이터 정규화 및 분석 작업을 수행한다.  
3. ControlComponent: 블랙보드 데이터를 기반으로 지식 소스를 순차적으로 호출한다.  

## 블랙보드 패턴의 장점

1. 확장성: 새로운 지식 소스를 추가하기 용이하다.  
2. 모듈화: 각 지식 소스가 독립적으로 동작하므로 재사용성이 높다.  
3. 복잡한 문제 해결 가능: 다양한 모듈이 협력하여 복잡한 문제를 점진적으로 해결할 수 있다.  

## 블랙보드 패턴의 단점

1. 복잡성: 단순한 문제에 적용하기에는 구조가 과도할 수 있다.  
2. 제어 로직 부담: 제어 컴포넌트가 블랙보드와 지식 소스 간의 상호작용을 관리해야 하므로 로직이 복잡해질 수 있다.  
3. 성능 문제: 지식 소스 간 데이터 전달 및 반복 작업이 많아질 경우 성능 저하가 발생할 수 있다.  

### 마무리

블랙보드 패턴은 복잡한 문제 해결을 위해 각기 다른 모듈이 협력해야 할 때 적합한 패턴이다.  
특히 AI와 패턴 인식 시스템에서 매우 유용하게 활용된다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
