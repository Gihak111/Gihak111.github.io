---
layout: single
title:  "알고리즘 정리 3. 브루트포스 알고리즘"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Bruteforce
오나전 탐색 알고리즘 이다.  
간단하게, 4자리 숫자 하나를 찾아야 한다면, 0000부터 9999까지 모두 넣어 본다는 뜻이다.  
비밀 번호 해킹할 떄 사용하면 좋다.  

간단하고 직관적이지만, 모든 경우를 다 시도하기 깨문에, 시간이 많이 들어가 비효율 적 일 수 있다.  
문제의 크기가 작거나, 최적의 해답이 필요없는 상황에선 나이스하게 사용할 수 있다.  

# 구현
1. for, while 같은거로 구현한다.  
2. 재귀함수  
웬만하면 재귀 함수 사용하자.  

간단하게, text 에서 특정 패턴이 존재하는지 찾아서 위피를 반환하는 프로그램 예실르 보자.  
```java
public class RecursiveBruteForce {

    // 재귀적으로 텍스트의 각 위치에서 패턴을 비교하는 메서드
    public static int recursiveSearch(String text, String pattern, int textIndex, int patternIndex) {
        // 패턴의 모든 문자가 일치했을 경우 시작 위치 반환
        if (patternIndex == pattern.length()) {
            return textIndex - patternIndex;
        }
        
        // 텍스트의 끝에 도달했거나 문자가 일치하지 않으면 -1 반환
        if (textIndex == text.length() || text.charAt(textIndex) != pattern.charAt(patternIndex)) {
            return -1;
        }

        // 다음 문자 비교
        return recursiveSearch(text, pattern, textIndex + 1, patternIndex + 1);
    }

    // 텍스트의 각 위치를 시작점으로 패턴을 찾는 메서드
    public static int findPattern(String text, String pattern, int startIndex) {
        // 텍스트의 남은 부분이 패턴보다 짧을 경우 더 이상 탐색할 필요 없음
        if (startIndex > text.length() - pattern.length()) {
            return -1;
        }

        // 현재 위치에서 패턴을 찾기 시도
        int result = recursiveSearch(text, pattern, startIndex, 0);
        
        // 패턴이 일치하면 그 위치 반환, 그렇지 않으면 다음 위치에서 재귀적으로 탐색
        if (result != -1) {
            return result;
        } else {
            return findPattern(text, pattern, startIndex + 1);
        }
    }

    public static void main(String[] args) {
        String text = "Hello, this is a simple example.";
        String pattern = "simple";

        // findPattern 메서드를 호출하여 패턴의 시작 위치를 찾음
        int result = findPattern(text, pattern, 0);

        if (result != -1) {
            System.out.println("Pattern found at index: " + result);
        } else {
            System.out.println("Pattern not found.");
        }
    }
}

```

위 코드를 보면,  
- **1. recursiveSearch 메서드**
    * textIndex에서 시작하여 pattern의 각 문자를 재귀적으로 비교한다.  
    * patternIndex가 패턴의 길이에 도달하면 패턴이 완전히 일치한 것이므로, textIndex - patternIndex를 반환  
    * 텍스트의 끝에 도달했거나 문자 비교가 실패하면 -1을 반환  
    * 문자 비교에 성공하면 다음 문자로 재귀 호출을 이어간다.  
- **2. findPattern 메서드**
    * 텍스트의 각 위치에서 패턴 매칭을 시도한다.  
    * 텍스트의 남은 부분이 패턴보다 짧아지면 더 이상 패턴을 찾을 수 없으므로 -1을 반환한다.  
    * 현재 위치에서 recursiveSearch를 호출하여 패턴을 찾는다. 패턴이 일치하면 위치를 반환하고, 일치하지 않으면 다음 위치에서 재귀적으로 탐색한다.  
- **3. main 메서드**
    * text와 pattern을 정의한 후, findPattern 메서드를 호출하여 결과를 출력한다.  

이런 구성이다.  

위 코드의 주축인 재귀 함수를 알아보자면,  
그냥 함수 스스로가 지를 호출한는 함수이다.  
위의 코드에도 스스로를 함수 내에서 호출한다.  
함숭서 빠져나오지 않으면 오버플로우가 생기기 때문에 꼭 빠져나오는 트리거를 준비해야 한다.  


1. 기본조건  
    재귀 호출을 멈추는 조건으로, 더 이상 자기 자신을 호출하지 않고 결과를 반환한다.  
2. 재귀 호출  
    더 작은 문제로 나누어 자기 자신을 호출하는 부분이다.  
    재귀 호출이 반복되며 기본 조건에 도달하면 재귀가 종료된다.  

예제를 보자면,  
```java
public class FactorialExample {

    // 재귀적으로 팩토리얼을 계산하는 함수
    public static int factorial(int n) {
        // 기본 조건: n이 0이면 1 반환 (0! = 1)
        if (n == 0) {
            return 1;
        }
        // 재귀 호출: n * (n-1)!
        return n * factorial(n - 1);
    }

    public static void main(String[] args) {
        int number = 5;
        int result = factorial(number);
        System.out.println("Factorial of " + number + " is: " + result);
    }
}

```

factorial 함수는 자기 자신을 호출하여 n!을 계산합니다. n이 0이 되면 재귀 호출이 종료되고, 그동안의 계산된 값들이 반환한다.  

반복문과 비교를 해 보자면,  
- **반복문 for, while**
    for: 정된 횟수만큼 반복할 때 사용된다. 반복 횟수가 명확한 경우에 주로 사용된다.  
    while: 조건이 참인 동안 계속 반복된다. 반복 횟수가 명확하지 않거나 조건에 따라 반복을 제어할 때 유용하다.  
- **재귀함수**
    문제를 더 작은 하위 문제로 나눌 수 있을 때 적합하다.  
    수학적 정의나 알고리즘을 자연스럽게 표현할 수 있다.  
    반복문보다 코드가 더 간결하고 직관적일 수 있다.  
    함수 호출이 스택에 저장되므로, 스택 오버플로우가 발생할 수 있다. 더해서 오버플로우 같은 문제가 발생할 수 있다.  

재귀함수 사용시기만을 보면,  
1. 문제의 자연스러운 분할: 문제를 하위 문제로 자연스럽게 분할할 수 있을 때
2. 알고리즘의 자연스러운 표현: 특정 알고리즘이 재귀적으로 정의될 때
3. 간결한 코드: 반복문으로 구현할 때보다 재귀적으로 표현하면 코드가 더 간결해지고 이해하기 쉬울 때.  

그 유명한 피보니치 수열을 재귀적으로 해보자.  
```java
public class FibonacciExample {

    // 재귀적으로 피보나치 수열을 계산하는 함수
    public static int fibonacci(int n) {
        // 기본 조건: n이 0 또는 1일 경우 그 값을 반환
        if (n == 0 || n == 1) {
            return n;
        }
        // 재귀 호출: (n-1)번째와 (n-2)번째 피보나치 수를 더함
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    public static void main(String[] args) {
        int number = 10;
        int result = fibonacci(number);
        System.out.println("Fibonacci number at position " + number + " is: " + result);
    }
}

```

위의 방법으로 간단하게 피보니치 수열을 재귀적으로 해결할 수 있다.  

정말 좋은 재귀함수지만, 단점도 있다.  
1. 성능 문제  
    단순한 재귀 구현은 중복된 계산을 많이 할 수 있어 비효율적입니다.
2. 스택 오버플로우  
    재귀 깊이가 너무 깊어지면 스택 오버플로우가 발생할 수 있다.  
    떄에 따라서 반복문이 더 효율적이다. 신은 아니라는 거다.  

재귀를 피해야 할 때는  
1. 반복 구조로 쉽게 해결 가능할 때  
    문제를 반복문으로 쉽게 해결할 수 있을 때.  
2. 깊은 재귀 호출이 필요할 때  
    너무 깊은 재귀 호출이 필요하여 스택 오버플로우가 발생할 수 있을 때.  
3. 성능이 중요한 경우  
    재귀가 성능 병목이 될 수 있는 경우, 특히 중복 계산이 많을 때.  

이런 상황에서는 반복문이나 메모이제이션 같은 기법을 사용하는 것이 더 나을 수 있다.  