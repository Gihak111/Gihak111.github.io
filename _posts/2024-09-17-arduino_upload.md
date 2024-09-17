---
layout: single
title:  "아두이노 시작하기"
categories: "arduino"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# arduino
아두이노 쉽고 재밌으니까 해보자.
아두이노의 기초적인 것을데 대해 서술하겠다.

## 사이트
(https://www.tinkercad.com/)[https://www.tinkercad.com/]
위 링크를 통해 진행하겠다.  

## 아두이노 기초 및 실습 가이드

아두이노는 다양한 전자 부품을 활용하여 다양한 프로젝트를 수행할 수 있는 오픈 소스 하드웨어 플랫폼이다.  
### 1. 아두이노 하드웨어 기본

**Arduino UNO R3**는 다음과 같은 주요 사양을 가지고 있다:  
- **마이크로컨트롤러**: ATmega328P  
- **동작 전원**: 5V  
- **입력 전압**: 7V ~ 12V (권장), 6V ~ 20V (제한)  
- **디지털 입출력 핀**: 14개 (그 중 6개는 PWM 출력 포함)  
- **아날로그 입력 핀**: 6개  
- **플래시 메모리**: 32KB (부트로더 0.5KB 사용)  
- **SRAM**: 2KB  
- **EEPROM**: 1KB  
- **클럭 속도**: 16MHz  

### 2. 아두이노의 기본 구문  

아두이노 프로그램은 두 가지 주요 함수로 구성된다:  
- **`setup()`**: 초기화 구문으로, 프로그램이 시작될 때 한 번만 실행된다.  
- **`loop()`**: 실제 동작 구문으로, `setup()`이 완료된 후 계속 반복 실행된다.  

#### 기본 코드 예제

```cpp
void setup() {
  Serial.begin(9600);
  Serial.println("Hello, World!");
}

void loop() {
  Serial.println("Loop is running...");
  delay(1000); // 1초 대기
}
```

### 3. 상수와 변수

**상수**와 **변수**는 데이터를 저장하고 조작하는 데 사용된다.  
상수는 값이 변경되지 않는 반면, 변수는 값을 변경할 수 있다.  

#### 상수와 변수 예제  

```cpp
const int LED_PIN = 13; // 상수
int ledState = LOW;     // 변수

void setup() {
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, ledState);
  delay(1000);
  ledState = (ledState == LOW) ? HIGH : LOW; // LED 상태 토글
}
```

### 4. 조건문과 반복문  

**조건문**은 특정 조건에 따라 코드의 실행 흐름을 제어하고, **반복문**은 코드를 반복 실행한다.  

#### 홀수 합 구하기

```cpp
int sumOdd(int maxNum) {
  int sum = 0;
  for (int i = 1; i <= maxNum; i += 2) { // 홀수만 합산
    sum += i;
  }
  return sum;
}
```

#### LED 제어 예제

```cpp
const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, HIGH); // LED 켜기
  delay(1000); // 1초 대기
  digitalWrite(LED_PIN, LOW);  // LED 끄기
  delay(1000); // 1초 대기
}
```

### 5. 함수와 데이터 전송

**함수**는 특정 작업을 수행하는 코드 블록이다. 
아두이노에서는 시리얼 통신을 통해 컴퓨터와 데이터를 주고받을 수 있다.  

#### 함수 예제

```cpp
int addNumbers(int a, int b) {
  return a + b;
}

void setup() {
  Serial.begin(9600);
}

void loop() {
  int result = addNumbers(5, 7);
  Serial.println(String("Result: ") + result);
  delay(2000); // 2초 대기
}
```

#### 시리얼 통신  

아두이노에서 시리얼 통신을 통해 데이터를 컴퓨터로 전송하거나 컴퓨터로부터 데이터를 수신할 수 있다.  

```cpp
void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    int data = Serial.parseInt();
    Serial.println(String("Received: ") + data);
  }
}
```

위와 같은 방식으로 구현할 수 있다.