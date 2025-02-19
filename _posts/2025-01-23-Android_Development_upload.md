---
layout: single
title: "Android App Bundle 업로드 시 서명 키 불일치 문제 해결하기"
categories: "ReactNative"
tags: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Android App Bundle 업로드 시 서명 키 불일치 문제 해결하기

Google Play Console에 App Bundle을 업로드할 때, **"사용된 서명 키가 Google Play Console에 등록된 키와 일치하지 않습니다"**라는 오류 메시지가 발생하는 경우가 있다.  
이 오류는 업로드된 App Bundle의 서명 키와 Google Play에 등록된 서명 키의 불일치로 인해 발생한다.  
이 글에서는 이 문제의 원인과 해결 방법을 자세히 알아본다.

---

## 1. **문제 원인**

### (1) **올바른 키로 서명되지 않음**
- App Bundle을 빌드할 때, Google Play Console에 등록된 키 대신 다른 서명 키를 사용한 경우 발생.

### (2) **키 저장소 변경 또는 분실**
- 이전에 사용했던 서명 키를 분실했거나, 다른 키 저장소를 사용하여 App Bundle을 서명한 경우 발생.

---

## 2. **오류 메시지 분석**

오류 메시지에는 업로드된 App Bundle의 서명에 사용된 키의 **SHA-1 지문**과 Google Play Console에 등록된 키의 **SHA-1 지문**이 제공된다.  
이 정보를 통해 두 지문이 일치하는지 확인하여 문제의 원인을 파악할 수 있다.

---

## 3. **해결 방법**

### (1) **현재 프로젝트에 사용된 키 확인**
먼저 프로젝트에서 사용하는 서명 키의 정보를 확인해야 한다.

1. **`keystore` 파일 확인**
   - 프로젝트에 설정된 `keystore` 파일의 경로와 alias(별칭)를 확인.

2. **SHA-1 지문 확인**
   - 아래 명령어를 사용하여 서명 키의 SHA-1 지문을 확인:
     ```bash
     keytool -list -v -keystore <your_keystore_file> -alias <your_alias_name>
     ```
     - `your_keystore_file`에 `keystore` 파일 경로를 입력.  
     - `your_alias_name`에 서명 키의 별칭을 입력.  

---

### (2) **Google Play Console에서 등록된 키 확인**
Google Play Console에 등록된 서명 키 정보를 확인한다.

1. Google Play Console에 로그인.  
2. **릴리스 > 앱 서명** 메뉴로 이동.  
3. 등록된 앱 서명 키(SHA-1, SHA-256) 정보를 확인.  

---

### (3) **일치 여부 확인**
위에서 확인한 SHA-1 지문과 Google Play Console에 등록된 지문을 비교한다.  
지문이 다르다면, 올바른 서명 키를 사용해 App Bundle을 다시 빌드해야 한다.

---

### (4) **올바른 키로 서명**
Google Play Console에 등록된 키로 App Bundle을 서명하여 문제를 해결한다.

1. **빌드 및 서명 설정**  
   프로젝트의 `build.gradle` 파일에서 다음과 같이 서명 정보를 설정:
   ```gradle
   signingConfigs {
       release {
           storeFile file("path/to/your/keystore")
           storePassword "your_password"
           keyAlias "your_alias"
           keyPassword "your_key_password"
       }
   }
   ```

2. **App Bundle 빌드**
   아래 명령어를 사용해 릴리스 빌드를 생성:
   ```bash
   ./gradlew assembleRelease
   ```

3. **업로드**
   새로 빌드된 App Bundle을 Google Play Console에 업로드.  

---

### (5) **서명 키를 분실한 경우**
서명 키를 분실한 경우, Google Play Console에서 **새 업로드 키 생성** 절차를 통해 문제를 해결한다.

1. Google Play Console에서 **릴리스 > 앱 서명** 메뉴로 이동.  
2. **새 업로드 키 생성 요청**을 진행.  
3. Google Play에서 제공한 새 업로드 키로 App Bundle을 서명한 후 업로드.  

---

### (6) **디버그 키 사용 여부 확인**
Google Play Console에 업로드할 App Bundle은 **릴리스 키**로 서명해야 한다.  
만약 디버그 키로 서명된 경우, 서명 설정을 변경하여 릴리스 키로 서명해야 한다.

---

## 4. **정리**

위 단계를 통해 서명 키 불일치 문제를 해결할 수 있다.  
Google Play Console의 서명 키는 앱의 신뢰성과 보안을 보장하는 중요한 요소이므로, 서명 키를 잘 관리하고 올바르게 사용하는 것이 중요하다.

> **팁**: 서명 키 분실을 방지하기 위해 `keystore` 파일을 안전한 곳에 백업해두고, 관련 정보(alias, password 등)를 기록해 두자.