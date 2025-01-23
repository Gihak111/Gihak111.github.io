---
layout: single
title: "RN프로젝트 명 바꾸기기"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 패키지명 변경

구글 플레이 스토어에 앱을 출시하려고 하는데, 앱 번들 파일의 명이 곂친다고 안된다 할 경우가 있다.  
이럴 때는, 다음과 같은 과정을 따르는 것으로 패키지 명을 변경할 수 있다.  
그냥 새로 만드는 것도 방법이긴하다;;  


### 1. **`app.json` 파일 수정**
`app.json` 파일에서 `expo` 프로젝트라면 `expo.name`과 `expo.android.package` 값을 변경하자자.  
Expo를 사용하지 않는다면 `android`와 `ios` 관련 작업만 진행하면 된된다.

```json
{
  "name": "YourAppName",
  "displayName": "YourAppName",
  "expo": {
    "android": {
      "package": "com.newpackage.name"
    }
  }
}
```

---

### 2. **Android 패키지명 변경**
1. **Java 파일 경로 변경**
   - `android/app/src/main/java` 경로 안의 폴더를 현재 패키지명 구조에 맞게 변경하자자.  
   예를 들어, `com.old.package.name`을 `com.new.package.name`으로 변경하려면:
   - 기존 `com/old/package/name` 디렉토리를 `com/new/package/name`으로 이동하거나 새 디렉토리를 생성 후 파일을 옮기면 된다.

2. **Java 파일 내부 패키지명 변경**
   - 변경한 패키지 디렉토리 안의 `MainActivity.java`, `MainApplication.java` 등의 파일에서 최상단의 패키지명을 새 패키지명으로 수정하자자.

   ```java
   package com.newpackage.name;
   ```

3. **`AndroidManifest.xml` 수정**
   - `android/app/src/main/AndroidManifest.xml`에서 패키지명을 새로 변경한 패키지명으로 업데이트 하자.
   ```xml
   <manifest xmlns:android="http://schemas.android.com/apk/res/android"
       package="com.newpackage.name">
   ```

4. **`build.gradle` 수정**
   - `android/app/build.gradle` 파일에서 `applicationId`를 새 패키지명으로 수정하자.
   ```gradle
   defaultConfig {
       applicationId "com.newpackage.name"
   }
   ```

---

### 3. **iOS 패키지명 변경**
1. **Xcode에서 프로젝트 열기**
   - `ios/YourAppName.xcworkspace` 파일을 Xcode로 열고, 왼쪽 상단의 프로젝트 이름을 선택하자.

2. **Bundle Identifier 변경**
   - Xcode 상단의 "Signing & Capabilities" 탭에서 `Bundle Identifier`를 새 패키지명으로 변경하자.

3. **프로젝트 폴더명 변경**
   - `ios` 폴더 내에서 프로젝트 이름을 패키지명에 맞게 변경하자.

---

### 4. **전체 코드에서 이전 패키지명 변경**
- VS Code나 IDE의 "찾기 및 바꾸기" 기능을 사용하여 프로젝트 전체에서 이전 패키지명(`com.old.package.name`)을 새로운 패키지명(`com.new.package.name`)으로 변경하자.

---

### 5. **캐시 삭제 및 빌드**
1. **캐시 클리어**
   ```bash
   npm start --reset-cache
   ```

2. **Android 빌드**
   ```bash
   cd android
   ./gradlew clean
   cd ..
   npm run android
   ```

3. **iOS 빌드**
   ```bash
   cd ios
   pod install
   cd ..
   npm run ios
   ```

---

이처럼 하면 프로젝트의 패키지 명을 바꿀 수 있다.
그냥 새로 프로젝트 만들고, 그거에 맞게 기존 코드의 패키지 명을 바꾸어 새로 만드는게 더 편할 수 도 있다.  