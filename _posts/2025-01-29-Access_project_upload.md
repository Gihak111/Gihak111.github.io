---
layout: single
title:  "리액트 네이티브  변경점과 엑세스 빌리티 서비스"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

RN에는, 청각장애인들을 위해 인터페이스를 가져와 글을 소리로 읽어주는 등의 기능을 제공하는 AccessibilityService라는 라이브러리가 있다. 이에 대해 잘 알아보자.  

## 1. 변경점

### 1. 생성 형식 변경
앱 생성 명령어가 다음과 같이 변경되었다.  
```bash
npx @react-native-community/cli init YourApp
```  

### 2. 속정 지원 중단
AndroidManifest.xml에서 package="com.yourapp" 속성은 더 이상 지원되지 않는다.  
따라서, AndroidManifest.xml에 선언되어 있는 패키지를 지우고,  
build.gradle에 namespace 속성을 추가해야 한다.  

## 2. AccessibilityService

먼저, AndroidManifest.xml에 서비스를 선언하자.  
```xml
<!-- 서비스 추가: Accessibility Service -->
<service
    android:name=".MyAccessibilityService"
    android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE">
    <intent-filter>
        <action android:name="android.accessibilityservice.AccessibilityService" />
    </intent-filter>
    <meta-data
        android:name="android.accessibilityservice"
        android:resource="@xml/accessibility_service_config" />
</service>    
```

이후, res/xml/accessibility_service_config.xml을 다음과 같이 하자.  
```xml
<accessibility-service xmlns:android="http://schemas.android.com/apk/res/android"
    android:accessibilityEventTypes="typeNotificationStateChanged"
    android:accessibilityFeedbackType="feedbackAllMask"
    android:notificationTimeout="100"
    android:canRetrieveWindowContent="true"
    android:description="@string/accessibility_service_description"
    android:settingsActivity="com.example.chatmonitor.MyAccessibilitySettings" />
```

위와 같은 속성을 사용해야 온전하게 AccessibilityService를 작동시킬 수 있다.  
이제 남은건, 위 내용을 사용하는 스크립트를 만들어 적용하는 것 뿐이다.  


안드로이드 12 이상에서부턴 service, activity, receiver 등에 intent-filter가 포함되면 반드시 android:exported 속성을 명시해야 한다.  
이 속성은 해당 컴포넌트가 외부에서 호출될 수 있는지 여부를 설정하는 거다.

만약 이 서비스가 다른 앱에서 호출되지 않도록 하려면 android:exported="false"로 설정하면 된다.  
만약 다른 앱이나 시스템에서 이 서비스를 호출할 수 있도록 하려면 android:exported="true"로 설정해야 한다.