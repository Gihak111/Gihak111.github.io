---
layout: single
title:  "RN 프로젝트와 AN의 연결"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## RN과 AN 연결

RN 프로제긑와 AN을 연결하는 과정을 브릿지가 아닌 모듈로 연결할 경우가 가끔 있다.  
브릿지를 통해서 간단하게 연결할 수 있지만,  
AN 프로젝트를 통째로 하나의 모듈로 하여 RN과 연결할 수 있다.  

---

### 1. 모듈 선언
```java
public class yourModule extends ReactContextBaseJavaModule
```

위와 같은 방식으로 하나의 모듈 클래스를 만든다.  
해당 모듈 클래스 내부에, 주요 함수들을 정의하거나, 내용을 오버라이딩 하면서 메인 엑티비티에 있어야 할 기능들을 작동하게 한다.  

### 2. 생성자 오류 안나도록, 모듈 클래스 선언

```java
package com.romanceslaughter;

import com.facebook.react.ReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.ViewManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MonitoringServiceModule implements ReactPackage {

    // 네이티브 모듈 등록
    @Override
    public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
        List<NativeModule> modules = new ArrayList<>();
        modules.add(new MonitoringServiceNativeModule(reactContext)); // NativeModule 추가
        return modules;
    }

    // ViewManager 등록 (현재는 비어 있음)
    @Override
    public List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
        return Collections.emptyList(); // ViewManager가 없으면 빈 리스트 반환
    }
}
```

위와 같은 코드를 통해서, 메인 엑티비티가 가져오려는 모듈이 원문이 아니라 생성자라 생기는 오류를 방지한다.
상속 관계가 아닌 구조 관계로 내용을 가져올 수 있도록 한다.

### 3. 메인 엑티비티에 App.js와 AN 연결
```java
    @Override
    protected String getMainComponentName() {
        return "yourApp"; // RN의 App.js와 연결
    }
```
위와 같이 선언하여, App.js와 AN을 연결한다.  
저 "  " 내부에 들어가느 ㄴ내용은 패키지 명과 일치해야 하며, 이는 대소문자 역시 일치시켜야 한다.  
만일, 대소문자를 일치시키지 않으면 AN의 메인 엑티비틸르 가져오지 못했다는 내용의 오류가 나오는데, 이 같은 오류와 모듈을 가져오지 못했다는 오류가 난다면 여기에서의 오타를 의심해 보자.  