---
layout: single
title:  "안드로이드 접근성 설정"
categories: "pynb"
tag: "ERROR"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 접근성 앱

RN을 통해서 앱을 만들경우, 장애인을 위한 서비스를 만들 떄 엑세스 빌리티 서비스를 사용할 경우가 있다.  
우리가 앱을 배포하지 않고, 단순 apk 다운로드로 접근성 권한을 가져오려면(즉, 풀퍼가 불명확한 앱), 다음과 같은 단계를 따라야 한다.  

1. 점근성 -> 앱으로 들어가서 앱의 접근성 권한을 허용하도록 시도. 이를 하면 애플리케이션 -> 다운받은 앱 -> 앱 정보에서 접근성 허용이 열린다.  
2. 애플리케이션 -> 다운받은 앱 -> 앱 정보에서 접근성 사용 허용을 구른다.  
3. 접근성 -> 앱 -> 내 앱에서 앱의 접근성 권한을 허용한다.  

위와 같은 3단계를 이행해야 앱의 접근섣 권한을 허용할 수 있다.  

하지만, 앱을 구글 플레이스토어나 원스토어 같은 곳에 배포 하였다면(즉, 출처가 명확한 앱),  
위와 같은 단계를 무시하고,  
바로 접근성 -> 앱 -> 접근성 권한 허용만 하면 된다.  
이 방법을 RN 코드로 보자면,  

```java
import android.provider.Settings;
import android.text.TextUtils;

// 접근성 설정이 활성화되었는지 확인하는 메서드
private boolean isAccessibilityServiceEnabled(Context context, String serviceName) {
    try {
        String enabledServices = Settings.Secure.getString(
            context.getContentResolver(),
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        );

        return enabledServices != null && enabledServices.contains(serviceName);
    } catch (Exception e) {
        return false;
    }
}

// 접근성 설정 여부를 체크하고, 설정이 안 되어 있으면 유도
if (!restrictedCheckDone) {
    String serviceName = context.getPackageName() + "/" + MyAccessibilityService.class.getName();

    if (!isAccessibilityServiceEnabled(context, serviceName)) {
        // 접근성 설정이 활성화되지 않음 → 설정 화면으로 이동
        Toast.makeText(context,
            "먼저 접근성 설정 화면에서 'Restricted Allow' 옵션을 활성화해 주세요.",
            Toast.LENGTH_LONG
        ).show();

        Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(intent);

        // "E_RESTRICTED" 오류 반환 → 현재 작업을 중단하고 설정하도록 유도
        promise.reject("E_RESTRICTED", "Need to allow restricted setting first");

        return;
    }

    // 접근성 설정이 활성화되었음 → restrictedCheckDone 저장
    prefs.edit().putBoolean(KEY_RESTRICTED_CHECK_DONE, true).apply();
}

```  

위와 같은 방법으로, 사용자가 접근성 권한을 허용하게 하고, 이 여부를 체크할 수 있다.  
