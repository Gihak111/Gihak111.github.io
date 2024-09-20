---
layout: single
title:  "Firebase Cloud Messaging "
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
 
# Firebase Cloud
무료 푸시 알림 서비스다.  
일단 무료인게 엄청 크다.  
이를 통해 사용자가 백엔드 서버에서 모바일 앱으로 푸시 알림을 보낼 수 있고, 이를 이용하여 SNS 앱이 열리거나 카메라가 켜질 때 백그라운드에서 알림을 보내는 기능을 구현할 수 있다.  

### FCM을 사용한 Android 푸시 알림 구현 방법

1. **Firebase 프로젝트 설정**
   - Firebase 콘솔에 로그인한 후 새로운 프로젝트를 만든다.
   - Android 앱을 Firebase 프로젝트에 추가한다. `google-services.json` 파일을 다운로드하고, Android 프로젝트의 `app` 폴더에 추가한다.

2. **Firebase SDK 추가**
   `build.gradle` 파일에서 Firebase SDK와 관련된 의존성을 추가한다.
   - 프로젝트의 `build.gradle`:
     ```gradle
     classpath 'com.google.gms:google-services:4.3.14' // 최신 버전 확인
     ```
   - 앱 모듈의 `build.gradle`:
     ```gradle
     implementation 'com.google.firebase:firebase-messaging:23.0.0' // 최신 버전 확인
     apply plugin: 'com.google.gms.google-services'
     ```

3. **FCM 토큰 수신**
   FirebaseInstanceId API를 사용하여 앱의 고유 토큰을 받아 백엔드 서버에 전달할 수 있다.
   ```java
   FirebaseMessaging.getInstance().getToken()
       .addOnCompleteListener(task -> {
           if (!task.isSuccessful()) {
               Log.w(TAG, "Fetching FCM registration token failed", task.getException());
               return;
           }
           // Get new FCM registration token
           String token = task.getResult();
           Log.d(TAG, "FCM Token: " + token);
       });
   ```

4. **푸시 알림 처리**
   FCM 메시지를 수신할 때, Android의 `FirebaseMessagingService` 클래스를 확장하여 메시지를 처리한다.
   ```java
   public class MyFirebaseMessagingService extends FirebaseMessagingService {

       @Override
       public void onMessageReceived(RemoteMessage remoteMessage) {
           // 메시지 처리
           if (remoteMessage.getNotification() != null) {
               sendNotification(remoteMessage.getNotification().getBody());
           }
       }

       private void sendNotification(String messageBody) {
           Intent intent = new Intent(this, MainActivity.class);
           intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
           PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_ONE_SHOT);

           NotificationCompat.Builder notificationBuilder =
                   new NotificationCompat.Builder(this, "channel_id")
                           .setSmallIcon(R.drawable.ic_notification)
                           .setContentTitle("New Message")
                           .setContentText(messageBody)
                           .setAutoCancel(true)
                           .setContentIntent(pendingIntent);

           NotificationManager notificationManager =
                   (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
           notificationManager.notify(0, notificationBuilder.build());
       }
   }
   ```

5. **백엔드에서 메시지 보내기**
   Express.js를 사용하는 백엔드 서버에서 Firebase Admin SDK를 통해 메시지를 보낼 수 있다.
   - `firebase-admin` 패키지를 설치하고, Firebase 프로젝트의 서비스 계정 키를 이용하여 메시지를 보낸다.
   ```javascript
   const admin = require('firebase-admin');
   admin.initializeApp({
       credential: admin.credential.cert(require('./path/to/serviceAccountKey.json'))
   });

   const message = {
       notification: {
           title: 'App opened',
           body: 'Your SNS app has been opened!'
       },
       token: 'user-device-fcm-token'
   };

   admin.messaging().send(message)
       .then(response => {
           console.log('Message sent successfully:', response);
       })
       .catch(error => {
           console.error('Error sending message:', error);
       });
   ```

이 방법을 사용하여 앱에서 SNS 앱이 열리거나 카메라가 켜질 때 알림을 보내는 기능을 구현할 수 있다. FCM을 통해 백그라운드 작업을 트리거하고, 프론트엔드의 상태 플래그를 업데이트하는 추가 로직도 작성할 수 있다.

### Firebase에서 `serviceAccountKey.json`와 `google-services.json` 파일 다운로드 방법

#### 1. `serviceAccountKey.json` 다운로드
이 파일은 Firebase Admin SDK에서 사용할 서비스 계정 키 파일로, 백엔드에서 Firebase 기능을 사용할 때 필요하다.

#### **단계:**
1. **Firebase 콘솔**에 로그인: [https://console.firebase.google.com](https://console.firebase.google.com)
2. 프로젝트 선택: Firebase 대시보드에서 해당 프로젝트를 선택한다.
3. **프로젝트 설정으로 이동**: 화면 왼쪽 상단의 톱니바퀴 모양 아이콘을 클릭하고 '프로젝트 설정'으로 이동한다.
4. **서비스 계정 탭** 선택: '프로젝트 설정' 페이지에서 상단에 '서비스 계정' 탭을 선택한다.
5. **새 비공식 키 생성**: 'Firebase Admin SDK' 섹션에서 '새 비공식 키 생성' 버튼을 클릭한다.
6. 파일 다운로드: 클릭하면 `serviceAccountKey.json` 파일이 자동으로 다운로드된다.

#### 2. `google-services.json` 다운로드
이 파일은 Android 애플리케이션에서 Firebase 기능을 연동할 때 필요하다.

#### **단계:**
1. **Firebase 콘솔**에 로그인: [https://console.firebase.google.com](https://console.firebase.google.com)
2. 프로젝트 선택: Firebase 대시보드에서 Android 프로젝트가 설정된 프로젝트를 선택한다.
3. **프로젝트 설정으로 이동**: 화면 왼쪽 상단의 톱니바퀴 모양 아이콘을 클릭하고 '프로젝트 설정'으로 이동한다.
4. **앱 등록**: '일반' 탭에서 'Firebase 설정 추가' 섹션으로 이동하여 Android 애플리케이션이 등록되어 있는지 확인한다.
    - 아직 등록하지 않았다면, 'Android 앱 추가' 버튼을 클릭하여 앱을 등록한다.
    - 등록할 때 패키지 이름을 입력하고, SHA-1 키를 추가할 수 있다 (선택 사항).
5. **google-services.json 파일 다운로드**: 앱이 등록되면 Firebase 콘솔에서 `google-services.json` 파일을 다운로드할 수 있는 버튼이 생성된다. 이를 클릭하여 파일을 다운로드한다.

이 두 파일은 각각 서버와 클라이언트에서 Firebase를 사용하기 위해 중요한 파일이니, 보안에 신경 써서 관리해야 한다.