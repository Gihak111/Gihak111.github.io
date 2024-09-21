---
layout: single
title:  "리액트 네이티브 Foreground"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Foreground
React Native에서는 백그라운드 작업을 수행할 수 있는 기능이 있지만, Foreground Service를 통해 앱이 사용자에게 지속적으로 알림을 제공하면서 실행되는 형태의 기능을 구현할 수 있다.  
Foreground Service는 사용자가 앱을 사용하지 않더라도 앱이 계속해서 중요한 작업을 수행할 수 있게 해주는 유용한 도구다.  

### Foreground Service를 사용한 Android 알림 구현 방법

1. **Android Native 코드 설정**
   - Foreground Service는 Android 네이티브 코드에서 설정해야 한다. React Native 프로젝트의 `android/app/src/main/java/com/yourapp` 폴더에 새로운 클래스를 생성하자.  
   
   예를 들어, `MyForegroundService.java` 파일을 만들고 아래와 같이 작성한다.  

   ```java
   package com.yourapp;

   import android.app.Notification;
   import android.app.NotificationChannel;
   import android.app.NotificationManager;
   import android.app.PendingIntent;
   import android.app.Service;
   import android.content.Intent;
   import android.os.IBinder;
   import androidx.annotation.Nullable;
   import androidx.core.app.NotificationCompat;

   public class MyForegroundService extends Service {
       private static final String CHANNEL_ID = "ForegroundServiceChannel";

       @Nullable
       @Override
       public IBinder onBind(Intent intent) {
           return null;
       }

       @Override
       public int onStartCommand(Intent intent, int flags, int startId) {
           createNotificationChannel();

           Intent notificationIntent = new Intent(this, MainActivity.class);
           PendingIntent pendingIntent = PendingIntent.getActivity(this,
                   0, notificationIntent, 0);

           Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                   .setContentTitle("Foreground Service")
                   .setContentText("The service is running in the foreground")
                   .setSmallIcon(R.drawable.ic_notification)
                   .setContentIntent(pendingIntent)
                   .build();

           startForeground(1, notification);

           // Service logic goes here
           return START_NOT_STICKY;
       }

       private void createNotificationChannel() {
           NotificationChannel serviceChannel = new NotificationChannel(
                   CHANNEL_ID,
                   "Foreground Service Channel",
                   NotificationManager.IMPORTANCE_DEFAULT
           );

           NotificationManager manager = getSystemService(NotificationManager.class);
           if (manager != null) {
               manager.createNotificationChannel(serviceChannel);
           }
       }

       @Override
       public void onDestroy() {
           super.onDestroy();
           stopForeground(true);
       }
   }
   ```

   **주요 코드 설명**:
   - **Notification Channel**: Android 8.0 이상에서는 Foreground Service를 위해 Notification Channel을 생성해야 한다.  
   - **Notification**: 서비스가 실행 중일 때 표시될 알림을 정의한다.  
   - **Foreground Service**: `startForeground` 메서드를 호출해 서비스를 foreground로 설정하고, 알림을 표시한다.  

2. **AndroidManifest.xml에 서비스 등록**
   서비스는 AndroidManifest.xml 파일에 등록해야 한다. 해당 파일을 열고, 아래 내용을 추가하자.  

   ```xml
   <service
       android:name=".MyForegroundService"
       android:enabled="true"
       android:exported="false" />
   ```

3. **JavaScript 코드에서 서비스 시작**
   React Native에서 Native 모듈을 호출해 Foreground Service를 시작할 수 있다. 이를 위해, `NativeModules`를 사용하자. 먼저, React Native 코드에서 JavaScript로 서비스를 시작하는 방법을 구현하자.  

   ```javascript
   import { NativeModules } from 'react-native';
   
   const startForegroundService = () => {
       NativeModules.MyForegroundService.startService();
   };
   
   export default function App() {
       return (
           <View>
               <Button title="Start Foreground Service" onPress={startForegroundService} />
           </View>
       );
   }
   ```

4. **알림 내용 커스터마이징**
   Foreground Service를 통해 실행 중인 동안 카메라가 열리거나 KakaoTalk 같은 SNS 앱이 열릴 때마다 알림을 표시할 수 있다.  
   예를 들어, 카메라가 열릴 때는 다음과 같은 알림을 보내자.  

   ```java
   Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
           .setContentTitle("Camera Opened")
           .setContentText("Your camera has been opened")
           .setSmallIcon(R.drawable.ic_camera)
           .setContentIntent(pendingIntent)
           .build();
   ```

   KakaoTalk이 열릴 때는 다음과 같은 알림을 설정할 수 있다.

   ```java
   Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
           .setContentTitle("KakaoTalk Opened")
           .setContentText("KakaoTalk has been opened")
           .setSmallIcon(R.drawable.ic_chat)
           .setContentIntent(pendingIntent)
           .build();
   ```

5. **주요 구현 흐름 요약**
   - **Foreground Service 구현**: Foreground Service는 Android 네이티브 코드에서 설정해야 한다. 서비스가 실행되면 알림을 통해 사용자에게 중요한 정보나 앱 상태를 알릴 수 있다.  
   - **JavaScript로 제어**: React Native에서는 네이티브 모듈을 통해 Foreground Service를 제어할 수 있다.  
   - **알림 커스터마이징**: 서비스가 실행 중일 때 표시되는 알림은 자유롭게 커스터마이징이 가능하며, 앱의 상태에 맞춰 적절한 정보를 표시할 수 있다.  

이 방법을 통해 React Native 앱에서도 Foreground Service를 쉽게 구현할 수 있다.  
Foreground Service는 배터리 최적화에 방해받지 않으면서 중요한 작업을 지속적으로 수행할 수 있도록 보장해주는 중요한 기능이다.  