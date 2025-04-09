---
layout: single
title:  "Expo 프로젝트 백그라운드에서 실행하기"
categories: "Expo"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Expo
일반적으로 Expo앱은 백그라운드에서 실행되지 않는다.  
하지만, 여러 API를 활용해서 구현할 수 있다.  

#### 의존성 다운
다음의 코드를 cmd에 실행시키자.  
```sh
expo install expo-task-manager expo-background-fetch
```

이어서, 백그라운드 작업 하면 된다.  
taskManager.js
```javascript
import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';

const BACKGROUND_FETCH_TASK = 'background-fetch-task';

TaskManager.defineTask(BACKGROUND_FETCH_TASK, async () => {
  try {
    // 이곳에 백그라운드에서 실행할 코드를 작성합니다.
    console.log('Background fetch task executed');

    // 성공적으로 완료되었음을 알립니다.
    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch (error) {
    // 오류가 발생했음을 알립니다.
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

const registerBackgroundFetchAsync = async () => {
  return BackgroundFetch.registerTaskAsync(BACKGROUND_FETCH_TASK, {
    minimumInterval: 15, // 작업이 실행될 최소 간격 (초 단위)
    stopOnTerminate: false, // 앱이 종료되어도 작업을 멈추지 않음
    startOnBoot: true, // 기기가 부팅되면 작업 시작
  });
};

const unregisterBackgroundFetchAsync = async () => {
  return BackgroundFetch.unregisterTaskAsync(BACKGROUND_FETCH_TASK);
};

export { registerBackgroundFetchAsync, unregisterBackgroundFetchAsync };
```

App.js  
```javascript
import React, { useEffect } from 'react';
import { Text, View } from 'react-native';
import { registerBackgroundFetchAsync } from './taskManager';

export default function App() {
  useEffect(() => {
    const initBackgroundFetch = async () => {
      await registerBackgroundFetchAsync();
    };

    initBackgroundFetch();
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Background Task Example</Text>
    </View>
  );
}

```  

이왕 만드는거 상단 바도 만들어 보자.  
백그라운드에서 실행중임을 알리기 위해 다음 패키지를 추가하자  
```sh
expo install expo-notifications
```
notifications.js  
```javascript
import * as Notifications from 'expo-notifications';
import * as Permissions from 'expo-permissions';

const scheduleNotification = async () => {
  // 알림 권한 요청
  const { status } = await Permissions.askAsync(Permissions.NOTIFICATIONS);

  if (status !== 'granted') {
    alert('Notification permissions not granted!');
    return;
  }

  // 알림 스케줄
  await Notifications.scheduleNotificationAsync({
    content: {
      title: "Background Task",
      body: "The app is running in the background.",
    },
    trigger: null, // 즉시 알림
  });
};

export { scheduleNotification };

```  

이어서, 앞서 만들었던 taskManager.js 코들르 업데이트 해 주면 된다.  
```javascript
import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';
import { scheduleNotification } from './notifications';

const BACKGROUND_FETCH_TASK = 'background-fetch-task';

TaskManager.defineTask(BACKGROUND_FETCH_TASK, async () => {
  try {
    // 백그라운드에서 실행할 코드
    console.log('Background fetch task executed');

    // 알림 스케줄
    await scheduleNotification();

    // 성공적으로 완료됨
    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch (error) {
    // 오류 발생
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});
```

이제, 이 앱은 백그라운드에서 실행될 수 있으며, 상단에 알림으로 알려준다.  
주석이 들어가 있는곳에 로직을 구현해 주는 것으로 간단하게 만들 수 있다.  