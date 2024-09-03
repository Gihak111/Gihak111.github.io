---
layout: single
title:  "리엑트 네이티브 버젼에 따른 homescreen 오류"
categories: "ERROR"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Home scewwn
Expo 앱 빌드시 내부의 리엑트 네이티브 버젼과 맞지 않을때, homescreen을 찾지 못한다는 오류가 나온다.  
이는 SDK 48 이후에 import 방식이 바뀌었기 때문이다.  

## 바뀐 import 형식
### Expo Router v1.0
`React Navigation` 같은 라이브러리에서 `Homescreen`과 같은 컴포넌트를 import하는 방식이 버젼마다 달라졌다.  
### 1. **기존 형식:**

예를 들어, 예전 `React Navigation v4`에서 homescreen을 import하는 방식은 다음과 같았다.

```javascript
import { createStackNavigator, createAppContainer } from 'react-navigation';
import HomeScreen from './screens/HomeScreen';

const AppNavigator = createStackNavigator({
  Home: {
    screen: HomeScreen,
  },
});

export default createAppContainer(AppNavigator);
```

### 2. **바뀐 형식:**

`React Navigation v5` 이후로는 `createStackNavigator`와 `createAppContainer`의 사용 방식이 바뀌었고, `import` 방법도 약간 달라졌다.

```javascript
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './screens/HomeScreen';

const Stack = createStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
```

이 경우, `createAppContainer`는 더 이상 필요하지 않으며, 대신 `NavigationContainer`가 사용된다.  

결론을 내자면, 리엑트 네이티브 v5 이후에는 import 방법이 바뀌었다. 최신 방법으로 import 하자.  


### 출처:
- [React Navigation v4 Documentation](https://reactnavigation.org/docs/4.x/getting-started/)  
- [React Navigation v5 Documentation](https://reactnavigation.org/docs/5.x/getting-started/)