---
layout: single
title:  "웹뷰 앱 만들기"
categories: "Expo"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 웹 사이트를 바로 앱으로 만들기
Expo를 통해 간단히 웹사이트를 앱으로 만들 수 있다.  
이 글에선, 다른 사람이 만들어 놓은 탬플릿으로 도메인을 만들고, 그 도메일을 앱으로 만드는 것을 해볼 것이다.  

#### 탬플릿을 통한 웹 만들기  
구글에 html template free 라고 쳐 보자.  
무료 페이지가 엄청나게 많이 나온다.  
그 중에, 마음에 드는걸로, 다운 받는다.  
저작권이 MIT로 붙어있는데, 이는 저작권에 큰 문제가 없다는 뜻이니 들어가서 내용을 한번 확인해 보자.  
파일을 다운받고 압축을 풀면,  
index html을 찾아서 실행하면 된다.  
저장한 사이트가 로컬에서 실행되는 것을 알 수 있다.  
내부 html의 타이틀이나, 헤더 같은거를 저장하면 된다.  

#### Deploy
이걸 이제 도메인에 배포해 보자.  
이런 과정을 Deploy 라고 한다.  
방법은 2가지가 있는데,  
1. 내 컴퓨터를 서버로 공개하기  
2. 외부 서버 빌리기  

1번 방법은 상시 내 컴퓨터를 켜놔야 하므로 2번으로 가자.  
클라우드 서버를 통해 진행할 수 있다.  
aws, microsoft 등이 제공하는 서버는 높은 질을 가지고 있지만, 돈은 내야 하는 단점이 잇다.  
간단하게 해 볼거기 떄문에,  netlify를 통해 무료로 진행해보자.  
netlify.com 으로  들어가서 회원가입 해 주면 된다.  
매뉴의 sites로 들어가서, 아까 index.html이 있는 폴더를 드래그 앤 드랍으로 업로드를 하면 된다.  
업로드가 끝나면, 주소가 생기는데, 그 주소로 접속하면 공개된 주소가 생긴다.  
자동 생선된 url을 바꿀려면, netlify로 돌아와서 site settings으로 가면 된다.  
Change sitename을 통해 이름을 바꾸는 것으로 url을 바꿀수 있다.  
수정은 깃허브 블로그 마냥, 수정 후 아까 홈페이지의 Deploys로 들어가서 그래그 앤 그랍을 하면 된다.  
이렇게 쉽게 웹사이트를 하나 만들 수 있다.  
기본적인 html 내용을 알고 잇으면 엄청 쉽게 할 수 있다.  

#### url을 활용한 모바일 앱 제작
이번엔 Expo로 할 꺼지만, 리액트 네이티브나, 다른걸로도 할 수 있다.  
일단, 가장 간단한 Expo로 해보자.  

일단, node.js를 설치 해야 한다. 인터넷에 검색하면, 쉽게 다운 받을 수 잇따.  
vscode, git도 있어야 한다. 웬만하면 다 깔려 있으니 스킵하겠다.  

Expo에 회원가입 할 필요가 있다.  
https://expo.dev/
위 링크로 들어가서 화원가입 하자.  
이어서, 핸드폰으로도 Expo 앱이 있으니 다운 받도록 하자.  

이어서 밑의 코드를 통해서, 프로젝트를 만들자.  이는 엑스포 프로젝트를 만드는 것이다.  
```bash
npm install -g expo-cli
npx creat-expo-app my-app
```
위 코들르 통해서 my-app을 만들 수 있다.  
오류가 엄청나게 남으로, c 드라이브에 영문 이름 폴더 하나 만들고 거기서 시작하자.  
cmd에서 cd 파일위치, 위의 코드를 통해 프로젝트를 만들고, vscode로 폴더를 열자.  

이어서, 설치가 도니 cmd에 다음을 입력해서 설치하자.  
```bash
npm install -g eas-cli
```
이어서, eas를 통해 로그인 하자.  
```bash
eas login
```
자신의 계정을 집어넣으면 된다.  
```bash
eas build:configure
```
위 코드를 통해서 설정한다.  
오류가 난다면, cd를 통해 폴더 안쪽으로 들어가서 실행하면 된다.  
Y 를 입력하여 나오는 설정들을 전부다 해 준다.
윈도우에서 진행한다면, ios빌드는 불가능 하므로 참고하자.  

#### apk 파일 만들기
먼저, apk 파일을 만들고 폰에서 실행해 보자.
보통 구글플레이스토어 같은데에는 aab 파일을 만들어 올리지만, 그 전에 먼저 폰에서 실행해 보자. 
https://docs.expo.dev/build-reference/apk/
밑의 링크에 가면 상세히 설명되어 있다.  
expo가 공식 문서가 잘 되어 있는게 참 좋다.  

시키는대로, eas.json에 다음의 코드를 추가하자.  
```json
{
  "build": {
    "preview": {
      "android": {
        "buildType": "apk"
      }
    },
    "preview2": {
      "android": {
        "gradleCommand": ":app:assembleRelease"
      }
    },
    "preview3": {
      "developmentClient": true
    },
    "preview4": {
      "distribution": "internal"
    },
    "production": {}
  }
}
```

이어서, ```eas build -p android --profile preview```를 cmd에 하는 것으로 빌드할 수 있다.  
id 설정하라고 나오는데, 이 주소는 고유해야 하며, com.~~~.앱이름 으로 진행하면 된다.  
보통 com. 회사이름. 앱 이름 이런식으로 한다. 당연히, 영어만 사용하도록 하자.  
keystore 역시 y를 눌러서 빌드 하도록 하자.  
이어서 나온 주소로 들어가서, Apk를 다운받고, 모바일에서 실행하면 설치되며 실행역시 잘 된다.  

#### 웹뷰 앱으로 만들기
위의 과정을 통해 모바일 앱을 만들어 봤으므로, 웹뷰 앱을 만들어 보자.  
https://docs.expo.dev/versions/latest/sdk/webview/
위 링크에 정말 자 ㄹ설명되어 있다. 엑스포가 좋긴 하네

먼저, ```npx expo install reat-native-webview``` 를 설치하자.  
과정은 항상 앱 안의 위치에서 실행되어야 한다.  
이어서, 조건에 맞게 수정하여 밑의 코드를 변경하면 된다.  
{% raw %}
```js
import { WebView } from 'react-native-webview';
import Constants from 'expo-constants';
import { StyleSheet } from 'react-native';

export default function App() {
  return (
    <WebView
      style={styles.container}
      source={{ uri: 'https://expo.dev' }}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: Constants.statusBarHeight,
  },
});

```
{% endraw %}
누가 봐도 변경하면 될 것 같은 부분이 있다.
App.js 파일을 열러서 니용을 전부 날리고, https://expo.dev 를 원하는 웹뷰 링크로 바꿔서 수정하자.  
css는 날려도 된다.  
asset  같은 파일들도 보면 대충 뭔지 알꺼다.  
다 쑤시면서 원하는 대로 바꾸면 된다.  
srs 폴더 만들어서 백인드 만들면 된다.  

돌아가서, 코드를 위처럼 수정했으면 ```eas build -p android --profile preview``` 코드를 통해서 빌드하면 된다.  
이어서, 생긴 링크로 apk를 다운받고, 설치해서 웹뷰가 잘 작동하는지 확인하면 된다.  
정말 간단하게 만들 수 있다.  