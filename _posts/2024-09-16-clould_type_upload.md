---
layout: single
title:  "Node.js 클라우드에 배포"
categories: "SQL"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Node.js 서버에 올리기
[https://app.cloudtype.io/](https://app.cloudtype.io/)
간단하게 무료로, 너가 만든 서버를 공개 도메인으로 무료로 바꿀 수 있게 해 주는 사이트가 있다.  
위 사이튼느 한국을 대상으로 무료로 동개된 도메인을 제공해 준다.  
아주 아름당누 사이트다.  

## cloudtype
이는 깃허브 레포지토리랑 연결되어, Dploy 할 수 있게 해 준다.  
프로젝틀르 만들고, 배포하기를 눌러서 필요한 정보를 집어넣는 것으로 쉽고 빠르게 배포할 수 있다.  
다만, Node.js 파일을 기준으로 배포할 떄, 확인해야 할 것들이 몇개 있다.  

1. Node.js 버전
cmd에 ```node -v```  를 치는 것으로 버젼을 맞춰서 배포하자.  

2. Start Vommand
일반적으론 ```npm start``` 가 기본으로 되어있다.  
파일을 만들고, 로컬에서 테스트 했을때 사용한 커맨드로 바꿔줘자.
예를 들면, ```node index.js``` 이런식으로 말이다.  

3. Install Command
```npm ci --production``` 코드를 통해서 package-lock에 있는 기준으로 설치한다는 커맨드다.  
위 코드로 집어넣으면 오류가 적어서 좋다.  

설정을 마치고, 배포한 후 도메인에 들어가 Api 테스트 하는 것으로 잘 배포 되었는지 확인하자.  
실행중 이라고 파란색으로 나오면 성공한거다.