---
layout: single
title:  " Expo 실행시 오류 설치"
categories: "pynb"
tag: "ERRORE"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Expo 실행 오류
Expo 앱을 실행할 때 이런 오류가 나오곤 힌다.  
```bash
TypeError: Invalid character in header content ["X-React-Native-Project-Root"]
    at ServerResponse.setHeader (node:_http_outgoing:662:3)
    at statusPageMiddleware (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\@react-native-community\cli-server-api\build\statusPageMiddleware.js:19:7)
    at call (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:239:7)
    at next (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:183:5)
    at next (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:161:14)
    at next (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:161:14)
    at next (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:161:14)
    at next (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:161:14)
    at nocache (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\nocache\index.js:11:5)
    at call (C:\Users\Desktop\공모전\AI공모전\AI공모전 K-PASS\ver3\app10\CameraApp\frontend\node_modules\connect\index.js:239:7)
```

위 오류를 보고 리액트 네이티브 관련 오류라고 착각하고 삽질 할 수 있다.  
하지만, 위 오류는 라이브러리 주소에 한국어가 들어가 있어서 나오는 오류이다.  
C 드라이브에 옮겨서 실행하면 완활하게 실행된다.  
