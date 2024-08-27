---
layout: single
title:  "FFmepeg 설치"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# FFmepeg 설치
이것저것 사이트를 만들 떄 FFmepeg 를 사용할 일이 있다.  
간단하게 FFmepeg를 설치하는 방법을 알아보자.  

## 1. 설피 파일 다운
아래 링크로 가서 파일을 다운 받자.  
[https://ffmpeg.org/](https://ffmpeg.org/)  
Download -> Windows -> ffmpeg-git-full.7z 을 클릭해서 다운받으면 된다.  

## 2. 환경변수 설정

위에서 받은 파일은 어디에 풀어도 상관 없다.  
그래도 가능하면  JDK 같은 파일들을 뭉펴 놓은 곳에 풀도록 하자.  
압축 해제한 후 bin 파일의 위치를 롼경변수에 추가하면 된다.  
사용자변수 -> Path 안에 bin 파일 위피를 집어넣으면 된다.  
이후, 다음 코드를 통해서 환경변숙가 잘 설정되었는지 아래 코드를 통해 확인하면 된다.  
```bash
ffmpeg -version
```

이상으로 ffmpeg를 다운받는 방법을 알아보았다.  