---
layout: single
title: "우분투 연결 가이드 3"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 복사 붙여넣기
일단, 리눅스의 붙여넣기는 Ctrl + shift + v 이다. 꼭 알고가자  
이걸 알아야 명령어 복사 붙여넣기도 하는거다.  

그런데, 본체 컴퓨터에서 리눅스의 cmd 를 열 순 있지만, 리눅스 내부 파일 내용에 붙여넣기가 안된다.  
이를 가능하게 하려면, 이걸 설치하면 된다.  

## 설치 방법
```bash
sudo apt update
sudo apt install autocutsel
```

실행은,  (이건 REALVNC로 연 cmd에서 쳐야 한다.)  
```
autocutsel -fork
```

이제, 본테에서 복사한게 리눅스 컴터로 잘 붙여넣어 질 것이다.  
이어서, 크롬, VS 코드 등 다 설지해 보자.  
