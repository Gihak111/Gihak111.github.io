---
layout: single
title: "tail scale"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Tail scale
이걸 쓰면 공짜로 포트 포워딩 없이 노트북 서버 등등을 vpn으로 연결할 수 있다.  
외부망이 아니라 디도스 받을 위험도 적고 방법도 구글 로그인 하나만으로 연결할 수 있어서 쉽고 편하다.  
아무튼 좋은 기능이니, 한 번 깔아보자.  

## 설치 + 실행

이걸로 설치하면 된다.  
```bash
curl -fsSL https://tailscale.com/install.sh | sh

```

그리고 이거 치면 로그인 하라고 링크 주니까 그거 들어가서 로그인 하면 된다.  
구글 로그인도 된다.  
```bash
sudo tailscale up

```

또 가끔 로그아웃 될 경우도 있고 한데, 그럴 때는 이거 쳐서 지금 상태도 확인할 수 있고, 그렇다.  
```bash
tailscale status

```

이걸로 아이피 딴 다음 다른 컴퓨터ㅓ에서 저 아이피로 접속하면 된다.  
```bash
ip addr show tailscale0

```


그냥 나도 계속 명령어 까먹어서 정리해 봤다.  