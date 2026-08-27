---
layout: single
title: "우분투 데스크탑에서 서버로 바꾸기"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Desk top
우분투 데스크탑은 처음 리눅스 접하는 사람들이 UI 없이 하는게 힘들어 서낵할 수 있는 좋은 선택지 이다.  
하지만, 어 마 나 리눅스 한다 게이야 라고 하기에는 UI가 있는거부터 짜칠 수 있다.  
근데 이미 우분투 데스크탑으로 이것저것 씹고 뜯고 다 했을텐데 그걸 또 언제 다 날리고 어 다시 다 설정하고 하냐 힘들게  
이럴 떄 간단하게 데스크탑을 서버로 쏙 바꿔치고 그 외 다른 내용들은 전부 유지할 수 있는 방법이 있다.  

## 야호야호
이건 정말 정석의 방법이니, 바로 진행해 보자.  
1. 먼저 업데이트 한다.  
```bash
sudo apt update && sudo apt upgrade
```

2. 우분투 서버 일단 깔자  
```bash
sudo apt install ubuntu-server
```

3. 이어서, 데스크탑에 있는 로그인 화면 등 리소스들을 제거하자  
```bash
sudo apt remove ubuntu-desktop gdm3
```

4. 이후 연관되어있는 관련 패키지들 싹 다 날리자  
```bash
sudo apt autoremove
```

5. 이제 부팅하고 바로 터미널로 시작되게만 설정하면 된다.  
```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

자 그런데 저래 하면 한 가지 문제가 있다.  
바로 XFCE4 같이 RealVNC로 접속하고 싶어도 우분투 자체에 UI 그리는 모든 내용이 싹 다 날라가버리기 때문에 저 기능이 막힌다는 것이다.  
이럴때를 대비해서 위 방법 말고 다른 방법이 있는데,  
앞선 1 ~ 5 실행하지 말고,  

1. 그냥 아래 명령어 실행한다.  
```
sudo systemctl set-default multi-user.target
sudo reboot
```

저래버리면 그냥 UI 그릴 수 있는 패키지는 남겨둔 채로 재부팅 시 바로 터미널로 켜져서 마 나 리눅스 촘 친다카이 하면서 꺼드럭 댈 수 있다.  

## 결론
야호야호  
퇴근하고 싶다 야호  