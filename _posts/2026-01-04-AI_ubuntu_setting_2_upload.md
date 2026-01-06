---
layout: single
title: "우분투 연결 가이드 2"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## VNC Remote Connection
우분투 설치가 완료되었다면, GUI(그래픽 사용자 인터페이스) 환경을 원격으로 제어하기 위해 VNC 설정을 진행한다.  
이 과정은 SSH 터미널에서 진행하며, 한 번 설정해 두면 이후에는 윈도우에서 화면을 보며 작업할 수 있다.  

### 1. 초기 설치 및 설정 (최초 1회)
SSH로 접속(`ssh joonmo-yeon@192.168.35.173`)한 상태에서 다음 명령어들을 순서대로 수행하여 환경을 구축한다.  

1. 필수 패키지 설치:  
가벼운 데스크탑 환경인 XFCE와 VNC 서버를 설치한다.   
```bash
sudo apt update
sudo apt install xfce4 xfce4-goodies tightvncserver -y
```

2. VNC 비밀번호 설정:  
서버를 처음 실행하면 암호를 설정한다. `view-only password`는 `n`을 선택한다.   
설정이 끝나면 설정을 위해 바로 세션을 종료한다.   
```bash
vncserver
vncserver -kill :1
```

### 2. 화면 설정 파일(xstartup) 수정
VNC가 XFCE 환경을 제대로 불러오도록 설정 파일을 수정해야 한다. 이 과정이 가장 중요하다.   

1. 파일 열기:  
```bash
nano ~/.vnc/xstartup
```

2. 내용 수정:  
**기존 내용을 모두 지우고** 아래 내용으로 채운다.   
```bash
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &
```
작성 후 저장(`Ctrl+O`, `Enter`)하고 종료(`Ctrl+X`)한다.  

3. 실행 권한 부여:  
```bash
chmod +x ~/.vnc/xstartup
```

### 3. VNC 서버 실행 (재부팅 시 필요)
컴퓨터가 재부팅되면 VNC 서버도 꺼지므로, SSH로 접속하여 다시 켜줘야 한다.   

1. 서버 실행:  
해상도를 지정하여 실행한다. (예: 1920x1080)   
```bash
vncserver -geometry 1920x1080
```

2. 포트 확인:  
실행 후 메시지에 `New 'X' desktop is ... :1`이 뜨는지 확인한다.  
숫자가 `:1`이면 포트는 `5901`, `:2`면 `5902`가 된다.   

### 4. Windows에서 접속하기

윈도우 PC에서 'RealVNC Viewer' 등을 실행하여 접속한다.   
1. 접속 정보 입력:  
- IP 주소 : `192.168.35.173`  
- 포트 번호 : `:5901` (세션이 `:1`일 경우)  
- 최종 입력값 : `192.168.35.173:5901`  

2. 인증:
앞서 설정해둔 VNC 비밀번호를 입력하면 화면이 뜬다.  


### 5. 유용한 관리 명령어
서버를 끄거나 목록을 확인할 때 사용한다. 

1. 서버 끄기 (세션 초기화):  
```bash
vncserver -kill :1
```
(세션 번호가 2번이면 `:2`로 입력)   

2. 현재 실행 중인 서버 목록 확인:  
```bash
vncserver -list
```