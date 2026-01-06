---
layout: single
title: "우분투 설치 가이드"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## ubuntu
이번에, 컴퓨터를 새로 하면서 기존의 컴퓨터는 우분투로 AI 서버를 만들었다.  
이 과정을 정리해 보았다.  

### 1. 설치 USB 제작 (윈도우 PC)
우분투를 설치하려면 부팅 가능한 USB 드라이브가 필요하다.  

1. ISO 파일 다운로드:  
우분투 공식 홈페이지에서 'Ubuntu 24.04 LTS Desktop' ISO 파일을 다운로드한다.  

2. 부팅 디스크 제작:  
8GB 이상의 USB를 꽂고, Rufus (또는 BalenaEtcher) 프로그램을 실행하여 다운로드한 ISO 파일을 USB에 굽는다.  



### 2. BIOS 진입 및 부팅 순서 변경

설치할 컴퓨터에 USB를 꽂고 전원을 켠다.

1. BIOS 진입:  
부팅 로고가 뜰 때 `F2`, `Del`, 또는 `F12` 키를 연타하여 BIOS 설정 화면으로 진입한다.  

2. Secure Boot 해제:  
원활한 드라이버 설치(특히 그래픽카드)를 위해 'Secure Boot' 항목을 `Disabled`로 변경하는 것을 권장한다.  

3. 부팅 순서 변경:  
'Boot Priority'에서 USB 드라이브를 1순위로 올린다.  

4. 저장 및 재부팅:  
설정 저장(`F10`) 후 재부팅하면 우분투 설치 화면이 나타난다.  



### 3. 우분투 설치 진행
'Try or Install Ubuntu'를 선택하여 설치 마법사를 시작한다.  

1. 언어 선택:  
'English'를 권장한다. 한글 설정 시 폴더명이 한글로 바뀌어 터미널 경로 입력이 불편해진다.  

2. 키보드 레이아웃:  
`Korean` - `Korean (101/104 key compatible)`을 선택한다.  

3. 업데이트 및 소프트웨어:  'Install third-party software'를 반드시 체크한다. 두 개 있는데, 둘 다 선택하는게 좋다. 이는 그래픽카드 및 와이파이 드라이버 설치를 위함이다.  

4. 설치 형식:  
윈도우를 지우고 단독으로 쓸 경우 `Erase disk and install Ubuntu`를 선택하고 , 윈도우와 같이 쓸 경우 `Install Ubuntu alongside Windows`를 선택한다.  

5. 사용자 설정:  
접속할 이름과 **비밀번호**를 설정한다. 이 비밀번호는 SSH 접속과 `sudo` 명령어 사용 시 필수이므로 잊지 않도록 주의한다.  



### 4. 설치 직후 필수 세팅 (원격 접속용)

설치가 끝나고 재부팅되면, 터미널(`Ctrl`+`Alt`+`T`)을 열고 아래 과정을 즉시 수행한다.  

1. 패키지 업데이트  
```bash
sudo apt update && sudo apt upgrade -y

```

2. SSH 서버 설치 및 활성화  
이것을 설치해야 다른 컴퓨터에서 `ssh` 명령어로 접속할 수 있다.  

```bash
sudo apt install openssh-server -y
sudo systemctl enable ssh
sudo systemctl start ssh

```

3. IP 주소 확인 도구 설치  
현재 내 IP를 확인하기 위해 `net-tools`를 설치한다.  

```bash
sudo apt install net-tools -y
ifconfig

```
출력된 `inet` 주소(예: `192.168.35.173`)가 접속 주소가 된다.  

### 5. 한글 입력기 설정  

기본 상태에서는 한글 입력이 안 될 수 있다.

1. `Settings` -> `Region & Language` -> `Manage Installed Languages` 클릭 후 설치 팝업이 뜨면 설치한다.  
2. `Keyboard Input Method System`을 **IBus**로 확인한다.  
3. `Settings` -> `Keyboard` -> `Input Sources`에서 `Korean (Hangul)`을 추가한다.  

이 과정을 마치면 앞서 진행했던 VNC 설정 및 원격 접속이 가능한 상태가 된다.  