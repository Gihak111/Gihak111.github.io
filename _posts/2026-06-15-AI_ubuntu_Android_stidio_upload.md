---
layout: single
title: "안드로이드 스튜디오 설치"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 우분투 안드로이드 스튜디오

예전과는 다르다.  
스쿱으로 깔고 JDK SDK 염병을 하던 예전의 그 개빡치는 안드로이드 스튜디오가 아니다.  
CUDNN도 그렇게 점점 편해지는 환경에 맞춰, 이제 안드로이드 스튜디오도 딸깍 설치가 가능하다.  
우분투 기준으로 알아가보자.  

## 방법

**1. 안드로이드 스튜디오 설치**  
Snap 패키지 매니저를 사용하여 터미널 명령어 한 줄로 설치한다.  
별도의 JDK 다운로드나 환경 변수 설정이 필요 없다.  

1. `Ctrl + Alt + T`를 눌러 터미널 연다.  
2. 아래 명령어를 입력하고 실행한다.  
```bash
sudo snap install android-studio --classic
```


**2. 실행 SDK 설치**  
cmd에  
```bash
android-studio
```
하면 안드로이드 스튜디오 켜질거다.  
이어서, 그냥 설치창 하라는대로 막 누르면 알아서 설치된다.  
이게 말이 안된다. build tool + SDK JDK 다 뇌 빼고 해도 알아서 된다.  

**3. 에뮬레이터 실행**
이제 디바이스 추가하고 실행하면 오류난다.  
그게 접근권한 없어서 그런거니까, 아래 코드 cmd에 쳐서 해결하자.  
```bash
sudo apt update
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils

sudo adduser $USER kvm
```
위에 저거 치고 재부팅 하면 된다.  

만일, 즉시 해결하고 싶으면,  
```bash
sudo chmod 777 /dev/kvm
```
또는  
```bash
sudo chown $USER /dev/kvm
```
근데 위 두 방법은 나중에 다시 오류 날 수도 있다.  
저러고 실행하면 잘 될것이다.  