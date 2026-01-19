---
layout: single
title: "우분투 연결 가이드 4"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 이제, 딥러닝 환경을 만들어 보자  
일단, 크롬 등 설치해야 한다.  
그런데, 우분투에 기본으로 깔려있는 파이어폭스가 VNC 환경에서 보안문제가 있어서 켜지지 않는 그런 문제도 있고 그러니까  
꼭 크롬을 깔면 좋다.  
때문에, wget 등 사용해서 설치를 해 보자.  

##  크롬 깔아보자.
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y
rm google-chrome-stable_current_amd64.deb
```
이렇게 설치하면 크롬이 깔리는데,  
설치 후에도 터미널에서 바로 실행하면 `Missing X server` 에러가 뜨곤 하는데,  
이건 VNC 화면 내의 메뉴(Applications → Internet)를 클릭하거나, 터미널에서 `DISPLAY=:1` 옵션을 붙여서 해결할 수 있다.  

## VSCODE
```bash
wget -O code_latest.deb 'https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64'
sudo apt install ./code_latest.deb -y
rm code_latest.deb
```
위 코드로 설치할 수 있다.  

##  python 가상환경 등
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
source ~/miniconda3/bin/activate
conda init --all
```
그런데, 이렇게 콘다깔면 "Terms of Service have not been accepted" 이러면서 막힌다.  
이는 정책이 바뀌어서 이런거다. 동의 해 주면 된다.  
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

이어서 가상환경 만들고,  
```bash
conda create -n start python=3.12.4 -y
conda activate start
pip install torch torchvision torchaudio

```
재부팅 하고 실행하면 된다
```bash
sudo reboot
```

