---
layout: single
title: "NVIDIA Isaac Sim with Tailscale"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Omniverse
엔비디아에서 제공하는 강력한 툴이다.  
로봇 강화학습하기 좋으니, 우리도 한번 써 보자.  


### 1단계: 리눅스 환경 준비 및 파이썬 3.10 설치  

```bash
# 1. 기존 아나콘다 환경 확실히 끄기
conda deactivate

# 2. 파이썬 3.10 및 venv 필수 모듈 설치sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
sudo apt install git

# 3. 가상환경 생성 및 활성화
python3.10 -m venv robot_sim
source robot_sim/bin/activate

# 4. 파이썬 버전 최종 확인 (여기서 반드시 Python 3.10.x 가 출력되어야 함)
python -V
```  

### 2단계: Isaac Sim 코어 및 Isaac Lab 설치  

```bash
# 1. pip 업데이트 및 Isaac Sim 코어 설치
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com

# 2. Isaac Lab 저장소 복제 및 리눅스 전용 설치 스크립트 실행
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 의존성 패키지 설치./isaaclab.sh --install
```  

### 3. 실행 테스트  
```bash
python -c "import isaacsim; print('suc')"
```

### 4. 이것 저것 테스트  
```bash
# 빈공간 띄우기
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# 로봇개 띄우기
./isaaclab.sh -p scripts/demos/quadrupeds.py

# 데모 파일 목록 확인하기
ls scripts/demos/

# 튜토리얼 파일 목록 확인하기
ls scripts/tutorials/
```  

### 5. 만일, 앞서 내가 한 방법인 VNC로 보고 있다면?  
이러면 또 GPU 가속 오류 나면서 안켜진다.  
이럴땐, 뒤에 다음을 붙여보자.  
```bash
--livestream 2
```  
예를 들면,  
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```  
이런식으로.  
저러면 잘 열릴 것이다.  

끄는 방법은, 컨트롤 c , 컨트롤 백슬래시로 꺼야 한다.  

아래를 실행에서 어디서 앱이 열려있는지 확인하자.  
```bash
ss -tulpn | grep python
```  
나온 결과의 포트를 뒤 포트에 입력하는걸로 접속할 수 있다.  

+ 만일, 본체에 글카다 두 개 라면,  
```bash
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```  
이러면 다른 글카로 켜지게 된다. 나이스  

### 6. 이제 크롬에서 접속을 진짜 해 보자.  
이거도 방법이 두 개가 있다. 먼저, 이걸 시도해 보자.  
1. 노트북에서 열기  
삽질을 엄청 했는데도 접속이 안된다.  
방법은 따로 있는데, 우선 아래 링크로 들어가자.  
[링크](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)  
더 정확히는, 아래의 링크이다.  
[더 정확한 링크](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release)  
여기서 Isaac Sim WebRTC Streaming Client 이걸 다운받고 설치하면 된다.  

영상이 지나갈 포트를 잘 열어주고,  
```bash
# 1. 시그널링 및 데이터 전송 포트 허용
sudo ufw allow in on tailscale0 to any port 49100:49200 proto udp
sudo ufw allow in on tailscale0 to any port 49100:49200 proto tcp

# 2. 스트리밍 기본 제어 포트 허용
sudo ufw allow in on tailscale0 to any port 8011 proto tcp

# 3. 설정 적용
sudo ufw reload
```  
1이 안되면, 그냥 리눅스에 뷰어 깔아서 리눅스로 봐야 방화벽 DISCONNECT 문제를 해결할 수 있다.  

2. 리눅스에 깔아서 해 보기  
아까 [더 정확한 링크](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release) 에 들어가서 리눅스 버전을 다운 받자.  
이어서, 아래를 리눅스 cmd에 치면 된다.  
```bash
sudo apt update
sudo apt install libfuse2 -y
cd ~/Downloads
chmod +x isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage --no-sandbox
./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
```

위 명령어 치면 리눅스에서 열리긴 한다.  
하지만, 아쉬우니, 다음엔 DISCONNECTON 문제를 해결해서 와 봐야 겠다.  
