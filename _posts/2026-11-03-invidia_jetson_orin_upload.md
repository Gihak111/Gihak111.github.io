---
layout: single
title: "엔비디아 젯슨 오린 초기 설정"
categories: "Lab"
tag: "nvidia"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 엔비디아 젯슨
솔직히 말하면, 가성비 면에서 라즈베리파이보다 구리다.  
뭔가 막 엄청 빠르고 쿠다코어도 막 들어있고 엄청 좋은데, ROS2 같은거 쓰는 상황이 아니면 그냥 라파 쓰는게 돈도 아끼고 더 좋다.  
하지만, 만일 강화학습 디지털트윈 기타 등등 좀 무거운걸 돌리고 싶다 그러면 젯슨도 나쁘지는 않은 것 같다.  
나는 드론에 박아두고 RIS 컨트롤 + ROS2 용도로 사용중인데 나름 좋다.  
os도 sd 카드에 설치할 수 도 있고, m.2 막 2T 이런거에 넣을 수 도 있는데, 그 방법을 전부 다 정리해 보겠다.  

## os 설치
일단 나는 우분투에서 설치했다.  
우분투 있어야 설치가 더 쉽더라.  
본 설명도 내가 진행하고 성공한 방식에 따라 우분투를 기준으로 작업하겠다.  

1. 먼저 os 파일을 다운받자.  
```bash
mkdir -p ~/jetson_flash && cd ~/jetson_flash
sudo apt update
sudo apt install -y wget lbzip2 build-essential bc zip unzip libxml2-utils

# 1. Jetson Linux BSP (드라이버 및 플래싱 툴) 다운로드
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/release/Jetson_Linux_R36.4.3_aarch64.tbz2

# 2. Sample Root Filesystem (우분투 22.04 기본 OS 이미지) 다운로드
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/release/Tegra_Linux_Sample-Root-Filesystem_R36.4.3_aarch64.tbz2

# 1. BSP 압축 해제 (Linux_for_Tegra 폴더가 생성됨)
tar xf Jetson_Linux_R36.4.3_aarch64.tbz2

# 2. 생성된 폴더 안의 rootfs 디렉토리에 OS 이미지 압축 해제 (반드시 sudo 사용)
sudo tar xpf Tegra_Linux_Sample-Root-Filesystem_R36.4.3_aarch64.tbz2 -C Linux_for_Tegra/rootfs/

# 3. 작업 폴더 이동 후 플래싱 필수 의존성 및 바이너리 적용
cd Linux_for_Tegra
sudo ./tools/l4t_flash_prerequisites.sh
sudo ./apply_binaries.sh
```

2. 초기 계정 미리 생성.  
아래와 같은 명령어 치면 된다.  
```bash
# 사용법: sudo ./tools/l4t_create_default_user.sh -u <아이디> -p <비밀번호> -n <호스트네임> --accept-license
sudo ./tools/l4t_create_default_user.sh -u joonmo -p 1234 -n jetson-orin --accept-license
```
위 내용에서 내 이름과 저 1234만 너가 쓰고싶은걸로 변경하면 된다.  

3. 젯슨 보드 강제 복구 모드로 설정.  
보드를 보면 핀이 있는데, 12번인 J14 헤덜르 찾아야 한다.  
라파 핀이 있고, 그거 말고 쿨러 쪽 밑에 핀이 하나 더 있는데 후잘를 말하는거다.  
그 핀에서 9번 FC REC 랑 10번 GND를 암암 케이블로 연결해 쇼트내준다.  
이래야 강제복구모드 활성화 된다.  
위 핀을 쇼트낸 상태에서, USB C to A로 컴퓨터와 연결하고 보드에 전원을 넣어주면 되는데,  
반드시 컴퓨터 뒷 쪽 메인보드와 바로 연결되는, 그 랜케이블 박는 쪽 근처에 연결해야 한다. 안그러면 컴퓨터가 감지를 못하더라.  
그러고 전원 넣고, 한 10초있다가 점퍼선 제거하고 싶으면 제거하면 된다. 안해도 무관함.  
이어서, 우분투에서,  
```bash
lsusb | grep -i nvidia
```
라고 치면 NVIDIA Corp. APX 이런게 뜰 꺼다. 뜨면 성공.  

4. 이제 펌웨어 올리자.  
만일, SD 카드에 굽고 싶다면?  
```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device mmcblk0p1 \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  --showlogs --network usb0 jetson-orin-nano-devkit internal
```

아니다 나는 돈이 많다 M.2에 굽고 싶다면?  
```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  --showlogs --network usb0 jetson-orin-nano-devkit internal
```
위 명령어들로 원하는 저장소에 펌웨어를 올리자.  

5. 우분투에서 젯슨으로 접속해보자.  
```bash
ssh joonmo@192.168.55.1
```
니들이 바꾼 이름, 아이피로 넣으면 된다.  

6. 이제 CUDA, TensorRT 등 JetPack를 다 깔아주자.  
```bash
sudo apt update
sudo apt install -y nvidia-jetpack
```

위 명령어 실행해서, 나머지 것 들도 전부 깔아주면된다.  

## 결론
간단하게 젯슨 os 올리는거 정리해 봤다.  
가성비는 좋지 않지만, 그래도 라파보다 성능은 더 좋다.  
암튼 젯슨 만관부.  