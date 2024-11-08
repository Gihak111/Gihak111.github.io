---
layout: single
title:  "CUDA, cuDNN 설치하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# cuda
딥러닝시, 글카로 딥러닝 하려면 필수적으로 있어야 하는 프로그램이다.  
이거 여러개 다운받으면, 중복되어 설피되니까 주의하자.  

## 1. 나한테 맞는 버젼 알기
cmd에 다음일 입력하자.  
```bash
nvidia-smi
```
이러면, 우측 상단에 CUDA Version이 나오는데, 이는 설피된 버젼이 아니라, 너가 설피해야 하는 버젼이다.  
나같은 경우에는 12.6이 나온다. 이 버젼을 다운받아주면 된다.  
만일, 이미 CUDA가 다운받아져 있다면,  
```bash
nvcc -version
```
위 명령어를 통해 다운받아져 있는 버젼을 알 수 있다.  

## 2. CUDA 설치
[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)  
위 링크에서, 너한테 맞는 버젼을 다운받아라.  
툴킷으로 다운받으면 된다.  
이어서, 실행시키면, 설치가 된다.  

## 3. cuDNN 설치
이는 CUDA 버젼에 맞게 설치해야 한다.  
[https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=12](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=12)  
위 링크의 버젼이 12에 맞는 버젼이다.  
다운받고, 압푹을 풀어서 앞서 다운받은 CUDA설피 폴더에 덮어쓰기 하자.  
CUDA 위피는 ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6``` 이런곳에 있다.  

## 4. 환경변수 추가
시스템 환경변수 편집 -> 고급 들어가서, path에 새로만들기로, bin, extras/CUPTI/Lib64, include 를 환경변수에 추가시킨다.  
이후 한번 재부팅 하고,  
```bash
nvcc -V
```
이 명령어 폈을 때 잘 잡히면 성공이다.  

이제 파이썬 스크립트 만들어서,  
```python
import torch
torch.cuda.is_available()
```
위 코드 실행시키고, true 나오면 진짜 성공이다. 이후에 진핸하는 딥러니 ㅇ들은 전부 GPU에서 가능해 진다.  

## 5. PyTorch
PyTorch에 따라 CUDA 가 실행이 되고 안되고 그런다.  
꼭 설치해 주자.  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
저 링크에서 다운받자. 내가 다운받은 조건에 맞게 해 주면 도니다.  

이어서 다 설피 했으면,  
```python
import torch
print(torch.version.cuda)  # 설치된 PyTorch에서 사용하는 CUDA 버전
print(torch.cuda.is_available())  # CUDA 사용 가능 여부
```
우 ㅣ코드를 통해 잘 되었는지 확인할 수 있다.  