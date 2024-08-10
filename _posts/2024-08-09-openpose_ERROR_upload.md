---
layout: single
title:  "openpose 설치"
categories: "pynb"
tag: "ERROR"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  
# 옾느 포즈를 설치하는 방법은 여러가지가 있다.
Cmake를 사용하는 방법은 cpp 파일에서 실행하는데엔 꼭 필요하지만, 파이썬에서 openpose를 사용하는거면 비교적 쉽게 설치할 수 있다.  
https://github.com/CMU-Perceptual-Computing-Lab/openpose
위 링크는 오픈포즈의 깃허브 링크이다. 일단 클론하자.  
파일을 받고, 안의 실행파일을 실행하면, 오류가 나온다.  
링크를 확인해 보면, 서버가 내려가서 모델이 다운로드 되지 않는 것을 확인할 수 잇다.  

https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2233
위 글을 통해 서버가 내려가서 다운이 되지 않는 오류를 많은 사람들이 겪고 있음을 알 수 있다.  
또한, 해당 글에는 다른 사람이 다운해서 올려놓은 모델이 구글 드라이브에 있는것을 확인할 수 있다.  
드라이브에서 모델을 다운받고, 클론한 오픈포즈의 모델 폴더에 덮어씌우는 것으로 오픈포즈를 파이썬에서 사용할 수 있다.  
import를 사용할 경우, 오류가 날 수 있다.  
이럴 땐, 직접 위치를 찍어주는 것으로 해결할 수 있다.  
```
# OpenPose 실행 경로
openpose_path = r"C:\Users\openpose\bin\OpenPoseDemo.exe"
```
위와 같은 코드를 통해 다운받은 오픈포즈를 사용할 수 있다.