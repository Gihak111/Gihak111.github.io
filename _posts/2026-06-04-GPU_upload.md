---
layout: single
title:  "AI 하기 위한 GPU 선택"
categories: "Vlog"
tag: "hello"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 요즘 주변에서 연구용
AI 연구용 GPU 뭐 사야 하냐고 많이 물어본다.  
이참에 정리해 둘 까 한다.  


RTX 5080  
VRAM: 16GB GDDR7 (960 GB/s) CUDA 코어: 10,752개 텐서코어: 336개  

RTX 5070 Ti  
VRAM:16GB GDDR7 (896 GB/s) CUDA 코어: 8,960개 텐서코어: 280개  

RTX 5070  
VRAM: 12GB GDDR7 (672 GB/s) CUDA 코어: 6,144개 텐서코어: 192개  

RTX 5060 Ti  
VRAM: 16GB16GB GDDR7 (448 GB/s) CUDA 코어: 4,608개 텐서코어: 144개  

5060ti 에서 1시간 걸리는 딥러닝이  
5070에선 40분  
5070ti에선 30분  
5080에선 25분  
정도로 시간이 줄어든다  
사실 이론상이고, 저정도로 체감되진 않는다.  
또한 VRAM 역시 12GB 로는 답답할 때가 많다.  
16GB 는 가는게 맞긴 한데 돈이 없네.  
슬프다.  
  