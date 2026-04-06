---
layout: single
title: "airllm"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## lyogavin/airllm
간단하게 말하면, 모델을 전부 로드하지 않고, 계산해야 하는 레이어만 불러와서 연산하는 기능이다.  
제일 큰 장점은, 120b 모델이건, 405b 모델이건 그냥 VRAM 4GB 짜리로도 전부 돌릴 수 있다는거다.  
이게 가능한 건, 모델은 SSD 같은 디스크에 들어가져 있는거고,  
그걸 램에 불러왔다가 계산하고 지우고 또 새로 불러오고 이걸 반복하기 때문이다.  

여기서 느껴지는게, 엄청 느리지 않을까? 인데,  
실제로도 엄청나게 느려서 실시간 챗은 불가능하다.  

아무리 생각해도 Multi-GPU가 훨신 좋은 것 같다.  
그래도 궁금하면 한 번쯤 사용해 보는 것도 좋을 것 같다.  
[링크](https://github.com/lyogavin/airllm)