---
layout: single
title:  "딥러닝 실패"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 너무 긴 딥러닝 시간
딥러닝 데이터셋이 크면, 그만큼 정확도가 늘어나지만, 반대로 시간이 엄청 오래걸린다.  
그렇게 딥러닝 박다가,  다음과 같은 로그가 뜨면 넌 개큰일난거다.  

![Image](https://github.com/user-attachments/assets/0aa0adea-50bd-4738-938e-1e8d061dfb12)  

이 아름다운 오류가 보이는가?  
이정도의 발산이면 태양계를 벗어난 거다.  
약 50시간의 딥러닝이 물거품이 된느 순간.  
로그를 보면,  Epoch5까지는 멀정하게 가다가, 이후 무너지는 모습을 보인다.  
이렇게 발산이 되면, 학습율을 낮추어 더 세밀하게 딥러닝 해야 하지만, 들어가는 시간이 늘어나기도 하고, 여러가지 문제가 또 생긴다.  
혀튼 딥러닝 쉽지 않다.  
발산이 뭐 한두번 나오는것도 아니지 않냐. 그냥 그려러니 하고 또 한번 하면 된다.  
  