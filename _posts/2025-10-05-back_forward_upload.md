---
layout: single
title:  "역전파 미분의 이해?"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 역전파 시, 미분이 일어난다
이게 미분이 일어나는 이유가, 기울기 얻기 위함인데,  
간단한 이해를 위해 예시를 들어서,  
Conv -> maxpool -> dense -> sofrmax라고 하자면,  

softmax + Cross Entropy 로 미분된 y hat - y 에서,  
계산된 결과가 Dense Layer의 출력 기울기가 되고, Sofrmax의 입력 기울기가 된다  
마찬가지로,  
Dense Layer의 출력 기울기는 앞서서 구한 값이 되고,  
Dense Layer를 미분해서 나온 기울기가 maxpool의 출력 기울기가 되는거다.  
이걸 이해해야 된다  
나중에 RNN cell t = 2 미분하거나 그랬을 때  
그게 t = 1의 기울기가 되면서 미래에서온 기울기가 되고 막 그러는데,  
이걸 잘 이해하려면 이 미분 개념을 잘 알고 가면 좋을 것 같아서 올려본다.  