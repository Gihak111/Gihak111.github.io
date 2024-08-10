---
layout: single
title:  "주피터 노트북 import 오류"
categories: "pynb"
tag: "ERROR"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  
# import cv
위 코드를 주피터에서 실행하면, 오류가 나곤 한다.  
이는, opencv가 잘 설치되지 않은것이 아니라, 주피터 노트북의 커널이 원인일 가능성이 높다.  
```python
import sys
print(sys.version)
```  
위 코드로 사용중인 커널의 파이썬 버젼을 확인할 수 있다.  
만일, 주피터에서 실행이 됮 않는다면,  
비주얼 코드로도 한번 실행해 보자.  
나는 여기서 잘 실행이 되었다.  
이런 겨로가는 파이썬 버젼차이, 가상환경 차이, 커널 차이 때문에 나올 수 있는 결과이다.  