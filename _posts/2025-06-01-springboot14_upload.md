---
layout: single
title:  "spring boot 14. 서버를 열었지만, 열리지 않을 경우"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 앱은 정상적으로 빌드가 되었지만, 서버가 열리지 않는다
분명히, ```bash mvn spring-boot:run``` 으로 실행시켰지만, build sucess 가 뜬 후 서버가 열리지 않고 바로 종료된다.  
이유는 간단하다 분명히, 엔티티에 오류가 있을 것이다.  
예를 들면, 쿼리에 빼먹을 값이 있다던가 그런 식으로  
나 역시, 엔티티로 정의해둔 클래스에 쿼리가 몉 개 누락되서 나온 오류였다.  
다들, 이러 ㄴ오류 나오면 엔티티부터 확인해 보자.  




























