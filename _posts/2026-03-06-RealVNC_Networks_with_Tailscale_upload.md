---
layout: single
title: "RealVNC With Tailscale"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## RealVNC with Tailscale

저번에 포트 포워딩을 통해서 외부 네트워크에서 내부 리눅스 서버에 들어가는걸 해 보았다.  
하지만, 이게 포트가 외부에 노출되어 위험하기도 하고, 쉽지 않기 때문에 보안 등 이슈가 있다.  
또한, 삼바 같은 라이브러리로 폴더 관리를 할 수 없기에, 아쉬움이 크다.  
이를 해결하기 위해 Tailscale을 사용해 보았다.  

## Tailscale
이건 외부 망인 두 컴퓨터를 마치 내부 망처럼 연결해주는 서비스이다.  
이게 참 좋다 계정 로그인만 하면 바로 되고, 만들어진 로컬 서버도 영원히 바뀌지 않느다.  
또한 기존 방식으론 ISP 차단 같은거로 못하는게 많은데, 이건 그냥 로컬이라 그런거 없다.  

리눅스쪽부터 바로 진행하자면,  
1. curl 이 없다면 설치하자.  
```bash
sudo apt install curl
```

2. 이제 설치하면 된다.  
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

3. 이러면 이제 로그인 하라는 내용이 뜨는데, 그냥 구글 연동으로 로그인 하면 된다.  

이어서, 컴퓨터 쪽도 해 보자.  
1. [링크](https://tailscale.com/download)로 들어가서 다운로드 받자.  
2. 위에 저거 설치하고, 같은 구글 아이디로 로그인한다.  
3. 그러면, 너의 리눅스 서버 사용자 이름으로 로컬 주소가 하나 뜰 것이다.
4. 이제 너의 컴퓨터에서 ```\\아까 본 로컬주소``` 하면 삼바로도 들어가 진다.  


## 결론
진짜 포트 포워딩 내다버리고 이거 써도 될 것 같다.  
외부망을 로컬로 엮다니 세상 참 많이 좋아졌다.  