---
layout: single
title: "컴퓨터 끄고 나와서 일 못하게 되어 개큰일날때"
categories: "Lab"
tag: "ubuntu"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 크아아아아악 조졌다
집 밖으로 나와 도착한 회사.  
이땐 이미 늦었다.  
까먹고 꺼두고 나온 컴퓨터.  
진짜 개조졌다 오늘은 일 못한다.  

다들 이런 경험 한 번씩 있지 않은가?  
이런 당신을 위한 해결책에 여기에 있다.  
내가 원하는 시간, 또는 Lan 핑이 들어오면 꺼져있던 컴퓨터가 켜지게 하는 방법이다.  
물론, 멀티탭까지 꺼져 있으면 망하지만, 멀티탭만 켜져있다 그러면 문제 없이 잘 켜진다.  

## 바이어스 설정
내 컴퓨터가 msi 이기 때문에, 이거 기준으로 설명하겠다.  




이어서, 랜으로 원격 컴퓨터 켜는 거도 해보자.  
1. 컴터 키고 Del 연타, BIOS에 들어간다. 바로 Advanced Mode로 들어가면 된다.  
2. Setting -> Advanced -> Wake Up Event Setup 으로 이동하자.  
3. Resume By PCI-E Device 항목을 Enabled 로 바꾼다. 저거 랜카드 라인이다.  
4. 위에 둘 다 활성화 하면 시간 설정할 수 있는데,  
5. Date Alarm: 매일 같은 시간에 켜지게 하고 싶으면 0 넣으면 된다.  
6. Time Alarm: 24시간 형식으로 넣으면 된다. 오전 8시 30은 08 : 30 : 00 이 되는 거다. 이제 재부팅 하면 적용된다.  
7.  Setting -> Advanced -> Power Management Setup 으로 이동하자.  
8. ErP Ready 항목이 있으면 이걸 Disable로 설정해야 한다. 저건 대기전력 막겠다는 거라 꼭 꺼야 한다. 

이어서, 랜으로 원격 컴퓨터 켜는 거도 해보자.  
이제 컴퓨터 꺼질 때 랜카드는 살아있도록 설정할꺼다. 여기선 리눅스 기준으로 한다.  
이건 앞에 들어간 바이어스 설정이 필요하니, 시간만 설정하지 말고 앞선 다 따라주어야 한다.  
1. 컴터 켜 주고 바로 cmd -> ```ip link``` 로 랜카드 이름, 주소 알아온다. lo 빼고 보통 enp3s0 이런거로 시작되는게 유선 랜카드 이다. 이 이름 잘 기억하자.  
2. link/ether 뒤에 나오는 XX:XX:XX:XX:XX:XX 형태의 값이 MAC 주소이다. 이거도 기억해야 한다.  
3. 여기서 랜카드 설정하려면 아래의 패키지가 필요하다.  
```bash
sudo apt update
sudo apt install ethtool -y
```
이어서, WOL 지원하는지 확인한다.  
```bash
sudo ethtool 인터페이스이름
# 예시: sudo ethtool enp3s0
```
Supports Wake-on: g 이런 식으로 나오면 매직 패킷인 g를 지원한다는 거다. g 대신 d 면 비활 상태인 거다.  
꺼져있다면, 이걸로 켜 주자.  
```bash
sudo ethtool -s 인터페이스이름 wol g
# 예시: sudo ethtool -s enp3s0 wol g
```
4. 앞서 한 랜카드 설정이 재부팅 해도 유지되도록 해야 한다.  
```bash
sudo nano /etc/systemd/system/wol.service
```
저 서비스에 아래 내용을 넣어 등록한다.  
```bash
  
[Unit]
Description=Enable Wake On LAN
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/ethtool -s 인터페이스이름 wol g

[Install]
WantedBy=basic.target
```


이어서 crtl + 0 -> Enter -> ctrl + x
이어서, 실행해 주면 된다.  

```bash
sudo systemctl daemon-reload
sudo systemctl enable wol.service
sudo systemctl start wol.service
```
이러면 랜 설정은 다 한거다.  
이제 이어서 공유기 설정 해야 한다.  
5. ip time 공유기 설정도 해야 한다. 먼저 WOL 기능에 컴터를 등록해야 한다.  
관리방법 -> 고급설정 -> 특수기능 -> WOL 기능 하면 된다.  
여기 컴터 ip  MAC 주소 입력해야 한다.  
6. 이어서, DDNS 설정도 해서 고정을 해야 한다.  
관리도구 -> 고급 설정 -> 특수기능 -> DDNS 설정  
호스트 이름에 원하는거 넣고 추가하면 된다.  
7. 이제, 공유기 원격 관리 포트를 열면 된다.  
관리도구 -> 구급설정 -> 보안기능 -> 공유기 접속/관리
예전에 포트포워딩 한다고 같은 짓거리 했었다.  
8. 이제 핸드폰으로 와서, iptime WOL 앱 깔면 된다.  
새 공유기 추가 -> 수동입력
여기에 앞서 만든 DDNS 주소, 포트번호 로그인 정보 넣으면 끝이다.  

## 결론
암튼 위 과정을 거친다 그러면 출근 전 컴터 끄고 나오는 참사를 겪어도 극복할 수 있다.  
과정이 좀 복잡하지만, 비슷한 실수로 피본 사람이라면 꼭 설정해두자.  