---
layout: single
title:  "내 컴퓨터 IP 바꾸기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 내 컴퓨터의 IP를 30분마다 변경하는 프로그램 만들기

인터넷을 사용할 때 우리의 컴퓨터는 특정한 IP 주소를 가지게 된다.  
이 IP 주소는 네트워크에서 장치를 식별하는 역할을 한다.  
하지만, 특정한 상황에서는 보안을 위해 일정 시간마다 IP를 변경해야 할 필요가 있다.  

- 특정 웹사이트에서 과도한 트래픽 차단을 피하기 위해
- 익명성을 유지하면서 인터넷을 사용하기 위해
- 특정 서비스에서 위치 기반 제한을 우회하기 위해

이 글에서는 **30분마다 내 컴퓨터의 IP를 자동으로 변경하는 프로그램**을 만들고, 이를 도메인과 연계하여 동작하도록 설정하는 방법을 설명하겠다.  

---

## 1. IP 변경 방법 이해하기

IP를 변경하는 방법에는 여러 가지가 있지만, 일반적으로 다음과 같은 방법이 있다.  

### 1) 인터넷 서비스 제공업체(ISP) 기반의 IP 변경
- 대부분의 가정용 인터넷 사용자는 **동적 IP(Dynamic IP)**를 할당받는다.  
- 모뎀/라우터를 재부팅하면 새로운 IP를 할당받을 가능성이 높다.  

### 2) VPN을 사용하여 IP 변경
- VPN을 사용하면 인터넷 트래픽을 다른 지역의 서버로 우회하여 IP를 변경할 수 있다.  
- OpenVPN, NordVPN, ExpressVPN 같은 서비스를 활용할 수 있다.  

### 3) 프록시 서버를 이용한 IP 변경
- 특정 프록시 서버를 사용하여 인터넷을 이용하면 IP를 변경할 수 있다.  
- 유료 또는 무료 프록시 리스트를 활용할 수 있다.  

### 4) Tor 네트워크를 이용한 IP 변경
- Tor 네트워크를 이용하면 일정 주기마다 IP를 자동으로 변경할 수 있다.  
- 보안성이 뛰어나지만 속도가 느릴 수 있다.  

이번 글에서는 **VPN 및 프록시를 활용한 IP 변경 프로그램**을 만들어보겠다.  

---

## 2. 30분마다 IP 변경하는 프로그램 구현

이제 Python을 사용하여 30분마다 IP를 변경하는 프로그램을 만들어 보겠다.  

### 1) VPN을 통한 IP 변경
우선, OpenVPN을 활용하여 IP를 변경하는 방법을 살펴보겠다.  

#### **(1) OpenVPN 설치 및 설정**
1. OpenVPN을 다운로드하고 설치한다.  
2. VPN 제공업체에서 `.ovpn` 설정 파일을 받아둔다.  
3. OpenVPN을 CLI에서 실행할 수 있도록 설정한다.  

#### **(2) Python 코드로 자동화**
아래 코드는 OpenVPN을 실행하여 30분마다 VPN 서버를 변경하는 방식이다.  

```python
import os
import time

def change_ip():
    print("IP 변경 중...")
    os.system("taskkill /F /IM openvpn.exe")  # 기존 VPN 연결 종료
    os.system("openvpn --config path/to/your.ovpn --daemon")  # 새 VPN 실행
    print("새로운 IP로 변경되었습니다!")

if __name__ == "__main__":
    while True:
        change_ip()
        time.sleep(1800)  # 30분 대기 (1800초)
```

위 코드를 실행하면 30분마다 OpenVPN이 재실행되면서 IP가 변경된다.  

### 2) 프록시 서버를 활용한 IP 변경
프록시 서버를 활용하는 경우, `requests` 라이브러리를 사용하여 특정 프록시를 적용할 수 있다.  

```python
import requests
import random
import time

PROXY_LIST = [
    "http://proxy1:port",
    "http://proxy2:port",
    "http://proxy3:port"
]

def change_proxy():
    proxy = random.choice(PROXY_LIST)
    print(f"새로운 프록시 사용: {proxy}")
    return {"http": proxy, "https": proxy}

if __name__ == "__main__":
    while True:
        proxies = change_proxy()
        try:
            response = requests.get("http://ipinfo.io/ip", proxies=proxies)
            print(f"현재 IP: {response.text.strip()}")
        except Exception as e:
            print(f"프록시 오류 발생: {e}")
        time.sleep(1800)  # 30분마다 변경
```

위 코드를 실행하면 30분마다 프록시 서버를 변경하여 IP를 바꾸는 기능이 작동한다.  

---

## 3. 도메인과 연계하여 동작하기

IP 변경 프로그램을 실행하면 내 컴퓨터의 IP는 변경되지만, 특정 도메인과 연계하여 동작할 수도 있다.  
예를 들어, `mydomain.com`이라는 도메인을 내 변경된 IP에 연결하고 싶다면, **Dynamic DNS(DDNS)** 서비스를 활용할 수 있다.  

### 1) Cloudflare를 활용한 DDNS 자동 업데이트
Cloudflare의 API를 사용하여 내 IP가 변경될 때마다 도메인 DNS 레코드를 자동으로 갱신할 수 있다.  

```python
import requests

CLOUDFLARE_API_KEY = "your_api_key"
ZONE_ID = "your_zone_id"
RECORD_ID = "your_record_id"
DOMAIN = "yourdomain.com"

headers = {
    "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
    "Content-Type": "application/json"
}

def update_dns(new_ip):
    url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/dns_records/{RECORD_ID}"
    data = {"type": "A", "name": DOMAIN, "content": new_ip}
    response = requests.put(url, headers=headers, json=data)
    print(response.json())

def get_current_ip():
    return requests.get("http://ipinfo.io/ip").text.strip()

if __name__ == "__main__":
    while True:
        new_ip = get_current_ip()
        update_dns(new_ip)
        print(f"도메인 {DOMAIN}의 IP가 {new_ip}로 변경되었습니다.")
        time.sleep(1800)
```

위 코드를 실행하면 **내 IP가 변경될 때마다 Cloudflare를 통해 도메인 IP를 자동으로 업데이트**할 수 있다.  

---

## 4. 결론

- VPN 또는 프록시를 사용하여 IP 변경  
- Python으로 자동화하여 주기적으로 실행  
- Cloudflare API를 활용하여 도메인과 연동  

이제 여러분도 원하는 방식으로 IP를 변경하고, 도메인과 연계하여 활용할 수 있다.  
추가적인 보안이 필요하다면, `Tor` 네트워크를 활용하는 방법도 고려할 수 있다.  

