---
layout: single
title:  "윈도우에서 디도스 막는 법"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 지난번의 디도스 연장선이다.
여전히 자신을 오나전히 숨긴 상태로 디도스를 날리는 방법을 알지 못했다.  
완전히 숨기려면 다으모가 같은 방법들을 전부다 하던지, 가능한 것들을 해서 자신을 숨길 수 있다.  



DDoS 공격을 방어하거나 공격의 출처를 추적하는 방법에 대해 설명드리겠습니다. 아래에는 기본적인 방어 기법과 함께 이를 구현할 수 있는 Python 코드 예시도 포함되어 있습니다.

### DDoS 공격 방어 방법

#### 1. **요청 수 제한 (Rate Limiting)**
- **설명**: 특정 IP 주소에서 발생하는 요청의 수를 제한하여 과도한 요청을 차단합니다.
- **코드 예시**:

```python
from flask import Flask, request, abort
from collections import defaultdict
import time

app = Flask(__name__)

# 요청 수를 저장할 딕셔너리
request_count = defaultdict(list)

@app.route('/api', methods=['GET'])
def api():
    client_ip = request.remote_addr
    current_time = time.time()

    # 최근 1분 동안의 요청 수를 카운트
    request_count[client_ip] = [timestamp for timestamp in request_count[client_ip] if timestamp > current_time - 60]

    # 요청 수가 100을 초과하면 차단
    if len(request_count[client_ip]) > 100:
        abort(429)  # Too Many Requests

    # 요청 기록 추가
    request_count[client_ip].append(current_time)
    return "Request Successful"

if __name__ == '__main__':
    app.run()
```

#### 2. **IP 차단 (IP Blacklisting)**
- **설명**: 악의적인 IP 주소를 사전에 차단하여 해당 IP에서의 요청을 차단합니다.
- **코드 예시**:

```python
from flask import Flask, request, abort

app = Flask(__name__)

# 차단할 IP 주소 목록
blacklisted_ips = {"192.168.1.100", "10.0.0.5"}

@app.route('/api', methods=['GET'])
def api():
    client_ip = request.remote_addr

    # IP가 차단 목록에 있으면 요청 차단
    if client_ip in blacklisted_ips:
        abort(403)  # Forbidden

    return "Request Successful"

if __name__ == '__main__':
    app.run()
```

#### 3. **패킷 필터링 (Packet Filtering)**
- **설명**: 네트워크 수준에서 비정상적인 트래픽을 차단하는 방어 기법입니다. 방화벽이나 IDS/IPS를 사용하여 특정 패턴의 트래픽을 필터링합니다.

#### 4. **트래픽 분석 및 로그 모니터링**
- **설명**: 서버 로그를 분석하여 비정상적인 트래픽 패턴을 감지하고, 공격을 사전 예방할 수 있습니다.

#### 5. **역 추적 (Tracing)**
- **설명**: 공격 출처를 추적하기 위해 소스 IP를 로깅하고, DDoS 공격이 발생한 IP에 대한 추가 분석을 수행합니다.
- **코드 예시**:

```python
import logging

# 로깅 설정
logging.basicConfig(filename='ddos_attack.log', level=logging.INFO)

@app.route('/api', methods=['GET'])
def api():
    client_ip = request.remote_addr

    # 요청을 로깅
    logging.info(f"Request from {client_ip}")

    # 요청 처리
    return "Request Successful"
```

### 결론
DDoS 공격 방어는 여러 기법을 통해 이루어지며, 위의 코드 예시들은 Flask 웹 프레임워크를 사용하여 간단한 방어 기법을 구현한 것입니다. 실무에서는 이와 같은 기본적인 방어 기법을 조합하고, 고급 방화벽, IDS/IPS, CDN 등을 활용하여 보다 강력한 방어 체계를 구축하는 것이 중요합니다.