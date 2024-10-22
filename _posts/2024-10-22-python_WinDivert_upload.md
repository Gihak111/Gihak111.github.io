---
layout: single
title:  "윈도우에서 네트워크 트래픽 차단하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 윈도우에서 네트워크를 차단해 보자.  
일반적으로, 다음과 같은 것을 통해 네트워크 트래픽을 차단할 수 있다.  
- **WinDivert**: 패킷 가로채기에 중점을 둔 라이브러리로, 네트워크를 세밀하게 조작할 수 있도록한다.  
- **Netsh**: 네트워크 설정 관리 프로그램으로, 명령줄 인터페이스를 통해 시스템 관리자가 네트워크 구성을 변경할 수 있도록 도와준다.  

### 1. Windpws에서 네트워크 트래픽 차단하기
URL 차단을 해 보자.  
사용자가 입력하거나, 파일을 통해 관리할 수 있도록 하였다.  

#### 기본 구조
- WinDivert를 사용해 패킷을 캡처한 후 HTTP(S) 트래픽을 분석  
- 분석된 URL이 설정된 차단 목록에 포함되면 해당 트래픽을 차단  

#### 코드 구현

1. **WinDivert 설치**
   [WinDivert](https://reqrypt.org/windivert.html)  
   페이지에서 필요한 파일을 다운로드 하자.  

2. **Python 코드**
   Python과 WinDivert를 사용해 URL을 차단하는 코드를 작성할 수 있다.  
   아래 예시는 기본적인 구조를 담고 있다:  

```python
import pydivert
import re

# 차단할 URL 목록을 사용자가 설정하도록
blocked_urls = [
    "example.com",
    "blockedsite.com",
]

# 패킷 캡처를 위해 WinDivert 초기화
with pydivert.WinDivert("tcp.DstPort == 80 or tcp.DstPort == 443") as w:
    for packet in w:
        if packet.is_outbound:
            try:
                # HTTP 요청에서 Host를 추출
                http_data = packet.payload.decode(errors="ignore")
                host = re.search(r"Host: ([\w\.]+)", http_data)
                
                if host:
                    domain = host.group(1)
                    if domain in blocked_urls:
                        print(f"차단된 URL 탐지: {domain}")
                        continue  # 패킷을 무시하고 차단함
            except Exception as e:
                print(f"패킷 처리 중 오류: {e}")
                
        # 패킷을 허용함
        w.send(packet)
```

### 3. URL 설정 기능 추가
차단할 URL 목록을 파일에서 가져오거나, 코드에서 사용자 입력을 받을 수 있도록 수정할 수 있다.  

```python
def load_blocked_urls_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"{filepath} 파일을 찾을 수 없습니다.")
        return []

# 차단할 URL 목록을 외부 파일에서 가져오기
blocked_urls = load_blocked_urls_from_file('blocked_urls.txt')

# 패킷 캡처 코드 반복...
```

### 4. 차단할 URL 파일 형식 예시
`blocked_urls.txt` 파일에는 차단할 URL을 한 줄씩 작성해두면 위 스크립트가 이를 차단해 준다.  

```
example.com
blockedsite.com
```

이 코드는 HTTP(S) 트래픽에서 URL을 분석해 차단 목록에 있는 경우 해당 트래픽을 차단하는 방식으로 동작한다.  WinDivert를 사용해 네트워크 패킷을 캡처하고 필터링하는 과정을 보여주며, 파일에서 차단할 URL을 동적으로 불러오는 방식도 구현할 수 있다.  