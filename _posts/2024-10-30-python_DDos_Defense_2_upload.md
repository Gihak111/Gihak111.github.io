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



DDoS 공격의 출처를 추적하기 위해 상대방의 IP 주소를 알아내는 방법에는 여러 가지가 있습니다. 그러나 이러한 방법은 법적 문제와 윤리적인 고려사항이 있을 수 있으므로 신중히 접근해야 합니다. 일반적으로 사용되는 방법들은 다음과 같습니다.

### 1. **로그 분석**
- **설명**: 서버에서 수집하는 로그를 분석하여 비정상적인 요청 패턴을 찾아 공격자의 IP 주소를 확인할 수 있습니다. 웹 서버 로그, 방화벽 로그 및 애플리케이션 로그를 모니터링하는 것이 좋습니다.
  
  ```python
  import logging

  # 로그 설정
  logging.basicConfig(filename='access.log', level=logging.INFO)

  @app.route('/api', methods=['GET'])
  def api():
      client_ip = request.remote_addr
      logging.info(f"Request from {client_ip} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
      return "Request Successful"
  ```

### 2. **패킷 스니핑**
- **설명**: Wireshark와 같은 패킷 스니퍼를 사용하여 네트워크 트래픽을 캡처하고, 공격자의 IP 주소를 확인할 수 있습니다. 이 방법은 네트워크의 흐름을 실시간으로 모니터링할 수 있습니다.
- **주의사항**: 패킷 스니핑은 법적으로 허용된 경우에만 사용해야 하며, 자신이 관리하는 네트워크에서만 사용해야 합니다.

### 3. **트래픽 분석 도구**
- **설명**: 특정 DDoS 공격 도구(예: `tcpdump`, `netstat`)를 사용하여 실시간 트래픽을 분석하고 의심스러운 IP 주소를 확인할 수 있습니다.

  ```bash
  # 특정 포트에서 트래픽 캡처
  sudo tcpdump -i eth0 port 80
  ```

### 4. **유입 IP 확인**
- **설명**: CDN(콘텐츠 전송 네트워크)을 사용할 경우, 요청 헤더에서 원래 클라이언트의 IP 주소를 확인할 수 있습니다. 예를 들어, Cloudflare나 AWS의 경우 `X-Forwarded-For` 헤더를 통해 원래의 IP를 알 수 있습니다.

  ```python
  @app.route('/api', methods=['GET'])
  def api():
      client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
      logging.info(f"Request from {client_ip}")
      return "Request Successful"
  ```

### 5. **법적 절차**
- **설명**: 특정 IP 주소가 악의적인 활동을 하는 경우, 법적 절차를 통해 ISP에 요청하여 해당 IP 주소의 주인을 확인할 수 있습니다. 이는 수사 기관의 협조를 받아야 가능하므로 일반 사용자에게는 접근이 어렵습니다.

### 주의사항
- IP 주소를 추적하는 과정은 법적인 문제가 발생할 수 있으며, 타인의 사생활을 침해할 수 있습니다. 따라서 적절한 권한을 가진 경우에만 수행해야 하며, 법적 절차를 따르는 것이 중요합니다.

이러한 방법들은 DDoS 공격의 출처를 추적하는 데 도움이 될 수 있지만, 실제로는 더 많은 기술적, 법적 장치가 필요할 수 있습니다.