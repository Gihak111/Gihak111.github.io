---
layout: single
title:  "보안 프로토콜 SHA-256과 HTTPS"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### SHA-256: 가용성, 무결성, 기밀성을 보장하는 방법
SHA-256은 현대 암호학에서 중요한 역할을 하는 해시 함수로, 데이터 무결성 검증, 암호화 프로토콜, 디지털 서명, 블록체인 등 다양한 분야에서 사용한다.  
전에 말했듯 SHA-256은 Secure Hash Algorithm의 줄임말로, 256비트 길이의 해시 값을 생성하는 함수다.  

### 1.가용성 (Availability)

가용성은 시스템이나 데이터가 언제나 접근 가능하도록 보장하는 것이다.  
SHA-256은 직접적으로 가용성을 보장하는 기술은 아니지만, 데이터 무결성을 보호함으로써 가용성 유지에 기여할 수 있다.  
서버나 애플리케이션이 정상적으로 작동하는지 확인하기 위해 전송 중인 데이터가 손상되지 않았는지 검증하는 도구로 활용할 수 있다.  

- 예시:  
파일이나 데이터를 전송하기 전 해시 값을 미리 계산한 후, 수신 후 동일한 해시 값을 다시 계산하여 무결성을 확인한다.  
데이터가 변조되지 않았다는 사실을 확인하면 시스템의 신뢰성과 가용성을 보장할 수 있다.  

### 2. 무결성 (Integrity)

무결성은 데이터가 전송되거나 저장되는 동안 손상되거나 변조되지 않았음을 보장하는 것이다.  
SHA-256의 주요 기능 중 하나가 바로 무결성을 확인하는 것이다.  
데이터를 해시 값으로 변환하고 이를 원본 데이터와 비교하여 중간에서 변조되었는지 확인한다.  

- SHA-256을 활용한 무결성 검증:
  - 프론트엔드에서 데이터를 서버로 전송하기 전에 SHA-256 해시 값을 생성한 후 데이터를 전송한다.  
  - 백엔드에서 동일한 해시 값을 생성하고 전송된 해시 값과 비교해 데이터가 중간에 손상되지 않았는지 확인한다.  

### 3. 기밀성 (Confidentiality)

기밀성은 데이터에 접근할 수 있는 권한이 있는 사람만 접근할 수 있도록 보장하는 것이다.  
SHA-256은 암호화 알고리즘은 아니지만, 암호화 프로토콜과 함께 사용하여 기밀성을 강화할 수 있다.  
SHA-256은 주로 데이터의 무결성을 확인하는 데 사용되며, HTTPS와 같은 암호화된 통신 프로토콜과 함께 기밀성을 보호한다.  

- 예시: 
HTTPS는 전송되는 데이터를 암호화하고, SHA-256은 데이터가 손상되지 않았음을 확인하는 데 사용한다.  
이 두 가지 기술을 결합하여 전송 중인 데이터를 안전하게 보호할 수 있다.  

### SHA-256의 실제 적용 예
#### 데이터 전송 시 SHA-256과 HTTPS 결합
- 프론트엔드에서 데이터를 전송하기 전 SHA-256으로 해시 값을 생성하고 서버로 전송한다.  
- 백엔드에서 동일한 해시 값을 계산해 무결성을 확인한 후 데이터가 변조되지 않았음을 보장한다.  
- 이 과정은 HTTPS를 통해 암호화되어 전송되므로 기밀성도 보장된다.  

```bash
npm install crypto-js axios
```

```js
import CryptoJS from 'crypto-js';
import axios from 'axios';

const createHash = (data) => {
  return CryptoJS.SHA256(data).toString(CryptoJS.enc.Hex);
};

const sendData = async (data) => {
  const hash = createHash(data);

  const response = await axios.post('https://your-backend-server.com/upload', {
    data,
    hash,
  });

  return response.data;
};

const imageData = '...';  // 이미지 데이터
sendData(imageData).then(response => {
  console.log('Response from server:', response);
});
```

### 결론

SHA-256은 가용성, 무결성, 기밀성을 보장하기 위한 필수적인 도구다.  
특히 데이터가 변조되지 않았는지 확인하는 데 매우 효과적이며, HTTPS와 같은 암호화 기술과 함께 사용하면 데이터 보호 수준을 한층 높일 수 있다.  
이를 통해 전송되는 데이터의 신뢰성을 유지하고 시스템의 안정성과 보안을 강화할 수 있다.  