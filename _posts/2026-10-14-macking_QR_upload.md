---
layout: single
title: "QR코드 만들기"
categories: "Lab"
tag: "hello"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## QR 코드
강의 자료 같은거 만들 때 은근히 많이 쓰이는게 QR 코드이다.  
이걸 사용하면 학생들에게 링크같은 데이터들을 쉽게 보낼 수 있어서 좋다.  
QR 만들 때 많은 사람들이 온라인 QR 생성 서비스 사용하곤 하는데,  
돈 내고 쓰고 그러더라.  
사실, QR 코드 라는게 그냥 2진수 데이터 이미지로 읽을 수 있게 해 주는게 전부라서 저딴거 돈 내고 사용할 이유가 없다.  
또, 저런 서비스 활용하면 막 광고도 뜨고 qr 삭제 될 수도 있고 그러는데 그냥 코드로 만들어서 쓰면 문제될게 완전히 없다.  
요즘 막 쇼츠에 인터넷 없일 QR로 이미지 전송 이딴거 뜨는데 그런거도 그냥 이미지 QR코드에 2신수로 만들어서 때려박은거다 누가누가 더 QR 덜 복잡하게 만드냐는 기술이 맞지만, 사실 그냥 할 수 있는거다.  
아래는 코드이다. 간단한 URL 정도는 저걸로 영구히 지속되느 QR 코드 만들 수 있다.

```python
import qrcode

# QR 코드에 심을 원본 URL
url = "https://example.com"

# 오류 복원력(ERROR_CORRECT_H: 최대 30% 손상되어도 인식 가능, 로고 삽입 시 유리)
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

qr.add_data(url)
qr.make(fit=True)

# 이미지 생성 및 저장
img = qr.make_image(fill_color="black", back_color="white")
img.save("permanent_qr.png")
```

이리 쉬운거 돈 쓰지말고  
그냥 직접 만들어서 쓰자.  