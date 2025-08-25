---
layout: single
title:  "CloudFlare"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## CloudFlare
과거에, 우리는 ngrok 라는 프로그램을 다룬 적 있다.  
아아 얼마나 아름다운 프로그램인가 공짜고 공개도메인으로 빼 준다니  
하지만, 개쳐느리고 무료 사용제한도 있어서 생각보다 많이 못쓴다.  
그런 당신을 위한게 CloudFlare다.  
일단 제일 큰 장점이, 엄청나게 빠르다는거다.  
버퍼만 좀 지우면, 진짜 빠르게 사용할 수 있다.  
대충, 자세히 알아보자.  

### 1\. Cloudflare Tunnel (cloudflared) 설치

Cloudflare가 제공하는 터널링 프로그램의 이름은 `cloudflared` 이다. 일단 이걸 먼저 설치해야 한다.

1.  **Cloudflared 다운로드**:

      * 웹 브라우저로 [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/) 로 이동해서 로그인 한다. (계정이 없다면 만들자. 공짜다.)
      * 로그인 후 왼쪽 메뉴에서 `Access` -\> `Tunnels` 로 이동한다.
      * `Create a tunnel` 버튼을 누르면 `cloudflared`를 설치하는 방법을 운영체제 별로 친절하게 안내해준다.
      * Windows의 경우 보통 `.msi` 설치파일을, macOS의 경우 Homebrew (`brew install cloudflare/cloudflare/cloudflared`)를 사용하는게 가장 편하다.

2.  **Path 환경 변수 추가**:

      * Ngrok 때와 마찬가지로, 터미널 어디서든 `cloudflared` 명령어를 사용하고 싶다면 설치 경로를 `PATH` 환경 변수에 추가해주자. Homebrew나 msi로 설치했다면 보통 자동으로 처리된다.

### 2\. Cloudflare 계정 연동

`cloudflared`는 당신의 Cloudflare 계정과 연동(로그인)을 해야 제대로 쓸 수 있다. 이 과정이 Ngrok보다 조금 더 귀찮지만, 한번만 하면 되니 참아보자.

1.  **로그인 명령어 실행**:

      * 터미널(cmd, PowerShell, Terminal 등)을 열고 아래 명령어를 입력한다.
        ```bash
        cloudflared tunnel login
        ```

2.  **브라우저 인증**:

      * 명령어를 실행하면 잠시 후 웹 브라우저가 열리면서 Cloudflare 로그인 창이 뜬다.
      * 로그인하고, 당신의 계정으로 이 `cloudflared` 클라이언트를 인증할 사이트를 선택하라고 나온다. 사용할 도메인이 있는 사이트를 선택해주면 된다.
      * 인증이 완료되면 터미널에 성공 메시지가 나타난다. 이제 당신의 PC는 Cloudflare와 연결될 준비를 마쳤다.

### 3\. Cloudflare Tunnel 실행 방법

이제 진짜 로컬 서버를 외부로 노출시킬 차례다. Ngrok과 사용법은 거의 똑같다.

1.  **터널 실행**:

      * 터미널을 열고 다음 명령어를 입력한다.
        ```bash
        cloudflared tunnel --url localhost:8000
        ```
      * 여기서 `8000`은 당신의 로컬 서버 포트 번호다. 만약 Flask 기본 포트인 `5000`을 쓴다면 `localhost:5000`으로 바꿔주면 된다.

2.  **출력 확인**:

      * 명령을 실행하면 터미널에 여러 정보가 표시된다. 그 중 아래와 같은 줄을 찾으면 된다.
        `https://<random-words>.trycloudflare.com`
      * 이 주소가 바로 Ngrok의 주소처럼 외부에서 당신의 로컬 서버로 접근할 수 있는 URL이다.

### 4\. Ngrok와 비교 및 장점

그래서 이게 Ngrok보다 뭐가 그렇게 좋은가?

  * **속도**: 서론에서 말했듯, 그냥 압도적으로 빠르다. 로컬에서 작업하는 것과 거의 차이를 느끼기 힘들 정도.
  * **지속적인 URL**: Ngrok 무료 버전은 실행할 때마다 주소가 바뀌지만, `cloudflared`는 한번 생성된 `trycloudflare.com` 주소를 한동안 계속 유지해준다. 프로그램을 껐다 켜도 같은 주소로 붙는 경우가 많아서 테스트하기 훨씬 편하다.
  * **안정성**: 세계 최대 CDN 기업이 운영하는만큼 연결 안정성이 뛰어나다.
  * **확장성**: 만약 당신이 개인 도메인을 가지고 있다면, `tunnel`을 정식으로 생성해서 `내도메인.com`으로 로컬 서버를 연결하는 미친 짓도 가능하다. 이건 좀 더 복잡하지만, 그만큼 강력하다.

#### 사용 시 주의사항

  * 결국 이 방법도 당신의 컴퓨터가 켜져 있고, 로컬 서버가 실행 중일 때만 유효하다.
  * 개발 및 테스트 용도로는 최고지만, 실제 서비스 배포는 클라우드 서버(AWS, GCP 등)에 하는 것이 정답이다.

이제 느려터진 Ngrok는 가끔씩만 쓰고, 빠르고 쾌적한 Cloudflare Tunnel의 세계로 넘어와보자.