---
layout: single
title: "클로드 코드"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## 오픈소스가 되어버진 클라우드 코드
역대 최단기로 10만 스타를 찍어버리면서 씬을 뒤집어 놓은 레포지토리. ultraworkers/claw-code.  
Claude Code 아키텍처를 오픈소스로 싹 다 까발려서 똑같이 다시 만들어버린 프로젝트다. 
저거 쓰면 구독료 안내고 쓸 수 있고, 패키지도 파이썬으로 리펙토링 되어 있어서 저작권에도 안걸린다.
나처럼 로컬 LLM 환경 구현하는 사람들에겐 가뭄의 단비라 할 수 있다.  
사실 이미 나온지는 1달가까이 된 것 같다.  
그 1달동안, 최소한 난 잘사용중이라.  
하지만 이것도 발전에 의해 도태될거라, 또 새로운게 나오면 갈아타야 한다.  


## 설치 해보기


1. 클론하고 냅다 빌드하기
   * 터미널 열고 쿨하게 복붙하자.
   ```bash
   git clone https://github.com/ultraworkers/claw-code
   cd claw-code/rust
   cargo build --workspace
   ```
2. 빌드 끝날 때까지 대기
   * Rust로 큰 걸 짜놔서 빌드하는 데 시간 좀 걸린다. `./target/debug/` 폴더에 `claw` (윈도우는 `claw.exe`) 실행 파일이 예쁘게 뽑혀 있을 거다.  


##  AI 이식하기
여기에 API키나 Gemma4 같은거 올리면 된다.  

1. Anthropic 모델 쓸 때 (가장 기본)
   * `export ANTHROPIC_API_KEY="sk-ant-..."`
2. OpenAI나 호환 모델 쓸 때
   * `export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"`
   * `export OPENAI_API_KEY="내_api_키"`
3. 무료 로컬 Ollama 쓸 때
   * `export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"` 치고, 키는 과감하게 `unset OPENAI_API_KEY`로 날려버린다.
   * 실행할 때 `--model "llama3.2"` 이런 식으로 모델 이름만 딱 지정해 주면 알아서 굴러간다.


## 돌려보면
1. 점검
   * 처음 켰으면 얘가 내 컴퓨터 상태랑 환경 변수 제대로 먹었는지 진단부터 받아야 한다.
   * `cd rust` 하고 `./target/debug/claw /doctor` 딱 쳐보자. 여기서 에러 안 나면 무사 통과다.
2. 명령 꽂아넣기
   * 그냥 딱 한 번 시키고 끄려면 이렇게 친다.
   * `./target/debug/claw prompt "이 레포지토리 요약해줘"`
3. 대화형 모드
   * 계속 티키타카 하면서 코딩 시킬 거면 그냥 실행 파일만 띡 부르면 된다.
   * `./target/debug/claw` (실행하면 터미널이 챗봇 창처럼 바뀐다. 거기다 말 걸면서 일 시키면 된다.)


## 결론
내가 저거 나오자 마자 무작정 실행하고 Gemma4로 계속 돌려봤는데 말이다.  
확실히 그냥 Gemma4 쓰는거랑은 차원이 다른 성능을 보인다.  
그래도 gtp4까지는 올라오는 느낌이여서 아직도 잘 사용하고 있다.  
앞으로 클로드 코드의 유출이 없다 가정해도, 계속해서 강한 모델들 나올텐데,  
로컬에서 돌릴 때 그냥 지금의 클로드만 입혀도 더 강해지니까 앞으로도 사용할 것 같다.  