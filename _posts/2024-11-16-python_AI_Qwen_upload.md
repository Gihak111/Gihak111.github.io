---
layout: single
title:  "GPT 4o를 넘는 AI 로컬에서 실행해보기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 1. Qwen
알리바바에서 낸 오픈 웨이트로 나와서 로컬에 모델을 다운받아서 사용할 수 있다.  
아래의 코드를 통해서 WebSocket에서 실행할 수 있다.  

```python
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import websockets

# 모델 초기화
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# WebSocket 서버 생성
async def chat_server(websocket, path):
    print("Client connected.")
    while True:
        # 클라이언트로부터 메시지 수신
        input_text = await websocket.recv()
        print(f"User: {input_text}")

        # 모델 출력 생성
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 응답 전송
        await websocket.send(response)
        print(f"Model: {response}")

# WebSocket 서버 시작
async def main():
    async with websockets.serve(chat_server, "localhost", 8765):
        print("Server started on ws://localhost:8765")
        await asyncio.Future()  # 서버를 종료하지 않도록 유지

asyncio.run(main())

```

아래의 코드를 통해서 실행할 수 있다.  
위 코드를 vs코드로 실행하고, 아래의 코드를 cmd로 실행시켜서 웹소켓에 접근해 코드를 사용할 수 있다.  

```python
import asyncio
import websockets

async def chat():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            user_input = input("You: ")
            await websocket.send(user_input)
            response = await websocket.recv() 
            print(f"AI: {response}")

asyncio.run(chat())
```
위 두 코드를 통해서 코드를 테스트 할 수 있다.  
일반적인 가정집에선 32b 모델은 안돌아 가지만, 3060ti에 8GB 이상의 VRAM이면 7b정도의 모델은 충분히 돌아간다.  
로컬에 두면 나만의 모델을 만들 수 있다. 프리러닝 못하는건 아쉽지만, 그래도 써보면 만족한다.  