---
layout: single
title:  "CUDA 사용시 RuntimeError 해결"
categories: "ERROR"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
 
# CUDA error: unknown error
딥러닝 시 볼 수 있는 오류이다. unknown error 로 나와서 이게 뭐지 싶을꺼다.  
내가 만났던 오류의 전문은 다음과 같다.  
```bash
예외가 발생했습니다. RuntimeError
CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
  File "C:\Users\Desktop\Llama 3.2 1B\Fine_tuning_Korean.py", line 64, in <module>
    trainer.train()
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
```

위와 같은 오류의 원인은 다음으로 추릴 수 있다.  
1. GPU 메모리 부족
  모델이 GPU 메모리를 초과하여 사용할 때 발생할 수 있다.  
  또는, 모델이 아슬아슬하게 VRAM에 올라가 있을 경우에도 발생할 수 있다.  
   배치 크기를 줄이거나 모델의 크기를 조절하여 해결할 수 있다.  

2. 드라이버, 라이브러리 문제
  CUDA 드라이버나 PyTorch의 버전이 맞지 않거나, 설치된 버전이 올바르지 않으면 이 오류가 발생할 수 있다.  
  하지만, 이럴 경우 이 오류가 아닌 다른 형태의 오류를 먼저 만나므로 아닐 가능성이 높다.  

3. 다중 프로세스 문제
  여러 프로세스가 GPU를 동시에 사용할 때 충돌이 발생할 수 있다.  
  이럴 경우, 유튜브나 게임을 끄고 실행하면 양호하게 돌아갈 수 있다.  

4. GPU 관련 하드웨어 문제
  GPU 자체에 하드웨어적인 문제가 있을 경우 이런 오류가 발생할 수 있다.  
  드문 경우로, 거의 해당되지 않는다.  
5. CUDA_LAUNCH_BLOCKING
  이 환경변수를 1로 설정하면 더 정확히 오류를 추적할 수 있다.  
  ```bash
  set CUDA_LAUNCH_BLOCKING=1
  ```

하지만, 위의 경우가 아닐 수 도 있다.  
나같은 경우에는, 듀얼모니터의 다른 한 쪽의 모니터를 껐더니 거기서 GPU의 상태가 바뀌며 오류가 난 것이다.  
이런 얼탱이 없는 경우도 있으니 참고하자.  