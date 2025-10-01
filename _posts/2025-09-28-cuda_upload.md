---
layout: single
title:  "CUDA, cuDNN 설치했는데도 글카가 안잡혀서 화가날 떄"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## ? 왜 안됨? 좆같네 ㅋㅋ
아니 이제 우리가 CUDA, CUDNN 설치했는데 왜 CUDA 못잡음??  
이럴 때 진짜 개좆같다  
보통,  
```bash
nvcc -V
```
해 봤는데 오 잘되노 하고  
아래 코드 돌려보지 않느냐  

```python
import torch
import logging

# --- 로깅 설정 ---
# 로그 레벨을 INFO로 설정하여 INFO, WARNING, ERROR, CRITICAL 로그를 모두 출력합니다.
# 로그 형식: [시간] - [로그 레벨] - [메시지]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_gpu_availability():
    """
    PyTorch에서 CUDA GPU 사용 가능 여부를 확인하고 결과를 로깅합니다.
    """
    logging.info("GPU 사용 가능 여부 확인을 시작합니다...")

    # torch.cuda.is_available()를 통해 GPU가 사용 가능한지 확인
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu_index)
        
        logging.info(f"성공: CUDA를 사용할 수 있습니다.")
        logging.info(f"사용 가능한 총 GPU 개수: {gpu_count}개")
        logging.info(f"현재 PyTorch가 사용할 GPU: {gpu_name} (장치 ID: {current_gpu_index})")
    else:
        logging.warning("실패: CUDA를 사용할 수 없습니다. PyTorch가 CPU를 사용합니다.")
    
    logging.info("GPU 확인 절차를 완료했습니다.")

# --- 메인 코드 실행 ---
if __name__ == "__main__":
    check_gpu_availability()
```  
근데 ?  
왜 ㅅㅂ 이번엔 GPU 못잡음?  
개좆같네 할 떄가 많다  
그건 이제 트랜스포머, 파이터치 버젼 안 맞는거다 어우 개빡치네 ㅋㅋ  

## 그래서 어떻게 해결해야 하냐
너가 다른 프로젝트 하다가 pip이나 트랜스포머 업데이트 했겠지 ㅋㅋ  

자 그래서 결론은,  

**원인은 다른 버전의 PyTorch가 설치되어 있기 때문이다.**  

터미널에서 `pip install torch`와 같이 간단하게 명령하면, 호환성 문제없이 어디서든 돌아가는 CPU 전용 버전을 설치하는 경우가 많다.  
GPU를 사용하려면 사용 중인 CUDA 버전에 맞는 특정 PyTorch 빌드를 설치해야 한다.  

#### 1. 기존 PyTorch 완전 제거

먼저, 현재 설치된 PyTorch 및 관련 라이브러리를 깨끗하게 삭제한다.  
아래 명령어를 터미널(CMD 또는 PowerShell)에 입력하자.  

```bash
pip uninstall torch torchvision torchaudio
```

삭제 과정에서 `(y/n)`을 물어보면 `y`를 입력하여 진행하면 된다.  

#### **2. 내 환경에 맞는 PyTorch 설치 명령어 확인 및 설치**

**가장 중요한 단계입니다.** PyTorch 공식 홈페이지에서 사용자의 환경에 맞는 설치 명령어를 직접 생성해야 한다.  

1.  \*\*[PyTorch 공식 홈페이지 시작 페이지](https://pytorch.org/get-started/locally/)\*\*로 이동한다.  

2.  아래와 같이 본인의 환경에 맞게 옵션을 선택한다.  

      * **PyTorch Build:** Stable (안정 버전)  
      * **Your OS:** Windows  
      * **Package:** Pip (pip를 사용)  
      * **Language:** Python  
      * **Compute Platform:** **CUDA 12.1** 또는 가장 높은 버전의 CUDA를 선택한다.  

3.  옵션을 선택하면 아래 **Run this Command:** 부분에 설치 명령어가 생성된다. **이 명령어를 복사**합니다. 현재 기준으로 생성되는 명령어는 다음과 같다.  

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    **(주의: 위 명령어는 예시이며, 반드시 공식 홈페이지에서 직접 생성된 최신 명령어를 사용해야 한다.)**  

4.  복사한 명령어를 터미널에 붙여넣고 실행하여 설치를 진행한다. GPU 버전은 다운로드 용량이 크므로(보통 2GB 이상), 인터넷 환경에 따라 시간이 걸릴 수 있다.  


### \#\# 재설치 후 확인

성공적으로 GPU를 인식했다면, 터미널에 다음과 같은 로그가 출력될 것이다.

```
2025-09-27 22:40:10,123 - INFO - GPU 사용 가능 여부 확인을 시작합니다...
2025-09-27 22:40:11,456 - INFO - 성공: CUDA를 사용할 수 있습니다.
2025-09-27 22:40:11,456 - INFO - 사용 가능한 총 GPU 개수: 1개
2025-09-27 22:40:11,456 - INFO - 현재 PyTorch가 사용할 GPU: NVIDIA GeForce RTX 4090 (장치 ID: 0)
2025-09-27 22:40:11,457 - INFO - GPU 확인 절차를 완료했습니다.
```

(GPU 모델명은 사용자 PC의 모델명으로 표시됩니다.)

### \#\# 만약 문제가 계속된다면?

위 방법으로도 해결되지 않는다면 아래 사항들을 추가로 점검해 보자.  

1.  **NVIDIA 드라이버 확인:** 터미널에 `nvidia-smi` 명령어를 입력했을 때 GPU 상태가 정상적으로 표시되는지 확인한다. 이 명령어가 실행되지 않는다면 NVIDIA 드라이버 문제일 수 있다.  
2.  **cuDNN 설치 확인:** cuDNN을 다운로드받아 압축을 푼 뒤, 내부의 `bin`, `include`, `lib` 폴더 안의 파일들을 CUDA Toolkit 설치 경로(보통 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`) 안의 동일한 이름의 폴더에 정확히 복사해 넣었는지 다시 한번 확인해 보자.  


또, 이렇게 파이터치 버젼을 호환시키면, 이번엔 트랜스포머 버젼이 꼬인다.  
그러면, 트랜스포머 다운그래이드 해 주면 된다.  
```bash
pip install "transformers<4.43.0"
```

또는 특정 버전을 지정해서 설치할 수도 있다.  

```bash
pip install transformers==4.42.4
```

#### **방법 2: `optimum` 패키지 버전을 높이기**

최신 버전의 `optimum`이 `transformers` 4.50.3 버전을 지원할 수도 있습니다.  
`optimum`을 최신 버전으로 업그레이드해 보자.  

```bash
pip install --upgrade optimum
```

**가장 먼저 1번 방법으로 `transformers` 버전을 낮춰서 시도해 보시는 것을 추천한다.**  

## 결론
암튼, 설정 해 놨는데 안되고 오랜만에 딥러닝 하면 또 설정해야 하고 아주 사람을 개빡치게 하지만,  
이게 또 딥러닝의 묘미이지 않을까  
암튼, 이번엔 진짜 확실하게 마스터 한 기분이 든다  
다들 오류 잘 해결하고 행복한 딥러닝 생활되기 바란다.  