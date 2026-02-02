---
layout: single
title: "PINN 환경 만들기"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### **PINN 실험 환경 구축 (Windows/myenv 기준)**

#### **1. Python 설치**

* **다운로드:** [Python 공식 홈페이지](https://www.python.org/downloads/)에서 **3.10.x** 또는 **3.11.x** 버전을 다운로드한다. (최신 3.12+는 호환성 문제가 있을 수 있다.)
* **주의사항:** 설치 초기 화면 하단의 **`Add Python to PATH`** 옵션을 **반드시 체크**하고 설치한다.
* **확인:** `cmd`(명령 프롬프트)에서 `python --version`을 입력해 버전이 나오면 성공이다.

#### **2. 가상환경(myenv) 구성**

* **생성:** 프로젝트 폴더로 이동한 뒤 아래 명령어로 가상환경을 만든다.
```bash
python -m venv myenv

```


* **활성화:** 아래 명령어로 가상환경을 켠다. (프롬프트 앞에 `(myenv)`가 떠야 한다.)
```bash
myenv\Scripts\activate

```


* **업그레이드:** `pip`를 최신화한다.
```bash
python -m pip install --upgrade pip

```



#### **3. CUDA 및 cuDNN 설정 (버전 매칭 필수)**

* **버전 확인:** [PyTorch 홈페이지](https://pytorch.org/get-started/locally/)에서 지원하는 CUDA 버전(예: **11.8**)을 먼저 확인한다.
* **CUDA 설치:** [NVIDIA 아카이브](https://developer.nvidia.com/cuda-toolkit-archive)에서 위에서 확인한 버전(11.8)을 다운로드하여 설치한다.
* **cuDNN 설정:** [cuDNN 아카이브](https://developer.nvidia.com/rdp/cudnn-archive)에서 CUDA 버전과 맞는 파일을 받는다. 압축을 푼 뒤 `bin`, `include`, `lib` 폴더 안의 파일들을 **CUDA 설치 경로**(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)에 덮어씌운다.

#### **4. PyTorch 및 라이브러리 설치**

* **PyTorch 설치:** 가상환경이 켜진 상태에서 CUDA 버전에 맞는 명령어를 입력한다. (CUDA 11.8 예시)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```


* **필수 패키지:** 수치 계산과 시각화를 위한 라이브러리를 설치한다.
```bash
pip install numpy matplotlib scipy

```


* **검증:** 파이썬 실행 후 `import torch; print(torch.cuda.is_available())` 입력 시 `True`가 나오면 GPU 연동 성공이다.

#### **5. 라플라스(Laplace) 방정식 정의**

* PINN의 핵심인 자동 미분을 이용해 $\nabla^2 u = u_{xx} + u_{yy}$를 정의하고 테스트한다.
* `test.py` 파일을 만들고 아래 코드를 실행해 본다.

```python
import torch

# 라플라스 잔차 계산 함수 (f = u_xx + u_yy)
def laplace_residual(u_pred, x, y):
    # 1차 미분 (create_graph=True 필수)
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    
    # 2차 미분
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    return u_xx + u_yy

# 테스트
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 임의의 입력값 (requires_grad=True 필수)
    x = torch.rand((5, 1), device=device, requires_grad=True)
    y = torch.rand((5, 1), device=device, requires_grad=True)
    
    # 임의의 예측값 (가정)
    u = torch.sin(x) * torch.cos(y)
    
    # 잔차 계산
    f = laplace_residual(u, x, y)
    print("계산된 잔차 f:\n", f.detach().cpu().numpy())

```