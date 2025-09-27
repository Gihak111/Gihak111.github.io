---
layout: single
title:  "CUDA CUDNN 파이터치 버젼에 따른 오류"
categories: "pynb"
tag: "ERROR"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## PyTorch와 torchvision 호환성 문제 해결

앱 개발 과정에서 PyTorch 및 torchvision을 활용하는 경우, 특정 버전 간의 호환성 문제가 발생할 수 있다.  
예를 들어 다음과 같은 오류가 발생할 수 있다:  

```python
RuntimeError: operator torchvision::nms does not exist
```

이 오류는 `torchvision`의 NMS(Non-Maximum Suppression) 연산자가 `torch`의 특정 버전과 호환되지 않을 때 발생한다.  

### 해결 방법

1. **PyTorch와 torchvision의 호환성 확인**
   - PyTorch와 torchvision은 항상 호환 가능한 버전을 설치해야 한다.
   - [PyTorch 공식 사이트](https://pytorch.org/get-started/previous-versions/)에서 버전별 호환성을 확인한다.  

#### 호환성 표 (2024년 최신 기준)
| PyTorch 버전 | torchvision 버전 |
|--------------|-------------------|
| 2.0.1        | 0.15.2            |
| 2.0.0        | 0.15.1            |
| 1.13.1       | 0.14.1            |
| 1.13.0       | 0.14.0            |

2. **현재 설치된 버전 확인**
   ```bash
   pip show torch torchvision
   ```

3. **권장 버전 설치**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2
   ```

4. **기존 PyTorch 및 torchvision 제거 후 재설치**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision
   ```

5. **CUDA 호환성 확인**
   - CUDA 지원이 필요한 경우:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     ```
   - CPU 전용 버전 설치:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```

6. **다른 안정적인 버전 시도**
   ```bash
   pip install torch==1.13.1 torchvision==0.14.1
   ```

7. **설치 확인**
   ```python
   import torch
   import torchvision
   print(torch.__version__)
   print(torchvision.__version__)
   ```

위 과정을 통해 PyTorch와 torchvision의 버전 문제를 해결할 수 있다.  
설치 후에도 문제가 지속되면 환경을 다시 구성하거나 상세 오류 메시지를 확인하여 추가적인 해결 방법을 고려해야 한다.  

