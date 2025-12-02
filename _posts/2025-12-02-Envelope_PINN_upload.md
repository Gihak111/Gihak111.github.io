---
layout: single
title:  "Envelope PINN (High-Frequency PDE)"
categories: "AI"
tag: "PINN"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## Envelope PINN
Envelope PINN은 딥러닝 모델(PINN)이 본질적으로 가진 Spectral Bias(고주파 학습 불가) 문제를 해결하기 위해 고안된 구조다.  
핵심은 진동하는 파동 전체를 학습하는 것이 아니라, 빠르게 진동하는 부분(Carrier)은 수학적으로 고정하고, 천천히 변하는 포락선(Envelope)만을 신경망이 학습하도록 유도하는 거다.  

### 구조적 접근 (The Ansatz Decomposition)
Envelope PINN은 네트워크의 출력을 물리적 직관에 기반하여 두 가지 요소로 분해한다.  

1.  **신경망 파트 (The Envelope Generator)**  
      * **수식:** $A(x) = \text{NeuralNet}(x; \theta)$  
      * **역할:** 파동의 진폭이나 위상의 느린 변화를 학습한다. 이것은 저주파(Low Frequency) 성분이므로, 일반적인 MLP나 SIREN으로 매우 쉽게 학습된다.  
      * **초기화:** 일반적인 Xavier 또는 SIREN 초기화 방식을 따른다.  

2.  **고정된 진동 파트 (The Carrier Wave)**  
      * **수식:** $C(x) = e^{i k x}$ 또는 $\sin(k x)$  
      * **역할:** 우리가 이미 알고 있는 지배적인 고주파수($k$) 정보를 주입한다. 네트워크는 이 $k$를 학습할 필요가 없다.  
      * **핵심:** 이 항은 학습되지 않는 상수(Non-trainable) 취급을 받는다.  

3.  **최종 합성 (The Composition)**  
      * **수식:** $u_{pred}(x) = A(x) \cdot C(x)$  
      * **역할:** 신경망의 출력(포락선)에 고주파 진동을 곱해 최종 물리량 $u$를 생성한다.  

## NumPy 구현: 포락선 결합 및 미분   
이 코드는 포락선 PINN의 핵심인 **'Product Rule(곱의 미분법)'**이 어떻게 작동하는지 보여준다.    
Carrier 주파수 $\omega$가 매우 클 때, 네트워크가 굳이 고주파를 쫓아갈 필요 없이 $A(x)$만 학습하면 됨을 구조적으로 강제한다.  

```python
import numpy as np

class EnvelopeWrapper:
    """
    Envelope PINN 모델 래퍼.
    기존 신경망(Envelope) 출력에 고주파 Carrier를 곱한다.
    Forward: A(x) * sin(w*x)
    Backward: Product Rule 적용
    """
    def __init__(self, envelope_net, omega=100.0):
        self.net = envelope_net   # 학습할 신경망 (A(x))
        self.omega = omega        # 고정된 고주파수 (Carrier Frequency)
        
        self.x_cache = None
        self.a_cache = None       # A(x) 출력값 캐시

    def forward(self, x):
        self.x_cache = x
        # 1. 신경망은 천천히 변하는 포락선 A(x)만 예측
        self.a_cache = self.net.forward(x)
        
        # 2. 고주파 진동 항 (Carrier)
        carrier = np.sin(self.omega * x)
        
        # 3. 최종 출력 u(x) = A(x) * sin(wx)
        return self.a_cache * carrier

    def backward(self, grad_output, learning_rate):
        """
        grad_output: dL/du (최종 출력에 대한 Loss 미분값)
        목표: dL/dA를 구해서 신경망(net)에 전달해야 함.
        """
        # u = A * C  (여기서 C = sin(wx))
        # dL/dA = dL/du * du/dA
        # du/dA = C (Carrier 값 그대로)
        
        carrier = np.sin(self.omega * self.x_cache)
        
        # 신경망으로 흘려보낼 기울기
        grad_for_net = grad_output * carrier
        
        # 신경망 내부 역전파 수행
        grad_input = self.net.backward(grad_for_net) 
        
        return grad_input

    # --- 물리 Loss 계산을 위한 미분 (Product Rule) ---
    def compute_physics_derivatives(self, x):
        """
        PINN Loss를 위해 u_xx 등을 계산할 때 사용되는 로직 설명 (NumPy 의사 코드)
        """
        # A = Net(x)
        # u = A * sin(wx)
        
        # 1계 미분 (u_x)
        # u' = A'sin(wx) + A*w*cos(wx)
        
        # 2계 미분 (u_xx)
        # u'' = A''sin(wx) + 2*A'*w*cos(wx) - A*w^2*sin(wx)
        
        pass # 실제 구현은 Autograd가 처리하나, 수치해석적 원리는 위와 같음

# --- 실행 예제 (개념적) ---
# 목표: u(x) = e^(-x) * sin(50x) 를 학습한다고 가정
# 신경망(net)은 e^(-x)만 배우면 된다. sin(50x)는 우리가 넣어준다.

# envelope_model = SirenNetwork(...) # 앞서 만든 SIREN 등 사용
# model = EnvelopeWrapper(envelope_model, omega=50.0)
```

## 왜 포락선이 필수적인가?

고주파 문제에서 일반 PINN과 Envelope PINN의 차이는 '최적화 난이도(Loss Landscape)'에서 극명하게 갈린다.  

### Step 1. Spectral Bias (일반 PINN의 실패 원인)  

일반적인 MLP는 저주파 성분부터 우선적으로 학습하는 경향이 있다(Rahaman et al., 2019).  

  * **Target:** $u(x) = \sin(100x)$  
  * **현상:** 네트워크는 $x$가 조금만 변해도 출력이 요동쳐야 하는데, 이를 부드러운 직선($y=0$)으로 근사해버리려는 성질이 강하다.  
  * **결과:** $Loss$가 줄어들지 않고, 학습이 수렴하지 않는다.  

### Step 2. 주파수 이동 (Envelope PINN의 해결책)  

Envelope PINN은 문제의 영역을 \*\*Baseband(기저 대역)\*\*로 이동시킨다. 통신 이론의 복조(Demodulation)와 동일한 원리다.  

  * **Ansatz:** $u(x) = A(x) \sin(\omega x)$  
  * **학습 목표:** 네트워크가 맞춰야 할 Target이 $u(x)$에서 $A(x)$로 바뀐다.  
  * **예시:** $u(x) = e^{-x} \sin(100x)$ 라면,  
      * **일반 PINN:** 주파수 100짜리 진동을 직접 그려야 함 (난이도: 최상).  
      * **Envelope PINN:** $A(x) = e^{-x}$ 라는 단순한 지수 감쇠 곡선만 그리면 됨 (난이도: 최하).  

### 수학적 이점 (미분 연산의 보조)  

위 코드의 `compute_physics_derivatives` 주석에서 보듯, 2계 미분($\nabla^2 u$) 계산 시 **확산 항**과 **진동 항**이 분리된다.  
$$\frac{\partial^2 u}{\partial x^2} = \underbrace{A'' \sin(\omega x)}_{\text{작은 항}} + \underbrace{2\omega A' \cos(\omega x)}_{\text{교차 항}} - \underbrace{\omega^2 A \sin(\omega x)}_{\text{지배적인 항}}$$  

일반 PINN은 신경망 혼자서 저 거대한 $-\omega^2$ 스케일을 감당해야 하지만, Envelope PINN에서는 $-\omega^2 \sin(\omega x)$ 부분이 해석적으로 이미 주어지므로, 신경망 $A$는 미세한 오차 보정에만 집중하면 된다.  

## 결론

Envelope PINN은 "네트워크가 못하는 것(고주파 진동)은 인간이 수식으로 넣어주고, 네트워크가 잘하는 것(천천히 변하는 패턴)만 학습시킨다"는 하이브리드 전략이다.  
특히 전자기학(Maxwell Eq.)이나 파동 시뮬레이션에서 파장($\lambda$)이 도메인 크기에 비해 매우 짧을 때, SIREN과 결합하여 사용하면 기존 방식보다 수십 배 빠른 수렴 속도를 보여준다.  
결과, 기존엔 너무 많은 $N_f$로 인해 학습하지 못했던 100m 안테나 등의 구조물도 학습할 수 있게 되었다.