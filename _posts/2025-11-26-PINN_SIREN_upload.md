---
layout: single
title:  "PINN SIREN"
categories: "AI"
tag: "PINN"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## SIREN
SIREN은 딥러닝에서 관습적으로 사용하던 ReLU 계열의 활성화 함수를 버리고, 주기 함수인 **Sine**을 채택함으로써 미분 방정식(PDE) 해결에 특화된 구조다.  
핵심은 **입력 좌표를 고주파 영역으로 매핑**하는 것과 **초기화 전략**에 있다.  

### 계층적 중첩 구조 (Layer Stacking)  
SIREN은 **Input Layer**, **Hidden Layers**, **Output Layer**의 3단계로 중첩되며, 각 단계의 수학적 역할은 엄격히 구분된다.  

1.  **입력층 (The Mapping Layer)**  
      * **수식:** $\mathbf{y}_1 = \sin(\omega_0 \cdot (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))$  
      * **역할:** 물리적 공간 좌표($x, y, t$)를 위상(Phase) 공간으로 확장한다.  
      * **핵심 ($\omega_0$):** 여기서 $\omega_0$(보통 30)는 '주파수 스케일링 팩터'다. 저주파인 입력 좌표(예: 0\~1 사이의 $x$)를 강제로 빠르게 진동시켜, 네트워크가 미세한 고주파 패턴을 학습할 수 있는 '해상도'를 확보한다.  
      * **초기화:** 가중치 $\mathbf{W}_1$은 $U(-1/n, 1/n)$ 범위에서 초기화한다.  

2.  **은닉층 (The Propagation Layers)**  
      * **수식:** $\mathbf{y}_i = \sin(\mathbf{W}_i \mathbf{y}_{i-1} + \mathbf{b}_i)$   
      * **역할:** 입력층에서 확보된 주파수 정보를 비선형적으로 합성하여 복잡한 물리적 파동이나 해를 근사한다.  
      * **특이점:** 두 번째 층부터는 $\omega_0$를 곱하지 않는다. 이미 첫 층에서 증폭되었기 때문이다.  
      * **초기화:** 가중치 $\mathbf{W}_i$는 $U(-\sqrt{6/n}, \sqrt{6/n})$ 범위에서 초기화한다. 이는 Sitzmann이 증명한 것으로, Sine 함수를 통과해도 신호의 분산이 유지되도록 하는 필수 조건이다.  

3.  **출력층 (The Linear Projector)**  
      * **수식:** $\mathbf{y}_{out} = \mathbf{W}_{last} \mathbf{y}_{last} + \mathbf{b}_{last}$  
      * **역할:** 추출된 특징들을 선형 결합하여 최종 물리량($u, E, H$ 등)으로 변환한다. 활성화 함수를 쓰지 않는다.  


## NumPy 구현: 순전파 및 역전파  
이 코드는 PyTorch의 `Autograd` 없이, 행렬 연산과 미분 연쇄 법칙(Chain Rule)을 직접 구현하여 SIREN의 작동 원리를 보여준다.  
PINN 학습을 위해서는 역전파(Gradient Descent를 위한 가중치 미분)가 필수적이므로, `forward`와 `backward` 메서드를 모두 구현했다.  

```python
import numpy as np

class SirenLayer:
    """
    SIREN의 단일 레이어.
    Forward: 선형 변환 -> Sine 활성화
    Backward: Chain Rule을 이용한 기울기 계산
    """
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        
        # 1. 초기화 (Sitzmann's Initialization)
        if is_first:
            limit = 1.0 / in_features
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            limit = np.sqrt(6.0 / in_features) / w0
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
            
        self.b = np.zeros((1, out_features))
        
        # 역전파를 위한 캐시 (입력값, 선형변환 결과값 저장)
        self.x_cache = None
        self.z_cache = None # z = Wx + b

    def forward(self, x):
        self.x_cache = x
        # 선형 변환: Z = XW + b
        self.z_cache = np.dot(x, self.W) + self.b
        # 활성화: A = sin(w0 * Z)
        return np.sin(self.w0 * self.z_cache)

    def backward(self, grad_output, learning_rate):
        """
        grad_output: 다음 레이어에서 넘어온 기울기 (dL/dA)
        """
        # 1. 활성화 함수 미분 (Chain Rule)
        # y = sin(w0 * z)  -->  dy/dz = w0 * cos(w0 * z)
        # dL/dz = dL/dy * dy/dz
        grad_activation = self.w0 * np.cos(self.w0 * self.z_cache)
        delta = grad_output * grad_activation # Element-wise multiplication
        
        # 2. 가중치(W) 기울기 계산
        # z = xW + b  -->  dz/dW = x
        # dL/dW = x.T * delta
        grad_W = np.dot(self.x_cache.T, delta)
        
        # 3. 편향(b) 기울기 계산
        grad_b = np.sum(delta, axis=0, keepdims=True)
        
        # 4. 입력(x)에 대한 기울기 계산 (이전 레이어로 전파)
        # z = xW + b  -->  dz/dx = W
        # dL/dx = delta * W.T
        grad_input = np.dot(delta, self.W.T)
        
        # 5. 가중치 업데이트 (Gradient Descent)
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        
        return grad_input

class SirenNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, w0=30.0):
        self.layers = []
        # Input Layer (w0=30, First)
        self.layers.append(SirenLayer(input_dim, hidden_dim, w0=w0, is_first=True))
        
        # Hidden Layers (w0=1, Not First)
        for _ in range(num_layers - 2):
            self.layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0, is_first=False))
            
        # Output Layer (Linear - 활성화 함수 처리는 backward 구조상 SirenLayer를 변형해 사용하거나 Linear만 따로 구현)
        # 편의상 여기서는 마지막 층도 SirenLayer를 쓰되 w0를 작게 하여 선형에 가깝게 하거나, 
        # 실제 구현에선 활성화 함수 없는 Linear Layer를 쓴다. (여기선 예시 단순화를 위해 생략하고 개념 전달에 집중)
        self.last_layer = SirenLayer(hidden_dim, output_dim, w0=1.0, is_first=False) 

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        out = self.last_layer.forward(out) 
        return out

    def backward(self, grad_loss):
        # 역전파: 마지막 층 -> 첫 번째 층
        grad = self.last_layer.backward(grad_loss, learning_rate=0.001)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate=0.001)

# --- 실행 예제 ---
# 간단한 목표: f(x) = sin(x) 를 학습
x_train = np.random.uniform(-np.pi, np.pi, (64, 1))
y_target = np.sin(x_train)

model = SirenNetwork(input_dim=1, hidden_dim=32, output_dim=1)

# 학습 루프 (1회만 예시)
y_pred = model.forward(x_train)
loss = np.mean((y_pred - y_target)**2) # MSE Loss

# dL/dy_pred 계산 (Loss 미분)
grad_loss = 2 * (y_pred - y_target) / x_train.shape[0]

# 역전파 수행
model.backward(grad_loss)

print(f"Loss: {loss:.6f}")
print("역전파 완료: 가중치 업데이트 됨")
```


## 심층 분석: 왜 PINN에서 '두 번' 미분하는가?
PINN에서 미분은 두 가지 다른 맥락에서 발생한다.  
이를 혼동하면 안 된다.  

### Step 1. 가중치 업데이트를 위한 미분 (Training)

위 코드의 `backward` 함수가 수행하는 것이다. Loss를 줄이기 위해 "가중치 $W$에 대해" 미분한다  
($\frac{\partial Loss}{\partial W}$). 이것은 일반적인 딥러닝과 동일하다.  

### Step 2. 물리 방정식 계산을 위한 미분 (Physics Loss)

이것이 PINN의 핵심이다. 물리 법칙(PDE)을 검증하기 위해 "입력 좌표 $x$에 대해" 출력 $y$를 미분한다  
($\frac{\partial y}{\partial x}$, $\frac{\partial^2 y}{\partial x^2}$).  

**왜 SIREN에서 이 두 번째 미분(2계 도함수)이 결정적인가?**  
1.  **ReLU의 한계 (정보 소멸):**  
      * 함수: $f(x) = \text{ReLU}(x)$  
      * 1계 미분: $f'(x)$는 0 또는 1 (Step Function)  
      * **2계 미분:** $f''(x) = 0$ (모든 곳에서 0)  
      * **결과:** 파동 방정식($\nabla^2 u = \dots$) 같은 2계 PDE를 학습할 때, ReLU를 쓰면 2계 미분값이 0이 되어 물리 정보를 학습할 수 없다.  

2.  **SIREN의 강력함 (정보 보존):**  
      * 함수: $f(x) = \sin(x)$  
      * 1계 미분: $f'(x) = \cos(x)$ (정보 유지)  
      * **2계 미분:** $f''(x) = -\sin(x)$ (정보 유지, 원래 형태 복원)  
      * **결과:** 미분을 아무리 많이 해도 0이 되지 않고 정보가 살아있다. 이 성질 덕분에 전자기장 방정식(Maxwell Eq.)이나 파동 방정식 같은 고차 미분 문제를 풀 때 SIREN이 압도적인 성능을 낸다.  

### 수학적 과정 (Chain Rule Breakdown)  
입력 $x$가 네트워크를 통과하여 $y$가 될 때 2계 미분이 어떻게 형성되는지 보자. (간단히 $y = \sin(Wx)$라 가정)  
1.  **Forward:** $y = \sin(Wx)$  
2.  **1st Derivative ($\nabla_x y$):**  
    $$\frac{\partial y}{\partial x} = W \cdot \cos(Wx)$$  
    (속미분 $W$가 튀어나온다)  
3.  **2nd Derivative ($\nabla^2_x y$):**  
    $$\frac{\partial^2 y}{\partial x^2} = W \cdot (-W \cdot \sin(Wx)) = -W^2 \sin(Wx)$$  
    (또다시 속미분 $W$가 나오며 $W^2$이 된다)  

## 결론
SIREN은 층이 깊어지거나 미분 차수가 높아져도 $\sin \leftrightarrow \cos$ 순환을 통해 그래디언트 소실 없이 물리적 정보를 끝까지 보존한다.  
PINN의 Loss 함수 내부에서는 위와 같은 입력에 대한 2계 미분 연산이 추가로 수행되어 물리 법칙을 강제하게 된다.  