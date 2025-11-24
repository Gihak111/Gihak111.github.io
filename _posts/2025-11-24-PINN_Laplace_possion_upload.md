---
layout: single
title:  "PINN + Laplace + possion"
categories: "AI"
tag: "PINN"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---


## PINN의 학습과 사용, 문제 해결 방법 
진짜 매력적인 PINN이다  
오늘은 간단하게, python으로 PINN Laplace랑 possion을 풀어볼 꺼다.  
Dense Layer 여러개로 만들었고,  
실행하면 png 하나 생기는데,  
결과 이미지에서 오른쪽이 FEM으로 나오는 결과, 왼쪽이 PINN으로 나온 결과라고 생각해라
먼저, Laplace 코드이다.  

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
# Qt 에러를 없애기 위해 'Agg' (비대화형 모드)로 설정합니다.
# 주의: 반드시 matplotlib.pyplot을 import 하기 '전'에 써야 합니다!
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. 딥러닝 모델 설계 (Neural Network)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 입력: (x, y) 좌표 2개
        # 은닉층: 20개의 뉴런을 가진 층 4개
        # 출력: u 값 1개
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),  # ★ 중요: ReLU 대신 Tanh 사용 (두 번 미분해도 값이 살아있어야 함)
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        # 좌표 x=(x,y)가 들어오면 u를 예측해서 뱉어냄
        return self.net(x)

# ==========================================
# 2. 물리 법칙 검사기 (Physics Loss Calculator)
# ==========================================
def physics_loss(model, x_f, mode='laplace'):
    """
    이 함수가 바로 PINN의 핵심입니다.
    데이터가 없는 허공(x_f)에서 모델이 물리 법칙을 어겼는지 검사합니다.
    """
    
    # [Step 1] 입력 좌표에 대한 미분 추적 활성화
    # x_f는 랜덤한 좌표점들입니다. 이 점에 대해 미분을 해야 하므로
    # requires_grad=True를 설정하여 PyTorch가 이 변수의 연산 과정을 기억하게 합니다.
    x_f.requires_grad = True
    
    # [Step 2] 모델 예측 (u_pred 구하기)
    u = model(x_f)
    
    # [Step 3] 1차 미분 (Gradient) 구하기: du/dx, du/dy
    # torch.autograd.grad 함수는 출력(u)을 입력(x_f)으로 미분합니다.
    # grad_outputs: 체인룰을 시작하기 위한 초기값 (1로 설정)
    # create_graph=True: ★ 중요! 2차 미분을 하려면 1차 미분 결과도 미분 가능한 그래프로 남겨둬야 합니다.
    grads = torch.autograd.grad(outputs=u, inputs=x_f,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    
    # grads는 [du/dx, du/dy] 형태의 벡터입니다.
    u_x = grads[:, 0:1] # x에 대한 편미분
    u_y = grads[:, 1:2] # y에 대한 편미분
    
    # [Step 4] 2차 미분 (Laplacian) 구하기: d^2u/dx^2, d^2u/dy^2
    # 구해둔 1차 미분값(u_x)을 다시 입력(x_f)으로 미분합니다.
    grads_x = torch.autograd.grad(outputs=u_x, inputs=x_f,
                                  grad_outputs=torch.ones_like(u_x),
                                  create_graph=True, retain_graph=True)[0]
    
    grads_y = torch.autograd.grad(outputs=u_y, inputs=x_f,
                                  grad_outputs=torch.ones_like(u_y),
                                  create_graph=True, retain_graph=True)[0]

    u_xx = grads_x[:, 0:1] # x로 두 번 미분
    u_yy = grads_y[:, 1:2] # y로 두 번 미분
    
    # [Step 5] 잔차(Residual) 계산: 물리 법칙 위배 정도
    # 라플라시안 = u_xx + u_yy
    laplacian = u_xx + u_yy
    
    if mode == 'laplace':
        # 식: ∇²u = 0
        # 잔차 = ∇²u - 0
        loss_f = torch.mean(laplacian ** 2)
        
    elif mode == 'poisson':
        # 식: ∇²u = f(x,y)
        # 여기서 f(x,y)는 우리가 설정한 소스 항입니다.
        # 예제: 정답이 sin(πx)sin(πy)가 되려면 f는 아래와 같아야 함
        x = x_f[:, 0:1]
        y = x_f[:, 1:2]
        source_term = -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        
        # 잔차 = ∇²u - f
        loss_f = torch.mean((laplacian - source_term) ** 2)
        
    return loss_f

# ==========================================
# 3. 학습 준비 및 실행
# ==========================================

# 모드 선택: 'laplace' 또는 'poisson'
MODE = 'poisson' 

# 모델 생성 및 Optimizer 설정
# GPU가 있으면 GPU 사용, 없으면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"--- Training Start ({MODE}) on {device} ---")

# 학습 루프 (Iteration)
for epoch in range(5001):
    optimizer.zero_grad()
    
    # --------------------------------------
    # A. 물리 법칙 손실 (Physics Loss)
    # --------------------------------------
    # 도메인 내부(-1 ~ 1)에서 무작위 점 2000개 추출 (데이터가 없는 곳!)
    x_f = torch.rand(2000, 2).to(device) * 2 - 1 
    loss_physics = physics_loss(model, x_f, mode=MODE)
    
    # --------------------------------------
    # B. 경계 조건 손실 (Boundary Loss)
    # --------------------------------------
    # 경계에서는 정답을 알려줘야 함 (Supervised Learning)
    # 상하좌우 벽면 좌표 생성 (간략화를 위해 전체 랜덤 중 벽 근처만 쓰는 대신, 정답 함수에서 샘플링)
    x_b = torch.rand(500, 2).to(device) * 2 - 1
    
    # 실제 정답값(Ground Truth) 계산 (경계 조건을 알기 위해)
    if MODE == 'laplace':
        # 라플라스 예제 정답: u = x^2 - y^2
        u_b_true = x_b[:, 0:1]**2 - x_b[:, 1:2]**2
    else:
        # 푸아송 예제 정답: u = sin(πx)sin(πy)
        u_b_true = torch.sin(np.pi * x_b[:, 0:1]) * torch.sin(np.pi * x_b[:, 1:2])
        
    # 모델 예측
    u_b_pred = model(x_b)
    
    # 경계에서의 오차 (MSE)
    loss_boundary = torch.mean((u_b_pred - u_b_true) ** 2)
    
    # --------------------------------------
    # C. 총 손실 및 역전파 (Total Loss & Backprop)
    # --------------------------------------
    # 총 손실 = 물리 손실 + 경계 손실
    total_loss = loss_physics + loss_boundary
    
    # 역전파: 이 한 줄이 실행될 때, PyTorch는
    # 2번 미분한 물리 식까지 거슬러 올라가서 웨이트를 수정합니다.
    total_loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f} (Physics={loss_physics.item():.6f}, Boundary={loss_boundary.item():.6f})")

# ==========================================
# 4. 결과 시각화
# ==========================================
# 그래프를 그리기 위해 격자 생성
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
input_tensor = torch.tensor(np.hstack((X.flatten()[:, None], Y.flatten()[:, None])), dtype=torch.float32).to(device)

# 모델 예측
with torch.no_grad(): # 예측만 할 때는 미분 불필요 (메모리 절약)
    u_pred = model(input_tensor).cpu().numpy().reshape(100, 100)

# 실제 정답 계산 (비교용)
if MODE == 'laplace':
    u_true = X**2 - Y**2
else:
    u_true = np.sin(np.pi * X) * np.sin(np.pi * Y)

# 플롯 그리기
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title(f"PINN Prediction ({MODE})")
plt.contourf(X, Y, u_pred, levels=50, cmap='jet')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title(f"Ground Truth ({MODE})")
plt.contourf(X, Y, u_true, levels=50, cmap='jet')
plt.colorbar()

# plt.show()

plt.savefig('result_poisson.png')  # 현재 폴더에 그림 파일로 저장
print("결과 이미지가 result_poisson.png로 저장되었습니다.")
```

이어서, possion 이다
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
# Qt 에러를 없애기 위해 'Agg' (비대화형 모드)로 설정합니다.
# 주의: 반드시 matplotlib.pyplot을 import 하기 '전'에 써야 합니다!
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 모델 설계 (아키텍처는 라플라스와 동일)
# ==========================================
class PoissonPINN(nn.Module):
    def __init__(self):
        super(PoissonPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),  # 미분 가능해야 하므로 Tanh 필수
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 소스 항 f(x,y) 정의 (푸아송의 핵심)
# ==========================================
def get_source_term(x, y):
    """
    ∇²u = f(x,y) 에서 우변인 f를 정의합니다.
    우리가 맞추고 싶은 답이 sin(πx)sin(πy)이므로,
    이를 두 번 미분한 결과인 아래 식을 소스 항으로 줍니다.
    """
    return -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# ==========================================
# 3. 물리 법칙 손실 계산 (Physics Loss)
# ==========================================
def compute_loss(model, x_f, x_b, u_b_true):
    # --- A. 내부(Physics) Loss ---
    x_f.requires_grad = True # 입력 좌표 미분 추적 시작
    u = model(x_f)
    
    # 1차 미분 (Gradient)
    grads = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    
    # 2차 미분 (Laplacian)
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x_f, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    # 라플라시안 계산: ∇²u
    laplacian = u_xx + u_yy
    
    # [차이점] 푸아송은 0이 아니라 소스 항 f와 같아야 함
    f = get_source_term(x_f[:, 0:1], x_f[:, 1:2])
    
    # 잔차 = ∇²u - f
    loss_pde = torch.mean((laplacian - f) ** 2)
    
    # --- B. 경계(Boundary) Loss ---
    u_b_pred = model(x_b)
    loss_bc = torch.mean((u_b_pred - u_b_true) ** 2)
    
    return loss_pde + loss_bc

# ==========================================
# 4. 학습 시작
# ==========================================
model = PoissonPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("--- 푸아송 방정식 학습 시작 ---")

for epoch in range(5001):
    optimizer.zero_grad()
    
    # 1. 내부 점 (Collocation Points) 생성: -1 ~ 1 사이 랜덤
    x_f = torch.rand(2000, 2).to(device) * 2 - 1
    
    # 2. 경계 점 (Boundary Points) 및 정답 생성
    # (간단히 랜덤 점을 경계 정답 함수에 넣어 경계 데이터로 사용)
    x_b = torch.rand(500, 2).to(device) * 2 - 1
    # 경계 조건 정답 (실제로는 경계 좌표만 써야 하지만 예제 단순화를 위해 함수값 사용)
    u_b_true = torch.sin(np.pi * x_b[:, 0:1]) * torch.sin(np.pi * x_b[:, 1:2])
    
    # 3. 손실 계산 및 역전파
    loss = compute_loss(model, x_f, x_b, u_b_true)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# ==========================================
# 5. 결과 시각화
# ==========================================
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
input_tensor = torch.tensor(np.hstack((X.flatten()[:, None], Y.flatten()[:, None])), dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred = model(input_tensor).cpu().numpy().reshape(100, 100)

# 실제 정답
u_true = np.sin(np.pi * X) * np.sin(np.pi * Y)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("PINN Prediction (Poisson)")
plt.contourf(X, Y, u_pred, levels=50, cmap='jet')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Ground Truth")
plt.contourf(X, Y, u_true, levels=50, cmap='jet')
plt.colorbar()
# plt.show()

plt.savefig('result_poisson.png')  # 현재 폴더에 그림 파일로 저장
print("결과 이미지가 result_poisson.png로 저장되었습니다.")
```

코드 실행해 보면 결과가 잘 나오는 것을 볼 수 있다.  

## 추가로, PCB등 검사하고 싶다면.
이제, 성능을 봤으니까 우리가 써 먹고 싶은대로 쓰고 싶지 않냐.  
간단하게, PCB의 유전값 , 위치값을 1차원 백터로 만들어서 집어넣으면 된다.  
그 전에, 맥스웰 방정식으로 사전학습해서 먼저 b, w 만드느 다음에 pcb 정보 넣으면, 맥스웰에 따른 결과를 얻을 수 있는 그런거도 할 수 있다.  

## 결론
너무 신기하고 재미있는 PINN이다.  