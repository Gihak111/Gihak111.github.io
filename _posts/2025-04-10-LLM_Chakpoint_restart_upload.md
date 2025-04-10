---
layout: single
title:  "딥러닝 엄추었을 때, 재시작과 CheckPoint"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 딥러닝 시 컴퓨터 멈춤

딥러닝을 돌리고 잠에 든 후, 일어났을 때 가장 두려운 것은 컴퓨터의 화면이 아름다운 자연 배경과 지금 시간 떠 있는 것이다.  
비밀번호를 치고 들어가면, 잘 돌아가던 cmd는 꺼져 있고, gpu는 차게 식어 있으며 메모리는 먹던것을 잃어버린듯 공허히 빈 공간만 남아있다.  
쏟아부은 몇 십 몇 백 시간이 허무아게 날라가는 이런 정말 아름다운 일은 가정에서 딥러닝을 돌릴 때 가끔가다 있는 일이다.  
내 경험상 자주 이런 일이 일어났었던 원인들을 생각해 보자면,  
1. 윈도우 업데이트.  
분명히 미뤄뒀는데, 새벽이 지 멋대로 재부팅 해댄다. 난 분명이 1주이 ㄹ뒤로 연기해 뒀는데 왜 이러는건지 모르겠다.  

2. 메모리 오프로드를 박아도 부족한 메모리로 인한 컴퓨터 꺼짐.  
가지고 있는 램 메모리가 아무리 크다 해도 이게 다 찰 때가 있다.  
그게 넘쳐버릴 만 할대 막아주는 코드를 만들어 두지 않았다면, 니 컴퓨터는 갑자기 꺼져버리는 의도치않은 억까를 당할 수 있다.  
물론, Epoch 8쯤에서 옴쳐버린거다 나도 왜 중간에 갑자기 지 혼자서 올라갔는지 모르겠다. 메모리가 혼자 뻥튀기 된로그를 보고 싶었지만, 슬프게도 로그는 날라가 있었다.  

3. 정전
내 방에는 벽면 콘센트가 하나다.  
그거 하나에 에어컨, 본체, 충전기 등등이 박혀 있다.  
여기서 먹는 전력이 많아서 일까, 가끔 에어컨과 본체를 동시에 돌리면 두꺼비 집이 내려간다.  
차단기가 달려있는 콘센트를 사용하지만, 정말 행복하게도 떨어지는 것은 콘센트 전체의 두꺼비 집이다.  
당연히, 본체도 떨어진다.  
돌리던 딥러닝은 날라간다.  

이러한 정말 아름다운 일이 비일비제로 일어난다.  
이럴 경우, 조금이라도 시간을 아끼고 싶다면,  
ChackPoint를 이용하는 방법이 있다.  
이게 정말 아름다운게, 예를들어, 총 2500의 체크 포인트를 두고,  
너의 컴퓨터가 2200에 꺼졌으며, 500마다 체크 포인트 가중치 값을 저장했다면, 2200의 값을 날리는게 아닌, 200의 값만 날리고, 2000번때 체크포인트에 저장된 값은 유지할 수 있다는 것이 정말 아름답다.  
물론, 새로 딥러닝을 시작할 때, 체크포인트를 저장해 놨기 때문에, 꺼진 코드를 다시 실행해도, 0이 아닌 2000부터 딥러닝 할 수 있다.  
코드를 통해 알아보자.  

---

## 예제 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델, 손실 함수, 최적화 함수 초기화
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 훈련 설정
num_epochs = 2500  # 총 훈련 횟수
checkpoint_interval = 500  # 체크포인트 저장 간격

# 임의의 데이터 (예시용)
X_train = torch.randn(100, 10)  # 100개의 샘플, 10개의 특성
y_train = torch.randint(0, 2, (100,))  # 0 또는 1의 레이블

# 훈련 루프
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 진행 상황 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 체크포인트 저장
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
        print(f'Checkpoint saved at epoch {epoch+1}')

print("훈련 완료!")

# 체크포인트 로드 예시
# checkpoint = torch.load('checkpoint_epoch_500.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']
```

정말 간단한 코드다.  
너무 간단히, 체크 포인트만 이해할 수 있게 구현하였다.  
앞서 말한 상황과 일치하게끔 만들었다.  

---

## 아름다운 체크포인트
이러한 체크 포인트는 그저 컴퓨터가 꺼졌을 때 값을 살리기 위함만은 아니다.  
딥러닝을 진행한다 해도, 때론 F1 스코어가 떨어지곤 한다.  
이럴 경우, 기존의 체크 포인트에서, F1스코어 값이 더 높았을 떄으 가중치 값을 가져와 그 상황으로 돌아가는게 가능하다.  
이를 통해서, 버벅이거나, 멈출 경우 빠르게 상황으로부터 벗어나 다른 학습율로 진행하여 대치 상황으로부터 벗어나는 초석이 될 수 있다.  
하지만, 체크포인트는 많은 자원을 소모한다.  
적게는 10GB, 많게는 몇백 기가까지도 잡아먹는다.  
얼마나 많이, 또 자주 체크 포인트를 세우느냐에 따라서 달라지지만,  
가중치 값을 저장한다는 특징은바뀌지 않으므로, 슬프게도 저장 공간자원은 뒤지게 많이 먹는다.  
하지만 이를 통해 얻을 수 있는, 마치 강파방과 같은 이 시스템을 안쓸 이유는 없을 것이다.  
물론, 딥러닝 속도도 전체 적으로 보면 살짝 느려진다.  
하지만, 대치 상황에서 보다 빠르게 벗어나게끔 할 수 있기 때문에, 상황에 따라 이게 더 속도가 빨리 나오곤 한다.  

## 결론

암튼 아름다운 우리 체크 포인트.  
많은 사랑 바란다. 