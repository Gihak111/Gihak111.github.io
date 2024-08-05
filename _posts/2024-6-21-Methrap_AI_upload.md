---
layout: single
title:  "매트랩으로 선형회귀 인공신경망 만들기"
categories: "Matlab"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
# 매트랩으로 인공 신경망 구성하기
다음과 같은 코드를 통해 매트랩으로 딥 러닝을 할수 있다.  
코드를 실행하기 전, 애드온으로 모듈은 다운 받아줘야 한다.  
# 코드
```c
% 데이터 생성
x = rand(1000, 2); % 100개의 2차원 데이터 포인트 생성
y = x(:, 1) + x(:, 2) > 1; % 간단한 선형 분류 기준
y = y'; % 레이블을 행 벡터로 변환

% 신경망 생성
hiddenLayerSize = 10; % 은닉층의 뉴런 수
net = feedforwardnet(hiddenLayerSize);

% 학습 데이터 설정
net.divideParam.trainRatio = 0.7; % 학습 데이터 비율
net.divideParam.valRatio = 0.15; % 검증 데이터 비율
net.divideParam.testRatio = 0.15; % 테스트 데이터 비율

% 신경망 학습
[net, tr] = train(net, x', y); % x와 y를 학습에 사용

% 테스트 데이터에 대한 예측
y_pred = net(x');

% 예측 결과를 이진화 (0 또는 1)
y_pred = y_pred > 0.5;

% 정확도 계산
accuracy = sum(y_pred == y) / length(y);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% 결과 시각화
figure;
hold on;

% 데이터 포인트 시각화
gscatter(x(:,1), x(:,2), y, 'rb', 'xo');

% 분류 경계 시각화
[x1Grid, x2Grid] = meshgrid(0:0.01:1, 0:0.01:1);
xGrid = [x1Grid(:), x2Grid(:)];
yGridPred = net(xGrid');
yGridPred = yGridPred > 0.5;
contourf(x1Grid, x2Grid, reshape(yGridPred, size(x1Grid)), 'LineStyle', 'none', 'FaceAlpha', 0.2);

% 기타 그래프 설정
title('Classification Result and Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 0', 'Class 1', 'Decision Boundary');
grid on;
hold off;

```

다음과 같은 결과를 얻을 수 있다.
make_AI  
Accuracy: 100.00%  
>> make_AI  
Accuracy: 99.80%  

![결과 표 1](https://github.com/Gihak111/Gihak111.github.io/assets/162708096/a26d0425-d1a9-493c-a705-f03eb50759fe)

![결과 표 2](https://github.com/Gihak111/Gihak111.github.io/assets/162708096/1d4d5fe4-8e6d-4796-a687-b517d7c24726)

