---
layout: single
title:  "매트랩으로 전방, 후방, 중앙 차분 구현"
categories: "Matlab"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 매트랩으로 구현한 전방, 중앙, 후방 차분 입니다.  

매트랩을 통해 간단히 구현했습니다.  

```cpp
x = linspace(0, pi/2, 10);
y = sin(x);

dydx_analytical = cos(x);

% 전방 차분법
dydx_forward = diff(y) ./ diff(x);
%dydx_forward = (y(2:end) - y(1:end-1)) ./ (x(2:end) - x(1:end));
dydx_forward = [dydx_forward, NaN];

% 후방 차분법
dydx_backward = diff(y) ./ diff(x);
%dydx_backward = (y(2:end) - y(1:end-1)) ./ (x(2:end) - x(1:end));
dydx_backward = [NaN, dydx_backward];

% 중앙 차분법
dydx_central = (y(3:end) - y(1:end-2)) ./ (x(3:end) - x(1:end-2));
%dydx_central = diff(y, 2) ./ diff(x, 2);
%하지만, 중앙 차분법을 구현할 때 diff(y, 2)와 같은 방법은 일반적인 중앙 차분법의 결과와 일치하지 않는다.
%그 이유는중앙 차분법의 원리가 주어진 지점의 양쪽 점 사이의 평균을 사용하는 반면, diff(y, 2)는 인접하지 않은 값들 사이의
%차이를 계산하기 때문이다.
%따라서 중앙 차분법을 시행할 때는 diff를 사용하지 않고 직접 식으로 계산한다.
dydx_central = [NaN, dydx_central, NaN];

% 전방 차분의 오차 백분율
error_forward = (dydx_analytical - dydx_forward) ./ dydx_analytical * 100;

% 후방 차분의 오차 백분율
error_backward = (dydx_analytical - dydx_backward) ./ dydx_analytical * 100;

% 중앙 차분의 오차 백분율
error_central = (dydx_analytical - dydx_central) ./ dydx_analytical * 100;

% 결과를 테이블
my_table = [x; dydx_analytical; dydx_forward; error_forward; dydx_backward; error_backward; dydx_central; error_central];
disp('x\tAnalytical\tForward Diff.\tForward Error (%)\tBackward Diff.\tBackward Error (%)\tCentral Diff.\tCentral Error (%)')
fprintf('%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\n', my_table);

% 그래프
figure;
plot(x, dydx_analytical, 'b-o', 'DisplayName', '분석적 도함수 (cos(x))');
hold on;
plot(x, dydx_forward, 'r--x', 'DisplayName', '전방 차분 근사');
plot(x, dydx_backward, 'g--x', 'DisplayName', '후방 차분 근사');
plot(x, dydx_central, 'm--x', 'DisplayName', '중앙 차분 근사');
xlabel('x');
ylabel('dy/dx');
title('sin(x)의 도함수: 분석적 vs 전방/후방/중앙 차분 근사');
legend;
hold off;

% 간격을 20개로 쪼개서 해보자.
x1 = linspace(0, pi/2, 20);
y1 = sin(x1);

% 전방 차분법
dydx_forward1 = diff(y1) ./ diff(x1);
dydx_forward1 = [dydx_forward1, NaN];

% 후방 차분법
dydx_backward1 = diff(y1) ./ diff(x1);
dydx_backward1 = [NaN, dydx_backward1];

% 중앙 차분법
dydx_central1 = (y1(3:end) - y1(1:end-2)) ./ (x1(3:end) - x1(1:end-2));
dydx_central1 = [NaN, dydx_central1, NaN];

% 20점 그래프
figure;
plot(x1, dydx_forward1, 'r--x', 'DisplayName', '전방 차분 근사 (20개의 점)');
hold on;
plot(x1, dydx_backward1, 'g--x', 'DisplayName', '후방 차분 근사 (20개의 점)');
plot(x1, dydx_central1, 'm--x', 'DisplayName', '중앙 차분 근사 (20개의 점)');
xlabel('x');
ylabel('dy/dx');
title('20개의 점으로 전방/후방/중앙 차분 근사');
legend;
hold off;

```
