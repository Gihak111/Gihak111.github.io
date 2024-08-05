---
layout: single
title:  "RLC 값을 활용한 여파기 구현"
categories: "Matlab"
tag: "code"
toc: true
author_profile: false
published: true
sidebar:
    nav: "docs"
---

# 매트랩으로 구현한 각종 주파수 차단필터 입니다.

매트랩의 filter 함수를 사용하지 않고 LPF, HPF, BPF, BRF를 구현해 보았습니다.
코드는 각 필터들의 공식을 활용하여 작성하였습니다.

```c
R.L.C 회로의 특징에 대해 주파수 특성 그래프를 그린다.
%여기서 R, L, C 값은 임의로 조정한다.
%RLC 회로의 파라미터 정의

%LPF 저역통과필터
% R과 C 값 설정
R = 1000; % 예시로 1000 옴 (저항 값)
C = 1e-6; % 예시로 1 마이크로패럿 (커패시터 값)
frequencies = logspace(1, 5, 100)

% 주파수 대신 R * C 값을 사용하여 전달 함수 재정의
%여기서의 fc는 R*C를 매개변수로 처리한다.
H = @(RC) 1 ./ (1 + 1j*frequencies*RC);

% 전달 함수를 이용하여 크기 이득 계산
gain = abs(H(R*C));

% 주파수 응답 곡선 그리기
figure;
subplot(2, 1, 1);
semilogx(R*C*frequencies, 10*log10(gain));
title('저역 통과 필터의 주파수 응답 곡선');
xlabel('RC 값');
ylabel('크기 이득 (dB)');
grid on;


%HPF 고역통과필터

% R과 C 값 설정
R = 1000; % 예시로 1000 옴 (저항 값)
C = 1e-6; % 예시로 1 마이크로패럿 (커패시터 값)

% 주파수 대신 R * C 값을 사용하여 전달 함수 재정의
%여기서의 fc는 R*C를 매개변수로 처리한다.
H = @(RC) (1j*frequencies*RC) ./ (1 + 1j*frequencies*RC);

% 전달 함수를 이용하여 크기 이득 계산
gain = abs(H(R*C));

% 주파수 응답 곡선 그리기
subplot(2, 1, 2);
semilogx(R*C*frequencies, 10*log10(gain));
title('고역 통과 필터의 주파수 응답 곡선');
xlabel('RC 값');
ylabel('크기 이득 (dB)');
grid on;


%BPF 대역통과필터

figure(2)
% R, C, L 값 설정
R = 10000; % 저항 값 (옴)
C = 1e-9; % 커패시터 값 (패럿)
L = 100; % 인덕터 값 (헨리)
% 로그 스케일로 주파수 범위 설정
frequencies = logspace(1, 5, 100); 

% 전달 함수 정의
H = R ./ (R + 1j * (1j * frequencies * L - 1./(1j * frequencies * C)));
gain = abs(H);

% 주파수 응답 곡선 그리기
subplot(2, 1, 1);
semilogx(frequencies,gain); % 로그 스케일로 x축 설정
title('대역 통과 필터의 주파수 응답 곡선');
xlabel('주파수 (Hz)');
ylabel('크기 이득 (dB)');
grid on;


%BRF 대역저지필터

R = 10000; % 저항 값 (옴)
C = 1e-9; % 커패시터 값 (패럿)
L = 100; % 인덕터 값 (헨리)
% 로그 스케일로 주파수 범위 설정
frequencies = logspace(1, 5, 1000);
w = j*frequencies;

% 전달 함수 정의 (대역 차단 필터)
H1 = (j * (w * L - 1./(w * C)));
H2 = abs(H1);

H = H2 ./(R + 1j * (w * L - 1./(w * C)));
gain = abs(H);

%주파수 응답 곡선 그리기
subplot(2, 1, 2);
semilogx(frequencies, H); % 주파수 응답의 크기 이득 그래프로 표시
title('대역 차단 필터의 주파수 응답 곡선');
xlabel('주파수 (Hz)');
ylabel('크기 이득 (dB)');
grid on;
x = [1, 2, 3, 4,5 ]
z = find(x > 3)
x(z)
```
다음과 같은 코드를 활용하면 주파수 값을 잘 막는 그래프를 결과로 얻을 수 있습니다.
