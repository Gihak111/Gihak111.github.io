---
layout: single
title: "AI 아키텍쳐 3. LCM"
categories: "AI"
tag: "Architecture"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## LCM
Latent Consistency Models(LCM)은 고품질 이미지 생성 모델인 잠재 확산 모델(Latent Diffusion Models, LDM)의 치명적인 단점인 '느린 추론 속도'를 해결하기 위해 제안된 프레임워크이다.  
기존의 Stable Diffusion과 같은 모델은 노이즈를 제거하기 위해 수십 번의 반복적인 샘플링(Iterative Sampling) 과정을 거쳐야 한다.  
이는 실시간 애플리케이션 적용에 큰 장벽이 된다.  
LCM은 미분 방정식의 궤적을 순차적으로 계산하는 대신, 임의의 시점에서 최종 해(Solution)인 원본 데이터로 **직접 매핑(Direct Mapping)**하는 함수를 학습함으로써 단 1~4 단계(Steps) 만에 고품질 이미지를 생성한다.  

## 2. 수식적 원리: PF-ODE와 일관성 조건
LCM의 이론적 토대는 확률 흐름 상미분 방정식(Probability Flow ODE, PF-ODE)이다.  
데이터 분포에서 노이즈 분포로의 변환 과정은 다음의 미분 방정식으로 표현된다.  

$$\frac{dx_t}{dt} = v_\Psi(x_t, t)$$

기존 확산 모델이 속도장 $v_\Psi$를 학습하여 $t=T$에서 $t=0$까지 수치 적분을 수행했다면, LCM은 궤적 상의 모든 점 $(x_t, t)$를 시작점 $x_0$로 매핑하는 **일관성 함수(Consistency Function) $f_\theta$**를 직접 모델링한다.  

$$f_\theta(x_t, t) = x_0$$

이 함수는 **일관성 조건(Self-Consistency Property)**을 만족해야 한다.  
즉, 동일한 PF-ODE 궤적 위에 있는 임의의 시점 $t, t'$에 대해 모델의 출력은 항상 동일해야 한다.  

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t')$$

결과적으로 모델은 적분 과정을 생략하고, 현재 상태 $x_t$에서 즉시 원본 $x_0$를 추론하는 능력을 갖게 된다.  

## 3. 학습 방법론: LCD와 Skipping-Step

LCM은 **Latent Consistency Distillation(LCD)**을 통해 학습된다.  
이는 사전 학습된 LDM(Teacher)의 지식을 LCM(Student)으로 전이하는 과정이다.  
효율적인 학습을 위해 인접한 단계가 아닌, $k$ 단계를 건너뛰는 **Skipping-Step** 기법을 적용한다.  

손실 함수 $\mathcal{L}_{LCD}$는 다음과 같이 정의된다.  

$$\mathcal{L}_{LCD}(\theta, \theta^-; \Psi) = \mathbb{E}_{x, t_n, z} \left[ d \left( f_\theta(z_{t_{n+k}}, t_{n+k}), f_{\theta^-}(\hat{z}_{t_n}^\Psi, t_n) \right) \right]$$

1.  **Student ($f_\theta$)**: 시점 $t_{n+k}$의 노이즈 데이터 $z_{t_{n+k}}$를 입력받아 $x_0$를 예측한다.  
2.  **Teacher Guide ($\hat{z}_{t_n}^\Psi$)**: ODE Solver($\Psi$)를 사용하여 $t_{n+k}$에서 $t_n$으로 진행한 상태를 계산한다.  
3.  **Target ($f_{\theta^-}$)**: 학습 안정성을 위해 파라미터의 지수 이동 평균(EMA)인 $\theta^-$를 사용하여 타겟값을 산출한다.  
4.  **최적화**: 두 예측값 간의 거리($d$)를 최소화함으로써, 모델이 Solver의 다단계 연산 결과를 단 한 번의 연산으로 모사하도록 한다.  

## 4. 핵심 기술: 잠재 공간 증류와 SK-Solver
LCM은 기존 Consistency Model(CM)의 한계를 극복하기 위해 두 가지 핵심 기술을 도입했다.  

* **잠재 공간(Latent Space) 활용**: 픽셀 공간(Pixel Space)에서의 연산은 계산 비용이 매우 높다. LCM은 VAE를 통해 압축된 저차원 잠재 공간에서 ODE 궤적을 학습함으로써 연산 효율성을 극대화했다.  
* **SK-Solver (Skipping-Step Solver)**: 미분 방정식의 국소적 오차(Local Error)가 누적되는 것을 방지하기 위해, 큰 타임스텝($k$)을 건너뛰며 학습한다. 이는 모델이 궤적의 전역적 구조(Global Structure)를 빠르게 파악하고 수렴 속도를 높이는 데 기여한다.  

## 5. 아키텍처 확장: LCM-LoRA
LCM의 가장 큰 기여 중 하나는 **LCM-LoRA(Low-Rank Adaptation)**의 도입이다.  
이는 '일관성 매핑' 능력을 전체 모델 파라미터가 아닌, 소수의 LoRA 파라미터로 분리하여 학습하는 방식이다.  

$$W_{new} = W_{pretrained} + A \times B$$

이 방식은 **범용 가속화(Universal Acceleration)**를 가능하게 한다.  
사용자는 수 기가바이트의 전체 모델을 다시 학습할 필요 없이, 사전 학습된 다양한 Stable Diffusion 모델(Base Model)에 작은 크기의 LCM-LoRA 어댑터만 부착함으로써 즉시 고속 추론 모델로 변환할 수 있다.  

## 6. 추론 및 멀티스텝 정제
LCM은 기본적으로 One-step 생성을 지향하나, 품질 향상을 위한 **Multistep Consistency Sampling**을 지원한다.  

1.  **초기 예측**: 노이즈 $z_T$에서 $x_0$를 예측한다.  
2.  **재주입 및 보정(Re-noising & Refinement)**: 예측된 $x_0$에 다시 노이즈를 추가하여 이전 시점의 상태를 만든 후, 다시 예측을 수행한다.  

이 과정을 2~4회 반복함으로써, 단일 단계 추론에서 발생할 수 있는 미세한 아티팩트를 제거하고 디테일을 보강할 수 있다.  

## 결론
LCM은 미분 방정식의 수치해석적 해법을 딥러닝 기반의 함수 근사(Function Approximation)로 대체한 혁신적인 시도이다.  
수십 단계의 연산을 단 1~4 단계로 압축함으로써 실시간 이미지 생성의 가능성을 열었으며, 특히 LoRA와의 결합을 통해 범용성과 효율성을 동시에 확보했다는 점에서 기술적 의의가 크다.  
이는 향후 생성형 AI가 연구실을 넘어 실제 산업 현장의 실시간 애플리케이션으로 확장되는 데 핵심적인 역할을 수행할 뻔 했지만, 더 좋은게 나와 안쓰이게 된다.  
요즘은 Flux.1 (Schnell)게 훤신 잘 나가고 잘 쓰이더라.  