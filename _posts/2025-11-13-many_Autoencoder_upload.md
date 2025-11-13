---
layout: single
title:  "다양한 Autoencoder"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## AutoEncoder
오토인코더에는 정말 다양한 형태가 있는데,  
기본적인 형태는 다들 잘 알고 있을 것 이므로, 오늘은 다른 파생혀응ㄹ 보자.  
오토인코더는 기본적인 형태 외에 다양한 변형 모델을 통해 그 활용도를 극대화한다.  

## 1. 노이즈 제거 오토인코더 (Denoising AE, DAE)

노이즈 제거 오토인코더(DAE)는 원본 입력 $\mathbf{x}$를 $\mathbf{z}$로 압축 후 $\hat{\mathbf{x}}$로 복원하는 대신, **원본 $\mathbf{x}$에 의도적으로 노이즈를 추가한 입력 $\tilde{\mathbf{x}}$**로부터 **원본 $\mathbf{x}$**를 복원하도록 학습한다.  
$\tilde{\mathbf{x}}$는 가우시안 노이즈를 추가하거나 입력 픽셀의 일부를 무작위로 제거(마스킹)하는 방식으로 생성된다.  

DAE의 손실 함수 $\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \| \mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}})) \|^2$는 노이즈가 주입된 $\tilde{\mathbf{x}}$를 인코더 $f_\theta$와 디코더 $g_\phi$에 통과시킨 결과가 노이즈가 없는 원본 $\mathbf{x}$와 같아지도록 강제한다.  

이러한 학습 방식은 모델이 입력을 그대로 복사하는 단순한 항등 함수를 학습하는 것을 방지한다.  
대신 모델은 입력의 손상된 부분을 "제거(denoise)"하고 원본을 복원하는 과정을 통해 데이터의 **더욱 강건한(robust) 잠재 표현 $\mathbf{z}$**를 학습하게 된다.  

## 2. 희소 오토인코더 (Sparse AE, SAE)

희소 오토인코더(SAE)는 잠재 차원 $k$를 입력 차원 $d$보다 작게 설정($k \ll d$)하는 일반적인 AE와 달리, $k \ge d$ 즉, **잠재 차원 $\mathbf{z}$를 입력보다 크게** 설정하는 과완비(Overcomplete) 구조를 갖는다.  

잠재 차원이 입력보다 크면 모델이 $\mathbf{x}$를 $\mathbf{z}$에 단순히 복사하여 의미 없는 항등 함수를 학습할 위험이 있다.  
SAE는 이 문제를 해결하기 위해 손실 함수에 **희소성 규제(Sparsity Regularization)** 항을 추가한다.  
\[ \mathcal{L}_{\text{total}} = \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) + \lambda \cdot \Omega(\mathbf{h}) \]
이때 규제 항 $\Omega(\mathbf{h})$는 잠재 변수 $\mathbf{z}$ 또는 인코더의 중간 활성화 값 $\mathbf{h}$에 대한 페널티로 작용한다.  
가장 간단한 방식은 L1 규제($\Omega(\mathbf{h}) = \sum |\mathbf{h}|$)를 사용하는 것이다.  
또는 $\mathbf{h}$의 평균 활성화 $\hat{\rho}_j$가 목표 희소성 $\rho$(예: 0.05)와 유사해지도록 강제하는 KL 발산($\Omega = \sum_{j=1}^k \text{KL}(\rho \| \hat{\rho}_j)$)을 사용할 수도 있다.  

이 규제를 통해 $\mathbf{z}$의 뉴런 대부분은 비활성화(0에 가깝게)되고 **소수의 뉴런만 활성화**되도록 강제된다.  
결과적으로 각 뉴런은 데이터의 특정 특징(예: 이미지의 엣지, 곡선)을 감지하는 "전문가" 뉴런처럼 작동하며, 모델은 데이터의 **특징 사전(Dictionary of features)**을 학습하게 된다.  

## 3. 변이형 오토인코더 (Variational AE, VAE)

변이형 오토인코더(VAE)는 AE를 **생성 모델(Generative Model)**로 발전시킨 핵심적인 형태이다.  
VAE는 표준 AE와 근본적인 차이점을 갖는다.  

표준 AE의 인코더가 입력 $\mathbf{x}$를 잠재 공간의 단일 벡터 $\mathbf{z}$로 매핑하는 것과 달리, VAE의 인코더는 $\mathbf{x}$에 대한 잠재 공간의 **확률 분포 $p(\mathbf{z}|\mathbf{x})$**를 학습한다.  
구체적으로 인코더는 이 분포를 정의하는 **평균($\boldsymbol{\mu}$)과 분산($\boldsymbol{\sigma}^2$)**을 출력한다.  

작동 과정은 다음과 같다. 먼저 인코더가 출력한 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}^2$를 따르는 정규 분포 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$에서 잠재 벡터 $\mathbf{z}$를 **샘플링**한다.  
(이 과정은 $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \cdot \boldsymbol{\sigma}$ (단, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$)의 Reparameterization Trick을 통해 수행된다.)  
이후 디코더는 샘플링된 $\mathbf{z}$를 입력받아 $\hat{\mathbf{x}}$를 재구성한다.  

VAE의 손실 함수는 다음과 같이 두 항으로 구성된다.  
$$\mathcal{L}_{\text{total}} = \underbrace{\mathbb{E}[\log p(\mathbf{x} | \mathbf{z})]}_{\text{Reconstruction Loss (BCE/MSE)}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL Divergence (Regularization)}}$$
첫 번째 항은 디코더가 $\mathbf{z}$로부터 $\mathbf{x}$를 잘 복원했는지를 측정하는 재구성 손실이다.  
두 번째 항인 KL 발산은 규제 항으로, 인코더가 만들어내는 잠재 분포 $q_\phi(\mathbf{z}|\mathbf{x})$를 표준 정규 분포 $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$에 가깝도록 강제한다.  

이 KL 발산 규제로 인해 잠재 공간 $\mathbf{z}$가 **연속적이고 조밀하게** 채워지는 효과가 발생한다.  
따라서 학습이 완료된 후, $\mathcal{N}(0, \mathbf{I})$에서  
새로운 $\mathbf{z}_{\text{new}}$를 무작위로 샘플링하여 학습된 디코더에 통과시키면,  
학습 데이터와 유사하지만 **완전히 새로운 데이터 $\hat{\mathbf{x}}_{\text{new}}$**를 생성할 수 있다.  


## 결론

오토인코더의 주요 변형들을 요약하면 다음과 같다.  

* **Dense AE** : 이 글에선 안다루었다. 1D 벡터를 입력받아 병목($k \ll d$) 잠재 공간을 통해 입력을 재구성하는 가장 기본적인 형태이다.  
* **CNN AE** :이 글에선 안다루었지만, 4D 텐서(이미지)를 입력받아 CNN을 활용하며, 공간 구조를 유지하고 파라미터 효율성을 높인다.  
* **Denoising AE** : 노이즈가 추가된 입력($\tilde{\mathbf{x}}$)을 받아 원본 $\mathbf{x}$를 복원하며 강건한 특징을 학습한다.  
* **Sparse AE** : 입력보다 큰 잠재 공간($k \ge d$)을 사용하되 희소성 규제($\Omega(\mathbf{h})$)를 추가하여 특징 사전을 학습하고 해석 용이성을 높인다.  
* **Attention AE** : 이 글에선 안다루었지만, 시퀀스나 피처를 입력받아 어텐션 메커니즘을 통해 동적으로 중요한 부분을 압축하며 해석 가능성을 제공한다.  
* **VAE** : 잠재 공간을 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ 분포로 모델링하고 재구성 손실과 $D_{KL}$ 손실을 함께 사용하여, **데이터 생성(Generative)**이 가능하다는 가장 큰 특징을 갖는다.  