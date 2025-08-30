---
layout: single
title:  "AI 입문 2편: 미분·적분·Dense Layer — 경사하강의 심장부"
categories: "AI"
tag: "Explanation"
toc: true
author\_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈

1편에서 **토큰화 → 임베딩 → 벡터/행렬 → 소프트맥스 → 크로스 엔트로피**까지 정리했었다.  
이번 2편에서는 모델이 실제로 **학습**한다는 것이 무엇인지, 그 수학적 핵심인 **미분(gradient)**과 **적분의 관점**, 그리고 신경망의 기본 단위인 **Dense(완전연결) Layer**를 **수식과 예제**로 깊이 파고들자.  
목표는 TR(트랜스포머)를 겨냥했지만, 그 바닥을 이루는 공통원리를 확실히 잡는 것이다.  
초심자도 읽고 따라오면 **“아, 신경망이 이렇게 움직이는구나”** 감각이 잡히게 하는것이 이 글의 목표이다.  

---

## 0. 이번 편의 핵심

* **학습 = 손실을 줄이는 방향으로 파라미터를 조금씩 움직이는 과정**이다.
* 그 방향과 크기를 알려주는 것이 \*\*미분(그래디언트)\*\*이다.
* Dense Layer는 **선형변환 + 편향 + 비선형 활성화**로 이해하면 된다.
* **배치 처리**와 **행렬 미분**을 알면 실전 계산이 훨씬 깔끔해진다.
* 적분은 “연속 누적”의 관점에서 확률·기댓값·규제(regularization) 직관을 준다.

---

## 1. 미분(derivative) — 왜 필요한가

### 1.1 손실 함수와 기울기

신경망이 갖는 학습 가능한 모든 값(가중치 $W$, 편향 $\mathbf{b}$ 등)을 파라미터 $\theta$라 하자.  
학습의 목적은 손실 함수 $L(\theta)$를 작게 만드는 것이다.  

* **문제**: $\theta$가 수천만·수억 개일 수 있는데, 어디로 움직여야 $L$이 줄어들까?  
* **해결**: **미분**이 알려준다.  

  $$
  \nabla_\theta L(\theta) = \left[ \frac{\partial L}{\partial \theta_1},\dots,\frac{\partial L}{\partial \theta_n} \right]^T
  $$

  이 벡터가 **가장 가파르게 증가하는 방향**이다. 그러면 **감소**시키려면 반대로 가면 된다.  

### 1.2 경사하강법(Gradient Descent)

가장 단순한 업데이트 규칙은 다음과 같다.  

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)
$$

* $\eta$는 **학습률(learning rate)**. 너무 크면 튀고, 너무 작으면 학습이 느리다.  

### 1.3 스칼라/벡터/행렬의 미분 감각

* 스칼라 $x$에 대한 $f(x)$의 미분: $f'(x) = \frac{df}{dx}$  
* 벡터 $\mathbf{x}\in\mathbb{R}^d$에 대한 스칼라 $f(\mathbf{x})$의 **그라디언트**:  

  $$
  \nabla_{\mathbf{x}} f = \begin{bmatrix}\frac{\partial f}{\partial x_1}\\ \vdots\\ \frac{\partial f}{\partial x_d}\end{bmatrix}
  $$
* 벡터-벡터 함수의 미분은 **야코비안(Jacobian)**, 스칼라의 2차 미분은 **헤시안(Hessian)**으로 일반화된다.  

---

## 2. 연쇄법칙(Chain Rule) — 신경망 역전파의 엔진  

신경망은 여러 연산이 **겹겹이 합성**되어 있다.  
합성함수 $f(g(h(\cdot )))$의 미분은 **연쇄법칙**으로 계산한다.  

* 스칼라 버전:

  $$
  \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
  $$
* 벡터/행렬 버전(개념):

  $$
  \frac{\partial L}{\partial \mathbf{x}} 
  = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^\top \frac{\partial L}{\partial \mathbf{y}}
  $$

  여기서 $\mathbf{y}$는 $\mathbf{x}$의 함수. 즉 **뒤에서 앞으로(출력→입력)** 그라디언트를 전달한다(**역전파**).  

> **직관**: “출력에서 조금 변화가 생기면, 입력 쪽에서는 어느 정도 변화가 필요했을까?”를 단계별로 역산하는 과정이 역전파이다.  

---

## 3. Dense Layer를 수식으로 뜯어보기

Dense Layer는 완전연결을 위해 존재한다.  
입력과 출력 사이의 모든 노드간 연결에 가중치 $W%를 부여하는 것이 Dense Layer의 역할이다.  
중요도를 통해 학습할 수 있게 하며, 추가로 편향을 더해 주는 것 까지가 Dense Layer의 역할이다.  

### 3.1 정의와 모양(shape)

Dense(완전연결) Layer는 다음을 계산한다.  

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b},\quad \mathbf{y} = f(\mathbf{z})
$$

* 입력 $\mathbf{x}\in\mathbb{R}^{d\_\text{in}}$
* 가중치 $W\in\mathbb{R}^{d\_\text{out}\times d\_\text{in}}$
* 편향 $\mathbf{b}\in\mathbb{R}^{d\_\text{out}}$
* 활성화 $f$는 요소별 비선형(예: ReLU, Sigmoid, GELU)

**배치(batch)** 입력 $X\in\mathbb{R}^{B\times d\_\text{in}}$에 대해:

$$
Z = X W^\top + \mathbf{1}\mathbf{b}^\top \quad (\in \mathbb{R}^{B\times d_\text{out}}), \quad Y = f(Z)
$$

* 여기서 $\mathbf{1}\in\mathbb{R}^{B\times 1}$은 1로 찬 열벡터(브로드캐스트용).  

### 3.2 손실과 출력층의 연결

1편에서 본 **크로스 엔트로피**와 **소프트맥스**를 출력층으로 사용하면:  

$$
\text{logits } Z = XW^\top + \mathbf{1}\mathbf{b}^\top,\quad P = \text{softmax}(Z)
$$

$$
L = -\frac{1}{B}\sum_{i=1}^{B} \sum_{k=1}^{K} y^{(i)}_k \log P^{(i)}_k
$$

여기서 $y^{(i)}$는 원-핫(혹은 확률분포) 레이블, $K=d\_\text{out}$.

---

## 4. Dense Layer의 역전파: 미분을 실제로 계산하자

### 4.1 소프트맥스+크로스엔트로피의 유명한 결과

소프트맥스와 크로스엔트로피를 결합하면 **출력 로짓(logit) $Z$에 대한 그라디언트가 간단**해진다:  

$$
\frac{\partial L}{\partial Z} = \frac{1}{B}(P - Y)
$$

이 결과는 실전 구현에서 매우 중요하다. (편의상 배치 평균을 포함한 형태로 표기)  

### 4.2 선형부에 대한 미분

$Z = XW^\top + \mathbf{1}\mathbf{b}^\top$이므로, 행렬 미분 규칙을 쓰면:  

* 가중치:

  $$
  \frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^\top X
  $$

  (모양: $d\_\text{out}\times d\_\text{in}$)
* 편향:

  $$
  \frac{\partial L}{\partial \mathbf{b}} = \left(\frac{\partial L}{\partial Z}\right)^\top \mathbf{1}
  $$

  (모양: $d\_\text{out}$, 즉 각 출력 차원별로 배치 방향 합)
* 입력(다음 레이어로 거슬러 갈 때):

  $$
  \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \, W
  $$

### 4.3 활성화가 있을 때  

중간에 활성화 $f$가 있으면 체인룰로 한 번 더 이어 붙이면 된다:  

$$
\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial Y} \odot f'(Z)
$$

* 예) ReLU: $f'(z)=\mathbb{1}[z>0]$
* 예) Sigmoid: $f'(z)=\sigma(z)(1-\sigma(z))$

---

## 5. 숫자로 따라가는 미니 예제 (손으로 풀어보기)

**설정**: 이진 분류(0/1), 배치 $B=2$, 입력 차원 $d\_\text{in}=2$, 출력 차원 $d\_\text{out}=2$(소프트맥스).
입력과 정답:  

* $X=\begin{bmatrix} 1 & 2 \ 0 & 1 \end{bmatrix}$
* 레이블(원-핫): $Y=\begin{bmatrix} 1 & 0 \ 0 & 1 \end{bmatrix}$

가중치·편향 초기값(작게 랜덤이라 가정):

* $W=\begin{bmatrix} 0.10 & -0.20 \ 0.05 & 0.10 \end{bmatrix}$
* $\mathbf{b}=\begin{bmatrix} 0.00 \ 0.00 \end{bmatrix}$

### 5.1 순전파(Forward)

로짓:

$$
Z = XW^\top + \mathbf{1}\mathbf{b}^\top 
= \begin{bmatrix} 1&2 \\ 0&1 \end{bmatrix}
\begin{bmatrix} 0.10 & -0.20 \\ 0.05 & 0.10 \end{bmatrix}^\top
= 
\begin{bmatrix}
1\cdot0.10 + 2\cdot0.05 & 1\cdot(-0.20) + 2\cdot 0.10 \\
0\cdot0.10 + 1\cdot0.05 & 0\cdot(-0.20) + 1\cdot 0.10
\end{bmatrix}
=
\begin{bmatrix}
0.20 & 0.00 \\
0.05 & 0.10
\end{bmatrix}
$$

소프트맥스: 각 행에 대해

* 1행: $\text{softmax}([0.20,0.00]) \approx [0.5498, 0.4502]$
* 2행: $\text{softmax}([0.05,0.10]) \approx [0.4875, 0.5125]$

즉 
$$
P = \begin{bmatrix}
0.5498 & 0.4502 \\
0.4875 & 0.5125
\end{bmatrix}
$$

크로스엔트로피(배치 평균):

$$
L = -\frac{1}{2} \left(\log 0.5498 + \log 0.5125\right) \approx 0.3306
$$

### 5.2 역전파(Backward)

출력 로짓에 대한 그라디언트:

$$
\frac{\partial L}{\partial Z} = \frac{1}{B}(P - Y) 
= \frac{1}{2}
\begin{bmatrix}
-0.4502 & 0.4502 \\
0.4875 & -0.4875
\end{bmatrix}
=
\begin{bmatrix}
-0.2251 & 0.2251 \\
0.2438 & -0.2438
\end{bmatrix}
$$

가중치/편향:

$$
\frac{\partial L}{\partial W} 
= \left(\frac{\partial L}{\partial Z}\right)^\top X
=
\begin{bmatrix}
-0.2251 & 0.2438\\
0.2251 & -0.2438
\end{bmatrix}
\begin{bmatrix}
1 & 2\\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
-0.2251 & -0.2064\\
0.2251 & 0.2064
\end{bmatrix}
$$

$$
\frac{\partial L}{\partial \mathbf{b}}
= \left(\frac{\partial L}{\partial Z}\right)^\top \mathbf{1}
=
\begin{bmatrix}
-0.2251 & 0.2438\\
0.2251 & -0.2438
\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
=
\begin{bmatrix}
0.0187\\
-0.0187
\end{bmatrix}
$$

입력 그라디언트(다음 레이어로 전달하고 싶을 때):

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W
=
\begin{bmatrix}
-0.2251 & 0.2251 \\
0.2438 & -0.2438
\end{bmatrix}
\begin{bmatrix}
0.10 & -0.20\\
0.05 & 0.10
\end{bmatrix}
=
\begin{bmatrix}
-0.0113 & 0.0675\\
0.0122 & -0.0729
\end{bmatrix}
$$

#### 역전파를 단계별로 유도해 보자
**설정**: 1차원(채널=1), $N=2$ (작은 배치). 입력 $x = [1.0, 3.0]$. $\gamma=2.0,\ \beta=0$. eps 아주 작다고 가정($\varepsilon=0$으로 근사). 손실 함수: $L = \frac{1}{2}(y_1^2 + y_2^2)$ (MSE with zero target).

**단계 1: 평균·분산 계산**

* $\mu = (1+3)/2 = 2.0$.
* $\sigma^2 = ((1-2)^2 + (3-2)^2)/2 = (1 + 1)/2 = 1.0$.
* $s = \sqrt{\sigma^2 + \varepsilon} = 1.0$.

**단계 2: 정규화, 출력 계산**

* $\hat{x} = [(1-2)/1, (3-2)/1] = [-1, +1]$.
* $y = \gamma \hat{x} + \beta = 2 \cdot [-1, +1] = [-2, +2]$.

**단계 3: 손실과 $\partial L/\partial y$**

* $L = 0.5((-2)^2 + 2^2) = 0.5(4+4) = 4.0$
* $\frac{\partial L}{\partial y} = [y_1, y_2] = [-2, +2]$ (왜냐하면 $d(0.5 y^2)/dy = y$).

**단계 4: $\partial L/\partial \gamma, \partial L/\partial \beta$**

* $\partial L/\partial \gamma = \sum_i (\partial L/\partial y_i) \hat{x_i} = (-2)(-1) + (2)(1) = 2 + 2 = 4.$
* $\partial L/\partial \beta = \sum_i (\partial L/\partial y_i) = -2 + 2 = 0.$

**단계 5: $\partial L/\partial x$** — 공식 사용:

공식:

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{s} \Big( \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{N} \sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \cdot \frac{1}{N} \sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \Big)
$$

여기서 $\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$ 이므로 먼저 계산:

* $\frac{\partial L}{\partial \hat{x}} = [-2, 2] \cdot 2 = [-4, 4]$.
* $\sum_j \frac{\partial L}{\partial \hat{x}_j} = -4 + 4 = 0$.
* $\sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j = (-4)(-1) + (4)(1) = 4 + 4 = 8$.

따라서, 각 원소에 대하여:

* $i=1:\quad \frac{\partial L}{\partial x_1} = \frac{2}{1} \Big( -4 - 0 - (-1) \cdot \frac{8}{2} \Big) / 2 ?$ — 여기서 꼼꼼히 단위를 맞추자. 편의를 위해 공식의 형태를 다시 쓰면:

정리된 형태:

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{s} \Big( \frac{\partial L}{\partial y_i} - \frac{1}{N} \sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \cdot \frac{1}{N} \sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \Big)
$$

(여기서는 이미 $\partial L/\partial y$를 쓰는 형태로 계산해도 된다.)

* $\sum_j \frac{\partial L}{\partial y_j} = -2 + 2 = 0$.
* (\sum\_j \frac{\partial L}{\partial y\_j} \hat{x}\_j = (-2)(-1) + 2(1) = 2 + 2 = 4.

따라서

* $\frac{\partial L}{\partial x_1} = \frac{2}{1} \Big( -2 - 0 - (-1) \cdot \frac{4}{2} \Big) = 2 \Big( -2 + 2 \Big) = 0.$
* $\frac{\partial L}{\partial x_2} = \frac{2}{1} \Big( 2 - 0 - (1) \cdot \frac{4}{2} \Big) = 2 \Big( 2 - 2 \Big) = 0.$

결과: $\partial L / \partial x = [0,0]$.

**해석**: 이 간단한 예에서는 대칭성 때문에 입력에 대한 그래디언트가 0이 된다(모든 변화가 $\gamma$ 쪽으로 흘러감).  
작은 예제로 역전파 공식을 직접 계산해 보면 수식이 틀리지 않았음을 확인할 수 있다.  

> 이 예제는 역전파 공식을 수치적으로 검증하는 데 좋은 연습이다.  
실제 네트워크에서는 그렇게 자주 0이 되지는 않는다.  

### 5.3 한 스텝 업데이트(학습률 $\eta=0.1$ 가정)

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W},\quad
\mathbf{b} \leftarrow \mathbf{b} - \eta \frac{\partial L}{\partial \mathbf{b}}
$$

수치 대입:

$$W_{new} \approx
\begin{bmatrix}
0.10 & -0.20\\
0.05 & 0.10
\end{bmatrix}
- 0.1 \cdot
\begin{bmatrix}
-0.2251 & -0.2064\\
0.2251 & 0.2064
\end{bmatrix}
=
\begin{bmatrix}
0.1225 & -0.1794\\
0.0275 & 0.0794
\end{bmatrix}$$

이후 같은 과정을 반복하면 손실이 점차 줄어든다.  
**이것이 학습의 미시적 모습**이다.

---

## 6. 활성화 함수와 미분 — 비선형이 필요한 이유  

Dense의 선형변환만으로는 **선형모델**에 불과하다.  
층을 여러 개 쌓아도 활성화가 없으면 전체가 하나의 선형변환으로 합쳐진다.  

$W\mathbf{x} + \mathbf{b}$ 이걸 선형변환이라고 한다.  
- $W\mathbf{x}$ : 가중치를 곱한다.  
- $+\mathbf{b}$ : 편향을 더한다.  
비유하자면, 이건 1자인 선을 늘리고, 이동시키는 것이다.  
구부리거나 접을 수 없는 그런 상태이다.  
근데 풀어야 할 문제가 구불구불 하면 이 상태만으론 못푸는거 아니냐
그래서 활성화 함수를 통해 비선형으로 만들어 해결할 수 있게 한다.  

**비선형 활성화**가 있어야 복잡한 함수 근사가 가능하다.  

선형변환을 거쳐 나온 결과값은 아직 정제도지 않은 값들이다.  
이 값을 받아 모양을 바꾸어 다음층에 전달하는것이 활성화함수이다.  


* **ReLU**: $$f(z)=\max(0,z)$$, $$f'(z)=\mathbb{1}[z>0]$$

  * 장점: 계산 단순, 기울기 소실 덜함
  * 주의: 음수영역에서 기울기 0(죽은 ReLU)
  * 음수 영역의 데이터를 모두 0으로 한다.  
* **Sigmoid**: $f(z)=\frac{1}{1+e^{-z}}\$, \$f'(z)=\sigma(z)(1-\sigma(z))$

  * 확률 해석에 유리하나 깊은 층에서는 기울기 소실
* **Tanh**: $(-1,1)$ 범위, 중심이 0이라 수렴에 유리한 경우  
* **GELU**(TR에서 자주 등장): 매끄럽고 성능 상 이점 보고됨  

> 다음 편(3편)에서 각 활성화의 수식·그래프·미분을 비교하고, **왜 특정 활성화가 언제 유리한지** 정리할 것이다.  

---

## 7. 적분의 관점 — 연속 누적, 기댓값, 규제

**적분**은 딥러닝에서 직접 미분처럼 매 스텝 계산되지는 않지만, **확률과 평균, 연속 누적**의 관점을 제공한다.  

### 7.1 기댓값(Expectation)

확률변수 $X$의 기댓값:  

$$
\mathbb{E}[X] = \int x \, p(x)\, dx
$$

미니배치를 평균내는 행위는, 샘플링을 통해 **기댓값을 근사**하는 것으로 이해할 수 있다.  

### 7.2 규제(Regularization) 직관

$L\_2$ 규제(가중치 감쇠):  

$$
L_{\text{total}} = L_{\text{data}} + \lambda \|W\|_2^2
$$

이는 **파라미터 공간**에서의 “에너지”를 누적해 벌점으로 더하는 행위로 볼 수 있다(적분적 누적의 관점).  
과적합을 줄이고 **일반화**를 돕는다.

### 7.3 연속 시간 관점

경사하강을 무한히 작은 스텝으로 본다면 **연립미분방정식**의 해석과 연결된다.  
실전에서는 이 관점이 **최적화 해석**이나 **연속정학습** 직관을 준다.  

---

## 8. 배치, 브로드캐스트, 그리고 행렬 미분 꿀팁

* **배치 평균**: 대부분의 프레임워크는 배치 축을 기준으로 평균을 취한다.  
손실과 그라디언트의 스케일에 영향을 주므로 학습률과 함께 고려해야 한다.  
* **브로드캐스트**: $\mathbf{b}$는 출력 차원과만 호환되면 자동으로 배치 전체에 더해진다.  
역전파에서 **편향 그라디언트는 배치 축으로 합산**된다.  
* **행렬 미분**: 실전 코드에서는 `dW = dZ^T @ X`, `db = dZ^T @ ones` 패턴이 자주 등장한다.  
모양을 먼저 맞추고 수식을 확인하면 실수 줄이기 쉽다.  

---

## 9. 입력 스케일과 초기화 — 기울기 폭주/소실 방지의 첫걸음

### 9.1 입력 정규화

입력 특징이 서로 다른 스케일이라면 **학습 표면**이 기울어져 경사하강이 비효율적이다.  
평균 0, 분산 1 스케일로 맞추면 보통 수렴이 빨라진다.  

### 9.2 가중치 초기화

레이어 깊이가 깊어질수록 출력 분산이 폭주/소실되지 않도록 초기화가 설계되었다(예: Xavier/He 초기화).  
Dense에도 동일한 원리 적용이다.  

---

## 10. 미분을 코드로 확인해보기 (NumPy 의사코드)

```python
import numpy as np

# ----- 하이퍼파라미터 -----
lr = 0.1                      # 학습률
B, d_in, d_out = 2, 2, 2      # 배치, 입력, 출력 차원

# ----- 데이터 -----
X = np.array([[1.0, 2.0],
              [0.0, 1.0]])               # (B, d_in)
Y = np.array([[1.0, 0.0],
              [0.0, 1.0]])               # (B, d_out) 원-핫

# ----- 파라미터 초기화 -----
W = np.array([[0.10, -0.20],
              [0.05,  0.10]])            # (d_out, d_in)
b = np.zeros((d_out,))                    # (d_out,)

def softmax(Z):
    # 수치안정성: 최대값 빼기
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ----- 순전파 -----
Z = X @ W.T + b[None, :]       # (B, d_out)
P = softmax(Z)                 # (B, d_out)
loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))  # CE

# ----- 역전파 -----
dZ = (P - Y) / B               # (B, d_out)  소프트맥스+CE 조합의 핵심
dW = dZ.T @ X                  # (d_out, d_in)
db = dZ.T @ np.ones((B,))      # (d_out,)

# ----- 업데이트 -----
W -= lr * dW
b -= lr * db
```

> 이 코드 한 블록이 **Dense + Softmax + Cross-Entropy**의 전과정을 드러낸다.  
> 실제 프레임워크(PyTorch/TF)는 이 과정을 \*\*자동미분(autograd)\*\*으로 수행한다.  

---

## 11. 자주 헷갈리는 포인트 정리

* **`W @ x` vs `x @ W`**: 본문에서는 \$W\$가 **행 = 출력차원**, **열 = 입력차원**이 되게 표기했다.  
코드에서는 데이터를 배치 우선으로 두어 `X @ W.T`가 자연스럽다. 모양을 먼저 확정하자.  
* **편향 미분**은 배치 방향 합: `db = dZ^T @ ones`.  
* **소프트맥스 안정성**: `logsumexp` 형태(최대값 빼기)로 구현해야 언더/오버플로를 피한다.  
* **배치 평균 여부**: 손실/그라디언트에 평균을 넣는지 합으로 둘지 일관성을 유지하자(학습률 조정에 영향).  

---

## 12. Dense Layer를 쌓으면 무슨 일이 생기나

두 개의 Dense를 활성화와 함께 쌓자:  

$$
\begin{aligned}
\mathbf{h} &= f_1(W_1 \mathbf{x} + \mathbf{b}_1) \\
\mathbf{y} &= f_2(W_2 \mathbf{h} + \mathbf{b}_2)
\end{aligned}
$$

역전파는:  

$$
\frac{\partial L}{\partial (W_2,\mathbf{b}_2)} \text{는 앞 장과 동일},\quad
\frac{\partial L}{\partial \mathbf{h}} = \left( \frac{\partial L}{\partial \mathbf{y}} \odot f_2'(Z_2) \right) W_2
$$

이 $\frac{\partial L}{\partial \mathbf{h}}$가 다시 첫 층으로 전파되어:  

$$
\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial \mathbf{h}} \odot f_1'(Z_1),\quad
\frac{\partial L}{\partial W_1} = \left(\frac{\partial L}{\partial Z_1}\right)^\top X, \quad
\frac{\partial L}{\partial \mathbf{b}_1} = \left(\frac{\partial L}{\partial Z_1}\right)^\top \mathbf{1}
$$

> **핵심**: 층이 더해져도 패턴은 같다. “선형 → 비선형 → (필요시) 정규화 → …”의 블록을 연쇄법칙으로 역전파하면 된다.

---

## 13. 최적화 한 숟갈 — SGD에서 Adam까지(개념만)

* **SGD**: 위에서 본 기본 경사하강. 간단하고 강력하다.  
* **Momentum**: 이전 기울기를 관성처럼 조금 유지해 진동을 줄이고 빠르게 수렴.  
* **RMSProp/Adam**: 차원별로 **적응적 학습률**을 추정해 스케일 문제를 완화.  

  * Dense 레이어 학습에서도 자주 기본값으로 Adam을 쓴다.  
  * 단, 과한 적응은 오버핏/일반화 문제를 낳기도 하므로 **검증 성능**으로 판단하자.  

---

## 14. 실무 감각: 수치 안정성과 디버깅  

* **폭주/소실 기울기**: 초기화·활성화 선택·정규화(배치/레이어 노름)로 완화.  
* **학습률 스케줄**: 초반 워밍업, 후반 코사인/지수 감소 등으로 안정성↑.  
* **그라디언트 확인**: 폭발 시 gradient clipping, NaN 시 입력/출력 분포·logarithm domain 확인.  
* **단위 테스트**: 아주 작은 네트워크·작은 데이터로 forward/backward가 기대대로 동작하는지 숫자로 검증하자(§5 스타일).  

---

## 15. Dense와 TR(트랜스포머)의 연결고리 미리보기

* TR의 각 블록(어텐션, MLP)은 결국 **선형변환+비선형+정규화**의 변주이다.  
* **어텐션 가중치** 계산에도 소프트맥스가 등장하며, 그 역전파 역시 본 장의 원리로 처리된다.  
* TR의 거대한 구조를 이해하려면, **지금 다룬 Dense/미분/적분 직관**이 반드시 필요하다.  

---

## 이번 편 요약(한 장으로)

* **학습 = 손실 $L(\theta)$를 줄이는 방향으로 $\theta$를 이동**하는 과정이다.  
* 그 방향과 크기를 주는 것이 그래디언트 $\nabla\_\theta L$이다.
* **연쇄법칙**으로 레이어를 따라 **뒤에서 앞으로** 그라디언트를 전달한다(역전파).  
* **Dense Layer**: $Z = XW^\top + b\$, \$Y=f(Z)$,  

  $$
  \frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^\top X,\quad
  \frac{\partial L}{\partial b} = \left(\frac{\partial L}{\partial Z}\right)^\top \mathbf{1}
  $$
* **소프트맥스+크로스엔트로피**의 조합은 $\partial L/\partial Z = (P-Y)/B$로 단순화된다.  
* **적분**은 기댓값·규제·연속누적의 직관을 제공한다.  
* 실전에서는 **수치 안정성, 초기화, 입력 정규화, 옵티마이저**가 함께 작동한다.  

---

## 결론

이번 편에서는 **미분과 연쇄법칙**을 통해 신경망이 어떻게 학습하는지, 그리고 **Dense Layer**가 어떻게 미분되고 업데이트되는지 **수식과 숫자 예제**로 해부하였다.  

다음 3편에서는 **활성화 함수**를 본격적으로 파고들 것이다.  
Sigmoid/Tanh/ReLU/GELU를 **그래프, 미분, 수치 안정성, 표현력** 관점에서 비교하고, **언제 무엇을 선택해야 하는지**를 원리로 설명하자.  
