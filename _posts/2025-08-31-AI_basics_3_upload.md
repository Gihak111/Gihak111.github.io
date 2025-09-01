---
layout: single
title:  "AI 입문 3편: 활성화 함수(Activation Functions) – Sigmoid, Tanh, ReLU, GELU"
categories: "AI"
tag: "Explanation"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

# AI 입문 시리즈
2편에서 미분·적분·Dense Layer를 통해 학습의 핵심을 정리했다.  
이번 3편에서는 신경망의 비선형성을 부여하는 **활성화 함수(Activation Functions)**를 본격적으로 알아보겠다.  

구체적으로 **Sigmoid, Tanh, ReLU, GELU**를 정리해 볼거다.  
목표는 TR(트랜스포머)를 겨냥했지만, 그 바닥을 이루는 초석를 확실히 잡는 것이다.  

---

## 0. 이번 편의 핵심  

* **활성화 함수 = 비선형성의 열쇠**: 선형만으로는 복잡한 패턴 학습 불가.  
* 각 함수의 **출력 범위·미분·문제점**을 알면 왜 특정 모델에서 특정 활성화가 선택되는지 이해.  
* **Sigmoid/Tanh**: 초기 신경망의 고전, 하지만 기울기 소실 문제.  
* **ReLU**: 현대의 기본, 단순·빠르지만 죽은 뉴런 문제.  
* **GELU**: TR 시대의 스타, 매끄럽고 성능 우위.  
* **선택 기준**: 깊이·데이터·태스크에 따라 다르다. 실험과 직관으로 익히자.  
* 적분 관점에서 확률·누적·스무딩을 연결해 보자.  

---

## 1. 활성화 함수가 왜 필요한가

### 1.1 선형성의 한계

Dense Layer의 선형 부분만 쌓으면(2편 참조): 

$$
\mathbf{y} = W_n ( \cdots (W_2 (W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) \cdots ) + \mathbf{b}_n = W_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}
$$

즉, 전체가 **하나의 선형 변환**으로 축소된다.  
복잡한 비선형 결정 경계(예: XOR 문제)를 학습할 수 없다.  

* **해결**: 각 층 사이에 비선형 함수 $f$를 끼워넣는다.  

  $$
  \mathbf{h}_1 = f(W_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{y} = f(W_2 \mathbf{h}_1 + \mathbf{b}_2)
  $$

이제 합성이 비선형이 되어 **보편 근사 정리(Universal Approximation Theorem)**에 따라 임의 함수를 근사할 수 있다.  

### 1.2 미분 가능성의 중요

학습은 역전파(2편)가 핵심이므로, 활성화 $f$는 대부분 **미분 가능**해야 한다. (예외: ReLU는 0에서 subgradient 사용)  

* **기울기 전달**: $f'(z)$가 너무 작으면(소실) 또는 폭주하면 학습이 어려워진다.  

### 1.3 그래프와 출력 범위의 역할  

* 출력 범위: 확률(0~1) 해석 vs 중심 0(평균화 유리).  
* 그래프 모양: 매끄러움 vs 날카로움 → 수렴 속도·안정성 영향.  

---

## 2. Sigmoid — 확률 해석의 고전  

### 2.1 정의와 그래프  

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

* 출력: (0, 1)  
* 그래프: S자 곡선, z→∞ 시 1, z→-∞ 시 0, z=0 시 0.5.  

직관: **로지스틱 함수**로, 이진 확률 모델링에 적합(예: 로지스틱 회귀).  

### 2.2 미분

$$
\sigma'(z) = \sigma(z) (1 - \sigma(z))
$$

* 최대값: z=0에서 0.25.  
* 문제: |z| 클 때 미분 → 0 (기울기 소실, vanishing gradient).  

### 2.3 장단점

* 장점: 매끄러움, 출력이 확률처럼 해석 가능.  
* 단점: 
  - 기울기 소실: 깊은 네트워크에서 초기 층 기울기 0 → 학습 정지.  
  - 출력 비중심: 평균 ~0.5 → 다음 층 입력이 양수 편향 → 지그재그 수렴.  
  - exp 계산 비용.  

### 2.4 사용 사례

* 출력층: 이진 분류 (소프트맥스 대신).  
* 초기 RNN/LSTM 게이트.  
* 현대: ReLU 계열로 대체됨, 하지만 이해를 위한 기본.  

---

## 3. Tanh — Sigmoid의 스케일 버전

### 3.1 정의와 그래프

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

* 출력: (-1, 1)
* 그래프: S자, z=0 시 0, 대칭적.

직관: Sigmoid를 (-1,1)로 스케일/시프트. 중심 0으로 수렴 유리.

### 3.2 미분

$$
\tanh'(z) = 1 - \tanh^2(z)
$$

* 최대값: z=0에서 1.
* 문제: |z| 클 때 미분 → 0 (기울기 소실).

### 3.3 장단점

* 장점: 중심 0 → 데이터 중앙화, Sigmoid보다 기울기 강함(최대 1 vs 0.25).
* 단점: 여전히 기울기 소실, exp 비용.

### 3.4 사용 사례

* 초기 신경망 히든 레이어.
* RNN/LSTM (게이트 제외).
* 현대: ReLU로 대체, 하지만 음수 허용이 필요한 경우.

---

## 4. ReLU — 현대 딥러닝의 기본

### 4.1 정의와 그래프

$$
\text{ReLU}(z) = \max(0, z)
$$

* 출력: [0, ∞)
* 그래프: z<0 시 0, z≥0 시 z (직선).

직관: **죽이기와 통과**. 음수 입력 무시 → 희소성(sparsity) 부여.

### 4.2 미분

$$
\text{ReLU}'(z) = \begin{cases} 0 & z < 0 \\ 1 & z > 0 \end{cases}
$$

(z=0은 0 또는 1로 정의, 실전 subgradient 0).

* 문제: z<0 시 기울기 0 (죽은 ReLU, dying ReLU).

### 4.3 장단점

* 장점: 
  - 계산 단순 (no exp), 빠름.
  - 기울기 소실 적음 (양수 영역 1).
  - 희소 활성화 → 효율·과적합 완화.
* 단점: 
  - 죽은 뉴런: 학습 중 일부 뉴런 영원히 0 → 용량 손실.
  - 출력 비중심: 평균 >0 → 편향.

### 4.4 사용 사례

* CNN 히든 레이어 (AlexNet부터 표준).
* 대부분의 딥 네트워크 기본.
* 변형: Leaky ReLU (음수에 작은 기울기 α=0.01).

---

## 5. GELU — TR의 매끄러운 선택

### 5.1 정의와 그래프

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

여기서 Φ(z)는 표준정규 누적분포함수(CDF)이다:

$$
\Phi(z) = \frac{1}{2} \left(1 + erf\left(\frac{z}{\sqrt{2}}\right)\right)
$$

근사식 (실전 구현):

$$
\text{GELU}(z) \approx 0.5 z \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 z^3\right)\right)\right)
$$

* 출력: (-∞, ∞), but 음수 약화.
* 그래프: ReLU 비슷 but 매끄럽고 음수 약간 허용.

직관: **Gaussian Error Linear Unit**. 확률적으로 게이팅 (z * 확률).

### 5.2 미분

$$
\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)
$$

ϕ(z)는 정규 밀도: ϕ(z) = (1/√(2π)) exp(-z²/2).

* 매끄러움: 모든 점에서 미분 가능, 기울기 소실 적음.

### 5.3 장단점

* 장점: 
  - ReLU의 희소성 + Sigmoid/Tanh의 매끄러움.
  - 음수 약간 허용 → 죽은 뉴런 적음.
  - 실증적 우위: TR(BERT/GPT)에서 성능 ↑.
* 단점: 
  - 계산 비용 (erf/tanh), but GPU 최적화로 무시.

### 5.4 사용 사례

* Transformer 모델 (MLP 부분).
* 현대 대형 모델 기본 (GELU/SiLU 등).

---

## 6. 숫자로 따라가는 미니 예제 (손으로 풀어보기)

**설정**: 단일 입력 z= [ -2, -1, 0, 1, 2 ], 각 활성화 적용 후 미분 계산.

### 6.1 Sigmoid

σ(z): [0.119, 0.269, 0.5, 0.731, 0.881]

σ'(z): [0.105, 0.197, 0.25, 0.197, 0.105]

### 6.2 Tanh

tanh(z): [-0.964, -0.762, 0, 0.762, 0.964]

tanh'(z): [0.071, 0.420, 1, 0.420, 0.071]

### 6.3 ReLU

ReLU(z): [0, 0, 0, 1, 2]

ReLU'(z): [0, 0, 0/1, 1, 1] (0에서 0 가정)

### 6.4 GELU (근사)

GELU(z) ≈ [-0.046, -0.159, 0, 0.841, 1.954]

GELU'(z) ≈ [0.023, 0.158, 0.5, 0.842, 0.977]

> 직관: ReLU는 음수 0, GELU는 약간 누출. 미분에서 Sigmoid/Tanh는 양끝 0, ReLU/GELU는 양수 1에 가까움.

### 6.5 역전파 시뮬 (간단 손실 L = (y - target)^2, target=1)

$y = f(z), dL/dy = 2(y-1), dL/dz = dL/dy * f'(z)$

$z=2$ 예: $ReLU y=2, dL/dy=2(2-1)=2, dL/dz=2*1=2$

Sigmoid y=0.881, $dL/dy=2(0.881-1)=-0.238$, $dL/dz=-0.238*0.105≈-0.025$ (작음)

> 소실 예시: Sigmoid에서 기울기 약화.

---

## 7. 그래프 비교와 시각화 팁

* Sigmoid/Tanh: S자, 대칭/비대칭.
* ReLU: 꺾인 직선, GELU: 부드러운 ReLU.
* 시각화: Matplotlib로 플롯 → 범위 [-5,5], f와 f' 함께.

직관: 깊은 층에서 f' 연곱(체인룰)이 1에 가까워야 기울기 유지.

---

## 8. 기울기 소실/폭주 — 문제와 해결

### 8.1 소실 (Vanishing)

Sigmoid/Tanh: |z|>5 시 f'≈0 → 체인룰로 초기 층 기울기 0.

해결: ReLU/GELU, 초기화(He/Xavier), 정규화(BN/LN).

### 8.2 폭주 (Exploding)

큰 초기화나 불안정 → 기울기 ∞.

해결: Gradient Clipping, 적절 학습률.

### 8.3 죽은 ReLU

학습 중 뉴런이 항상 음수 → 0 출력.

해결: Leaky ReLU (αz for z<0), PReLU(α 학습), ELU.

---

## 9. 적분의 관점 — 확률과 스무딩

적분(2편)은 누적. $GELU = z Φ(z)$에서 $Φ$는 적분(누적 확률).

* 직관: GELU는 z를 "확률적으로 통과" (dropout-like).
* 규제: 활성화의 누적 효과로 과적합 완화.
* 기댓값: $E[f(z)]$로 모델 안정성 해석.

---

## 10. 초기화와 활성화의 궁합

* Xavier (Glorot): $Sigmoid/Tanh$용, 분산 $1/fan_avg$.
* He: $ReLU/GELU$용, 분산 $2/fan_in$ (음수 절반 고려).

직관: 층 통과 시 분산 유지 → 소실/폭주 방지.

---

## 11. 활성화를 코드로 확인해보기 (NumPy 의사코드)

```python
import numpy as np
from scipy.special import erf  # GELU용

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def gelu(z):
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def dgelu(z):
    phi = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * z**2)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    return Phi + z * phi

# 테스트
z = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", sigmoid(z), dsigmoid(z))
print("Tanh:", tanh(z), dtanh(z))
print("ReLU:", relu(z), drelu(z))
print("GELU:", gelu(z), dgelu(z))
```

> 이 코드로 각 함수의 출력/미분을 숫자로 확인. 실전 프레임워크는 내장.

---

## 12. 자주 헷갈리는 포인트 정리

* **출력 범위 영향**: Sigmoid (0,1) → 게이트, ReLU [0,∞) → 양수 특징.
* **미분 at 0**: ReLU 불연속 but 실전 OK.
* **수치 안정**: exp 오버플로 → 클리핑 or 근사.
* **변형 선택**: 태스크별 (이미지: ReLU, NLP: GELU).

---

## 13. 활성화 쌓기와 네트워크 동역학

다층: h = f(W x + b), y = f(V h + c)

역전파: dL/dh = dL/dy * f'(V h + c) * V, then dL/dx via f'(W x + b).

직관: f'가 체인에서 곱셈 → 안정적 f' 필요.

---

## 14. 실무 감각: 선택과 튜닝

* **선택 기준**: CNN → ReLU, RNN → Tanh, TR → GELU.
* **실험**: Ablation study (활성화 바꿔 성능 비교).
* **디버깅**: 히스토그램으로 활성화 분포 모니터 → 소실 시 변경.
* **트렌드**: Swish/SiLU (z sigmoid(z)), Mish 등 신흥.

---

## 15. 활성화와 TR(트랜스포머)의 연결고리 미리보기

* TR MLP: 두 Dense 사이 GELU (희소 + 매끄러움).
* 어텐션: 소프트맥스 (Sigmoid-like).
* 왜 GELU? 대형 모델에서 미세한 누적 효과로 성능 ↑.

---

## 이번 편 요약(한 장으로)

* **활성화 = 비선형 부여**: 선형 한계 극복.
* **Sigmoid**: σ(z)=1/(1+e^{-z}), 범위 (0,1), 소실 문제.
* **Tanh**: tanh(z), 범위 (-1,1), 중심 0 but 소실.
* **ReLU**: max(0,z), 빠름 but 죽은 뉴런.
* **GELU**: z Φ(z), 매끄럽고 TR 표준.
* **미분**: 체인룰 전달 핵심, 소실/폭주 주의.
* **적분**: GELU의 확률 누적 직관.
* 실전: 초기화·정규화와 함께, 태스크별 선택.

---

## 결론

이번 편에서는 **활성화 함수**를 Sigmoid부터 GELU까지 비교하며 왜 각자가 언제 유리한지 해부하였다. 이 감각이 생기면, 네트워크 설계가 훨씬 자유로워진다.

다음 4편에서는 정규화(Normalization)를 다룬다. BatchNorm/LayerNorm/GroupNorm을 **수식·직관·TR 적용**으로 설명하고, 왜 대형 모델에서 필수인지 알아보자.