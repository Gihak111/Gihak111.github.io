---
layout: single
title:  "작은 크기로 높은 성능을 내기 위한 순차적 몬테카를로 기법"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 제한된 자원과 AI의 발전속도 저하

최근 AI 연구에서 놀라운 사실 중 하나는, 거대한 매개변수를 가진 대형 모델이 항상 최고의 성능을 보장하지 않는다는 것이다.  
특히, **순차적 몬테카를로(Sequential Monte Carlo, SMC)** 기법을 활용하면 **더 작은 모델이 더 높은 추론 성능**을 발휘할 수 있다는 연구 결과들이 속속 등장하고 있다.  
이번 포스트에서는 SMC가 무엇인지, 왜 효과적인지, 그리고 어떻게 소형 모델의 경쟁력을 높여주는지 수식과 함께 살펴보는 것으로 하자.  

---

## 1. Sequential Monte Carlo란?

SMC는 **시간에 따라 변화하는 확률 분포를 샘플링 기반으로 추적**하는 기법이이다.  
**Bayesian 추론**, **Hidden Markov Models**, **비선형 상태 공간 모델** 등에 활용된다.  
딥러닝 분야에서의 응용은 비교적 최근에 활발해졌으며, Transformer나 diffusion 모델 등 대규모 아키텍처의 inference 성능을 **작고 효율적인 모델**로 재현하는 데 핵심 도구가 된다.  

---

## 2. 핵심 개념: 샘플 기반 확률 추론

SMC는 다음과 같은 순차적인 과정을 따른다:

1. **초기화**: \( \{x_0^{(i)}\}_{i=1}^N \sim p(x_0) \) — 초기 샘플을 분포로부터 생성  
2. **예측**: \( x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}) \) — 이전 상태로부터 다음 상태 예측  
3. **가중치 갱신**: \( w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t \mid x_t^{(i)}) \) — 관측값에 기반한 중요도 계산  
4. **리샘플링**: 중요도가 낮은 샘플을 제거하고 높은 것 중심으로 재구성  

최종적으로 모델은 다음과 같은 근사치를 얻게 된다:  

$$
p(x_{1:t} \mid y_{1:t}) \approx \sum_{i=1}^N w_t^{(i)} \delta_{x_{1:t}^{(i)}}
$$

---

## 3. 딥러닝에서의 적용

최근 논문에서는 SMC를 Transformer와 같은 디코더 기반 모델에 결합하여 inference의 정확도와 다양성을 동시에 향상시키는 방법을 제안한다.  

### (1) SMC Transformer
기존 Transformer는 학습된 확률 분포에서 **한 번의 greedy sampling 또는 beam search**로 출력을 생성한다.  
하지만 이 방식은 단일 모드에 치우치는 경향이 있다.  
SMC Transformer는 다수의 샘플(trajectory)을 추론에 사용하고, 가중치를 기반으로 후처리한다:  

```text
입력 → N개의 샘플 경로 생성 → 중요도 기반 재샘플링 → 다중 모드 추론 결과
```

### (2) 효과

* 작은 모델에서도 **다양한 예측 분포**를 유지 가능  
* **추론 불확실성에 민감**하게 반응  
* Sampling 기반이므로 더 유연한 표현력 확보  

---

## 4. 왜 대형 모델보다 유리한가?

| 기준      | 대형 모델 (Greedy)     | 소형 + SMC          |
| ------- | ------------------ | ----------------- |
| 추론 다양성  | 낮음 (mode collapse) | 높음 (multi-sample) |
| 계산 자원   | 매우 큼               | 상대적으로 작음          |
| 불확실성 추적 | 거의 불가능             | 명시적으로 추적          |
| 구조적 복잡도 | 높음                 | 상대적으로 단순          |

SMC는 **추론 과정에서 불확실성을 명시적으로 표현하고** 이를 기반으로 더 다양한 출력 후보를 관리하기 때문에, 작은 모델이라도 상황에 따라 더 정밀한 결과를 도출할 수 있다.  

---

## 5. 수식 정리

기본 SMC 알고리즘 수식 요약:  

1. **Transition Sampling**:  
   $x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})$  

2. **Weight Update**:  
   $w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t \mid x_t^{(i)})$  

3. **Normalization**:  
   $\tilde{w}_t^{(i)} = \frac{w_t^{(i)}}{\sum_{j=1}^N w_t^{(j)}}$  

4. **Resampling (if needed)**:  
   Resample $x_t^{(i)}$ according to $\tilde{w}_t^{(i)}$  

---

## 6. 실제 예시: Language Modeling  

논문 "Sequential Monte Carlo for Probabilistic Transformers" (Maddison et al., 2023)에서는 기존 GPT 계열의 greedy decoding보다 SMC 방식이  
더 높은 BLEU / ROUGE 점수를 획득하고,  
의미론적으로 일관된 문장을 생성하며,  
적은 파라미터 수로도 SOTA 모델과 비슷한 성능을 보였음을 보고했습니다.  

---

## 7. 결론  
아무튼 경제적인 측면에서 바라보면, 순차적 몬테카를로 기법은 쓸만하다.  
특히, 물량, 돈으로써으ㅢ 한계점에 다다른 최근 같은 경우,  
이런 다양한 기법들이 AI 성능을 급진적으로 올리는 데 혁신적인 열쇠가 될 수 도 있다.  