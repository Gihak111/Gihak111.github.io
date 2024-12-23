---
layout: single
title:  "딥러닝 시 나오는 로그와 마주치는 오류들 with ONNX"
categories: "ERROR"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 딥러닝시의 시리얼 모니터  

딥러닝 시 시리얼 모니터에 많은 로그들이 나온다.  
일반저긍로 버젼오류, 또는 자신이 설정한 디버그 로그들이 나온다.  
또한, 텐서플로우를 사용하면 꼭 나오는 것들이 있는데, 이는 다음과 같다.  

### 로그 데이터 항목 설명

1. `loss`:  
   - 의미: 현재 학습 단계에서의 손실 값. 손실 함수의 결과로, 모델의 예측과 실제 값의 차이를 측정한다.  
   - 해석:  
     - 값이 크면 모델이 데이터를 잘 학습하지 못하고 있음을 의미한다.  
     - 값이 작아지면 모델이 점점 더 정확히 데이터를 학습하고 있다는 신호이다.  

2. `grad_norm`:    
   - 의미: 모델 매개변수(가중치) 업데이트를 위한 그래디언트(Gradient)의 노름(Norm) 크기.  
   - 해석:  
     - 매우 큰 값(예: `1495043.625`)은 학습 불안정성을 나타낼 수 있다.  
     - 값이 작으면 그래디언트가 안정적인 상태를 의미한다.  
     - 그러나 값이 0에 가까우면 학습이 멈추는 현상(Gradient Vanishing)이 발생할 가능성이 있다.  

3. `learning_rate`:  
   - 의미: 현재 사용 중인 학습률(Learning Rate). 학습률은 모델 매개변수를 업데이트하는 속도를 조정한다.  
   - 해석:  
     - 너무 크면 학습이 불안정해질 수 있고, 너무 작으면 학습이 느려진다.  
     - 학습률이 단계적으로 증가(`1.5e-05` → `3.4e-05`)하며 모델이 안정적으로 학습하도록 설할 수 있다.  

4. `epoch`:  
   - 의미: 전체 데이터셋을 한 번 학습한 정도를 나타내는 지표.  
   - 해석:  
     - `0.14`, `0.15`처럼 소수점으로 표시된 것은, 전체 에포크(1.0) 중 몇 %만큼 진행되었는지를 나타낸다.  
     - 예: `epoch 0.14`는 14% 진행된 상태를 의미한다.  

---

### 로그 데이터 해석

- 초기 단계 (`epoch 0.14 ~ 0.16`):  
  - `loss` 값이 매우 크고(`18305.9719`), `grad_norm` 값도 급격히 변동(`1495043.625`)하고 있다.  
  - 이는 모델이 아직 학습 초기 단계이며, 학습이 불안정하다는 것을 의미한다.  

- 중간 단계 (`epoch 0.17 ~ 0.25`):  
  - `loss` 값이 급격히 감소하며 (`114.0165` → `0.186`), 모델이 데이터를 점점 더 잘 학습하고 있음을 보여준다.  
  - `grad_norm` 값도 점차 안정화되고 있다.  

- 후기 단계 (`epoch 0.26 ~ 0.32`):  
  - `loss` 값이 거의 0에 가까워지며 (`0.0008`, `0.0`), 모델이 학습을 거의 완료했음을 나타낸다.  
  - `grad_norm` 값도 매우 작아졌다(`0.0020479606464505196`).  
위와 같은 양상을 보이는 것이 일반적이다.  

### 문제점 및 개선  
1. 학습 초기에 불안정한 `grad_norm`:  
   - `grad_norm` 값이 너무 커서 학습이 불안정할 가능성이 크다. 이를 방지하기 위해:  
     - 학습률 감소: 학습 초기 학습률을 더 작게 설정.  
     - 그래디언트 클리핑(Gradient Clipping): `grad_norm` 값이 일정 이상 커지지 않도록 제한.  

2. `loss`의 불규칙적 상승:  
   - 예: `epoch 0.22`에서 `loss = 4.6215`로 갑자기 상승.  
   - 데이터 불균형, 이상치, 또는 모델의 과적합(overfitting) 가능성을 봐야 한다.  

### 요약  
- 일반적으로 잘 만든 AI 로그는 딥러닝 모델이 데이터를 점점 더 잘 학습하고 있다는 과정을 보여준다.  
- 초기에 불안정한 학습 단계가 있었지만, 후반으로 갈수록 `loss`와 `grad_norm`이 안정화되고, 모델이 학습을 잘 마무리하는 양상을 띈다.  


## pyyTorch 경고
PyTorch에서 텐서 복사 방식과 관련된 경고가 뜨곤 한다.  
아래의 메시지는 `torch.tensor()`를 사용해 이미 존재하는 텐서를 복사할 때 잘못된 사용 방식을 알려주고 있다.  

### 경고 메시지 의미
```plaintext
To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
```  

#### 핵심 내용:
- `torch.tensor(sourceTensor)` 사용 문제:  
  - `sourceTensor`(기존 텐서)를 `torch.tensor()`에 전달하면 새로운 텐서를 생성한다.  
  - 하지만 이 과정에서 기존 텐서의 그래디언트 추적 정보가 제대로 유지되지 않을 수 있다.  
  - 결과적으로 학습 과정에서 예상치 못한 동작을 유발할 수 있다.  

- 권장 방식:  
  - 기존 텐서를 복사하려면 `clone()` 및 `detach()` 메서드를 사용해야 한다.  
  - 이를 통해 새로운 텐서를 생성하되, 기존 텐서의 그래디언트 추적 상태를 명확히 제어할 수 있다.  

### 올바른 코드 수정 방법

#### 기존 코드 예시 (문제가 발생하는 코드):  
```python
new_tensor = torch.tensor(existing_tensor)
```  

#### 수정된 코드 (권장 방식):  
```python
new_tensor = existing_tensor.clone().detach()
```  

#### 경우에 따라 `requires_grad_(True)` 추가:  
- 만약 새로운 텐서에서 그래디언트 추적을 활성화해야 한다면:  
```python
new_tensor = existing_tensor.clone().detach().requires_grad_(True)
```  

---

### 왜 이 경고가 중요한가?
1. 그래디언트 추적:
   - 딥러닝 학습 과정에서는 자동 미분을 위해 텐서의 그래디언트를 추적한다.  
   - `torch.tensor()`를 사용하면 원래 텐서의 그래디언트 추적 상태가 무시될 수 있다.  

2. 불필요한 메모리 사용:  
   - `torch.tensor()`는 텐서를 복사하며 새로운 메모리를 할당한다.  
     이는 효율적이지 않을 수 있다.  

3. 학습 과정에서 오류 발생 가능성:  
   - 잘못된 텐서 복사는 학습 중 예기치 않은 동작(예: 그래디언트 계산 오류)을 유발할 수 있다.  

### 요약
이 경고는 텐서를 복사할 때 `torch.tensor()` 대신 `clone().detach()`를 사용하라는 의미이다.  
코드를 수정하여 권장 방식을 따르도록 하면 경고를 제거하고 모델 학습 과정에서 발생할 수 있는 문제를 방지할 수 있다.  

## ONNX 형식 변환시 오류

이 메시지는 PyTorch 모델을 ONNX 형식으로 변환할 때 발생한 경고와 상태 로그를 보여준다. 아래에서 각 메시지의 의미를 자세히 설명하겠다.  

### 로그 내용 분석 및 의미  

#### 1. `TracerWarning: torch.tensor results are registered as constants in the trace.`  
- 이 경고는 `torch.jit.trace()` 또는 ONNX 변환 과정에서 `torch.tensor()`가 상수로 처리되었다는 의미이다.  
- PyTorch에서 ONNX로 변환 시, `torch.tensor(1000)`와 같이 동적으로 생성된 텐서를 상수 값으로 간주하여 ONNX 그래프에 삽입한다.  
- 이 경고는 주로 모델에서 반복적으로 동일한 값을 생성하는 경우에는 무시해도 된다.  
  
해결 방법:  
- 만약 이 값이 매번 다르게 생성되어야 한다면, 이를 입력으로 처리하거나 동적 연산으로 작성해야 한다.  
- 예를 들어, `torch.tensor()` 대신 모델 입력으로 처리:  
  ```python
  dynamic_value = torch.tensor(input_value)  # 모델 외부에서 생성하여 전달
  ```  

#### 2. `UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.`  
- 이 경고는 ONNX 변환 중 Constant Folding 최적화가 적용되지 않았다는 것을 알려준다.  
- Constant Folding은 변환 과정에서 상수 값을 미리 계산하여 그래프를 최적화하는 기법이다.  
- 여기서 Slice 연산(`onnx::Slice`)에서 `steps=1` 이외의 값이 사용되었기 때문에 최적화가 제한되었다.  
- opset(ONNX의 연산 지원 버전) 10 이상에서 이러한 조건을 만족하지 못하면 경고가 발생한다.  

해결 방법:  
- 이 경고는 최적화의 한계를 알리는 것이므로 무시해도 모델의 동작에는 문제가 없다.  
- 하지만 필요하면 모델 코드에서 Slice 연산의 `steps` 값을 검토하고, `steps=1`로 조정하거나 연산을 단순화할 수 있다.  

#### 3. 모델이 성공적으로 저장된 로그:  
```plaintext
모델이 ONNX 형식으로 저장되었습니다: C:\mobile_bert\AI\mobilebert_scam_classifier.onnx
```  
- 최종적으로 경고에도 불구하고 모델이 성공적으로 **ONNX 형식**으로 변환되었음을 알리는 메시지 이다.  
- 변환된 ONNX 모델은 `C:\mobile_bert\AI\mobilebert_scam_classifier.onnx`에 저장되었다.  

---  

### 요약 및 다음 단계
1. 경고 무시 여부:  
   - 경고는 대부분 변환 및 실행에 영향을 주지 않으므로 무시해도 괜찮다.
   - 하지만 `torch.tensor()` 관련 경고는 동적으로 생성된 값이 필요한 경우 코드 수정이 필요하다.  

2. ONNX 모델 테스트:  
   - 변환된 ONNX 모델을 ONNX Runtime 또는 다른 지원 라이브러리로 테스트하여 결과를 검증하자.  
   - 예를 들어, ONNX Runtime으로 추론 실행:  
     ```python
     import onnxruntime as ort

     session = ort.InferenceSession("C:\\mobile_bert\\AI\\mobilebert_scam_classifier.onnx")
     input_name = session.get_inputs()[0].name
     result = session.run(None, {input_name: input_tensor})
     print(result)
     ```  

3. Slice 연산 최적화 필요 여부:  
   - 성능 최적화가 중요한 경우, Slice 연산을 간소화하거나 Constant Folding 문제를 해결해야 한다.  

모델 자체에는 문제가 없으므로 저장된 ONNX 파일을 활용해 작업을 계속 진행해도 된다.  

위와 같은 과정을 통해 발생중인 오류를 해결하였다