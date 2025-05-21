---
layout: single
title:  "[논문 리뷰] Quartet: Native FP4 Training Can Be Optimal for Large Language Models"
categories: "AI"
tag: "review"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 논문 리뷰
시대의 흐름은 진짜 엄청나게 빠르다  
DeepGEMM으로 FP8을 효율적으로 사용한게 불과 5개월 전인데,  
벌써 FP4르 ㄹ활용해서 대규모 언어모델을 학습시키는 논문이 나왔다.  
훈련이 점점 더 많은 컴퓨터 자원을 잡아먹고 있는 요즘, 이 논문은 아주 적은 비트로 모델을 훈련시키는 방법을 제안한다.  
초저정밀도 포맷을 사용해서 훈련 속도를 높이고 메모리도 아끼는 방법이며, **Quartet**라는 알고리즘을 통해 정밀도를 챙긴다.  

논문 링크: [https://arxiv.org/abs/2505.14669](https://arxiv.org/abs/2505.14669)  

이 논문은 NVIDIA의 최신 Blackwell GPU에서 지원하는 FP4 포맷을 활용해서, 대규모 언어 모델을 처음부터 끝까지 FP4로 훈련시키는 방법이다.  
이전에는 FP4로 훈련하면 정확도가 많이 떨어지거나, 중간에 FP8이나 FP16 같은 고정밀도 포맷으로 돌아가야 했는데, Quartet은 이 문제를 깔끔하게 해결하였다.  
초심자도 이해할 수 있게 쉽게 풀어서 설명해 보겠다.  


## 대규모 언어 모델과 자원 문제
이미 대규모 모델 만들려면, 많은 학습량으로 인해 정신나간 양의 GPU가 소모되는 것을 우리 모두는 알고 있다.  

이 문제를 해결하려고 연구자들은 계산을 더 적은 비트로 하는 **저정밀도 훈련**을 연구했지만, 결국 FP8만으로 학습하지 못하고 중간에 FP816이나 FP 32를 섞는 방식으로 문제를 해결해 왔다.  
예를 들어, FP16(16비트)나 FP8(8비트) 같은 포맷은 이미 많이 쓰이고 있다.  
하지만, 뇌절되 계속되면 예술이라고, 이 논문은 한 단계 더 나아가서 **FP4(4비트)**를 사용해서 더 적은 비트수, 더 빠른 계산 속도더 적은 에너지 소모를 보여준다.  
하지만, 당연히 정확도는 엄청나게 떨어진다.  

이 문제를 Quartet알고리즘을 통해서 해결했다.  
어떻게? FP4로 모든 주요 계산(예: 행렬 곱셈)을 처리하면서도, 기존 FP8이나 FP16과 비슷한 정확도를 유지하는 방법을 찾아냈다.  
게다가 NVIDIA의 Blackwell GPU에 최적화된 구현까지 제공해서 속도도 엄청 빠르다.  


## Quartet의 핵심 아이디어
Quartet은 FP4로 대규모 언어 모델을 훈련시키는 데 필요한 몇 가지 똑똑한 기술을 조합한 거다.  

1. **스케일링 법칙으로 최적화 찾기**  
   이 논문은 **스케일링 법칙(scaling law)**라는 개념을 도입해서, FP4 훈련이 언제 최적일지 분석했다.  
   스케일링 법칙은 모델 크기(파라미터 수)와 데이터 양에 따라 손실(loss)이 어떻게 변하는지를 예측하는 방법이다.  
   여기서 Quartet은 두 가지 효율성을 측정한다다:
   - **파라미터 효율성(eff_N)**: 모델이 얼마나 효율적으로 파라미터를 활용하는지.  
   - **데이터 효율성(eff_D)**: 주어진 데이터로 얼마나 잘 학습하는지.    
   이 두 가지를 분석해서, FP4가 FP8보다 나은 상황(즉, 속도는 빠르면서 정확도 손실은 거의 없는 경우)을 찾아냈다.  
   이건 마치 최소의 자원으로 낼 수 있는 최적의 효율을 뽑는 방식인 거다.  

2. **전방 패스에서 오류 최소화**  
   전방 패스(forward pass)는 모델이 입력을 받아 출력을 내는 과정이다.  
   여기서 FP4를 사용하면 값이 뭉개질 위험이 있는데, Quartet은 **QuEST**라는 기술을 사용하여 이를 해결한다.  
   QuEST는 **Hadamard 변환**이라는 수학적 트릭을 써서 값을 변환한 뒤, 평균 제곱 오차(MSE)를 최소화하는 방식으로 값을 양자화 한다.  
   이렇게 하면 FP4로도 정확한 계산이 가능해 진다.  
   논문에 따르면, QuEST는 다른 방법들보다 파라미터 효율성이 더 높아서 모델 성능을 잘 유지해 준다고 한다.  

3. **역방향 패스에서 편향 줄이기**  
   역방향 패스(backward pass)는 모델이 오류를 줄이기 위해 가중치를 조정하는 과정이다.  
   여기서도 FP4를 쓰면 gradient(가중치 업데이트 방향을 알려주는 값)에 편향(bias)이 생길 수 있다.  
   Quartet은 **Stochastic Rounding(확률적 반올림)**을 사용해서 이 편향을 줄였다.  
   이 방법은 값을 반올림할 때 무작위로 위나 아래로 올리거나 내려서, 장기적으로 편향이 쌓이지 않게 해주는 거다.  
   논문에 따르면, 이 방식이 긴 훈련 과정에서 특히 더 안정적인 성능을 보여주었다.  

4. **Blackwell GPU에 최적화된 구현**  
   Quartet의 진짜 강점은 NVIDIA의 Blackwell GPU에 맞춘 고효율 CUDA 커널이다.  
   FP4 포맷(MXFP4)을 사용해서 행렬 곱셈을 초고속으로 처리하고, Hadamard 변환 같은 추가 계산도 효율적으로 한다.  
   이 구현 덕분에 Quartet은 FP8보다 **최대 2배 빠른 속도**를 내면서도 비슷한 정확도를 유지할 수 있었다.  
   예를 들어, Blackwell의 B200 GPU에서는 FP4 계산이 FP8보다 에너지를 절반 가까이 줄여준다.  

## ## Quartet
논문 내용을 토대로 직접 레이어를 구현해 보았다.  
오픈소스라도 했지만, 찾는데 실패했다.  

```python
import torch
import numpy as np
from scipy.linalg import hadamard
import transformer_engine.pytorch as te
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.cuda.amp import autocast

# FP4 양자화 클래스
class FP4Quantizer:
    def __init__(self, exponent_bits=2, mantissa_bits=1):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.max_value = 6.0
        self.levels = 2 ** (exponent_bits + mantissa_bits)
        self.lookup_table = self._create_lookup_table()

    def _create_lookup_table(self):
        values = []
        for sign in [-1, 1]:
            for e in range(2 ** self.exponent_bits):
                for m in range(2 ** self.mantissa_bits):
                    value = sign * (1 + m / 2 ** self.mantissa_bits) * 2 ** (e - 1)
                    if abs(value) <= self.max_value:
                        values.append(value)
        return torch.tensor(values, dtype=torch.float32).cuda()

    def quantize(self, tensor):
        n = tensor.shape[-1]
        if n % 2 != 0:
            n = 2 ** int(np.ceil(np.log2(n)))
            tensor = torch.nn.functional.pad(tensor, (0, n - tensor.shape[-1]))
        H = torch.tensor(hadamard(n), dtype=torch.float32).cuda()
        transformed = torch.matmul(tensor, H)
        quantized = torch.zeros_like(transformed)
        for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                value = transformed[i, j]
                closest = self.lookup_table[torch.argmin(torch.abs(self.lookup_table - value))]
                quantized[i, j] = closest
        return torch.matmul(quantized, H.t())[:, :tensor.shape[-1]]

# 확률적 반올림
def stochastic_round(tensor, quantizer):
    quantized = quantizer.quantize(tensor)
    floor = torch.floor(tensor / quantizer.max_value * quantizer.levels) / quantizer.levels * quantizer.max_value
    ceil = torch.ceil(tensor / quantizer.max_value * quantizer.levels) / quantizer.levels * quantizer.max_value
    prob = (tensor - floor) / (ceil - floor + 1e-10)
    mask = torch.rand_like(tensor) < prob
    return torch.where(mask, ceil, floor)

# 스케일링 법칙
def scaling_law_loss(model_size, data_size, loss):
    eff_N = model_size / loss
    eff_D = data_size / loss
    return eff_N, eff_D

# FP4 선형 레이어
class FP4Linear(te.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, precision="fp4")
        self.quantizer = FP4Quantizer()

    def forward(self, input):
        with autocast(dtype=torch.float4_e2m1):
            weight = self.quantizer.quantize(self.weight)
            output = torch.matmul(input, weight.t())
            if self.bias is not None:
                output += self.bias
            return output

# 모델 및 데이터 준비
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b").cuda()
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
dataset = load_dataset("c4", split="train")
dataloader = DataLoader(dataset, batch_size=8)

# FP4 선형 레이어로 교체
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        setattr(model, name, FP4Linear(module.in_features, module.out_features))

# 훈련 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
quantizer = FP4Quantizer()

# 훈련 루프
for epoch in range(3):
    for batch in dataloader:
        inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True).to("cuda")
        outputs = model(**inputs)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad = stochastic_round(param.grad, quantizer)
        optimizer.step()
        eff_N, eff_D = scaling_law_loss(7e9, len(dataset), loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item()}, eff_N: {eff_N}, eff_D: {eff_D}")

```  
실행해 보진 않았지만,  
일단 논문에 나온대로 구현해 보긴했다.  


## Quartet의 장점
Quartet이 왜 개쩌는지 정리하자면,  

1. **속도와 효율성**  
   Quartet은 FP4를 사용해서 계산 속도를 최대 2배까지 끌어올렸다.  
   특히 Blackwell GPU에서는 행렬 곱셈이 FP8보다 훨씬 빠르고, 에너지 소비도 절반 수준으로 줄었다.  
   이건 훈련 비용을 엄청 아낄 수 있다.  

2. **정확도 유지**  
   FP4로 훈련하면 정확도가 떨어질 거라 생각하기 쉽지만, Quartet은 Llama 계열 모델을 C4 데이터셋으로 훈련시켰을 때 FP8과 비슷한 성능을 냈다.  
   심지어 70억 파라미터 같은 대규모 모델에서도 정확도 손실이 거의 없었다고 한다.  

3. **스케일링 법칙 분석**  
   Quartet은 스케일링 법칙을 통해 언제 FP4가 최적인지 알려준다.  
   예를 들어, 데이터가 많거나 모델 크기가 클 때 FP4가 더 효율적이라는 걸 수학적으로 증명했다.  


## 실제로 어땠을까?
논문에서는 Quartet을 Llama 계열 모델(30M부터 7B까지)에 적용해서 C4 데이터셋으로 테스트한 내용이 있다.  
결과는 FP4로 훈련한 모델이 FP8과 거의 동일한 정확도를 내면서, 속도는 최대 2배 빨랐다.   
특히, 긴 훈련(데이터-파라미터 비율이 400 이상)에서는 Quartet의 Stochastic Rounding이 안정적인 성능을 보여주었다.    

또, Quartet은 기존 방법들(Switchback, Jetfire, HALO 등)보다 더 나은 정확도와 안정성을 보여주었다.   
예를 들어, 다른 방법들은 FP4로 훈련할 때 1~2% 정확도 손실이 있었는데, Quartet은 이 손실을 거의 없애주었다.  


## 결론
Quartet은 FP4라는 초저정밀도로 대규모 언어 모델을 훈련시킬 수 있는 길을 열어줬다.  
Blackwell GPU에 최적화된 구현 덕분에 속도도 빠르다.    

무엇보다, 이 논문은 FP4 훈련이 단순히 "가능하다"를 넘어 "최적"일 수 있다는 걸 보여줬다.  
스케일링 법칙을 통해 언제 FP4가 FP8보다 나은지 명확히 알려줬고,  
오픈소스로 공개된 CUDA 커널은 누구나 활용할 수 있게 해주었다.  
앞으로 AI 훈련 비용을 획기적으로 줄이는 데 큰 역할을 할 것이다.  
하지만, 정작 나는 오픈소스를 찾지 못했기 때문에, 이게 진실인진 나도 모르겠다.  

하지만, 논문의 데이터를 보면, FP8 과의 비교 뿐이다.  
기존의 FP8 역시 이것 저것 덕지덕지 붙여서 사용했던 만큼,  
FP4를 활용할 때에도 Quartet 말고도 더 많은 알고리즘, 예를 들어 DeepGEMM에서의 FP8 대신 FP4와 Quartet을 활용하는 방식으로 사용하는 것이 합리적으로 보인다.  
하지만, 이역시 직접 해 보고 수지타산을 비교해 본 것이 아니기 때문에,  
확정적으로 말할 순 없다.  
암튼, FP4를 활용하는 신기한 논문중 하나 였다.  