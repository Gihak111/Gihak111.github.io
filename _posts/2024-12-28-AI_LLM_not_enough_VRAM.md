---
layout: single
title:  "VRAM이 부족한 경우 해결법"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# VRAM 부족

가정집에서 고출력 LLM 같은 모델을 딥러닝 하려면 VRAM 용량이 부족할 때가 있다.  
3070같은 고사양 글카의 VRAM도 8GB이며, 16GB를 가지도고 끝내 딥러닝을 마치지 못하는 경우가 있다.  
이럴 경우, 악으로 깡으로 딥러닝을 돌리는데, 이런 상황에서 할 수 있는 꼼수들이 있다.  


### 주요 변경 사항 및 최적화

1. BitsAndBytes 라이브러리를 활용한 양자화  
   - 모델의 일부를 8-bit 양자화하여 VRAM 사용량을 크게 줄인다.  
   - `BitsAndBytesConfig(load_in_8bit=True)` 옵션을 추가로 사용한다.  

   설치:  
   ```bash
   pip install bitsandbytes
   ```  

2. 메모리 최적화를 위한 `torch_dtype` 조정  
   - VRAM 사용량이 많은 경우 `torch.float16` 대신 `torch.bfloat16`으로 설정 가능하다.   
   - `torch.bfloat16`은 성능 손실 없이 메모리를 더욱 절약할 수 있다.  

3. `accelerate`를 통한 메모리 관리:  
   - `transformers` 라이브러리에서 내부적으로 사용하는 `accelerate`로 대규모 모델의 메모리 효율성을 자동으로 관리한다.  

   설치:  
   ```bash
   pip install accelerate
   ```  

### 추가 주의 사항

- VRAM 확인:  
  `torch.cuda.get_device_properties(0).total_memory`를 사용해 GPU의 VRAM 크기를 확인하자.  
  
- 모델 크기 문제 지속 시:  
  - 양자화를 8-bit에서 4-bit로 조정(`load_in_4bit=True`)  
  - GPU와 CPU 간의 메모리 스왑을 효율적으로 관리하기 위해 `offload_folder`를 추가 설정:  
    ```python
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        offload_folder="./offload"  # CPU와 디스크 스왑 폴더
    )
    ```  

위 설정을 통해서 문제를 해결할 수 있다.  
이걸로 해결 못하면, VRAM을 추가하거나 경량 모델을 사용하는 것이 최선이다.  



또한, 딥러닝 중, 이런 오류가 나오면서 뻗을 수 있다.  

`ValueError: Some modules are dispatched on the CPU or the disk` 오류는 여전히 모델의 일부 모듈이 GPU에 적재되지 못하고 CPU나 디스크로 오프로드되었음을 의미한다.  
GPU VRAM이 부족한 경우 또는 `device_map`과 양자화 설정이 잘못된 경우 발생한다.  


### 오류 해결 전략

1. GPU와 CPU 오프로드 설정 강화  
   - `load_in_8bit_fp32_cpu_offload=True`를 설정하고 `device_map`을 명시적으로 설정한다.  

2. 디스크 오프로드 추가  
   - 모델이 디스크를 사용하도록 `offload_folder`를 설정해 더 큰 모델도 실행 가능하게 한다.  

3. VRAM 사용 최적화  
   - 일부 모듈만 GPU에 올리고 나머지는 CPU/디스크에 두는 방식을 활용한다.  

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 모델 경로
model_name = "your/model"  #여기선 Qwen 모델 사용

# 양자화 설정
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                  # 8-bit 양자화
    load_in_8bit_fp32_cpu_offload=True  # CPU로 FP32 오프로드
)

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                  # 디바이스 자동 매핑
    quantization_config=quant_config,  # 양자화 설정
    trust_remote_code=True,
    offload_folder="./offload"          # 디스크 오프로드 폴더 설정
)

# 입력 데이터가 GPU 또는 CPU에 맞춰지도록 강제 배치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 응답 생성 함수
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # 입력 배치
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        temperature=0.7,  # 창의성 조절
        top_k=50,         # 상위 k 후보 선택
        top_p=0.9,        # 누적 확률 기반 선택
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 대화 시뮬레이션
print("안녕하세요! 대화를 시작해 주세요.")
while True:
    user_input = input(">> ")
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("대화를 종료합니다. 다음에 또 만나요!")
        break
    response = generate_response(f"User: {user_input}\nAI: ")
    print(f"AI : {response}")
```

### 변경 사항 설명

1. 디스크 오프로드 추가  
   - `offload_folder="./offload"` 설정으로 GPU와 CPU에도 적재할 수 없는 경우 디스크로 오프로드.  

2. `device_map` 자동 매핑  
   - `device_map="auto"`로 자동 배치 사용.  
   - 디스크, CPU, GPU를 혼합해 사용하도록 설정.  

3. `load_in_8bit_fp32_cpu_offload=True`  
   - 일부 모듈이 GPU에 적재되지 않아도 CPU에서 FP32로 오프로드 실행.  

4. VRAM 절약  
   - `BitsAndBytesConfig`로 8-bit 양자화를 유지해 VRAM 요구량을 최소화.  


### 디버깅 팁  

1. 모듈별 장치 확인:  
   모델의 각 모듈이 적재된 장치를 출력해 문제가 발생하는 모듈을 확인한다.  
   ```python
   for name, param in model.named_parameters():
       print(f"{name} is on {param.device}")
   ```  

2. GPU 메모리 확인:  
   실행 전후에 VRAM 사용량을 확인한다.  
   ```bash
   nvidia-smi
   ```  

3. 다른 모델 사용:  
   여전히 GPU 메모리가 부족하면 작은 모델(`Qwen-7B`, `GPT-J-6B`)로 테스트 해보자.   

4. 오프로드 폴더 정리:  
   디스크 오프로드 폴더(`./offload`)가 과도하게 커지지 않도록 필요 시 정리한다.  
