---
layout: single
title:  "딥시크 DeepGEMM-main에 대해 알아보기기"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## DeepGEMM
FP8(8-bit 부동소수점) 연산을 활용한 General Matrix Multiplication (GEMM) 연산 최적화 라이브러리이다.  
DeepSeek-V3에서 제안된 기술을 기반으로 하며, 일반적인 GEMM뿐만 아니라 Mix-of-Experts (MoE) 모델을 위한 그룹화된 GEMM도 지원한한다.  

CUDA 기반으로 작성되었으며, 설치 시 사전 컴파일이 필요하지 않고 경량 Just-In-Time (JIT) 컴파일러러를 통해 실행 중에 모든 커널을 빌드하는 방식으로 동작한한다.  

현재 DeepGEMM은 NVIDIA Hopper 아키텍처의 Tensor Core 전용으로 설계되어 있으며, FP8 Tensor Core의 부정확한 누적 문제를 해결하기 위해 CUDA Core 기반의 이중 누적(promotion) 방식을 사용한다.  
CUTLASS 및 CuTe에서 일부 개념을 차용했지만, 복잡한 템플릿 및 수학적 연산에 의존하지 않고 단순하고 가벼운 구조를 유지한한다.


## DeepGEMM의 동작 방식

### 1. FP8 연산 최적화
DeepGEMM은 FP8 형식의 행렬 곱셈을 실행할 때, 기존의 CUDA 라이브러리보다 더 높은 성능을 제공하도록 설계되었다.  
FP8은 메모리 사용량을 줄이고 연산 속도를 높이는 장점이 있지만, 낮은 정밀도로 인해 누적 오류가 발생할 수 있다.  
이를 보완하기 위해 DeepGEMM은 이중 누적(promotion) 기법을 적용하여 정확도를 개선한다.  

이중 누적 기법이란, 간단히 말해 곱셈은 FB8에서 수행하지만, 합산은 더 높은 정밀도를 사진 형식(FB16, FB32)에서 수행하고, 결과를 다시 FB8로 변환하여 저장하는 방식이다.  

이해를 돕기 위해 기존 방식과 비교해서 보자  

#### 1. 기존방식
```python
import torch

# FP8 형식의 두 개의 행렬 곱셈
x = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float16).cuda()
y = torch.tensor([[5.5, 6.5], [7.5, 8.5]], dtype=torch.float16).cuda()

# FP8로 변환 후 행렬 곱셈
x_fp8 = x.to(torch.float8_e5m2)  # FP8 변환
y_fp8 = y.to(torch.float8_e5m2)

result_fp8 = x_fp8 @ y_fp8  # FP8 연산
result_fp8_fp32 = result_fp8.to(torch.float32)  # FP32로 변환하여 확인

print(result_fp8_fp32)

```

##### 특징:  
FP8은 기존 FP16, FP32보다 메모리 사용량이 적고 연산 속도가 빠름
하지만 정밀도가 낮아 연산을 반복하면 오차가 누적됨.  
특히, GEMM(행렬 곱셈)에서 많은 요소가 곱해지고 더해지기 때문에 FP8의 낮은 정밀도로 인해 심각한 누적 오차 발생.  

##### 문제점:
FP8에서 연산한 후 FP32로 변환하면 원래 값과 차이가 큼.  
오차가 누적되면서 학습이나 추론에서 잘못된 결과를 초래할 가능성이 높음.  


#### 2. 이중 누적 방식
```python
import torch

# FP8을 사용하되, 누적 연산을 FP16 또는 FP32에서 수행
x = torch.randn((1024, 1024), dtype=torch.float8_e5m2, device='cuda')
y = torch.randn((1024, 1024), dtype=torch.float8_e5m2, device='cuda')

# 곱셈 연산 후 FP16으로 변환하여 누적
result_fp16 = (x @ y).to(torch.float16)  # 이중 누적 기법 적용

# 다시 FP8로 변환하여 저장
result_fp8 = result_fp16.to(torch.float8_e5m2)
print(result_fp8)

```

##### 특징:  
FP8 연산을 CUDA Core에서 FP16 또는 FP32로 변환하여 계산   
2️곱셈은 FP8에서 수행하지만, 합산(accumulation)은 더 높은 정밀도를 가진 형식에서 수행   
3️최종 결과를 다시 FP8로 변환하여 저장   

##### 해결책:
```x @ y```를 수행할 때 FP8이 아니라 FP16으로 변환 후 합산(accumulation)  
최종 결과를 다시 FP8로 변환하여 저장  
결과적으로 정밀도를 높이면서도 FP8의 속도와 메모리 이점을 활용 가능  

#### CUDA에 이중누적 기법 활용방식을 적용해 보자.  

FP8 연산을 Tensor Core에서 수행  
누적(accumulation)은 CUDA Core에서 FP16 또는 FP32로 처리  
이 방식으로 낮은 정밀도를 보완하면서도 속도를 유지  

흐름
```cpp
__global__ void fp8_gemm_kernel(...) {
    // FP8 입력 행렬을 FP16으로 변환
    half A_fp16 = __half2float(A_fp8);
    half B_fp16 = __half2float(B_fp8);
    
    // FP16에서 곱셈 연산
    float C_fp32 = A_fp16 * B_fp16;
    
    // FP32에서 누적 연산 수행
    atomicAdd(&C, C_fp32);

    // 최종 결과를 FP8로 변환하여 저장
    C_fp8 = __float2half_rn(C_fp32);
}

```
FP8 → FP16 변환 후 연산 수행
FP16 → FP32에서 합산
최종적으로 FP8로 변환하여 저장

따라서, 이중 누적 기법을 적용하면,  
1. FP8의 속도와 메모리 절약 효과를 유지  
2. FP8의 낮은 정밀도로 인한 누적 오차 문제 해결  
3. 기존 FP8 연산보다 훨씬 정확한 결과 제공  
4. NVIDIA Hopper 아키텍처에 최적화된 방식  
5. DeepGEMM이 기존 GEMM 라이브러리보다 우수한 성능을 보이는 핵심 이유  

와 같은 이점을 가져올 수 있다.  
즉, FP8의 속도 + FP16&FP32의 정확도를 모두 잡을 수 있는 핵심 기법이다.  


### 2. JIT 컴파일을 통한 성능 최적화
일반적인 CUDA 연산 라이브러리는 컴파일된 상태로 제공되지만,  
DeepGEMM은 JIT(Just-In-Time) 컴파일을 사용하여 실행 시점에서 가장 적절한 설정을 선택하고 최적의 코드로 변환한다.    
이를 통해 행렬 크기, 블록 크기, 파이프라인 스테이지 개수 등을 실행 환경에 따라 최적화할 수 있다.  
즉, 컴파일 시점이 아닌 실행 시점에 커널을 생성하여 불필요한 연산 낭비를 줄이고 성능을 극대화한한다.  

### 3. Tensor Memory Accelerator(TMA) 활용
NVIDIA Hopper 아키텍처에서 제공하는 TMA(Tensor Memory Accelerator) 기능을 적극 활용한다.  
- TMA Load: LHS, Scaling Factor, RHS 행렬을 고속으로 로드  
- TMA Store: 결과 행렬을 고속으로 저장  
- TMA Multicast: LHS 행렬을 여러 워프(warp)로 효율적으로 분배  
- TMA Prefetching: 사전 로딩을 통해 캐시 활용도를 극대화  

이러한 기법은 연산 병목을 최소화하고 데이터 이동 속도를 향상시킨다.  

### 4. Mix-of-Experts (MoE) 모델 지원
일반적인 Dense GEMM뿐만 아니라, Mix-of-Experts (MoE) 모델에서 사용되는 Grouped GEMM 연산도 지원한다.    
Grouped GEMM은 하나의 큰 행렬 연산을 여러 개의 작은 그룹으로 나누어 수행하는 방식이며, DeepGEMM에서는 M 축을 기준으로 그룹화하여 최적화를 수행한다.  

이를 통해 모델이 전문가 레이어(Experts)별로 서로 다른 크기의 행렬을 처리하는 과정에서 성능을 극대화할 수 있다.  


## DeepGEMM의 경쟁력 (기존 라이브러리 대비 장점)

###  1. CUTLASS 대비 간결한 구조
DeepGEMM은 단 하나의 핵심 커널 함수 (~300줄) 만으로 구성되어 있어, 읽기 쉽고 유지보수가 용이하다.    
반면, CUTLASS는 복잡한 템플릿을 사용하여 코드가 매우 방대하고 학습 곡선이 가파르다.  

### 2. JIT(Just-In-Time) 컴파일 기반 최적화
- 실행 시점에서 최적의 블록 크기, 파이프라인 스테이지 개수 등을 자동 선택  
- GEMM 연산을 완전히 언롤링(Unrolling)하여 컴파일러의 최적화 기회를 극대화  
- 작은 행렬 크기에 대해서도 성능 최적화 가능 (Triton과 유사한 방식)  

### 3. NVIDIA Hopper 아키텍처 최적화
- Tensor Core의 FP8 연산을 CUDA Core 기반 이중 누적(Promotion) 방식으로 보정  
- TMA (Tensor Memory Accelerator) 기능 활용으로 데이터 이동 최적화  
- Hopper 전용 FFMA(병렬 부동소수점 곱-합) 명령어 최적화  

### 4. 경쟁 라이브러리 대비 성능 우위
DeepGEMM은 자체적으로 튜닝된 CUTLASS 3.6 기반 구현과 비교했을 때 1.4~2.7배의 속도 향상을 기록했다.  

| M | N | K | 연산량(TFLOPS) | 메모리 대역폭(GB/s) | 속도 향상 배율 |
|--|--|--|--|--|--|
| 64 | 2112 | 7168 | 206 | 1688 | **2.7x** |
| 128 | 2112 | 7168 | 352 | 1509 | **2.4x** |
| 128 | 4096 | 7168 | 533 | 2221 | **2.0x** |
| 128 | 7168 | 2048 | 510 | 2277 | **1.7x** |


## DeepGEMM의 사용 방법

### 설치 방법
```bash
python setup.py install
```

###  개발 환경 설정
```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
python setup.py develop
python tests/test_jit.py  # JIT 컴파일 테스트
python tests/test_core.py # 모든 GEMM 연산 테스트
```

### 사용 가능한 함수
1. 기본 FP8 GEMM 연산 (Dense GEMM)
   ```python
   deep_gemm.gemm_fp8_fp8_bf16_nt(lhs, rhs)
   ```
   - NT 포맷 (LHS: Non-Transposed, RHS: Transposed)만 지원  
   - LHS 스케일링 팩터는 별도 정렬 필요  

2. Grouped GEMM (MoE 모델 - Contiguous Layout)
   ```python
   deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(...)
   ```
   - 같은 크기의 전문가(Experts) 그룹을 처리할 때 사용  
   - M 축을 기준으로 그룹화  

3. Grouped GEMM (MoE 모델 - Masked Layout)
   ```python
   deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(...)
   ```
   - Inference 단계에서 CUDA 그래프 기반으로 동작  
   - CPU가 각 전문가(Experts)의 입력 토큰 개수를 모르는 경우 사용  

## 결론적으로,  

DeepGEMM은 FP8 기반 GEMM 연산의 성능을 극대화하는 NVIDIA Hopper 최적화 라이브러리 이다.  
CUTLASS 대비 간결한 구조  
JIT 컴파일 기반 실행 시 최적화  
Tensor Memory Accelerator(TMA) 활용  
최신 NVIDIA Hopper 아키텍처에 맞춘 성능 최적화  
경쟁 라이브러리 대비 1.4~2.7배 속도 향상  

즉, DeepGEMM은 Hopper 기반 AI 모델의 연산 속도를 극대화할 수 있는 강력한 솔루션 이다.  


## 이제 코드를 뜯어보자.  
이 라이브러리으 핵심 기능은, 앞서 말한 FB8의 연산을 누적연산을 통해 강화하는 것이다.  
이 내용의 코드는 ```fp8_gemm.cuh``` 안에 들어있다.  
cuh 라는 확장자가 생소할 수 있는데,  
이는 ```cuda Header```파일을 의미한다.  
.h가 c언어의 헤더인 것 처럼, .cuh는 CUDA에서 사용하기 위한 헤더인 것이다.  

특징은,  
- CUDA 커널 함수 및 GPU 실행 관련 코드 포함  
- 템플릿을 활용한 다양한 GPU 환경 대응 가능  
- #pragma once를 사용해 중복 포함 방지  
- GPU에서 동작하는 __global__, __device__ 키워드 사용 가능  
- CUDA 라이브러리 및 GPU 전용 연산자를 포함할 수 있음  

fp8_gemm.cuh 파일은 DeepGEMM의 핵심 FP8 행렬 곱셈 연산을 담당하는 CUDA 커널을 정의한 파일이라 보면 된다.  

### 코드 분석: DeepGEMM fp8_gemm.cuh 연산 코드
주요 헤더부터 전부 뜯어보자.  

#### 1. 헤더하필
```cpp
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

```

- CUTLASS 및 CuTe(CUDA Utility) 라이브러리를 활용하여 Tensor Core 연산을 최적화  
- 행렬 연산(MMA, Matrix Multiply-Accumulate) 관련 유틸리티 포함  
- 스케줄러, TMA(Tensor Memory Accelerator) 관련 최적화 코드 포함  

#### 2. get_num_threads_per_sm()
```cpp
template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

```

블록 내 스레드 수를 설정한다.  
- 각 SM(Stream Multiprocessor)에 할당할 스레드 수를 계산  
- 128개의 수학 연산 전용 스레드를 유지하면서, 추가적인 TMA(Tensor Memory Accelerator) 스레드를 고려  
- 행렬 크기(block_m)에 따라 스레드 그룹 크기를 조정  

#### 3. fp8_gemm_kernel() 
```cpp
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {

```

FP8 행렬 곱셈 CUDA 커널이다.  
이 함수의 역할  
- FP8 행렬 곱셈(GEMM) 연산을 수행하는 CUDA 커널  
- CUDA Hopper 아키텍처(SM90) 전용으로 설계됨  
- Tensor Memory Accelerator(TMA) 사용하여 메모리 로딩 최적화  
- CUTLASS 기반으로 WGMMA(Warp Group Matrix Multiply-Accumulate) 연산 수행  

핵심 내용  
__global__ 키워드 사용 → GPU에서 실행되는 커널 함수  
__launch_bounds__() → 스레드 블록 크기를 최적화하여 설정  
gmem_d, tensor_map_a, tensor_map_b 등 → 전역 메모리에서 읽고 쓰는 행렬 데이터  


#### 4. TMA 기반 메모리 최적화  
```cpp
// Prefetch TMA descriptors at very beginning
if (threadIdx.x == kNumMathThreads) {
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
}
__syncwarp();

```
TMA(Tensor Memory Accelerator)를 사용하여 메모리 접근을 최적화  
CUDA Core가 FP8 연산을 수행하는 동안 데이터 로딩을 비동기적으로 수행  


#### 5. FP8 데이터 변환 및 WGMMA 연산  
```cpp
auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

// WGMMA 실행
#pragma unroll
for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
    auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
    auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
    WGMMA::wgmma(desc_a, desc_b, accum, k);
}

```

- FP8 데이터를 FP16 또는 FP32로 변환 후 연산(이중 누적 기법, Promotion)  
- Tensor Core의 WGMMA 연산 사용하여 고속 행렬 연산 수행  
- CUDA Shared Memory 최적화하여 연산 속도 향상  

#### 6. GEMM 실행을 위한 Gemm 클래스
```cpp
class Gemm {
public:
    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // 실행할 CUDA 커널 선택
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, kGemmType>;

```

- CUDA 커널을 실행하기 위한 인터페이스 제공  
- TMA를 활용한 최적화된 데이터 로딩 방식 사용  


### 결론
- DeepGEMM의 FP8 GEMM 연산을 수행하는 핵심 CUDA 커널이 포함된 파일  
- CUDA Hopper 아키텍처(SM90)에서 최적화된 Tensor Core 활용  
- 이중 누적(Promotion) 기법을 사용하여 FP8 연산 정확도 향상  
- TMA(Tensor Memory Accelerator) 기반으로 빠른 메모리 로딩 및 공유 메모리 활용  

## 마무리
암튼, FP8 의 장점만 가져오는, 훌륭한 방법이다.  
