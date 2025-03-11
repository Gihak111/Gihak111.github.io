---
layout: single
title:  "딥시크 FlashMLA-main 알아보기"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## FlashMLA-main
딥시크가 무료로 공개한 알고리즘 중 하나이다.  
코드를 분해하여 이해하는 시간을 가져보자.


딥시크가 무료로 공개한 FlashMLA는 Hopper GPU를 위한 효율적인 MLA 디코딩 커널로, 가변 길이 시퀀스를 최적화하여 처리할 수 있다.

현재 공개된 버전에서는 다음 기능을 지원한다.
- **BF16, FP16** 지원  
- **Paged kvcache** 방식 사용 (블록 크기: 64)  
- **CUDA 12.3 이상** 지원 (권장: 12.8 이상)  
- **PyTorch 2.0 이상** 필요  

고성능 설정에서 **메모리 대역폭 3000GB/s**, **연산 성능 580 TFLOPS**를 달성할 수 있다.  

---

## 설치

다음 명령어로 설치할 수 있다.

```bash
python setup.py install
```

이후, 성능 테스트를 위해 다음을 실행하면 된다.

```bash
python tests/test_flash_mla.py
```

> 💡 CUDA **12.3 이상**에서 동작하지만, **12.8 이상**을 권장한다.  
> 💡 VRAM은 **프레임당 최소 3GB**가 필요하다.

---

## 사용법

아래 예제는 **FlashMLA를 활용한 kvcache 기반 연산**을 수행하는 코드이다.

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

### `scaled_dot_product_attention` 함수

FlashMLA의 핵심 연산 중 하나인 **Scaled Dot-Product Attention**을 구현한 예제이다.

```python
import torch

def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    """
    Scaled Dot-Product Attention 구현.

    query: 쿼리 행렬 (Batch, Head, Seq_len_q, Dim)
    key: 키 행렬 (Batch, Head, Seq_len_k, Dim)
    value: 값 행렬 (Batch, Head, Seq_len_v, Dim)
    h_q: 쿼리 헤드 개수
    h_kv: 키-값 헤드 개수
    is_causal: 미래 토큰 마스킹 여부 (기본값: False)
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5

    if is_causal:
        seq_len = scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(query.device)
        scores = scores.masked_fill(mask == 1, float('-inf'))

    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, value)
```

이 함수는 FlashMLA 내부의 **flash_mla_with_kvcache** 연산과 유사한 방식으로 작동한다.

---

## 코드 분석

FlashMLA의 연산 로직은 `brench_flash_mla`에 구현되어 있다.  
주요 연산 흐름을 이해하려면 `get_mla_metadata`, `flash_mla_with_kvcache` 함수 분석이 필요하다.  

---

## 관련 프로젝트

FlashMLA는 **FlashAttention 2&3** 및 **NVIDIA Cutlass** 프로젝트의 영향을 받았다.

| 플랫폼        | 공식 사이트 | 관련 FlashMLA 버전 |
|--------------|-----------|-------------------|
| **MetaX** | [MetaX](https://www.metax-tech.com) | [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA) |
| **Moore Threads** | [Moore Threads](https://www.mthreads.com/) | [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA) |
| **Hygon DCU** | [Hygon Developer](https://developer.sourcefind.cn/) | [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention) |
| **Intellifusion** | [Intellifusion](https://www.intellif.com) | [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py) |
| **Iluvatar Corex** | [Iluvatar Corex](https://www.iluvatar.com) | [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla) |

---

## 인용

FlashMLA를 연구 또는 논문에서 활용할 경우 아래 BibTeX을 사용할 수 있다.

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernels},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```

## 주요함수

### 1. `get_mla_metadata`  
FlashMLA의 타일 스케줄링 정보를 생성하는 함수이다.   
쉽게 말해서, MLA 연산을 최적화하기 위해 입력 데이터를 어떻게 쪼갤지(Tiling) 결정하는 역할이다.  
 
- 시퀀스 길이(`cache_seqlens`)와 쿼리-키 관계를 계산해서 타일 크기랑 병렬 연산 개수(`num_splits`)를 정해줌.  
- FlashMLA가 기존 Attention보다 빠른 이유 중 하나가 이 타일링 구조 때문이다다.  

코드 흐름:  
```python
tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)
```  
- `cache_seqlens`: 각 시퀀스의 길이 정보  
- `s_q * h_q // h_kv`: 쿼리 길이와 키-값 길이 조정  
- `h_kv`: 키-값 헤드 개수  

FlashMLA는 연산을 한 번에 다 하는 게 아니라, 메모리 효율을 극대화하기 위해 타일 단위로 나눠서 실행한다.  
이걸 결정하는 게 `get_mla_metadata` 함수고, 여기서 `num_splits`가 얼마나 적절하게 설정되는지가 속도와 성능에 직접적인 영향을 준다.  

---

### 2. `flash_mla_with_kvcache`   
FlashMLA의 핵심 연산이 돌아가는 함수이다.    
쿼리(`query`), 키-값 캐시(`kvcache`), 그리고 블록 테이블(`block_table`)을 받아서 최적화된 MLA 연산을 수행한다.  
  
- Paged kvcache 구조를 사용해서 메모리 효율을 극대화  
- 기존 Attention보다 훨씬 빠른 연산 가능  
- `causal=True` 설정 시, Auto-Regressive Transformer(GPT 같은 모델) 지원  

코드 흐름:  
```python
o_i, lse_i = flash_mla_with_kvcache(
    q_i, kvcache_i, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits, causal=True,
)
```  
- `q_i`: 쿼리 텐서  
- `kvcache_i`: Key-Value 캐시  
- `block_table`: 블록 구조 관리 테이블  
- `cache_seqlens`: 각 시퀀스 길이  
- `dv`: Value의 변화량  
- `tile_scheduler_metadata, num_splits`: 위에서 `get_mla_metadata`로 얻은 타일 정보  
- `causal=True`: Decoder-Only Transformer(GPT 같은 모델)에서 필수 설정  
  
FlashMLA의 가장 핵심적인 로직이 여기 들어있어.  
- 기존 Attention은 모든 토큰을 한 번에 계산하는데, FlashMLA는 Paged kvcache를 활용해 부분적으로 연산한다.  
- 메모리 사용량을 줄이면서도 빠르게 계산할 수 있는 이유가 여기에 있다.  


### 3. `scaled_dot_product_attention` (기본 Attention 비교용)  
이 함수는 FlashMLA랑 직접적으로 관련은 없다.  
하지만, 기존 Attention이 어떻게 작동하는지 알고 있어야 FlashMLA가 얼마나 최적화됐는지 비교할 수 있다.  

코드 흐름:  
```python
def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5

    if is_causal:
        seq_len = scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(query.device)
        scores = scores.masked_fill(mask == 1, float('-inf'))

    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, value)
```  
- 기본적인 Scaled Dot-Product Attention이 수행하는 연산 과정을 볼 수 있다.    
- `query * key^T` → `softmax` → `value` 연산을 거치는 전통적인 방식이다.  
 
FlashMLA는 이 과정을 타일링과 Paged kvcache 방식으로 최적화해서, 메모리 절약 + 빠른 연산을 동시에 가능하게 만든다.    



| 함수 | 역할 | 중요한 이유유 |
|------|------|-------------|
| `get_mla_metadata` | 타일 스케줄링 & 병렬 연산 설정 | MLA 연산을 최적화하는 핵심 요소 |
| `flash_mla_with_kvcache` | FlashMLA의 핵심 연산 수행 | 메모리 효율성과 속도를 극대화하는 핵심 로직 |
| `scaled_dot_product_attention` | 기존 Attention과 비교 | FlashMLA 최적화 효과를 이해하는 데 필수 |

결국, `flash_mla_with_kvcache`랑 `get_mla_metadata` 이게 제일 중요하다.  

## 마무리  
다음에는, 위 저 두 함수에 대해서 알아보자.  