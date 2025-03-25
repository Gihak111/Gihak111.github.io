---
layout: single
title:  "메모리 오프로드와 flash_mla_with_kvcache"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## CPU 메모리 오프로드와 flash_mla_with_kvcache

CPU 메모리 오프로드는 컴퓨팅 시스템에서 중요한 역할을 한다.  
이 기술은 CPU의 부하를 줄이고 메모리를 효율적으로 활용하게 해준다.  
오늘은 딥시크(DeepSeek)에서 공개한 `FlashMLA` 프로젝트의 주요 로직 중 하나인 `flash_mla_with_kvcache` 함수를 분석해보자.  
이 함수는 메모리 효율성과 속도를 극대화하는 핵심 로직이다.  
코드가 짧으니, 먼저 전체 코드를 살펴보자.  

```python
from typing import Optional, Tuple
import torch
import flash_mla_cuda

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)

def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
        q, k_cache, None, head_dim_v, cache_seqlens, block_table, softmax_scale, causal, 
        tile_scheduler_metadata, num_splits
    )
    return out, softmax_lse
```

이 코드는 CPU 메모리 오프로드와 GPU 병렬 처리를 결합한 MLA(Multi-Head Latent Attention) 구현의 핵심이다.  
이제 단계별로 분석해보자.  


### 1. `get_mla_metadata` 함수 분석

`get_mla_metadata`는 MLA 계산에 필요한 메타데이터를 준비하는 함수이다.  
이 함수를 잘 이해하면 GPU 병렬 처리를 최적화하는 방법을 알 수 있다.  
코드를 구문별로 나눠서 살펴보자.  

#### 함수 시그니처와 타입 힌트
```python
def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

- **목적**: MLA 계산에 필요한 메타데이터를 생성한다.
- **입력**:
  - `cache_seqlens`: 배치 크기(`batch_size`)에 해당하는 1차원 텐서로, 각 샘플의 캐시된 시퀀스 길이를 나타낸다. 타입은 `torch.int32`이다.
  - `num_heads_per_head_k`: 쿼리 시퀀스 길이(`seq_len_q`)와 쿼리 헤드 수(`num_heads_q`)를 키 헤드 수(`num_heads_k`)로 나눈 값이다. 즉, 키 헤드당 처리할 쿼리 헤드 수를 의미한다.
  - `num_heads_k`: 키(Key) 벡터의 헤드 수이다.
- **출력**: 두 개의 텐서를 튜플로 반환한다.
  - `tile_scheduler_metadata`: 타일 스케줄링을 위한 메타데이터이다.
  - `num_splits`: 배치별 작업 분할 수를 나타낸다.

#### Docstring
```python
"""
Arguments:
    cache_seqlens: (batch_size), dtype torch.int32.
    num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
    num_heads_k: num_heads_k.

Returns:
    tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: (batch_size + 1), dtype torch.int32.
"""
```

- **설명**: 입력과 출력의 구조를 명확히 정의한다.
  - `tile_scheduler_metadata`: GPU의 SM(Streaming Multiprocessor) 단위로 작업을 나누기 위한 메타데이터로, 크기는 `(num_sm_parts, TileSchedulerMetaDataSize)`이다.
  - `num_splits`: 배치별로 작업을 몇 개의 조각으로 나눌지 나타내는 텐서로, 크기는 `(batch_size + 1)`이다.

#### 함수 본문
```python
return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
```

- **역할**: 실제 계산은 CUDA 확장 모듈 `flash_mla_cuda.get_mla_metadata`에서 처리한다. 이 함수는 GPU 병렬 처리를 최적화하기 위한 메타데이터를 반환한다.  

#### 요약
`get_mla_metadata`는 MLA 계산에서 GPU의 병렬성을 극대화하려는 준비 작업이다. CPU에서 최소한의 연산을 하고, 무거운 작업을 GPU로 오프로드하자.  

---

### 2. `flash_mla_with_kvcache` 함수 분석

이제 본론인 `flash_mla_with_kvcache`를 살펴보자. 이 함수는 캐시된 키(Key) 데이터를 활용해 메모리 효율성과 속도를 높이는 핵심 로직이다.  

#### 함수 시그니처와 입력
```python
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

- **입력**:
  - `q`: 쿼리 텐서로, 크기는 `(batch_size, seq_len_q, num_heads_q, head_dim)`이다.
  - `k_cache`: 캐시된 키 텐서로, 크기는 `(num_blocks, page_block_size, num_heads_k, head_dim)`이다.
  - `block_table`: 배치별 블록 매핑 테이블로, 크기는 `(batch_size, max_num_blocks_per_seq)`이다.
  - `cache_seqlens`: 각 샘플의 캐시된 시퀀스 길이로, 크기는 `(batch_size)`이다.
  - `head_dim_v`: 값(Value)의 헤드 차원이다.
  - `tile_scheduler_metadata`와 `num_splits`: `get_mla_metadata`에서 받은 메타데이터이다.
  - `softmax_scale`: 소프트맥스 전 스케일 값으로, 기본값은 `1 / sqrt(head_dim)`이다.
  - `causal`: 인과적 어텐션 마스크 적용 여부이다.

#### 주요 로직
```python
if softmax_scale is None:
    softmax_scale = q.shape[-1] ** (-0.5)
out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
    q, k_cache, None, head_dim_v, cache_seqlens, block_table, softmax_scale, causal, 
    tile_scheduler_metadata, num_splits
)
return out, softmax_lse
```

- **작업**:
  - `softmax_scale`이 없으면 헤드 차원의 제곱근 역수를 기본값으로 설정한다.
  - `flash_mla_cuda.fwd_kvcache_mla`를 호출해 GPU에서 MLA 연산을 수행한다.
  - 출력 `out`과 소프트맥스 로그 합 `softmax_lse`를 반환한다.

#### 요약
이 함수는 캐시된 키 데이터를 활용해 CPU의 메모리 부담을 줄이고, GPU로 작업을 오프로드한다. 이렇게 하면 속도와 메모리 효율성을 동시에 챙길 수 있다.

---

### 마무리
`flash_mla_with_kvcache`는 CPU 메모리 오프로드의 좋은 예시이다. 메타데이터 준비부터 GPU 연산까지, 효율적인 설계가 돋보인다. 이 코드를 깃허브 블로그에 올려서 다른 개발자들과 공유해보자. MLA의 메모리 최적화를 이해하고, 우리 프로젝트에도 적용해보자!
