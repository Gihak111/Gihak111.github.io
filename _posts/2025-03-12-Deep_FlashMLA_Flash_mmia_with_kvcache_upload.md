---
layout: single
title:  "딥시크 FlashMLA-main 더 깊게 알아보기"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## FlashMLA-main_flash_mla_with_kvcache
오늘은 주요 로직 중, flash_mla_with_kvcache 함수에 대해서 알아보자.  
메모리 효율성과 속도를 극대화하는 핵심 로직으로 중요한 로직이다.  
코드가 짧으므로, 코드 전문을 먼저 보자.  

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
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse

```

위 코드는 딥시크가 공개한 코드의 전문이다.  
단계별로 로직을 분석해 보자.  

### 1. `get_mla_metadata` 함수 분석

#### 코드 전체
```python
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
```

#### 구문별 분석
1. 함수 시그니처와 타입 힌트
   ```python
   def get_mla_metadata(
       cache_seqlens: torch.Tensor,
       num_heads_per_head_k: int,
       num_heads_k: int,
   ) -> Tuple[torch.Tensor, torch.Tensor]:
   ```
   - 목적: 이 함수는 MLA 계산에 필요한 메타데이터를 생성한다.  
   - 입력:  
     - `cache_seqlens`: 배치 크기(`batch_size`)에 해당하는 1차원 텐서로, 각 샘플의 캐시된 시퀀스 길이를 나타낸다. 데이터 타입은 `torch.int32`.
     - `num_heads_per_head_k`: 쿼리 헤드 수(`num_heads_q`)와 시퀀스 길이(`seq_len_q`)를 키 헤드 수(`num_heads_k`)로 나눈 값. 즉, 키 헤드당 처리해야 할 쿼리 헤드의 개수를 의미한다.  
     - `num_heads_k`: 키(Key) 벡터의 헤드 수.  
   - 출력: 두 개의 텐서를 포함한 튜플:  
     - `tile_scheduler_metadata`: 타일 스케줄링을 위한 메타데이터.  
     - `num_splits`: 각 배치에 대한 분할 수.  

2. Docstring  
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
   - 설명: 입력과 출력의 구조를 명확히 설명한다.  
   - `tile_scheduler_metadata`: GPU의 SM(Streaming Multiprocessor) 단위로 작업을 나누기 위한 메타데이터로, `(num_sm_parts, TileSchedulerMetaDataSize)` 크기를 가진다.  
   - `num_splits`: 배치별로 작업을 몇 개의 조각으로 나눌지 나타내는 텐서이다. 크기는 `(batch_size + 1)`.  

3. 함수 본문
   ```python
   return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
   ```
   - 역할: 실제 계산은 CUDA 확장 모듈인 `flash_mla_cuda.get_mla_metadata`에서 수행된다.   

#### 요약
- 목적: MLA 계산에서 GPU의 병렬 처리를 최적화하기 위한 메타데이터를 생성.  
- 핵심 로직: CUDA 확장을 통해 GPU에서 실행되며, 타일 단위로 작업을 분할하는 데 필요한 정보를 제공.  
- 메모리 효율성: 메타데이터를 사전에 계산해 불필요한 연산을 줄이고, 캐시된 시퀀스 길이를 활용해 동적 메모리 할당을 최적화.  


### 2. `flash_mla_with_kvcache` 함수 분석

#### 코드 전체
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
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse
```

#### 구문별 분석
1. 함수 시그니처와 타입 힌트
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
   - 목적: 키-값 캐시를 활용한 멀티헤드 어텐션 연산을 수행.  
   - 입력:  
     - `q`: 쿼리 텐서, 크기는 `(batch_size, seq_len_q, num_heads_q, head_dim)`.  
     - `k_cache`: 캐시된 키 텐서, 크기는 `(num_blocks, page_block_size, num_heads_k, head_dim)`.  
     - `block_table`: 캐시 블록의 인덱스를 나타내는 테이블, 크기는 `(batch_size, max_num_blocks_per_seq)`.  
     - `cache_seqlens`: 각 배치의 캐시된 시퀀스 길이, 크기는 `(batch_size)`.  
     - `head_dim_v`: 값(Value) 벡터의 헤드 차원.  
     - `tile_scheduler_metadata`와 `num_splits`: `get_mla_metadata`에서 생성된 메타데이터.  
     - `softmax_scale`: 소프트맥스前の 스케일링 값(기본값은 `1/sqrt(head_dim)`).  
     - `causal`: 인과적 어텐션 마스크 적용 여부.  
   - 출력: 두 개의 텐서를 포함한 튜플:  
     - `out`: 어텐션 출력, 크기는 `(batch_size, seq_len_q, num_heads_q, head_dim_v)`.  
     - `softmax_lse`: 소프트맥스 로그 합계, 크기는 `(batch_size, num_heads_q, seq_len_q)`.  

2. Docstring
   ```python
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
   ```
   - 설명: 입력과 출력 텐서의 차원 및 역할에 대한 상세 설명이다.  
   - 특이점: `k_cache`와 `block_table`을 사용해 키-값 캐시를 효율적으로 관리하며, `softmax_lse`를 반환해 후속 계산(예: 로그 확률)에서 재사용 가능.  

3. 소프트맥스 스케일링 초기화
   ```python
   if softmax_scale is None:
       softmax_scale = q.shape[-1] ** (-0.5)
   ```
   - 역할: `softmax_scale`이 명시되지 않은 경우, 헤드 차원의 제곱근 역수를 기본값으로 설정한다.  
   - 의미: 이는 어텐션 메커니즘에서 표준 스케일링 방식(`1/sqrt(d_k)`)으로, 쿼리와 키의 내적이 너무 커지는 것을 방지해 소프트맥스 출력의 분포를 안정화 한다.  

4. CUDA 호출
   ```python
   out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
       q,
       k_cache,
       None,
       head_dim_v,
       cache_seqlens,
       block_table,
       softmax_scale,
       causal,
       tile_scheduler_metadata,
       num_splits,
   )
   ```
   - 역할: CUDA 확장 모듈을 호출해 실제 MLA 연산을 수행한다.  
   - 특이점: 세 번째 인자로 `None`이 전달되는데, 이는 `v_cache`(값 캐시)가 없음을 의미할 수 있다. 코드가 `k_cache`만 사용하므로 값 벡터는 동적으로 계산되거나 별도로 처리되는 것으로 보인다.  
   - 과정:    
     1. 쿼리(`q`)와 캐시된 키(`k_cache`)를 사용해 어텐션 스코어를 계산.  
     2. `tile_scheduler_metadata`와 `num_splits`를 활용해 작업을 타일 단위로 분할 및 병렬 처리.  
     3. `causal=True`일 경우, 인과적 마스크를 적용해 미래 토큰을 참조하지 않음.  
     4. 소프트맥스 적용 후 출력(`out`)과 로그 합계(`softmax_lse`)를 계산.  

5. 출력 반환  
   ```python
   return out, softmax_lse
   ```
   - 역할: 계산된 어텐션 출력과 소프트맥스 로그 합계를 반환한다.  

### 결론
- 목적: 키-값 캐시를 활용해 MLA 연산을 효율적으로 수행하며, GPU에서 타일 기반 병렬 처리를 최적화.  
- 핵심 로직:  
  - 캐시(`k_cache`)와 블록 테이블(`block_table`)을 사용해 메모리 접근을 최적화.  
  - CUDA 확장을 통해 고속 연산을 구현.  
  - `softmax_lse`를 반환해 후속 계산에서 재사용 가능.  
- 메모리 효율성: 캐시된 키를 재사용하고, 타일 단위로 작업을 분할해 GPU 메모리 사용을 최소화.  
- 속도 최적화: 타일 스케줄링과 CUDA 기반 병렬 처리를 통해 연산 속도를 극대화.  

1. 메모리 효율성:  
   - `k_cache`와 `block_table`을 활용해 키 데이터를 캐싱하고 재사용함으로써 메모리 사용량을 줄임.  
   - 동적 시퀀스 길이(`cache_seqlens`)를 처리해 불필요한 메모리 할당을 방지.  

2. 속도 최적화:  
   - `tile_scheduler_metadata`와 `num_splits`를 통해 GPU의 SM 단위로 작업을 분할, 병렬 처리 효율을 극대화.  
   - CUDA 확장을 사용해 CPU-GPU 간 데이터 전송 오버헤드를 최소화하고, GPU 네이티브 연산을 활용.  

3. 유연성:  
   - `causal` 옵션으로 인과적/비인과적 어텐션을 지원.  
   - `softmax_scale`을 커스터마이징 가능해 다양한 어텐션 메커니즘에 적응 가능.  

여기까지 왔으니까, 내부동작, 타일 스케줄링의 세부 사항도 알아보자.  

### 1. `flash_mla_cuda`의 내부 동작  

이 헤더 파일은 두 개의 주요 기능을 정의합니다:  
1. `get_mla_metadata_func`: MLA 연산에 필요한 메타데이터를 생성.  
2. `run_mha_fwd_splitkv_mla`: MLA의 순방향 연산을 수행.  

#### 1.1 구조체: `Flash_fwd_mla_params`  
```cpp
struct Flash_fwd_mla_params {
    // 주요 파라미터
    int b, seqlen_q, d, d_v;  // 배치 크기, 쿼리 시퀀스 길이, 헤드 차원(QK), 헤드 차원(V)
    int h, h_h_k_ratio, ngroups;  // 헤드 수(Q), 헤드 비율(h_q/h_k), 그룹 수
    bool is_causal;  // 인과적 어텐션 여부
    float scale_softmax, scale_softmax_log2;  // 소프트맥스 스케일링 값(일반 및 log2 기반)
    int *__restrict__ cu_seqlens_k;  // 캐시된 키의 시퀀스 길이

    // 데이터 포인터
    void *__restrict__ q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr;  // Q, K, V, 출력, 소프트맥스 로그 합계

    // 스트라이드 (배치, 행, 헤드 단위 메모리 오프셋)
    index_t q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride;
    index_t q_row_stride, k_row_stride, v_row_stride, o_row_stride;
    index_t q_head_stride, k_head_stride, v_head_stride, o_head_stride;

    // 블록 테이블 (캐시 관리)
    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // 타일 스케줄링 메타데이터
    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    // 누적 버퍼
    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;
};
```
- 주요 역할: MLA 연산에 필요한 모든 파라미터와 메모리 포인터를 정의.  
- 특징:  
  - `h_h_k_ratio`: `num_heads_q / num_heads_k`로, 쿼리 헤드와 키 헤드 간의 비율을 나타냄 (PyTorch 코드의 `num_heads_per_head_k`와 동일).  
  - `cu_seqlens_k`: 키 캐시의 시퀀스 길이를 GPU 메모리에 저장.  
  - 스트라이드: Q, K, V, 출력 텐서의 메모리 레이아웃을 정의하며, 배치/행/헤드 단위로 접근 가능.  
  - `block_table`: 페이지 기반 KV 캐시를 관리하며, `page_block_size` 단위로 블록을 구성.  
  - 누적 버퍼: `softmax_lseaccum_ptr`와 `oaccum_ptr`는 타일별 중간 결과를 누적하기 위한 임시 버퍼로 보임.  

#### 1.2 구조체: `Mla_metadata_params`
```cpp
struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;  // 키 시퀀스 길이
    int *__restrict__ tile_scheduler_metadata_ptr;  // 타일 스케줄링 메타데이터
    int *__restrict__ num_splits_ptr;  // 분할 수
    int batch_size, block_size_n, fixed_overhead_num_blocks, num_sm_parts;  // 배치 크기, 블록 크기, 오버헤드 블록 수, SM 파트 수
};
```  

- 주요 역할: `get_mla_metadata_func`에 전달되는 파라미터로, 타일 스케줄링과 작업 분할을 위한 입력을 정의.  
- 특징:  
  - `block_size_n`: 키 캐시의 블록 크기(아마 `page_block_size`와 동일).  
  - `fixed_overhead_num_blocks`: 고정된 오버헤드 블록 수로, 메타데이터 계산 시 고려되는 상수.  

#### 1.3 함수: `get_mla_metadata_func`  
```cpp
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
```  
- 입력: `Mla_metadata_params`와 CUDA 스트림.  
- 출력: `tile_scheduler_metadata_ptr`와 `num_splits_ptr`를 채움.  
- 내부 동작:  
  1. 시퀀스 길이 분석: `seqlens_k_ptr`를 기반으로 각 배치의 작업량 계산.  
  2. 작업 분할: `batch_size`, `block_size_n`, `num_sm_parts`를 고려해 작업을 타일로 나눔.  
  3. 메타데이터 생성: `tile_scheduler_metadata_ptr`에 타일별 정보(시작/끝 인덱스 등)를 기록.  
  4. 분할 수 기록: `num_splits_ptr`에 배치별 분할 수를 저장.  

#### 1.4 함수: `run_mha_fwd_splitkv_mla`
```cpp
template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);
```  

- 입력: `Flash_fwd_mla_params`와 CUDA 스트림.  
- 템플릿 파라미터:  
  - `T`: 데이터 타입(예: `float`, `half`).  
  - `Headdim`: 헤드 차원(컴파일 타임 상수로 최적화).  
- 내부 동작:  
  1. 타일 단위 연산: `tile_scheduler_metadata_ptr`와 `num_splits_ptr`를 기반으로 작업을 타일로 분할.  
  2. QK^T 계산: `q_ptr`와 `k_ptr`를 사용해 어텐션 스코어 계산.  
  3. 소프트맥스: `scale_softmax` 적용 후 소프트맥스 계산, `is_causal`에 따라 마스크 적용.  
  4. 출력 생성: `v_ptr`를 사용해 최종 출력(`o_ptr`) 계산.  
  5. 로그 합계 저장* `softmax_lse_ptr`에 결과 기록.  
  6. 누적 처리: `softmax_lseaccum_ptr`와 `oaccum_ptr`를 사용해 타일 간 결과 집계.  


### 2. 타일 스케줄링의 세부 사항

#### 2.1 `TileSchedulerMetaDataSize` 정의
```cpp
static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]
```
- 의미: 각 타일의 메타데이터는 8개의 `int`로 구성.  
- 구성:  
  1. `begin_idx`: 타일이 시작하는 배치 또는 블록 인덱스.  
  2. `begin_seqlen`: 타일의 시작 시퀀스 위치.  
  3. `end_idx`: 타일이 끝나는 배치 또는 블록 인덱스.  
  4. `end_seqlen`: 타일의 끝 시퀀스 위치.  
  5. `begin_n_split_idx`: 해당 타일이 속한 분할 그룹의 시작 인덱스.  
  6-8. 예약 필드: 현재 사용되지 않거나, 디버깅/최적화를 위한 추가 정보.  
- **크기**: `(num_sm_parts, 8)`로, GPU의 SM 수에 따라 타일 메타데이터가 생성됨.

#### 2.2 타일 스케줄링의 설계  
- 목적: MLA 연산을 GPU의 SM 단위로 분할해 병렬 처리하며, SRAM을 활용해 HBM 접근을 최소화.  
- 과정:  
  1. 작업 분할:
     - `seqlens_k_ptr`와 `batch_size`를 기반으로 전체 시퀀스를 분석.  
     - `num_sm_parts`에 따라 작업을 SM별로 나눔.  
  2. 타일 정의:  
     - 각 타일은 `begin_idx` ~ `end_idx`와 `begin_seqlen` ~ `end_seqlen`으로 정의.  
     - `block_table`과 `page_block_size`를 사용해 캐시 블록 단위로 매핑.  
  3. 분할 수 계산:  
     - `num_splits_ptr`는 배치별로 몇 개의 타일로 나뉘는지를 기록.  
     - 예: 긴 시퀀스(`seqlen_k > page_block_size`)는 여러 타일로 분할.  
  4. SM 할당:  
     - `tile_scheduler_metadata_ptr`를 통해 각 SM에 타일을 배정.  
     - `num_sm_parts`는 GPU의 SM 수와 작업 부하에 따라 동적으로 조정될 가능성.  

#### 2.3 타일 단위 연산
- 메모리 관리:  
  - `q_ptr`, `k_ptr`, `v_ptr`의 타일 부분을 SRAM에 로드.  
  - 스트라이드(`q_row_stride`, `k_head_stride` 등)를 사용해 메모리 접근 최적화.  
- 연산 흐름:  
  1. QK^T 계산: 타일 단위로 행렬 곱셈 수행.  
  2. 소프트맥스: `scale_softmax_log2`를 활용해 로그 도메인에서 안정적으로 계산.  
  3. V 적용: 값 벡터를 곱해 출력 생성.  
  4. 누적: `oaccum_ptr`와 `softmax_lseaccum_ptr`에 타일별 결과 저장.  
- **병렬성**: 각 SM이 독립적으로 타일을 처리하며, 워프 단위로 세부 연산 병렬화.  

#### 2.4 최적화 기법  
- 페이지 기반 캐시: `block_table`과 `page_block_size`를 통해 KV 캐시를 페이지 단위로 관리, 메모리 효율성 향상.  
- 온라인 소프트맥스: 타일별로 소프트맥스를 계산하고 누적 버퍼에 저장해 HBM 쓰기를 줄임.  
- 비동기 처리: `cudaStream_t`를 활용해 데이터 로드와 연산을 오버랩.  


### 3. PyTorch 코드와의 매핑
- `get_mla_metadata` → `get_mla_metadata_func`:  
  - PyTorch의 `cache_seqlens` → `seqlens_k_ptr`.  
  - `tile_scheduler_metadata` → `tile_scheduler_metadata_ptr`.  
  - `num_splits` → `num_splits_ptr`.  
- `flash_mla_with_kvcache` → `run_mha_fwd_splitkv_mla`:  
  - `q`, `k_cache`, `block_table` → `q_ptr`, `k_ptr`, `block_table`.  
  - `head_dim_v` → `d_v`.  
  - 출력 `out`, `softmax_lse` → `o_ptr`, `softmax_lse_ptr`.  

차이점:  
- PyTorch 코드에서는 `v_cache`가 생략되었지만, 헤더 파일에는 `v_ptr`가 포함되었다다.  
이는 `flash_mla_with_kvcache`가 값 벡터를 동적으로 생성하거나 별도로 처리할 가능성을 시사한다.  


### 4. 결론  
- 내부 동작: `flash_mla_cuda`는 페이지 기반 KV 캐시와 타일 스케줄링을 결합해 MLA를 최적화하며, SRAM 중심 연산과 온라인 소프트맥스를 통해 메모리와 속도를 극대화한다.  
- 타일 스케줄링: `tile_scheduler_metadata_ptr`는 타일의 시작/끝 위치와 분할 정보를 정의하며, `num_splits_ptr`와 함께 SM 단위 병렬 처리를 관리한다.  


### 진짜 결론  
아무튼, 메모리 효율성과 속도를 극대화 하기 위해 타일 스케줄링 형식을 활용하는 방법이다.  