---
layout: single
title:  "KoBART 기반 대화 요약 모델 레이어 분석 및 활용"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## PyTorch로 KoBART 모델의 레이어 구조를 분석하고 활용하기

KoBART 모델을 활용해 한국어 대화 요약을 학습시킨 후, 모델의 레이어 구조를 이해하고 예측 결과를 분석하며 활용하는 것은 프로젝트의 핵심이다.  
단순히 모델을 학습시키는 데 그치지 않고, 각 레이어가 어떻게 작동하며 결과를 어떻게 실무에 적용할 수 있는지 파악해야 한다.  
이번 글에서는 PyTorch와 KoBART를 활용해 모델 레이어의 역할을 정리하고, 예측 결과를 추출하며, 실전에서 활용하는 방법을 다룬다.  
레이어 구조 분석, 예측 성능 평가, 결과 활용에 초점을 맞춘다.

### 1. 학습된 모델 로드하기
예측을 위해 학습된 KoBART 모델을 로드한다.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_save_path = './kobart_korean_dialogue_summary_model'
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path)
```

저장된 KoBART 모델과 토크나이저를 로드해 대화 요약 예측을 준비한다.

### 2. 테스트 데이터 준비하기
대화 요약에 사용할 테스트 데이터를 준비한다.

```python
import pandas as pd

# 예시 테스트 데이터
test_data = {
    'ID': ['dlg1', 'dlg2', 'dlg3'],
    'dialogue': [
        "안녕하세요, 오늘 회의 일정 확인하려고 합니다.\n네, 오전 10시에 회의가 있습니다.",
        "제품이 언제 도착하나요?\n내일 오후에 배송 예정입니다.",
        "서비스가 너무 느려서 불만입니다.\n죄송합니다, 개선하겠습니다."
    ]
}
test_df = pd.DataFrame(test_data)

# 토큰화
def tokenize_function(texts):
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    return encodings

test_encodings = tokenize_function(test_df['dialogue'].tolist())
```

테스트 대화를 토큰화하여 모델 입력으로 변환한다.

### 3. 예측 수행하기
KoBART 모델로 대화 요약을 생성한다.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with torch.no_grad():
    inputs = {key: val.to(device) for key, val in test_encodings.items()}
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        max_length=128,
        early_stopping=True
    )
    summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]

test_df['predicted_summary'] = summaries
```

모델을 평가 모드로 전환하고, Beam Search를 활용해 요약을 생성하여 데이터프레임에 추가한다.

### 4. 레이어 구조 분석하기: 입력 단계
KoBART 모델의 각 레이어 역할을 이해한다.

- **Word Embedding Layer**
  - **구성**: `model.shared.weight` [30,000, 768]
  - **역할**: 대화 단어를 768차원 벡터로 변환해 의미를 표현.
- **Positional Embedding Layer**
  - **구성**: `embed_positions.weight` [1,028, 768]
  - **역할**: 단어 위치 정보를 추가해 순서 이해를 돕는다.
- **LayerNorm**
  - **구성**: `layernorm_embedding` [768]
  - **역할**: 입력 벡터를 정규화해 학습 안정성을 높인다.

### 5. 레이어 구조 분석하기: Transformer (Encoder)
Encoder는 대화의 의미를 추출한다.

- **Self-Attention**
  - **구성**: `self_attn.q_proj`, `k_proj`, `v_proj`, `out_proj` [768, 768]
  - **역할**: 대화 내 단어 간 관계(예: "회의"와 "10시")를 학습.
- **LayerNorm**
  - **구성**: `self_attn_layer_norm` [768]
  - **역할**: Attention 출력을 정규화해 안정화.
- **Feed-Forward Network (FFN)**
  - **구성**: `fc1` [3,072, 768], `fc2` [768, 3,072]
  - **역할**: 단어 의미를 깊이 처리해 정보 추출.
- **LayerNorm**
  - **구성**: `final_layer_norm` [768]
  - **역할**: FFN 출력을 정규화해 다음 레이어로 전달.

### 6. 레이어 구조 분석하기: Transformer (Decoder)
Decoder는 요약을 생성한다.

- **Self-Attention**
  - **구성**: `self_attn.q_proj`, `k_proj`, `v_proj`, `out_proj` [768, 768]
  - **역할**: 생성 중인 요약의 단어 간 관계를 학습.
- **Encoder-Decoder Attention**
  - **구성**: `encoder_attn.q_proj`, `k_proj`, `v_proj`, `out_proj` [768, 768]
  - **역할**: 대화에서 중요한 부분을 요약에 연결.
- **LayerNorm**
  - **구성**: `self_attn_layer_norm`, `encoder_attn_layer_norm` [768]
  - **역할**: 각 Attention 출력을 정규화.
- **Feed-Forward Network (FFN)**
  - **구성**: `fc1` [3,072, 768], `fc2` [768, 3,072]
  - **역할**: 요약 단어의 의미를 강화.
- **LayerNorm**
  - **구성**: `final_layer_norm` [768]
  - **역할**: 최종 출력 정규화.

### 7. 레이어 구조 분석하기: 출력 단계
출력 레이어는 요약 텍스트를 생성한다.

- **Linear Layer**
  - **구성**: `model.shared.weight` [30,000, 768], `final_logits_bias` [1, 30,000]
  - **역할**: Decoder 출력을 단어 점수(로짓)로 변환.
- **Softmax Layer**
  - **구성**: 연산 레이어 [30,000]
  - **역할**: 로짓을 확률로 변환해 다음 단어 예측.

### 8. 예측 결과 저장하기
예측된 요약을 저장한다.

```python
test_df.to_csv('summary_results.csv', index=False)
print("예측 요약이 'summary_results.csv'에 저장되었습니다.")
```

CSV 파일로 저장해 결과를 보존한다.

### 9. 오차 사례 분석하기
실제 요약과 비교해 오차를 분석한다.

```python
# 실제 요약이 있다고 가정
true_summaries = [
    "오전 10시 회의가 있다.",
    "제품은 내일 오후 도착한다.",
    "서비스가 느리다."
]
test_df['true_summary'] = true_summaries
test_df['is_correct'] = test_df['predicted_summary'] == test_df['true_summary']

errors = test_df[~test_df['is_correct']]
print("오차 사례:")
print(errors[['ID', 'dialogue', 'predicted_summary', 'true_summary']])
```

오차 사례를 확인해 모델 개선점을 찾는다.

### 10. 요약 품질 평가하기
ROUGE 점수로 요약 품질을 평가한다.

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(true, pred) for true, pred in zip(test_df['true_summary'], test_df['predicted_summary'])]

avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
print(f"평균 ROUGE-1: {avg_rouge1:.4f}, 평균 ROUGE-L: {avg_rougeL:.4f}")
```

ROUGE 점수로 요약의 정확성과 유사성을 평가한다.

### 11. 예측 결과를 비즈니스에 활용하기
요약 결과를 실무에 적용한다.

```python
def suggest_action(summary):
    if "회의" in summary:
        return "회의 자료 준비"
    elif "배송" in summary or "도착" in summary:
        return "배송 상태 확인"
    else:
        return "고객 피드백 요청"

test_df['action'] = test_df['predicted_summary'].apply(suggest_action)
print(test_df[['ID', 'dialogue', 'predicted_summary', 'action']])
```

요약 내용에 따라 실무 액션을 제안한다.

### 12. 결과 요약 보고서 작성하기
분석 결과를 요약한다.

```python
summary = {
    '총 대화 수': len(test_df),
    '정확도': test_df['is_correct'].mean() if 'is_correct' in test_df.columns else 'N/A',
    '평균 ROUGE-1': avg_rouge1,
    '평균 ROUGE-L': avg_rougeL,
    '오차 사례 수': len(errors) if 'is_correct' in test_df.columns else 'N/A'
}
print("예측 결과 요약:", summary)
```

결과를 한눈에 보여주는 보고서를 작성한다.

### 결론
그냥, 레이어 어떻게 작동하는지 보려고 만들어 봤다.
위 내용은 파인튜닝이지만, 그래도, 각 레이어에 대해 알기 쉽다.
