---
layout: single
title:  "대규모 데이터셋 활용시"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## PyTorch로 대규모 텍스트 데이터셋을 탐색하고 시각화하기

딥러닝 프로젝트를 시작하기 전에 데이터셋을 이해하는 것은 필수적이다.  
특히 대규모 텍스트 데이터셋은 그 규모와 복잡성 때문에 체계적인 탐색과 시각화가 필요하다.  
이는, cc-100처럼, 너무 큰 규모의 데이터셋은 여는것 조차 불가하기 때문이다.  
이번에는 PyTorch와 코드를 활용하여 대규모 텍스트 데이터셋을 로드하고, 탐색하며, 시각화하는 방법을 정리한다.  
데이터의 구조를 파악하고 인사이트를 얻는 데 초점을 맞추며, 실무에서 바로 적용할 수 있는 가이드를 제공하고자 한다.  

### 1. Parquet 파일로 데이터셋 로드하기
대규모 데이터셋은 효율적인 포맷으로 저장된다.  
Parquet 파일은 빠른 읽기와 압축을 지원한다.  

```python
import os
import pyarrow.parquet as pq
from tqdm import tqdm

def load_all_parquet_data(folder_path):
    data_dict = {}
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for folder in tqdm(folders, desc="폴더 로드 진행"):
        folder_full_path = os.path.join(folder_path, folder)
        data_dict[folder] = {}
        files = [f for f in os.listdir(folder_full_path) if f.endswith('.parquet')]
        for file in tqdm(files, desc=f"{folder} 파일 로드", leave=False):
            file_path = os.path.join(folder_full_path, file)
            data_dict[folder][file] = pq.read_table(file_path).to_pandas()
    return data_dict

train_folder_path = './데이터셋/train'
train_data_dict = load_all_parquet_data(train_folder_path)
```

폴더 구조를 유지하며 데이터를 딕셔너리에 로드한다.  
`tqdm`으로 진행 상황을 확인하여 대규모 데이터 로드 과정을 모니터링한한다.  

### 2. 고객별 텍스트 데이터 통합하기
데이터셋을 고객 단위로 묶어 탐색 준비를 한다.  

```python
def consolidate_customer_data(data_dict):
    all_ids = set()
    for folder in data_dict:
        for file in data_dict[folder]:
            if 'ID' in data_dict[folder][file].columns:
                all_ids.update(data_dict[folder][file]['ID'].unique())
    
    customer_data = {}
    for customer_id in tqdm(all_ids, desc="고객 데이터 통합 진행"):
        customer_text = []
        for folder in data_dict:
            for file in data_dict[folder]:
                df = data_dict[folder][file]
                if 'ID' in df.columns and customer_id in df['ID'].values:
                    row = df[df['ID'] == customer_id].iloc[0].drop('ID', errors='ignore')
                    customer_text.append(f"{folder}/{file}: {' '.join(map(str, row.values))}")
        customer_data[customer_id] = " ".join(customer_text) if customer_text else "no_data"
    return customer_data

customer_data = consolidate_customer_data(train_data_dict)
```

고객 ID별로 텍스트를 통합하여 데이터셋의 개별 엔터티를 파악한한다.  

### 3. 데이터프레임으로 구조화하기
탐색을 위해 데이터를 pandas 데이터프레임으로 변환한다.  

```python
import pandas as pd

train_df = pd.DataFrame({
    'ID': list(customer_data.keys()),
    'text': list(customer_data.values())
})
```

ID와 텍스트를 열로 구성하여 데이터 구조를 한눈에 확인한다.  

### 4. 텍스트 길이 분포 분석하기
텍스트 데이터의 길이를 분석하면 데이터셋의 특성을 이해할 수 있다.  

```python
train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))
print(train_df['text_length'].describe())
```

`describe()`로 평균, 최소, 최대 길이를 확인하여 데이터셋의 분포를 파악한한다.  

### 5. 텍스트 길이 히스토그램 시각화하기
길이 분포를 시각화하면 직관적인 인사이트를 얻는다.  

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(train_df['text_length'], bins=50, color='skyblue', edgecolor='black')
plt.title('텍스트 길이 분포')
plt.xlabel('단어 수')
plt.ylabel('빈도')
plt.grid(True)
plt.show()
```

히스토그램으로 텍스트 길이의 분포를 시각화하여 이상치나 패턴을 탐지한한다.  

### 6. 고빈도 단어 추출하기
텍스트에서 자주 등장하는 단어를 분석한다.   

```python
from collections import Counter

all_words = " ".join(train_df['text']).split()
word_counts = Counter(all_words)
top_words = word_counts.most_common(20)
print("상위 20개 단어:", top_words)
```

`Counter`로 상위 20개 단어를 추출하여 데이터셋의 주요 주제를 파악한다.  

### 7. 워드클라우드로 단어 빈도 시각화하기
고빈도 단어를 시각적으로 표현한다.  

```python
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('상위 단어 워드클라우드')
plt.show()
```

워드클라우드로 단어 빈도를 시각화하여 데이터셋의 핵심 키워드를 강조한다.  

### 8. 결측 데이터 탐색하기
결측값은 데이터 품질에 영향을 미친다.  

```python
missing_data = train_df['text'].apply(lambda x: x == "no_data").sum()
print(f"결측 데이터 수: {missing_data}")
```

`"no_data"`로 표시된 결측값의 개수를 확인하여 데이터 완전성을 점검한다.  

### 9. 고객별 데이터 건수 분석하기
고객별 데이터 분포를 파악한다.  

```python
customer_counts = {folder: sum(len(df) for df in data_dict[folder].values()) for folder in train_data_dict}
print("폴더별 데이터 건수:", customer_counts)
```

폴더별 데이터 건수를 계산하여 데이터셋의 불균형 여부를 확인한다.  

### 10. 폴더별 데이터 건수 막대그래프 그리기
데이터 분포를 시각화한다.  

```python
plt.figure(figsize=(12, 6))
plt.bar(customer_counts.keys(), customer_counts.values(), color='lightgreen')
plt.title('폴더별 데이터 건수')
plt.xlabel('폴더')
plt.ylabel('데이터 건수')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

막대그래프로 폴더별 데이터 건수를 비교하여 데이터셋의 구조를 이해힌다.  

### 11. 샘플 텍스트 확인하기
실제 데이터를 살펴보면 데이터셋의 내용을 직관적으로 알 수 있다.  

```python
for i in range(5):
    print(f"샘플 {i+1}: {train_df['text'].iloc[i][:200]}...")
```

상위 5개 샘플을 출력하여 데이터의 구체적인 내용을 확인한다.  

## 결론
대규모 텍스트 데이터셋을 탐색하고 시각화하는 것은 모델 학습 전 필수 단계이다.  
Parquet 파일로 데이터를 로드하고, 고객별로 통합하며, 데이터프레임으로 구조화한다.  
텍스트 길이와 고빈도 단어를 분석하고, 히스토그램과 워드클라우드로 시각화하여 데이터셋의 특성을 파악한다.  
결측값과 폴더별 분포를 점검하며, 샘플 데이터를 확인하여 데이터의 실체를 이해한다.  
이 과정을 통해 데이터셋에 대한 깊은 인사이트를 얻고, 이후 학습 전략을 세우는 데 활용할 수 있다.  
아 암튼 알아두면 좋다는 거다.  
