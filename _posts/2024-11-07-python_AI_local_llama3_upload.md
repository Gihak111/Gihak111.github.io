---
layout: single
title:  "lama, Falcon 로컬 설치 및 사용해보기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# llama3.1
메타에서 라마 3.1을 뿌렸다.  
그저 고트. 그래서 이번에는 lama3를 로컬에 다운 받고, 하이퍼 파라미터를 조절해 자신만의 AI를 만들어 보자.  


# 모델 종류 및 규격
다음과 같은 모델들이 있다.  

## 1. 8B  
권장 사양  
- RAM: 32GB  
- VRAM: 24G  
- storage: 5G  

## 2. 70B  
권장 사양  
- RAM: 120GB  
- VRAM: 80G  
- storage: 5G  

## 3. 45B  
권장 사양  
- RAM: 500GB  
- VRAM: 400G  
- storage: 5G  

위 사양보다 조금은 낮아도 돌아가기는 한다. 8B 모델은 웬만하면 잘 돌릴 수 있다.  
따라서, 8B로 진행하도록 하자.  

## 1. 설치 방법
Ollama에서 다운받자.  
[https://ollama.com/download](https://ollama.com/download)  
위 링크에 들어가서, Ollama를 다운받고, 설치하자.  

위에서 다운받은 파일이 실행이 되지 않는다면, 처음 화면에서 밑에 작게 있는 Lama3.1을 누르고, 원하는 모델을 눌러 8b, 4.7GB를 선택해 명령어를 받자.  
[아니면, 이 링크로 들어가도 있다.](https://ollama.com/library/llama3.1:8b)
```bash
ollama run llama3.1:8b
```
이런식의 문구가 있는데, 이를 terminal에 집어넣는 것으로 간단하게 라마1을 다운받을 수 있다.  
작은 모델이기 때문에, 금방 다운로드 된다.  
이 방법으로 해도, 잘 다운받아지며 설치된다.  
스테이블 디퓨전도 알아보자.  
아주 나이스 하다.  

더 최근 모델인 라마 3.2도 있으니 이걸 다운받아도 된다.  
3.2가 성능, 효율 전부다 앞선다. 최적화에 경량화 가지 되어있으니, 3.2 사용을 적극 권장한다.  

---

다운받은 라마를 보면, 헤시 함수 SHA 256으로 암호화 되어 있는 것을 볼 수 있다.  
따라서 일반적인 방식으로 파인튜닝 하기 힘들기 때문에,  
이후부터는  Falcon-7B 모델로 진행하였다.  
## 2. 한국어 패치하기
일단, 한국어를 사용할 수 있게 딥러닝 시켜보자.  
공개되어 있는 한국어 데이터 셋에는 모두의 말뭉치, AI Hub의 한국어 데이터셋, Kaist 말뭉치 등이 있다.  
이에 앞서, 우선 한번 사용해 보자.  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 및 토크나이저 로드
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 텍스트 생성 함수
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 모델이 사용할 디바이스 설정 (GPU가 있으면 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # 모델을 사용해 답변 생성
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 생성된 답변을 디코딩하여 반환
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 실시간 대화
print("실시간 대화를 시작합니다. 종료하려면 '종료'라고 입력하세요.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["종료", "exit", "quit"]:
        print("대화를 종료합니다.")
        break
    
    # 사용자 입력에 대한 모델 응답 생성
    response = generate_response(user_input)
    print("Bot:", response)

```
위 코드로 실시간 대화가 가능하다.  
물론, 컴퓨터가 상당히 좋아야 한다.  

허깅페이스에서 트랜스포머 라이브러리르 ㄹ통해 파인튜닝 할 수 있다.  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 및 토크나이저 로드 (Falcon-7B 모델 경로)
model_name = "tiiuae/falcon-7b"
logger.info(f"Loading tokenizer and model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 패딩 토큰 설정 (필요에 따라 eos_token을 pad_token으로 사용하거나 새로운 토큰을 추가)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 또는 새로 추가하려면 `{'pad_token': '[PAD]'}` 사용
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 한국어 데이터셋 로드 (KLUE MRC 태스크 사용)
logger.info("Loading KLUE MRC dataset...")
dataset = load_dataset("klue", "mrc")

# 데이터셋 텍스트 변환 (질문과 답변을 이어 붙인 형태로 파인튜닝용 데이터 생성)
def preprocess_function(examples):
    texts = []
    for question, answers in zip(examples["question"], examples["answers"]):
        # 질문 텍스트 추출
        question_text = question if isinstance(question, str) else " ".join(question)
        # 첫 번째 답변 텍스트 추출
        answer_text = answers["text"][0] if answers and "text" in answers and answers["text"] else ""
        
        # 질문과 답변이 모두 존재하는 경우만 텍스트 생성
        if question_text.strip() and answer_text.strip():
            combined_text = question_text + " " + answer_text
            texts.append(combined_text)
    return {"text": texts}

logger.info("Preprocessing dataset...")
# 데이터셋에 대해 텍스트로 변환 및 빈 샘플 제거
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# 텍스트가 유효하지 않은 샘플 필터링 (텍스트가 비어 있거나 너무 짧은 경우 제거)
logger.info("Filtering invalid samples...")
tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["text"].strip()) > 10)

# 필터링 후 데이터셋의 크기 확인
for split in ["train", "validation"]:
    split_size = len(tokenized_dataset[split])
    logger.info(f"'{split}' split size after filtering: {split_size}")

    if split_size == 0:
        raise ValueError(f"'{split}' split has no valid samples after filtering.")

# 토크나이저를 사용하여 데이터셋 토크나이징
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

logger.info("Tokenizing dataset...")
tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)

# 필요없는 컬럼 제거
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# DataCollator 설정 (언어 모델링을 위한 데이터 콜레이터)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 설정
training_args = TrainingArguments(
    output_dir=r"C:\Falcon_AI",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,  # 더 자주 저장하도록 조정
    save_total_limit=2,
    learning_rate=5e-5,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="steps",
    logging_dir=r"C:\Falcon_AI\logs",
)

logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 모델 학습
logger.info("Starting training...")
trainer.train()

# 학습된 모델 저장
logger.info("Saving model and tokenizer...")
model.save_pretrained(r"C:\Falcon_AI")
tokenizer.save_pretrained(r"C:\Falcon_AI")
logger.info("Training and saving completed successfully.")

```
위 방법으로 데이터를 딥러닝 시켜 모델을 업그레이드 할 수 있다.  
이걸 계속 반복하는 것으로 로컬에 있는 내 AI는 점점 강해진다.  

## 3. 내가 원하는 대로
예를들어, 나의 AI가 이런 말투로 대답을 해 줬으면 좋겠다 싶은게 있으면 다음과 같은 방법으로 이를 구현할 수 있다.  

1. 데이터 수집
원하는 캐릭터의 어투와 성격을 반영한 데이터를 준비한다.  
자주 사용하는 표현, 단어, 무장 스타일을 포함하여 텍스트 데이터를 만들자.  
이걸 json이나, txt로 만들어 활용하면 도니다.  

txt 예시
```bash
안녕하세요! 오늘은 정말 기분이 좋아요! 당신은 어때요? 혹시 커피 한 잔 같이 마시고 싶으세요?
저는 완벽하지 않지만, 조금씩 나아지고 있어요!
기분 좋은 날에는 세상이 다 잘 되는 것 같죠? 계속 웃어요, 저는 항상 당신을 응원해요! 😁
오늘은 뭐하고 지내요?

```

json 예시
```json
{
  "conversations": [
    {
      "user": "안녕하세요! 오늘 기분 어때요?",
      "character": "안녕하세요! 오늘은 정말 기분이 좋아요! 당신은 어때요? 혹시 커피 한 잔 같이 마시고 싶으세요?"
    },
    {
      "user": "저 오늘 완전히 망했어요!",
      "character": "어머, 뭔 일이 있었나요? 아마도 그냥... 그럴 수도 있어요! 누구나 실수는 하니까요. 이제 그 실수도 웃을 수 있는 좋은 추억이 될 거예요! 😄"
    },
    {
      "user": "이렇게 복잡한 문제는 처음이에요.",
      "character": "복잡한 문제라니, 걱정 마세요! 사실, 인생도 복잡하긴 하죠! 😅 다 해결될 거예요, 제가 도와줄게요!"
    }
  ]
}

```

저기에 자주 사용하는 키워드 같은걸 집어넣어서 성격을 반영시킬 수 있다.  
위처럼 하면 대화내용, 어투, 특징적인 표형이 포함되게끔 하여 구현할 수 있다.  

2. 데이터 추가 학습  
이제, 파인튜닝 하자.  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 모델과 토크나이저 로드 (Falcon-7B)
model_name = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 캐릭터 대화 데이터셋 불러오기
dataset = load_dataset("text", data_files="character_data.txt")  # 캐릭터 데이터 경로 지정

# 텍스트 전처리 함수 (필요시 수정 가능)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 데이터셋 전처리 적용
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 트레이닝 설정
training_args = TrainingArguments(
    output_dir="./falcon_character_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# 모델 학습
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./falcon_character_finetuned")
tokenizer.save_pretrained("./falcon_character_finetuned")

```  
위 방법으로 추가적인학습을 시행한 모델을 만들 수 있다.  

3. 프롬프트 엔지니어링링  
후에 사용할 때, 질문시 약간 다르게 설정하여 캐릭터의 느낌도 살릴 수 있다.  
예를들어, 다음과 같이 할 수 있다.  

```python
prompt = "주인공처럼 말해줘: 안녕하세요! 오늘 기분이 어떤가요?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

```  

전체 코드로 보면,  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Falcon-7B 모델과 토크나이저 로드
model_name = "tiiuae/falcon-7b"  # Falcon-7B 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 프롬프트 정의
prompt = "주인공처럼 말해줘: 안녕하세요! 오늘 기분이 어떤가요?"

# 입력 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 모델 실행 및 출력 생성
output = model.generate(**inputs, max_length=50)

# 출력 디코딩
print(tokenizer.decode(output[0], skip_special_tokens=True))

```
이런식이다.  
팁을 주자면, 캐릭터가 자주 사용하는 특정 표현이나 반응을 데이터셋에 반복적으로 포함시키면 모델이 이를 자연스럽게 사용하게된다.  
감정을 살리고 싶으면 해당 감정을 나타내는 문장 패턴을 다양하게 포함해 모델이 이를 학습하도록 하면 된다.  

## 4. 코딩 지식 주입하기
위에서 한 것과 마찬가지로, 파인 튜닝을 통해서 너가 원하는 기능을 걔속 추가해 나갈 수 있다.  
이번엔 코딩 지식을 주입해 보자.  
이미 코딩 관련 데이터 셋은 엄청나게 많이 있다. 종류를 알아보자면
- CodeSearchNet
    깃허브에서 수집된 내용이며, Python, Java, JavaScript, PHP, Ruby, Go등을 포함한다.  
    사용은, 
    ```python
    from datasets import load_dataset

    dataset = load_dataset("code_search_net", "python")

    ```
    이렇게 할 수 있다.  

- Github Repositories
    깃허브의 오픈소스 프로젝트의 내용을 포함한다.  
    Hugging Face와 BigCode 프로젝트를 통해 데이터셋을 확보할 수 있다.  

- CodeParrot
    이건 파이썬 단일이며 파이썬 코드 생성에 최적화 시킬 수 있다.  
    ```python
    dataset = load_dataset("codeparrot/codeparrot-clean")
    ```

- Python Dataset by Hugging Face
    허깅페이스에서 수집된, 즉, ai 생성에 특화된 내용들이 담겨져 있다.  
    코드 예제와 주석, 문서화된 코드 등이 포함되어 있어 자연스럽고 유용한 코딩 표현을 학습하는 데 유리하다.  
    ```python
    dataset = load_dataset("codeparrot/github-code")
    ```

바로 사용할 수 있는 데이터 셋들로 구성하여 만들어 보자.  
```python
# 필요한 라이브러리 설치
!pip install transformers datasets torch

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch

# 모델과 토크나이저 로드 (Falcon-7B 모델)
model_name = "tiiuae/falcon-7b"  # Falcon-7B 모델 경로
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 1: 다양한 코딩 데이터셋 불러오기
print("Loading datasets...")

# CodeSearchNet 데이터셋 (Python 코드)
codesearchnet = load_dataset("code_search_net", "python", split="train[:5%]")

# CodeParrot 데이터셋 (Python 코드)
codeparrot = load_dataset("codeparrot/codeparrot-clean", split="train[:5%]")

# APPS 데이터셋 (Python 코드 문제와 솔루션)
apps = load_dataset("HuggingFaceH4/APPS", split="train[:5%]")

# 데이터셋 병합
dataset = concatenate_datasets([codesearchnet, codeparrot, apps])

# Step 2: 데이터 전처리 함수 정의
def preprocess_function(examples):
    # 코드가 있는 열을 선택 (dataset마다 열 이름이 다를 수 있음)
    code_samples = examples["code"] if "code" in examples else examples["text"]
    inputs = tokenizer(code_samples, padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # 언어 모델 학습을 위한 labels 설정
    return inputs

# 전체 데이터셋에 전처리 적용
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Step 3: 트레이닝 설정
training_args = TrainingArguments(
    output_dir="./falcon_code_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=10,
)

# Step 4: Trainer 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 학습 시작
print("Starting training...")
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./falcon_code_finetuned")
tokenizer.save_pretrained("./falcon_code_finetuned")
print("Training completed and model saved.")

```
위와 같은 방법으로 자신만의 AI를 계속해서 강화할 수 있다.