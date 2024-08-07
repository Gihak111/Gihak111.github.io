---
layout: single
title:  "spring boot 8. AI모델과 병합"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# Ai와 병합  
계속된 발전으로 인해 많은 Ai가 만들어 졌다.  
Ai를 배포하려면, 허깅페이스나 깃허브에 올리는 방법도 있지만, 직접 밴앤드를 만들어 배포하는 방법도 있다.  
이번엔 간단히 my_modle.keras 라는 텍스트 분류 모델이 있다고 생각하고, 코드를 만들어 보겠다.  

다음 기준으로 파일을 만들어 보자.  
Group: com.example  
Artifact: aicall  
Name: AI Model Caller  
Description: A project to call AI model using Python script  
Package name: com.example.aicall  

Dependencies  
Spring Web  
Spring Boot DevTools  

만들 파일 구조는 다음 같다.  
```arduino
aicall
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── aicall
│   │   │               ├── AicallApplication.java
│   │   │               ├── controller
│   │   │               │   └── AIController.java
│   │   │               └── service
│   │   │                   └── AIService.java
│   │   └── resources
│   │       └── application.properties
│   └── python-scripts
│       ├── my_model.keras
│       └── predict.py
└── pom.xml

```  

predict.py 파일은 다음과 같은 코드를 가지고 있어야 한다.  
```python
import sys
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 모델과 토크나이저 로드
model = load_model('src/main/python-scripts/my_model.keras')
tokenizer = Tokenizer()
# 토크나이저를 미리 훈련된 상태로 로드하는 과정이 필요.
# 예: tokenizer.fit_on_texts(training_texts)

def classify_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # 시퀀스 길이는 모델 훈련 시 사용된 길이와 일치해야 한다.
    prediction = model.predict(padded_sequences)
    label = 'positive' if prediction > 0.5 else 'negative'
    return label

if __name__ == "__main__":
    input_text = sys.argv[1]
    result = classify_text(input_text)
    output = {"text": input_text, "classification": result}
    print(json.dumps(output))

```
모델을 로드하고 이용한는 간단한 코드이다.  

이어서, 스프링 부트 코드이다.  
익숙한 코드일 것이다.  
AicallApplication.java  
```java
package com.example.aicall;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class AicallApplication {

    public static void main(String[] args) {
        SpringApplication.run(AicallApplication.class, args);
    }
}

```

AIController.java  
```java
package com.example.aicall.controller;

import com.example.aicall.service.AIService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class AIController {

    @Autowired
    private AIService aiService;

    @PostMapping("/classify")
    public Map<String, String> classifyText(@RequestBody Map<String, String> request) throws IOException {
        String text = request.get("text");
        return aiService.classifyText(text);
    }
}

```

AIService.java
```java
package com.example.aicall.service;

import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

@Service
public class AIService {

    public Map<String, String> classifyText(String text) throws IOException {
        // 파이썬 스크립트를 호출하여 결과를 받아옵니다.
        String pythonScriptPath = "src/main/python-scripts/predict.py"; // 실제 파이썬 스크립트 경로
        ProcessBuilder processBuilder = new ProcessBuilder("python", pythonScriptPath, text);
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        // 파이썬 스크립트의 출력을 읽습니다.
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line);
        }

        // JSON 형식의 출력을 파싱하여 반환합니다.
        Map<String, String> result = new HashMap<>();
        result.put("text", text);
        result.put("classification", output.toString());

        return result;
    }
}

```  
이 코드를 통해서 아까 만든 파이썬 스크립트와 통신한다.  

위의 구성으로 스프링 부트를 실행하고, Postman 또는 cURL을 사용하여 /api/classify 엔드포인트에 POST 요청을 보내는 것으로 태스트 할 수 있다.  
예시
```json
{
  "text": "This is a good example."
}
```
```json
{
  "text": "This is a good example.",
  "classification": "positive"
}
```

위와 같은 방식으로, Ai 오멜을 사용할 수 있다.