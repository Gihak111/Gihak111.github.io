---
layout: single
title:  "매트랩 트랜스포머 활용시 특성 줄이기 오류"
categories: "Matlab"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 매트랩으로 AI 구현
매트랩으로 트랜스포머 모델을 구현하려면TransformerLayer와 wordEmbeddingLayer를 직접 구현해야 한다.
애초에 트랜스포머 모델을 자 ㄹ활용할 수 없는 환경인 만큼 여러 오류가 난다.  
그 중에서 한가지 해결하지 못한 오류가 있는데, 특성 줄이기 이다.
다음 코드는 FlattenLayer로 특성을 줄여서 ClassificationLayer로 전달한다.  
FlattenLayer는 입력으로 128 * 128인 구조이며 출력인 Z는 특성 곱하기 1의 형식으로 출력한다.  
이후에 ClassificationLayer는 FlattenLayer가 만든 Z를 입력으로 받는데, 여기서 오류가 나온다.

# 코드
```c
classdef FlattenLayer < nnet.layer.Layer
    methods
        function layer = FlattenLayer(name)
            % 생성자: 이름 설정
            if nargin > 0
                layer.Name = name;
            end
        end
        
        function Z = predict(~, X)
            disp(size(X))
            Z = reshape(X, [], 1);  % X를 [특징 수 x 1] 형태로 변환
            disp(size(Z))
            %ClassificationOutputLayer는 하나의 차원만 가져야 한다.
        end
    end
end

```
다음의 코드를 실행하면 트랜스포머 레이어를 포함한 모든 레이어가 잘 작동하고, 마지막 출력층에서 다음과 같은 오류가 나온다.  
   128   128  

       16384           1  

다음 사용 중 오류가 발생함: trainNetwork (191번 라인)  
신경망이 유효하지 않습니다.  

오류 발생: train_transformer_model (64번 라인)  
    model = trainNetwork(X, Y, layers, options);  

오류 발생: start (12번 라인)  
train_transformer_model(X, Y, enc); % 모델 학습 및 저장  

원인:  
    계층 7: Invalid input data for classification layer. The input data must have  
    spatial dimension sizes equal to one.  

해당 오류는 7계층인 ClassificationLayer에 들어오는 값이 잘못된 형식이기 때문에 생기는 오류이다.  

 