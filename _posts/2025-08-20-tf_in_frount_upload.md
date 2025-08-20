---
layout: single
title:  "프론트 엔드에서 모델 딥러닝시의 텐서플로우 로드"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 딥러닝
앞서서, 나는 인덱스 디비를 소개한 적 있다.  
이는, 디비를 대신해서 사용할 위치를 찾기 위함이였는데,  
그 글을 보고 백엔드에서 딥러닝 진행 -> 인덱스 디비에 저장으 ㄹ시도해 본 사람들은 이가 불가능 하단것을 깨닭았을 것이다.  
인덱스 디비에 모델을 저장하기 위해선, 브라우저에서 모델 ㄷ비러닝이 돌아가야 한다.  
이를 위해 딥러닝 코드를 만들어서 따로 빼고, 이를 브라우저에서 불러서 돌리기 위해 임포트 tensorflow을 시도하면, 당연히 오류난다 이는 브라우저에서 돌아가기 때문에, 당연한 거다.  

## 그러면 어떻게 해야 텐서플로우를 넘길 수 있을까??
방법은 간단하다  
프론트엔드 코드를 main.js, 프롵느 엔드의 html 파일을 index.html, 딥러닝 코드를 depp_trainer.js라 하자.  
그러면, 다음과 같은 코드를 index.html에 넣는다.  
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
```
위 코드는 설치 없이 tf를 사용할 수 있게 해주는 코드이다 저걸 헤더에 넣으면 된다.  

이어서, 프론트 엔드 코드에선 저 html 을 참조하므로, tf 함수를 사용할 수 있다.  

그러면, main.js에서 depp_trainer.js의 함수를 사용할 떄 tf 인자를 넘겨준다면?  
당연히 depp_trainer.js 내부 코드에서도 tf를 사용할 수 있다. 넘기는 것도 간단하다 그냥 함수 선언 을 이런식으로 하고, 넘길때도 같은 방식으로 넘기면 그만이다.  
```javascript
async function trainTextModel(tf, modelInfo, onProgress, supabase, createSupabaseIOHandler) {
```

이런 방법으로 tf 뿐 만 아니라 슈파베이스도 넘길 수 있다 이 얼마나 좋은 방식이냐

물론, 딥러닝 이라는 것 자체가 브라우저에서 돌릴 만한 것도 아니고,  
백엔드에서 돌리는게 맞지만,  
나처럼 어쩔 수 없는 상황이거나,  
또는 인자를 불러와야 할 경우 이런 방식을 사용하면 편하다는걸 알리고 싶었다.  

## 결론
이게 변수들은 백 <-> 프론트 이동이 자유롭지만, 이거도 많이 하는게 좋지도 않고  
또 이런 의존성은 안넘어가지고 import 나 requore 도 안되니까 이런 방식 쓰면 될 것 같다.  