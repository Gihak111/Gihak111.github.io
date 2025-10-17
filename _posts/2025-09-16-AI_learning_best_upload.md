---
layout: single
title:  "AI 입문용 글 모음"
categories: "AI"
tag: "Explanation"
toc: true
author_profile: false
sidebar:
nav: "docs"
---

## AI YEAH  
최근 들어서, AI 기초에 대한 글을 정말 많이 만들었다.  
이 글들을 누구나 한 눈에 볼 수 있게 정리해 두었다.  


## AI 책 추천  
[책 추천 1편](https://gihak111.github.io/ai/2025/09/14/Best_Ai_Book_upload.html)  


## AI 논문 리뷰  
<ul style="list-style-type: none;">
{% for post in site.tags.review %}
  <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>  


## AI Basics  
<ul style="list-style-type: none;">
{% for post in site.tags.Explanation %}
  <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>  


## Linear Algebra  
<ul style="list-style-type: none;">
{% for post in site.tags.['linear algebra'] %}
  <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>  


## AI 강의
[링크](https://gihak111.github.io/ai/2025/09/26/AI_class_best_upload.html)  


## AI Architecture  
<ul style="list-style-type: none;">
{% for post in site.tags.Architecture %}
  <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>  


## AI 직접 구현  
[Bert](https://huggingface.co/gihakkk/bert_nupy_model)  
[MLP](https://huggingface.co/gihakkk/MLP_test)