---
layout: single
title:  "파이썬으로 만드는 간단한 AI. 트랜스포머 모듈"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

#
해커뉴스와 레딧의 개시글을 크롤링 해와서 요약, 정리 해주는 코드 입니다.  


# 코드
```python
import requests
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch

# 해커뉴스 크롤링
def hackernews_crawler():
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty'
    response = requests.get(url)
    story_ids = response.json()
    hn_data = []
    
    for i in range(min(10, len(story_ids))):
        story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_ids[i]}.json?print=pretty'
        story = requests.get(story_url).json()
        hn_data.append({'title': story.get('title'), 'link': story.get('url')})
    
    return hn_data

# 레딧 크롤링
def reddit_crawler():
    url = 'https://www.reddit.com/r/news/top/.json?limit=10'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    data = response.json()
    reddit_data = []

    for post in data['data']['children']:
        title = post['data']['title']
        link = post['data']['url']
        reddit_data.append({'title': title, 'link': link})
    
    return reddit_data

# 텍스트 요약 함수
def summarize_texts(texts, summarizer):
    summaries = summarizer(texts, max_length=50, min_length=25, do_sample=False)
    return [summary['summary_text'] for summary in summaries]

# 텍스트 분석 함수
def analyze_texts(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.tolist()

# 메인 함수
def main():
    # 데이터 크롤링
    hackernews_data = hackernews_crawler()
    reddit_data = reddit_crawler()

    hackernews_titles = [item['title'] for item in hackernews_data]
    reddit_titles = [item['title'] for item in reddit_data]

    # 요약 파이프라인 로드
    summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")

    # 텍스트 요약
    summarized_hackernews = summarize_texts(hackernews_titles, summarizer)
    summarized_reddit = summarize_texts(reddit_titles, summarizer)

    # 모델 로드 및 텍스트 분석
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    analyzed_hackernews = analyze_texts(summarized_hackernews, model, tokenizer)
    analyzed_reddit = analyze_texts(summarized_reddit, model, tokenizer)

    # 결과 출력
    print("Hacker News Summaries and Analysis:")
    for item, summary, analysis in zip(hackernews_data, summarized_hackernews, analyzed_hackernews):
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Summary: {summary}")
        print(f"Analysis: {'Positive' if analysis == 1 else 'Negative'}")
        print()

    print("Reddit Summaries and Analysis:")
    for item, summary, analysis in zip(reddit_data, summarized_reddit, analyzed_reddit):
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Summary: {summary}")
        print(f"Analysis: {'Positive' if analysis == 1 else 'Negative'}")
        print()

if __name__ == "__main__":
    main()
```