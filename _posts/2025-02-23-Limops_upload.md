---
layout: single
title:  "LLMOps 플랫폼"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## LangWatch: LLM 모니터링 및 최적화를 위한 LLMOps 플랫폼

### LangWatch 소개
LangWatch는 LLM 애플리케이션을 운영하는 개발자와 연구자를 위한 모니터링 및 최적화 플랫폼이다.  
일반적으로 LLM(대형 언어 모델)을 활용한 애플리케이션을 운영할 때 성능 최적화와 모니터링이 중요한 과제로 떠오른다.  

LangWatch는 이러한 문제를 해결하기 위해 Stanford DSPy 프레임워크 기반의 시각적 인터페이스를 제공하며, LLM 파이프라인의 성능을 분석하고 개선할 수 있는 올인원 솔루션을 지원한다.   
특히, LLM 워크플로우의 모니터링, 실험, 평가 및 최적화를 지원하여 단순 배포를 넘어 비용 분석, 성능 평가, 품질 보장, 실험 및 개선 등의 기능을 제공하여 보다 효율적인 운영이 가능하도록 한다.  

### LangWatch 주요 기능

#### LLM 최적화 스튜디오
- Stanford DSPy 기반의 Drag-and-Drop 방식 최적화 인터페이스를 제공한다.  
- 자동 프롬프트 생성 및 Few-shot 예제를 자동으로 추천한다.  
- 실험 추적 및 버전 관리 기능을 지원하여 성능 평가를 돕는다.  

#### 품질 보장 (QA)
- 30가지 이상의 사전 구축된 평가 지표를 제공한다.  
- 사용자가 직접 평가 지표를 정의할 수 있는 도구(Builder)를 지원한다.  
- 데이터셋 관리 및 안전성 검사를 제공하며, DSPy 시각화를 지원한다.  

#### 모니터링 & 분석
- 비용 및 성능 추적 기능을 지원한다.  
- 실시간 디버깅 및 추적이 가능하다.  
- 사용자 분석, 커스텀 지표, 맞춤형 대시보드 및 알림 시스템을 제공한다.  

### LangWatch 사용 방법
LangWatch Cloud에 가입하여 즉시 사용할 수 있으며, 로컬 실행을 원할 경우 다음과 같이 Docker를 활용하면 된다:

```bash
git clone https://github.com/langwatch/langwatch.git
cp langwatch/.env.example langwatch/.env
docker compose up --build
```

이후, `http://localhost:5560`에서 LangWatch에 접속할 수 있다.
기업 환경에서 자체 서버에 배포하여 운영하려면 Self-Hosting 가이드를 참고해야 한다.

### 라이선스
LangWatch 프로젝트는 Business Source License 1.1에 따라 배포된다.   
이 라이선스는 오픈소스 라이선스가 아니며, 비영리적 사용(Non-Production Use)에 한해서만 복사, 수정, 파생 저작물 생성 및 재배포가 가능하다.  

LangWatch 프로젝트의 상업적 사용은 금지되며, 이를 위해서는 별도의 상용 라이선스를 구매해야 한다.  
자세한 내용은 라이선스 원문 및 LangWatch 홈페이지 하단의 Contact 메뉴를 참고하면 된다.  

### 참고 자료  
- [LangWatch 공식 홈페이지](https://langwatch.ai)  
- [LangWatch GitHub 저장소](https://github.com/langwatch/langwatch)  
- [LangWatch 공식 문서](https://docs.langwatch.ai)  

