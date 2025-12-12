---
layout: single
title: "LAM"
categories: "AI"
tag: "Framework"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## LAM
Large Action Models는 텍스트 생성에만 머물던 기존 거대 언어 모델의 한계인 생각과 행동의 괴리를 해결하기 위해 제안된 프레임워크이다.  
기존 모델들은 복잡한 추론은 잘해내지만, 실제로 웹사이트 버튼을 클릭하거나 API를 호출하는 등의 구체적인 실행 단계에서는 환각(Hallucination)을 보이거나 인터페이스를 이해하지 못하는 경우가 빈번했다.  
LAM은 언어적 의도(Intent)를 구체적인 실행 가능한 행동(Executable Action)으로 변환하는 데 특화되어 있으며, 신경망의 추론 능력과 심볼릭 시스템의 정확성을 결합하여 디지털 에이전트의 실용성을 비약적으로 향상시켰다.  

## 2. 수식적 원리: 궤적 모델링과 결정 과정
LAM의 이론적 토대는 마르코프 결정 과정(MDP)에 기반한 순차적 행동 생성 확률 분포이다.  
사용자의 목표 $g$를 달성하기 위해, 단일 응답 $y$ 대신 일련의 행동 궤적(Trajectory) $\tau = \{a_1, s_1, a_2, s_2, ..., a_T, s_T\}$를 생성한다.  

행동 생성 과정은 현재 상태 $s_t$와 이전까지의 히스토리 $h_{t-1}$이 주어졌을 때, 최적의 행동 $a_t$를 선택하는 정책 함수(Policy Function)로 표현된다.  

$$\pi_\theta(a_t | s_t, g, h_{t-1}) = \text{softmax}(f_\theta(s_t, g, h_{t-1}))$$

여기서 $s_t$는 현재의 화면 상태(Screenshot, DOM Tree 등)를 의미하며, $a_t$는 '클릭', '타이핑', '스크롤'과 같은 이산적인 행동 공간(Action Space) 내의 원소이다.  
기존 LLM이 $P(text|prompt)$를 학습했다면, LAM은 동적인 환경과의 상호작용을 포함하는 **상태 전이 확률(State Transition Probability)**을 내재화하여, 행동의 결과까지 예측하며 최적의 경로를 탐색한다.  

## 3. 학습 방법론: Imitation Learning과 Online RL
LAM은 인간의 행동 패턴을 모방하는 동시에, 실패했을 때 스스로 교정하는 능력을 학습해야 한다.  
이를 위해 **Behavioral Cloning(BC)**과 **Reinforcement Learning(RL)** 전략을 혼합하여 사용한다.  

학습 손실 함수는 전문가의 데모를 따르는 지도 학습 손실과, 목표 달성 여부에 따른 보상 손실의 합으로 구성된다.  

$$\mathcal{L}_{LAM} = \mathcal{L}_{BC}(\tau_{demo}, \pi_\theta) - \lambda \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

1.  **Demonstration Pre-training**: 인간이 실제로 웹사이트나 앱을 조작하는 로그 데이터를 통해 기본적인 UI 조작법과 워크플로우를 학습한다.  
2.  **Environment-Aware Fine-tuning**: 시뮬레이터 환경에서 모델이 직접 행동해보고, 에러가 발생하거나 목표를 달성하지 못했을 때 페널티를 부여하여 강건성(Robustness)을 기른다.  
3.  **최적화**: 행동 공간이 매우 크기 때문에, 모든 행동을 탐색하기보다 유효한 행동(Valid Actions) 마스킹을 적용하여 탐색 효율을 높이는 방식을 취한다.  

## 4. 핵심 기술: 뉴로-심볼릭 그라운딩과 DOM 파싱
LAM이 실제 소프트웨어를 제어할 수 있는 비결은 두 가지 핵심 기술에 있다.  

* **멀티모달 UI 인코더 (Multimodal UI Encoder)**:
    단순히 화면을 이미지로 보는 것이 아니라, HTML DOM 트리나 접근성 트리(Accessibility Tree)를 함께 분석하여 버튼의 기능적 의미를 파악한다.  
    이는 $s_t$가 단순한 픽셀 덩어리가 아닌, "로그인 버튼(ID: submit_btn)"과 같은 구조적 정보를 담도록 강제한다.  
* **뉴로-심볼릭 그라운딩 (Neuro-Symbolic Grounding)**:
    모델이 생성한 자연어 계획을 실제 시스템이 이해할 수 있는 코드나 좌표로 변환한다.  
    추상적인 명령("비행기표 예매해줘")을 구체적인 함수 호출(`select_date('2025-12-11')`, `click('#search')`)로 맵핑하여 실행의 정확성을 보장한다.  

## 5. 아키텍처 확장: LAM-Verifier
LAM의 구조는 잘못된 행동이 시스템에 치명적인 결과를 초래할 수 있다는 리스크를 안고 있다.  
이를 보완한 **LAM-Verifier** 아키텍처는 행동 실행 직전에 검증 단계를 추가한다.  

$$a_{final} = \begin{cases} a_{pred} & \text{if } \mathcal{V}(a_{pred}, s_t) > \delta \\ \text{abort} & \text{otherwise} \end{cases}$$

* **안전성 제어**: 결제나 데이터 삭제와 같은 민감한 작업(Critical Action)이 감지되면, 별도의 검증 모듈 $\mathcal{V}$가 개입하여 사용자의 추가 승인을 요구하거나 룰 기반의 안전 장치를 작동시킨다.  
* **자기 교정 (Self-Correction)**: 행동 실행 후 화면 상태가 예상과 다를 경우(예: 팝업창 발생), 즉시 이전 단계로 롤백하거나 우회 경로를 생성하는 루프를 포함한다.  
    이러한 특성 덕분에 LAM은 복잡한 RPA(Robotic Process Automation) 작업에서 기존 스크립트 방식보다 월등한 유연성을 보인다.  

## 6. 추론 및 Interleaved Execution
LAM의 추론 과정은 생각과 행동이 교차하는 **Interleaved Execution** 방식을 따른다.  

1.  **Perception**: 현재 화면 상태를 캡처하고 DOM 트리를 파싱하여 구조화된 상태 $s_t$를 얻는다.  
2.  **Reasoning**: 목표 $g$와 비교하여 다음 행동 $a_t$를 계획한다. 이때 CoT(Chain of Thought)를 통해 "장바구니가 비었으니 상품을 먼저 담아야 한다"는 식의 내부 추론을 거친다.  
3.  **Actuation**: 결정된 행동을 마우스/키보드 이벤트나 API 호출로 실행한다.  
4.  **Observation**: 행동의 결과로 변한 화면 $s_{t+1}$을 다시 관측하고, 목표 달성 시까지 1~3 과정을 반복한다.  

이 방식은 API 연동이 없는 레거시 소프트웨어까지 제어할 수 있다는 강력한 장점이 있지만,  
실시간으로 화면을 분석해야 하므로 추론 지연(Latency)이 발생할 수 있다.  

## 결론
LAM은 "말만 번지르르하게 하는" 기존 언어 모델의 한계를 넘어, 실제로 손발이 되어 움직이는 행동 중심의 아키텍처이다.  
디지털 비서가 단순히 정보를 검색하는 것을 넘어 예약, 주문, 업무 자동화까지 수행할 수 있다는 점에서 AGI로 가는 중요한 교두보라 할 수 있다.  
특히 Rabbit R1 같은 디바이스나 차세대 OS 인터페이스에서 그 가능성이 주목받고 있다.  
하지만 UI가 조금만 바뀌어도 모델이 헤매는 경우가 많고, 속도와 안정성 면에서 아직 사람이 직접 클릭하는 것을 따라가기 벅차다.  
결국 연구실에서는 LAM으로 'General Agent' 논문을 쓰지만, 현업에서는 그냥 셀레니움(Selenium)이나 매크로 짠다.  
매크로 만큼의 가성비가 없다 진짜 사기는 사실 AI고 나발이고 원초적인 매크로다  
그게 제일 싸고 확실하다.  