---
layout: single
title:  "하둡 강좌 5편 YARN"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 5편: **YARN 개념 및 사용법**  
이번 강좌에서는 하둡에서 리소스 관리 및 작업 스케줄링을 담당하는 **YARN**의 개념과 작동 원리를 설명하고, YARN을 통해 작업을 실행하는 실습을 해보자.  

## 1. YARN이란?  
YARN은 하둡의 핵심 컴포넌트로, 클러스터의 리소스를 관리하고 작업을 스케줄링하는 역할을 한다.  
이전에는 하둡의 **MapReduce**가 리소스 관리와 작업 실행을 모두 담당했지만, **하둡 2.x** 버전부터 YARN이 리소스 관리 역할을 분리해 더욱 효율적인 분산 처리가 가능하게 되었다.  

### 1.1 YARN의 주요 역할  
- **리소스 관리**: 클러스터의 CPU, 메모리 등의 리소스를 관리하고 할당.  
- **작업 스케줄링**: 여러 애플리케이션 간 리소스를 균등하게 분배하고, 작업을 스케줄링.  
- **확장성**: 다양한 유형의 분산 처리 애플리케이션을 지원하여 확장성이 향상.  

## 2. YARN의 구조
YARN의 구조는 크게 **ResourceManager**, **NodeManager**, **ApplicationMaster** 세 가지 컴포넌트로 나눌 수 있다.  

### 2.1 ResourceManager
YARN의 중앙 관리자로, 클러스터 전체의 리소스를 관리하고 스케줄링하는 역할을 한다.  
ResourceManager는 다음 두 가지 핵심 기능을 담당한다.  
- **Scheduler**: 리소스를 여러 작업에 분배하는 역할을 한다. 작업의 우선순위를 고려하여 리소스를 할당한다.  
- **ApplicationManager**: 애플리케이션의 라이프사이클을 관리하며, 각 애플리케이션의 실행을 조정한다.  

### 2.2 NodeManager
NodeManager는 각 노드에서 실행되며, 해당 노드의 리소스를 관리하고, ResourceManager로부터 받은 명령에 따라 컨테이너(Container)를 시작하고 종료하는 역할을 한다.  

### 2.3 ApplicationMaster  
각 애플리케이션은 자체적인 **ApplicationMaster**를 가지며, 애플리케이션의 실행과 리소스 요구를 관리한다.  
각 작업의 실행을 조율하고, ResourceManager와 통신하여 필요한 리소스를 요청한다.  

## 3. YARN 실행 흐름  
YARN에서 작업이 실행되는 기본적인 흐름은 다음과 같다:  
1. 클라이언트가 **ResourceManager**에게 애플리케이션을 제출한다.  
2. ResourceManager는 클러스터에서 리소스를 할당하여, 애플리케이션을 관리할 **ApplicationMaster**를 시작한다.  
3. ApplicationMaster는 작업 실행을 위해 필요한 리소스를 **NodeManager**에 요청하고, 작업이 컨테이너(Container)에서 실행된다.  
4. 작업이 완료되면, ApplicationMaster는 ResourceManager에 완료 보고를 하고 종료된다.  

## 4. YARN 관련 주요 명령어  
하둡에서는 YARN과 관련된 다양한 명령어를 제공한다.  
YARN에서 작업을 실행하고 관리하는 방법을 알아보자.  

### 4.1 YARN 클러스터 상태 확인  
YARN 클러스터의 상태를 확인하려면 다음 명령어를 사용할 수 있다.  
```bash
yarn cluster
```  
이 명령어는 YARN 클러스터의 상태와 활성 노드, 작업 수 등을 보여준다.  

### 4.2 YARN 애플리케이션 상태 확인  
현재 실행 중인 애플리케이션을 확인하려면:  
```bash
yarn application -list
```  
이 명령어는 클러스터에서 실행 중인 모든 애플리케이션의 목록을 출력한다.  

### 4.3 YARN 애플리케이션 종료  
특정 애플리케이션을 종료하려면:  
```bash
yarn application -kill <application ID>
```  
애플리케이션 ID는 `yarn application -list` 명령어로 확인할 수 있다.  

### 4.4 YARN 작업 제출  
YARN에서 작업을 실행하려면 `yarn jar` 명령어를 사용하여 애플리케이션을 제출한다.  
```bash
yarn jar <JAR 파일 경로> <메인 클래스> <입력 파일 경로> <출력 경로>
```  
예를 들어, WordCount 프로그램을 YARN에서 실행하려면 다음과 같이 한다.  
```bash
yarn jar wordcount-1.0.jar WordCountDriver /user/hadoop/input /user/hadoop/output
```  
이 명령어는 HDFS에 저장된 `/user/hadoop/input` 데이터를 읽어 단어 빈도수를 계산하고, 결과를 `/user/hadoop/output` 디렉토리에 저장한다.  

## 5. YARN 실습: MapReduce 작업 실행  
이번 실습에서는 YARN을 통해 MapReduce 작업을 실행하고, 작업 상태를 모니터링하는 방법을 알아보자.  

### 5.1 데이터 준비
HDFS에 사용할 샘플 데이터를 업로드한다.  
```bash
echo "Hadoop YARN is a resource manager" > input.txt
echo "MapReduce runs on top of YARN" >> input.txt

hdfs dfs -put input.txt /user/hadoop/input
```

### 5.2 WordCount 프로그램 실행  
YARN을 통해 WordCount 프로그램을 실행해보자.  
```bash
yarn jar target/wordcount-1.0.jar WordCountDriver /user/hadoop/input /user/hadoop/output
```
이 명령어는 HDFS에서 입력 파일을 읽어 단어 빈도수를 계산한 후, 결과를 출력한다.  

### 5.3 작업 상태 확인  
실행 중인 애플리케이션의 상태를 확인하기 위해 다음 명령어를 사용한다.  
```bash
yarn application -list
```  
애플리케이션이 완료된 후에는 출력 디렉토리에서 결과를 확인할 수 있다.  
```bash
hdfs dfs -cat /user/hadoop/output/part-r-00000
```
출력은 다음과 같이 각 단어의 빈도수를 나타낸다.  
```text
Hadoop 1
YARN 2
is 1
a 1
resource 1
manager 1
MapReduce 1
runs 1
on 1
top 1
```

### 5.4 YARN 웹 인터페이스 사용  
YARN은 **웹 인터페이스**를 통해 애플리케이션 상태와 클러스터 상태를 모니터링할 수 있다. 브라우저에서 YARN의 웹 UI에 접속하려면:  
- ResourceManager UI: `http://<ResourceManager_IP>:8088`  
- NodeManager UI: `http://<NodeManager_IP>:8042`  
여기서 각 작업의 세부 정보, 리소스 사용량, 오류 메시지 등을 확인할 수 있다.  

## 6. YARN의 리소스 설정  
YARN은 클러스터의 리소스를 효율적으로 사용하기 위해 다양한 설정을 제공한다. 주요 설정은 다음과 같다:  
- **yarn.nodemanager.resource.memory-mb**: 각 NodeManager에서 사용할 수 있는 총 메모리 양.  
- **yarn.nodemanager.resource.cpu-vcores**: 각 NodeManager에서 사용할 수 있는 CPU 코어 수.  
- **yarn.scheduler.maximum-allocation-mb**: YARN이 각 작업에 할당할 수 있는 최대 메모리.  
이 설정들은 `yarn-site.xml` 파일에서 조정할 수 있다.  

## 7. 마무리  
이번 강좌에서는 YARN의 개념과 구조를 살펴보고, 실제로 YARN을 통해 작업을 실행하고 관리하는 방법을 배웠다.  
YARN은 하둡 클러스터에서 리소스를 관리하고 여러 작업을 동시에 실행할 수 있도록 도와주는 중요한 역할을 한다.  

---

다음 강좌에서는 **하둡 에코시스템**의 또 다른 중요한 컴포넌트인 **Hive**를 다루며, 대규모 데이터 쿼리 처리와 분석에 대해 알아보자.  