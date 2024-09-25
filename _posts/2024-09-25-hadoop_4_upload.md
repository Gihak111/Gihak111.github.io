---
layout: single
title:  "하둡 강좌 4편 HDFS"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 4편: **HDFS 구조 및 데이터 처리 실습**  
이번 강좌에서는 하둡 분산 파일 시스템인 **HDFS**(Hadoop Distributed File System)의 구조와 역할을 이해하고, 실제로 데이터를 업로드하고 관리하는 실습을 진행해보자.  
HDFS는 하둡에서 데이터를 저장하고 분산 처리할 수 있도록 설계된 파일 시스템이다.  

## 1. HDFS란?  
HDFS는 대용량 데이터를 저장하고 관리하는 분산 파일 시스템이다.  
여러 대의 서버에 데이터를 분산 저장하여, 데이터를 중복하고 장애에 대비할 수 있다.  
HDFS의 특징은 다음과 같다:  
- **대용량 파일 처리**: HDFS는 수 기가바이트에서 페타바이트에 이르는 대용량 파일을 처리할 수 있다.  
- **데이터 복제**: 기본적으로 각 데이터를 3개의 복제본으로 저장하여 고가용성을 제공한다.  
- **읽기 최적화**: 자주 읽는 데이터를 빠르게 읽을 수 있도록 최적화되어 있다.  

### 1.1 HDFS의 주요 컴포넌트  
- **NameNode**: 파일 시스템의 메타데이터를 관리한다. 파일의 위치, 크기, 블록 위치 등을 저장하며, 파일에 대한 모든 작업을 조정한다.  
- **DataNode**: 실제 데이터를 저장하는 서버이다. 각 DataNode는 NameNode의 명령에 따라 데이터를 저장하고 복제한다.  
- **Secondary NameNode**: NameNode의 데이터를 주기적으로 백업하는 역할을 한다.  

## 2. HDFS 파일 저장 및 블록 구조  
HDFS에서 파일은 여러 개의 **블록**으로 나뉘어 저장된다.  
기본 블록 크기는 128MB이며, 파일이 이 블록 단위로 분할되어 여러 DataNode에 분산 저장된다.  
이로 인해 대용량 데이터를 효율적으로 처리할 수 있다.  

### 2.1 파일 복제  
각 블록은 기본적으로 3개의 복제본으로 저장된다.  
이를 통해 서버 하나가 장애를 일으켜도 다른 서버에 저장된 복제본으로 데이터를 복구할 수 있다.  
복제본의 수는 설정에 따라 조정할 수 있다.  

## 3. HDFS 기본 명령어  
HDFS에서 파일을 관리하기 위한 기본 명령어들을 알아보자.  

### 3.1 HDFS에 파일 업로드  
로컬 파일 시스템에서 HDFS로 파일을 업로드하려면 다음 명령어를 사용한다.  
```bash
hdfs dfs -put <로컬 파일 경로> <HDFS 경로>
```  

예를 들어, `input.txt` 파일을 HDFS의 `/user/hadoop/` 디렉토리에 업로드하려면:  
```bash
hdfs dfs -put input.txt /user/hadoop/
```  

### 3.2 HDFS에서 파일 다운로드  
HDFS에서 로컬 파일 시스템으로 파일을 다운로드하려면 `-get` 명령어를 사용한다.  
```bash
hdfs dfs -get <HDFS 파일 경로> <로컬 파일 경로>
```  

예를 들어, HDFS에 저장된 `output.txt` 파일을 로컬로 가져오려면:  
```bash
hdfs dfs -get /user/hadoop/output.txt ./output.txt
```  

### 3.3 HDFS에서 파일 목록 보기  
HDFS에 저장된 파일 목록을 확인하려면 `-ls` 명령어를 사용한다.  
```bash
hdfs dfs -ls <HDFS 경로>
```  

예를 들어, `/user/hadoop/` 디렉토리의 파일 목록을 확인하려면:  
```bash
hdfs dfs -ls /user/hadoop/
```

### 3.4 HDFS 파일 삭제  
HDFS에서 파일을 삭제하려면 `-rm` 명령어를 사용한다.  
```bash
hdfs dfs -rm <HDFS 파일 경로>
```  
예를 들어, `/user/hadoop/input.txt` 파일을 삭제하려면:  
```bash
hdfs dfs -rm /user/hadoop/input.txt
```  

## 4. HDFS에 데이터 업로드 및 처리 실습  
이번 실습에서는 HDFS에 데이터를 업로드하고, 간단한 MapReduce 작업을 통해 데이터를 처리해보자.  

### 4.1 샘플 데이터 생성  
먼저 로컬에서 사용할 샘플 데이터를 생성하자.  
간단한 텍스트 파일을 생성해보자.  
```bash
echo "Hadoop is a big data framework" > input.txt
echo "HDFS is the storage layer of Hadoop" >> input.txt
echo "MapReduce processes the data in parallel" >> input.txt
```  

### 4.2 HDFS에 데이터 업로드  
생성한 `input.txt` 파일을 HDFS에 업로드해보자.  
```bash
hdfs dfs -put input.txt /user/hadoop/
```  
이 명령어를 통해 `input.txt` 파일이 HDFS의 `/user/hadoop/` 디렉토리에 저장된다.  

### 4.3 MapReduce 작업 실행  
이전에 작성한 WordCount MapReduce 프로그램을 사용하여 HDFS에 업로드한 데이터를 처리해보자.  
```bash
hadoop jar target/wordcount-1.0.jar WordCountDriver /user/hadoop/input.txt /user/hadoop/output
```  
위 명령어는 `input.txt` 파일을 입력으로 받아 단어 빈도수를 계산한 후, 결과를 `/user/hadoop/output` 디렉토리에 저장한다.  

### 4.4 결과 확인  
MapReduce 작업이 완료되면, 결과 파일을 확인해보자.  
```bash
hdfs dfs -cat /user/hadoop/output/part-r-00000
```  
이 명령어를 통해 각 단어의 빈도수가 출력될 것이다.  
```bash
Hadoop 1
is 2
a 1
big 1
data 1
framework 1
HDFS 1
the 2
storage 1
layer 1
of 1
MapReduce 1
processes 1
in 1
parallel 1
```  

## 5. HDFS의 고급 기능  
HDFS는 기본적인 파일 저장 및 관리 기능 외에도 여러 고급 기능을 제공한다.  

### 5.1 복제본 설정  
파일 업로드 시 복제본 수를 설정할 수 있다. 예를 들어, 파일을 업로드하면서 복제본 수를 2개로 설정하려면:  
```bash
hdfs dfs -D dfs.replication=2 -put input.txt /user/hadoop/
```

### 5.2 HDFS 웹 인터페이스  
HDFS는 웹 기반의 사용자 인터페이스를 제공한다. 브라우저에서 `http://<NameNode_IP>:50070` 주소에 접속하면, 파일 시스템 상태, 복제본 상태, 클러스터 상태 등을 확인할 수 있다.  

## 6. 마무리  
이번 강좌에서는 HDFS의 구조와 주요 기능을 살펴보고, 기본 명령어를 통해 데이터를 업로드하고 처리하는 방법을 익혔다.  
HDFS는 하둡에서 데이터를 저장하고 분산 처리하는 핵심 역할을 하며, 대용량 데이터 처리에 적합한 파일 시스템이다.  
다음 강좌에서는 하둡의 또 다른 중요한 컴포넌트인 **YARN**(Yet Another Resource Negotiator)을 다루고, 이를 통해 리소스 관리와 작업 스케줄링을 배우도록 하겠다.  

---

다음 강좌에서는 **YARN의 개념 및 사용법**에 대해 알아보자. YARN은 하둡 클러스터에서 자원을 관리하고, 작업을 실행하는 데 중요한 역할을 한다.  