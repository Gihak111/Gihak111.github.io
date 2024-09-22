---
layout: single
title:  "하둡 강좌 2편 HDFS 기본 개념"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 2편: **HDFS 파일 업로드 및 파일 시스템 명령어**  
이번 강좌에서는 하둡의 분산 파일 시스템인 HDFS(Hadoop Distributed File System)에 대해 알아보고, 데이터를 업로드하고 관리하는 방법을 다루겠다. 또한 HDFS에서 사용할 수 있는 다양한 명령어들을 살펴보자.  

## 1. HDFS 기본 개념  

HDFS는 대규모 데이터를 분산 저장하는 시스템이다.   
데이터를 여러 블록으로 나누어 여러 노드에 분산하여 저장하며, 데이터를 복제해 데이터 손실에 대비한다.  
HDFS의 기본 구조는 크게 **NameNode**와 **DataNode**로 이루어져 있다.  

- **NameNode**: 파일과 블록의 메타데이터를 관리한다.  
- **DataNode**: 실제 데이터를 블록 단위로 저장한다.  

## 2. 파일 업로드 및 다운로드  
HDFS에 데이터를 업로드하는 가장 기본적인 작업을 해보자.  

### 2.1 HDFS에 파일 업로드  
로컬 파일을 HDFS에 업로드하기 위해 다음 명령어를 사용한다. 이때, `put` 명령을 사용해 로컬 파일을 HDFS에 업로드한다.  
```bash
hdfs dfs -put /path/to/local/file /user/hadoop/directory/
```

예시:  
```bash
hdfs dfs -put ~/sample.txt /user/hadoop/
```

이 명령어는 `sample.txt` 파일을 HDFS의 `/user/hadoop/` 디렉토리에 업로드한다.  

### 2.2 HDFS에서 파일 확인  
파일이 제대로 업로드되었는지 확인하려면 `ls` 명령을 사용한다.  
```bash
hdfs dfs -ls /user/hadoop/
```

HDFS 경로 내 파일 목록을 확인할 수 있다.  

### 2.3 HDFS에서 파일 다운로드  
HDFS에 있는 파일을 로컬로 다운로드하려면 `get` 명령을 사용한다.  
```bash
hdfs dfs -get /user/hadoop/sample.txt /path/to/local/directory/
```  
이 명령은 HDFS에 있는 `sample.txt` 파일을 로컬 경로로 다운로드한다.  

## 3. HDFS 파일 시스템 명령어  
HDFS에서 사용할 수 있는 다양한 파일 관리 명령어들을 살펴보자.

### 3.1 디렉토리 생성  
HDFS에 새 디렉토리를 생성하려면 `mkdir` 명령을 사용한다.  
```bash
hdfs dfs -mkdir /user/hadoop/new_directory
```  
이 명령은 HDFS에 `new_directory`라는 디렉토리를 생성한다.  

### 3.2 파일 삭제  
HDFS에서 파일을 삭제하려면 `rm` 명령을 사용한다.  
```bash
hdfs dfs -rm /user/hadoop/sample.txt
```  
이 명령은 HDFS에 있는 `sample.txt` 파일을 삭제한다.  
디렉토리를 삭제하려면 `-r` 옵션을 사용해 재귀적으로 삭제할 수 있다.  
```bash
hdfs dfs -rm -r /user/hadoop/new_directory
```  

### 3.3 파일 내용 확인  
HDFS에 있는 파일의 내용을 확인하려면 `cat` 명령을 사용한다.  
```bash
hdfs dfs -cat /user/hadoop/sample.txt
```  
이 명령은 HDFS에 있는 `sample.txt` 파일의 내용을 출력한다.  

### 3.4 파일 복사  
HDFS에서 파일을 복사하려면 `cp` 명령을 사용한다.  
```bash
hdfs dfs -cp /user/hadoop/sample.txt /user/hadoop/sample_copy.txt
```  
이 명령은 `sample.txt` 파일을 `sample_copy.txt`라는 이름으로 복사한다.  

### 3.5 파일 이동  
HDFS 내에서 파일을 이동하려면 `mv` 명령을 사용한다.  
```bash
hdfs dfs -mv /user/hadoop/sample.txt /user/hadoop/archive/sample.txt
```  
이 명령은 `sample.txt` 파일을 `archive` 디렉토리로 이동한다.  

### 3.6 파일 크기 확인  
HDFS에 저장된 파일의 크기를 확인하려면 `du` 명령을 사용한다.  
```bash
hdfs dfs -du /user/hadoop/sample.txt
```  
이 명령은 `sample.txt` 파일의 크기를 출력한다.  

## 4. HDFS 사용 예시  
### 4.1 파일 업로드 및 다운로드  
로컬에 있는 파일을 업로드하고, 다시 다운로드해보자.  
1. `sample.txt` 파일을 로컬에서 HDFS로 업로드한다.  
```bash
hdfs dfs -put ~/sample.txt /user/hadoop/
```  

2. HDFS에 업로드된 파일을 확인한다.  
```bash
hdfs dfs -ls /user/hadoop/
```  

3. 업로드된 파일을 로컬로 다시 다운로드한다.  
```bash
hdfs dfs -get /user/hadoop/sample.txt ~/downloaded_sample.txt
```  

4. 파일 내용을 확인한다.  
```bash
hdfs dfs -cat /user/hadoop/sample.txt
```  

## 5. 마무리  

이번 강좌에서는 HDFS에 파일을 업로드하고, 파일 시스템 명령어를 사용하는 방법을 알아보았다.  
다음 강좌에서는 하둡의 주요 컴포넌트인 **MapReduce**를 사용하여 데이터를 처리하는 방법을 알아보겠다.

---

다음 강좌에서는 MapReduce를 통한 데이터 처리 방법을 다루면서, 하둡의 핵심 기능을 이해하는 데 도움을 줄 것이다.  