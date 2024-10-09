---
layout: single
title:  "하둡 간좌 1편"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 10편: **HDFS 데이터 압축**  
이번 강좌에서는 **HDFS (Hadoop Distributed File System)**에서 데이터를 **압축**하는 방법과 그로 인해 얻을 수 있는 장점들을 알아보겠다.  
대규모 데이터 처리에서는 **저장 공간 절약**과 **데이터 전송 속도 향상**이 중요한데, HDFS는 이러한 목표를 위해 다양한 압축 방식을 지원한다.  

## 1. 데이터 압축의 필요성  
대규모 데이터 처리를 하다 보면, 데이터 저장소와 네트워크 대역폭이 한계에 다다를 수 있다.  
데이터 압축은 이런 문제를 해결하는 데 도움을 준다.  
압축된 데이터는 더 적은 저장 공간을 차지하고, 네트워크를 통해 전송될 때에도 속도를 높일 수 있다.  

### 1.1 압축의 장점  
- **저장 공간 절약**: 데이터를 압축하면 파일 크기가 줄어들어, 동일한 용량의 디스크에서 더 많은 데이터를 저장할 수 있다.  
- **네트워크 트래픽 감소**: 압축된 파일은 크기가 작기 때문에 HDFS 클러스터 간 데이터 전송 시 네트워크 부하를 줄여준다.  
- **성능 향상**: 압축된 파일을 다룰 때 입출력(I/O) 성능이 향상되어 처리 속도가 빨라질 수 있다.  

## 2. 하둡에서 지원하는 압축 코덱  
하둡은 여러 종류의 압축 코덱을 지원하며, 각 코덱은 파일을 다른 방식으로 압축하고 해제한다.  
주요 압축 코덱은 다음과 같다:  
- **Gzip**: 매우 보편적인 압축 방식으로, 높은 압축률을 제공하지만 압축 및 해제 속도가 느리다.  
- **Bzip2**: Gzip보다 높은 압축률을 제공하지만, 속도는 상대적으로 느리다.  
- **LZO**: 빠른 압축 및 해제를 제공하지만 압축률은 낮다. 실시간 데이터 처리에 적합하다.  
- **Snappy**: 빠른 속도와 적절한 압축률을 제공하는 압축 코덱으로, 대규모 데이터 처리에 자주 사용된다.  
- **Zlib**: 압축률과 속도 간 균형을 유지하는 압축 방식이다.  

## 3. 압축 코덱 설정  
### 3.1 압축 코덱 설정 방법  
하둡에서 압축 코덱을 설정하는 것은 두 가지 방식으로 가능하다:  
1. **파일 수준에서 설정**: 특정 파일을 압축된 형식으로 저장하거나, 특정 파일에 대해 압축 코덱을 선택할 수 있다.  
2. **맵리듀스 작업에서 설정**: 맵리듀스 작업에서 출력 데이터를 압축할 때, 코덱을 선택할 수 있다.  

### 3.2 압축 코덱 설치  
Hadoop에는 기본적으로 Gzip, Bzip2 등이 포함되어 있지만, LZO나 Snappy와 같은 코덱은 추가 설치가 필요할 수 있다.  
여기서는 LZO 코덱 설치 방법을 간단히 설명하겠다.  

#### LZO 설치  
```bash
sudo apt-get install lzop
# Native LZO를 빌드해야 한다.
cd $HADOOP_HOME/src/native
mvn package -Pnative -DskipTests
# 빌드된 LZO 라이브러리를 HDFS에 등록
hdfs dfs -put /path/to/lzo_library /user/hadoop/
```  
이 과정은 하둡 설치 환경에 따라 달라질 수 있으니, 환경에 맞는 방법으로 진행해야 한다.  

## 4. 파일 압축  
HDFS에 파일을 업로드할 때, 압축된 형식으로 파일을 저장하는 방법을 배워보자.  
이때 `gzip`을 사용하여 파일을 압축한다.  

### 4.1 Gzip을 사용한 파일 압축  
  1. 로컬 파일을 압축:  
  ```bash
  gzip -k example.txt
  ```  
  이 명령어는 **example.txt.gz** 파일을 생성하며, 원본 파일을 보존하면서 압축한다.  

  2. 압축된 파일을 HDFS에 업로드:  
  ```bash
  hdfs dfs -put example.txt.gz /user/hadoop/
  ```  

  3. 파일 확인:  
  ```bash
  hdfs dfs -ls /user/hadoop/
  ```  

압축된 파일이 HDFS에 정상적으로 업로드된 것을 확인할 수 있다.  

### 4.2 압축된 파일 읽기  
하둡의 맵리듀스 작업에서는 압축된 파일을 바로 읽을 수 있다.  
하둡은 파일 확장자를 기반으로 압축된 데이터를 인식하고, 자동으로 압축을 해제하여 처리한다.  

## 5. 맵리듀스 작업에서 압축 사용  
맵리듀스 작업에서도 출력 데이터를 압축할 수 있다.  
이를 통해 작업 결과가 더 적은 공간을 차지하도록 만들 수 있다.  

### 5.1 출력 데이터 압축  
맵리듀스 작업에서 출력 데이터를 압축하려면 다음 설정을 사용한다:  
```xml
<property>
  <name>mapreduce.output.fileoutputformat.compress</name>
  <value>true</value>
</property>
<property>
  <name>mapreduce.output.fileoutputformat.compress.codec</name>
  <value>org.apache.hadoop.io.compress.GzipCodec</value>
</property>
```  
위 설정은 맵리듀스 작업에서 **Gzip** 코덱을 사용하여 출력 데이터를 압축하는 설정이다.  

### 5.2 중간 데이터 압축  
맵리듀스 작업에서 **중간 데이터** 역시 압축할 수 있다.  
중간 데이터 압축은 디스크 I/O 성능을 개선하고 네트워크 대역폭을 줄이는 데 유용하다.  
```xml
<property>
  <name>mapreduce.map.output.compress</name>
  <value>true</value>
</property>
<property>
  <name>mapreduce.map.output.compress.codec</name>
  <value>org.apache.hadoop.io.compress.SnappyCodec</value>
</property>
```  
이 설정은 맵 작업의 중간 결과를 **Snappy** 코덱으로 압축한다.  

## 6. 실습: 파일 압축 및 맵리듀스 작업에서의 사용  
### 6.1 파일 압축 및 업로드 실습  
  1. 로컬 파일을 Gzip으로 압축:  
  ```bash
  gzip -k sample.txt
  ```  

  2. 압축된 파일을 HDFS에 업로드:  
  ```bash
  hdfs dfs -put sample.txt.gz /user/hadoop/
  ```  

  3. 업로드한 파일 확인:  
  ```bash
  hdfs dfs -ls /user/hadoop/
  ```  

### 6.2 맵리듀스 작업에서 압축 사용 실습  
맵리듀스 작업에서 데이터를 압축하는 설정을 추가한 후, 간단한 워드카운트 작업을 실행해보자.  
```bash
hadoop jar /path/to/hadoop-streaming.jar \
  -D mapreduce.output.fileoutputformat.compress=true \
  -D mapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec \
  -input /user/hadoop/input \
  -output /user/hadoop/output \
  -mapper /bin/cat \
  -reducer /usr/bin/wc
```  
출력 파일이 Gzip으로 압축된 상태로 생성된다.  

---

## 7. 마무리  
이번 강좌에서는 하둡에서 **데이터 압축**을 사용하는 방법과, 이를 통해 얻을 수 있는 장점에 대해 살펴보았다. 압축을 통해 저장 공간을 절약하고, 네트워크 대역폭을 줄이며, 입출력 성능을 향상시킬 수 있다. 다음 강좌에서는 **하둡에서 보안 설정 및 권한 관리**에 대해 다루겠다.  