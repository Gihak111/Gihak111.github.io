---
layout: single
title:  "하둡 강좌 1편 하둡 설치 및 시작"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## 1. 하둡 설치
### 1.1 자바 설치
하둡은 자바로 작성된 소프트웨어이므로, 자바 설치가 필요하다. 최신 하둡 버전은 자바 8 또는 자바 11을 요구한다.  
```bash
sudo apt update
sudo apt install openjdk-11-jdk -y
```

설치 후 자바 버전을 확인한다.  
```bash
java -version
```

### 1.2 하둡 다운로드
아파치 하둡 공식 웹사이트에서 하둡을 다운로드한다.  
```bash
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
tar -xzvf hadoop-3.3.6.tar.gz
sudo mv hadoop-3.3.6 /usr/local/hadoop
```

### 1.3 환경 변수 설정
하둡을 사용할 수 있도록 환경 변수를 설정하자. `.bashrc` 파일을 열고 하둡 및 자바 관련 변수를 추가한다.  
```bash
sudo nano ~/.bashrc
```

파일 하단에 다음 내용을 추가한다.  
```bash
# Hadoop 환경 변수
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin

# Java 환경 변수
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

환경 변수를 적용한다.  
```bash
source ~/.bashrc
```

### 1.4 하둡 설정 파일 수정

1. **hadoop-env.sh** 수정  
```bash
sudo nano $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

`JAVA_HOME` 경로를 설정한다.  
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

2. **core-site.xml** 설정  
하둡이 데이터를 저장할 디렉토리를 설정하자.    
```bash
sudo nano $HADOOP_HOME/etc/hadoop/core-site.xml
```

다음 내용을 `<configuration>` 태그 사이에 추가한다.  
```xml
<property>
  <name>fs.defaultFS</name>
  <value>hdfs://localhost:9000</value>
</property>
<property>
  <name>hadoop.tmp.dir</name>
  <value>/usr/local/hadoop/tmp</value>
</property>
```

3. **hdfs-site.xml** 설정  
HDFS에 필요한 데이터 저장소 및 복제 설정을 추가한다.  
```bash
sudo nano $HADOOP_HOME/etc/hadoop/hdfs-site.xml
```

다음 내용을 추가한다.  
```xml
<property>
  <name>dfs.replication</name>
  <value>1</value>
</property>
<property>
  <name>dfs.namenode.name.dir</name>
  <value>file:///usr/local/hadoop/hdfs/namenode</value>
</property>
<property>
  <name>dfs.datanode.data.dir</name>
  <value>file:///usr/local/hadoop/hdfs/datanode</value>
</property>
```

### 1.5 HDFS 포맷팅
설정 후, HDFS 파일 시스템을 포맷한다.  
```bash
hdfs namenode -format
```

## 2. 하둡 클러스터 시작  
하둡 클러스터를 시작하려면 다음 명령어를 사용한다.  
```bash
start-dfs.sh
```  
명령어 실행 후, 네임노드 및 데이터노드가 제대로 실행되는지 확인할 수 있다.   
```bash
jps
```  
`NameNode`, `DataNode`, `SecondaryNameNode` 등의 프로세스가 실행 중인지 확인한다.  
## 3. 웹 인터페이스 확인  
하둡은 웹 UI를 제공하여 클러스터 상태를 확인할 수 있다. 브라우저에서 다음 주소로 접속한다.  
- 네임노드 UI: http://localhost:9870  
이를 통해 파일 시스템 상태와 클러스터 정보를 확인할 수 있다.  

---

이제 하둡 설치가 완료되었고, 간단한 클러스터 설정을 마쳤다. 다음 시간에는 하둡의 파일 시스템(HDFS)에 파일을 업로드하고 작업을 실행하는 방법을 알아보자.  