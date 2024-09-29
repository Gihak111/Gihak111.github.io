---
layout: single
title:  "하둡 강좌 6편 Hive"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 6편: **Hive 설치 및 기본 사용법**
이번 강좌에서는 하둡 에코시스템에서 데이터 웨어하우징 도구로 널리 사용되는 **Hive**에 대해 알아보겠다.  
Hive는 SQL-like 쿼리를 사용해 하둡 분산 저장소에서 데이터를 처리하고 분석할 수 있도록 도와준다.  
이번 시간에는 Hive의 개념과 설치, 그리고 기본적인 사용법을 설명하겠다.  

## 1. Hive란?  
**Hive**는 하둡에서 SQL과 유사한 언어인 **HiveQL**을 통해 대규모 데이터를 분석할 수 있는 도구다.  
Hive는 MapReduce 작업을 추상화하여 사용자가 복잡한 코드를 작성할 필요 없이 SQL로 데이터를 처리할 수 있게 도와준다.  
Hive는 대규모 데이터를 효율적으로 처리하며, 하둡 HDFS에 저장된 데이터를 대상으로 쿼리를 실행한다.  

### 1.1 Hive의 주요 기능  
- **SQL-like 언어**: SQL과 유사한 HiveQL을 사용하여 간편하게 데이터 분석 가능.  
- **대규모 데이터 처리**: 하둡 클러스터의 분산 저장소를 이용해 수 페타바이트(PB) 이상의 데이터를 처리할 수 있음.  
- **테이블 형식 데이터 지원**: Hive는 테이블 형식의 데이터를 지원하며, 데이터는 하둡의 HDFS에 저장됨.  
- **확장성**: 하둡의 분산 구조를 사용하여 높은 확장성을 제공.  

## 2. Hive 설치  
Hive를 설치하려면 하둡이 먼저 설치되어 있어야 한다.  
하둡이 설치된 상태에서 Hive를 설치하는 절차는 다음과 같다.  

### 2.1 의존성 설치  
Hive를 설치하기 전에 **Java**, **Hadoop**, **MySQL** 등이 설치되어 있어야 한다.  
Hive의 메타데이터를 저장하기 위해 **MySQL**을 설치하고 설정하는 과정도 필요하다.  

#### 2.1.1 Java 설치 확인  
Java는 Hive 실행에 필수적인 요소다.  
다음 명령어로 Java 설치를 확인한다:  
```bash
java -version
```  
Java가 설치되어 있지 않다면 설치해 주자.  

#### 2.1.2 Hadoop 설치 확인  
Hadoop이 설치되어 있어야 하며, 하둡 네임노드(NameNode)와 데이터노드(DataNode)가 실행 중이어야 한다.  
```bash
hdfs dfs -ls /
```  
이 명령어가 제대로 작동하는지 확인하고, 하둡 클러스터가 정상적으로 작동하고 있는지 확인하자.  

### 2.2 Hive 다운로드 및 설치  
Hive는 아파치 소프트웨어 재단에서 제공하는 오픈소스 프로젝트다.  
최신 버전의 Hive를 다운로드받고 설치해보자.  

#### 2.2.1 Hive 다운로드  
Hive 최신 버전은 다음 URL에서 다운로드할 수 있다:  
[https://hive.apache.org/downloads.html](https://hive.apache.org/downloads.html)  
```bash
cd /opt/
wget https://downloads.apache.org/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz  
```  

#### 2.2.2 압축 해제 및 환경 변수 설정  
```bash
tar -xvzf apache-hive-3.1.2-bin.tar.gz
mv apache-hive-3.1.2-bin hive
```  
환경 변수 설정을 위해 `.bashrc` 파일에 Hive 경로를 추가한다.  
```bash
nano ~/.bashrc
```  
다음 줄을 추가한 후, 저장하고 나가자:  
```bash
export HIVE_HOME=/opt/hive
export PATH=$HIVE_HOME/bin:$PATH
```  
변경 사항을 적용하기 위해 `source` 명령어를 사용한다.  
```bash
source ~/.bashrc
```  

### 2.3 MySQL 설치 및 설정  
Hive는 메타스토어(Metastore)로서 **MySQL**을 사용한다. MySQL을 설치하고 Hive와 연결하는 과정을 보자.  

#### 2.3.1 MySQL 설치  
```bash
sudo apt-get install mysql-server
sudo service mysql start
```  

#### 2.3.2 MySQL에 데이터베이스 생성  
MySQL에 접속하여 Hive용 데이터베이스를 생성한다.  
```bash
mysql -u root -p

CREATE DATABASE metastore;
CREATE USER 'hiveuser'@'localhost' IDENTIFIED BY 'hivepassword';
GRANT ALL PRIVILEGES ON metastore.* TO 'hiveuser'@'localhost';
FLUSH PRIVILEGES;
```  

### 2.4 Hive 설정  
Hive가 MySQL과 통신할 수 있도록 설정 파일을 수정하자.  

#### 2.4.1 Hive 설정 파일 수정  
Hive 설치 경로의 `hive-site.xml` 파일을 설정한다.  
```bash
cp $HIVE_HOME/conf/hive-default.xml.template $HIVE_HOME/conf/hive-site.xml
nano $HIVE_HOME/conf/hive-site.xml
```  
`hive-site.xml`에 다음 내용을 추가하여 MySQL 설정을 한다.  
```xml
<property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://localhost/metastore</value>
    <description>JDBC connect string for a JDBC metastore</description>
</property>

<property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.cj.jdbc.Driver</value>
    <description>Driver class name for a JDBC metastore</description>
</property>

<property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>hiveuser</value>
</property>

<property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>hivepassword</value>
</property>
```  

#### 2.4.2 MySQL JDBC 드라이버 설정  
MySQL과 Hive를 연결하기 위해 JDBC 드라이버를 다운로드하여 `hive/lib` 디렉토리에 복사한다.  
```bash
wget https://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-java-8.0.23.tar.gz
tar -xvzf mysql-connector-java-8.0.23.tar.gz
cp mysql-connector-java-8.0.23/mysql-connector-java-8.0.23.jar $HIVE_HOME/lib/
```  

### 2.5 Hive 초기화  
Hive 메타스토어를 초기화하여 MySQL과의 연결을 테스트한다.  
```bash
schematool -dbType mysql -initSchema
```  
이 명령어가 성공적으로 실행되면 Hive가 MySQL에 메타데이터를 저장할 수 있는 환경이 완성된다.  

## 3. Hive 기본 사용법  
Hive 설치가 완료되었다면 이제 기본적인 Hive 사용법을 알아보자.  

### 3.1 Hive CLI 접속  
다음 명령어로 Hive CLI에 접속할 수 있다.  
```bash
hive
```  
Hive 셸이 실행되면 SQL-like 쿼리를 실행할 수 있는 환경이 제공된다.  

### 3.2 테이블 생성  
Hive에서 테이블을 생성하려면 `CREATE TABLE` 문을 사용한다.  
```sql
CREATE TABLE employee (
    id INT,
    name STRING,
    salary FLOAT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
```  

### 3.3 데이터 로드  
HDFS에 저장된 데이터를 Hive 테이블로 로드할 수 있다.  
```bash
hdfs dfs -put /local/path/to/employee.csv /user/hive/warehouse/
```  
```sql
LOAD DATA INPATH '/user/hive/warehouse/employee.csv' INTO TABLE employee;
```  

### 3.4 데이터 쿼리  
Hive에서 SQL-like 쿼리를 사용하여 데이터를 조회할 수 있다.  
```sql
SELECT * FROM employee;
```  
이외에도 다양한 집계 함수와 조인을 사용할 수 있다.  
```sql
SELECT name, salary FROM employee WHERE salary > 5000;
```  

## 4. Hive 웹 UI 사용  
Hive는 웹 UI를 통해 관리할 수 있다. Hive **Beeline** 인터페이스를 사용하여 메타스토어 상태를 확인하거나 쿼리를 실행할 수 있다.  
```bash
beeline
```  

---

## 5. 마무리  
이번 강좌에서는 Hive의 개념과 설치, 기본 사용법을 다뤘다. Hive는 하둡 분산 파일 시스템에서 SQL-like 쿼리를 통해 데이터를 쉽게 분석할 수 있게 해주며, 대규모 데이터 처리에서 매우 유용한 도구다. 다음 강좌에서는 HiveQL을 더 깊이 이해하고 복잡한 쿼리 실행 방법에 대해 다루겠다.  