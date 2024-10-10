---
layout: single
title:  "하둡 강좌 11편 보안 설정 및 권한 관리"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 11편: **하둡에서 보안 설정 및 권한 관리**  
이번 강좌에서는 **하둡(Hadoop)** 환경에서 **보안 설정**과 **권한 관리**를 어떻게 하는지에 대해 살펴보겠다.   
하둡은 기본적으로 대규모 데이터 처리를 목적으로 하기에, 데이터의 안전한 저장과 접근을 위해 보안 설정이 매우 중요하다.  
이를 위해 하둡에서는 다양한 보안 메커니즘을 제공한다.  

## 1. 하둡의 보안 필요성  
하둡 클러스터는 일반적으로 많은 사용자와 다양한 애플리케이션이 접근할 수 있는 대규모 데이터 저장소이기 때문에 보안이 중요하다.  
적절한 보안 설정이 없을 경우 데이터 도난, 손상, 부적절한 접근 등이 발생할 수 있다.  
주요 보안 요구 사항은 다음과 같다:  
- **데이터 접근 제어**: 특정 사용자가 특정 데이터에만 접근할 수 있도록 권한을 제한해야 한다.  
- **데이터 전송 보안**: 하둡 클러스터 내에서 주고받는 데이터가 중간에 가로채이지 않도록 해야 한다.  
- **사용자 인증**: 클러스터에 접근하는 사용자가 실제로 인증된 사용자임을 확인해야 한다.  

## 2. 하둡 보안 모델  
기본적으로 하둡은 **Kubernetes 기반의 보안 모델**을 제공하며, 주요 보안 기능은 다음과 같다:  
- **사용자 인증(Authentication)**: 하둡에서 사용자는 기본적으로 시스템 사용자와 동일한 ID로 관리된다.  
- **권한 부여(Authorization)**: 하둡 파일 시스템(HDFS)과 맵리듀스(MR)는 UNIX와 유사한 권한 체계를 제공한다.  
- **암호화(Encryption)**: 하둡은 데이터 저장 및 전송 시 암호화를 지원하여 데이터 보안을 강화한다.  
- **Kerberos 통합**: 하둡은 Kerberos를 통한 강력한 인증 메커니즘을 제공하여 네트워크 레벨의 보안을 제공한다.  

## 3. 사용자 인증  
### 3.1 Kerberos 기반 사용자 인증  
하둡에서는 보통 **Kerberos**를 통해 사용자 인증을 수행한다.  
Kerberos는 네트워크 상에서 안전하게 사용자 인증을 처리할 수 있도록 해주는 프로토콜이다.  
하둡에서 Kerberos 인증을 설정하려면 다음 단계가 필요하다:  
  1.  Kerberos 설치 및 설정  
  클러스터에서 Kerberos를 사용하려면 먼저 Kerberos를 설치하고 설정해야 한다. 설치는 클러스터의 모든 노드에 수행되어야 하며, 주로 **Key Distribution Center(KDC)**와 **Kerberos 클라이언트**가 필요하다.

  2.  hdfs-site.xml 설정  
  Kerberos 인증을 하둡에 적용하려면 **hdfs-site.xml** 파일에서 다음과 같은 설정을 추가한다:  
  ```xml
  <property>
    <name>dfs.namenode.kerberos.principal</name>
    <value>hdfs/_HOST@YOUR-REALM.COM</value>
  </property>
  <property>
    <name>dfs.datanode.kerberos.principal</name>
    <value>hdfs/_HOST@YOUR-REALM.COM</value>
  </property>
  ```  

  3. core-site.xml 설정  
  Kerberos 인증을 활성화하기 위해 **core-site.xml** 파일에서도 설정이 필요하다:  
  ```xml
  <property>
    <name>hadoop.security.authentication</name>
    <value>kerberos</value>
  </property>
  <property>
    <name>hadoop.security.authorization</name>
    <value>true</value>
  </property>
  ```  
  이렇게 설정한 후, 클러스터를 재시작하면 Kerberos 기반 인증이 활성화된다.  

## 4. 권한 부여 (Authorization)  
하둡의 HDFS는 UNIX 파일 시스템과 유사한 **소유자**, **그룹**, **권한** 구조를 따르고 있다. 이를 통해 사용자와 그룹에 대한 권한을 설정할 수 있다.  

### 4.1 파일 및 디렉터리 권한  
HDFS에서 각 파일과 디렉터리는 **소유자(owner)**, **그룹(group)**, **기타 사용자(others)**에게 권한을 부여할 수 있다.   
권한은 다음과 같이 읽기(r), 쓰기(w), 실행(x)으로 나뉜다:  
- **r**: 읽기 권한  
- **w**: 쓰기 권한  
- **x**: 실행 권한  
예를 들어, 다음과 같은 명령으로 HDFS 파일의 권한을 확인할 수 있다:  
```bash
hdfs dfs -ls /user/hadoop
```  
이 명령어는 `/user/hadoop` 디렉터리의 파일 권한 정보를 보여준다.  

### 4.2 파일 권한 설정  
파일 권한을 설정하려면 `hdfs dfs -chmod` 명령을 사용하면 된다. 예를 들어, 특정 파일에 대해 모든 사용자에게 읽기 권한을 부여하려면 다음과 같이 한다:  
```bash
hdfs dfs -chmod 755 /user/hadoop/sample.txt
```  
이 명령은 **소유자에게 읽기, 쓰기, 실행 권한**을, **그룹과 기타 사용자에게는 읽기 및 실행 권한**을 부여한다.  

### 4.3 소유자 및 그룹 변경  
파일의 소유자와 그룹을 변경하려면 `hdfs dfs -chown` 명령을 사용한다.  
예를 들어, 파일의 소유자를 `user1`, 그룹을 `group1`으로 변경하려면:  
```bash
hdfs dfs -chown user1:group1 /user/hadoop/sample.txt
```

## 5. 데이터 전송 보안 (Encryption)  
하둡에서는 데이터를 전송할 때 **암호화**를 통해 데이터가 네트워크 상에서 안전하게 전송되도록 할 수 있다.  

### 5.1 RPC 암호화  
하둡은 **RPC(Remote Procedure Call)**를 사용하여 클라이언트와 서버 간 통신을 한다.  
이 통신을 암호화하기 위해 **core-site.xml** 파일에서 다음 설정을 추가한다:  
```xml
<property>
  <name>hadoop.rpc.protection</name>
  <value>privacy</value>
</property>
```  
위 설정은 모든 RPC 통신을 암호화하여 전송하는 역할을 한다.  

### 5.2 데이터 암호화  
하둡은 **데이터 암호화** 기능을 제공하여, 저장된 데이터를 암호화할 수 있다.  
HDFS에서 암호화 영역(encryption zone)을 생성하여, 해당 영역 내의 모든 파일을 자동으로 암호화하도록 설정할 수 있다.  

#### 암호화 영역 생성  
1. 먼저 **키 관리**를 위해 Hadoop Key Management Server(KMS)를 설정한다.  
2. KMS에서 생성한 키를 사용하여 암호화 영역을 생성한다:  
```bash
hdfs crypto -createZone -keyName myKey -path /user/hadoop/encrypted_zone
```  
이 명령은 `/user/hadoop/encrypted_zone` 디렉터리를 암호화 영역으로 설정한다.  
이 안에 저장되는 모든 파일은 자동으로 암호화된다.  

## 6. 실습: Kerberos 인증 및 파일 권한 설정

### 6.1 Kerberos 인증 실습  
1. **Kerberos 설치**: 클러스터의 모든 노드에 Kerberos를 설치하고 설정한다.  
2. **hdfs-site.xml 및 core-site.xml 설정**: Kerberos 관련 설정을 추가한 후 클러스터를 재시작한다.  
3. **Kerberos 인증 확인**: 클라이언트에서 Kerberos 티켓을 받아 인증이 잘 되는지 확인한다.  
```bash
kinit hadoop_user
```  

### 6.2 파일 권한 설정 실습  
  1. `/user/hadoop` 디렉터리에 새로운 파일을 생성한다:  
  ```bash
  hdfs dfs -touchz /user/hadoop/test_file.txt
  ```  

  2. 파일 권한을 변경하여 그룹과 기타 사용자에게 읽기만 허용한다:  
  ```bash
  hdfs dfs -chmod 744 /user/hadoop/test_file.txt
  ```  

  3. 권한이 잘 적용되었는지 확인한다:  
  ```bash
  hdfs dfs -ls /user/hadoop/
  ```  

---

## 7. 마무리  
이번 강좌에서는 하둡 클러스터에서 보안 설정 및 권한 관리를 다루었다. 보안을 설정함으로써 하둡 클러스터의 데이터와 리소스를 보호할 수 있으며, Kerberos 인증과 권한 관리를 통해 적절한 사용자만 데이터에 접근하도록 제어할 수 있다. 다음 강좌에서는 **하둡 클러스터의 성능 최적화** 방법을 다루겠다.  