---
layout: single
title:  "아키텍처 패턴 시리즈 15. 서비스 지향 아키텍처 (Service-Oriented Architecture, SOA)"
categories: "Architecture_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 아키텍처 패턴 시리즈 15: 서비스 지향 아키텍처 (Service-Oriented Architecture, SOA)

서비스 지향 아키텍처(SOA)는 애플리케이션의 기능을 서비스 단위로 분리하여 유연하게 통합 및 재사용할 수 있도록 설계된 아키텍처 패턴이다.  
각 서비스는 표준 인터페이스를 통해 독립적으로 제공되며, 다양한 애플리케이션에서 이를 호출할 수 있다.  

## 서비스 지향 아키텍처의 필요성

다음과 같은 문제를 해결하기 위해 SOA가 도입된다:  

1. 복잡한 시스템 간 통합 문제: 서로 다른 기술 스택이나 플랫폼을 사용하는 시스템 간 통신 문제 해결.  
2. 재사용성 부족: 기존 모놀리식 시스템에서 특정 기능을 재사용하기 어려움.  
3. 유연성 부족: 빠르게 변화하는 비즈니스 요구사항에 맞게 시스템을 수정하고 확장하는 데 제약.  

### 예시: 금융 서비스 플랫폼

금융 플랫폼은 다음과 같은 서비스로 구성될 수 있다:  

- 고객 관리 서비스  
- 계좌 관리 서비스  
- 거래 처리 서비스  

SOA를 사용하면 각 서비스를 독립적으로 관리하고 다른 시스템에서도 동일한 서비스를 재사용할 수 있다.  

## 서비스 지향 아키텍처의 구조

### 주요 컴포넌트

1. 서비스 제공자 (Service Provider): 서비스의 구현 및 배포를 담당.  
2. 서비스 소비자 (Service Consumer): 필요한 서비스를 호출하여 사용하는 주체.  
3. 서비스 레지스트리 (Service Registry): 서비스의 위치 및 메타데이터를 관리.  
4. 서비스 계약 (Service Contract): 서비스 제공자와 소비자 간의 통신 규약을 정의.  

### 구조 다이어그램

```
[Service Consumer]
       |
[Service Registry] <---> [Service Provider]
       |
  [Messaging Protocols]
```

### 동작 원리

1. 서비스 제공자는 서비스 레지스트리에 자신의 서비스 정보를 등록.  
2. 서비스 소비자는 레지스트리를 조회하여 필요한 서비스를 찾고 호출.  
3. 서비스는 표준화된 프로토콜(SOAP, REST 등)을 통해 통신.  

## 서비스 지향 아키텍처 예시

### SOAP 기반 SOA 예시

#### 1. 서비스 계약 (WSDL 파일)

SOAP 기반의 SOA에서는 WSDL(Web Services Description Language)로 서비스 계약을 정의한다.  

```xml
<definitions name="AccountService"
    xmlns="http://schemas.xmlsoap.org/wsdl/"
    xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"
    xmlns:tns="http://example.com/account"
    targetNamespace="http://example.com/account">
    
    <message name="GetAccountRequest">
        <part name="accountId" type="xsd:string"/>
    </message>
    
    <message name="GetAccountResponse">
        <part name="accountDetails" type="xsd:string"/>
    </message>
    
    <portType name="AccountServicePortType">
        <operation name="GetAccount">
            <input message="tns:GetAccountRequest"/>
            <output message="tns:GetAccountResponse"/>
        </operation>
    </portType>
    
    <binding name="AccountServiceBinding" type="tns:AccountServicePortType">
        <soap:binding style="rpc" transport="http://schemas.xmlsoap.org/soap/http"/>
        <operation name="GetAccount">
            <soap:operation soapAction="http://example.com/account/GetAccount"/>
            <input>
                <soap:body use="literal"/>
            </input>
            <output>
                <soap:body use="literal"/>
            </output>
        </operation>
    </binding>
    
    <service name="AccountService">
        <port name="AccountServicePort" binding="tns:AccountServiceBinding">
            <soap:address location="http://example.com/accountService"/>
        </port>
    </service>
</definitions>
```

#### 2. 서비스 제공자 (Java)

```java
@WebService
public class AccountService {
    @WebMethod
    public String getAccount(String accountId) {
        // 예제: ID에 따라 계좌 정보 반환
        return "Account details for ID: " + accountId;
    }
}
```

#### 3. 서비스 소비자 (Java)

```java
import javax.xml.namespace.QName;
import javax.xml.ws.Service;
import java.net.URL;

public class AccountClient {
    public static void main(String[] args) throws Exception {
        URL url = new URL("http://example.com/accountService?wsdl");
        QName qname = new QName("http://example.com/account", "AccountService");
        Service service = Service.create(url, qname);
        AccountService accountService = service.getPort(AccountService.class);

        String accountDetails = accountService.getAccount("12345");
        System.out.println(accountDetails);
    }
}
```

### 실행 방법

1. 서비스 제공자를 실행하여 SOAP 서비스를 배포.  
2. 소비자는 WSDL을 기반으로 서비스를 호출.  

## 서비스 지향 아키텍처의 장점  

1. 재사용성: 서비스는 독립적으로 설계되므로 여러 애플리케이션에서 재사용 가능.  
2. 유연성: 새로운 서비스 추가 및 변경이 쉬움.  
3. 표준화된 통신: 다양한 플랫폼 간의 상호 운영성을 제공.  
4. 모듈성: 서비스별로 분리되어 있어 독립적인 배포 및 관리 가능.  

## 서비스 지향 아키텍처의 단점

1. 높은 초기 비용: 설계 및 구현에 많은 시간과 비용이 필요.  
2. 성능 오버헤드: 서비스 간 통신에서 프로토콜 변환으로 인한 성능 저하 가능.  
3. 복잡한 관리: 서비스 수가 많아지면 레지스트리 관리가 복잡.  
4. 표준의 제약: 특정 표준(SOAP, WSDL 등)에 종속될 수 있음.  

### 마무리

서비스 지향 아키텍처는 이질적인 시스템 통합과 재사용성을 극대화하는 데 적합하다.  

아래 글에서 다른 아키텍쳐 패턴들을 확인할 수 있다.  
[아키텍처 패턴 모음](https://gihak111.github.io/architecture_patterns/2024/12/04/Type_of_Architecture_Patterns_upload.html)  
