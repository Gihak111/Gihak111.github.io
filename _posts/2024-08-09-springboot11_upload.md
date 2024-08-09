---
layout: single
title:  "spring boot 11.애플리케이션 테스트"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  

# 스프링 부트 테스트
스프링 부트는 테스트를 위한 다양한 기능을 제공하여 애플리케이션의 품질을 높일 수 있다.  
이 문서에서는 @SpringBootTest를 통한 전체 애플리케이션 테스트와 JUnit을 활용한 단위 테스트를 다룬다.  
초심자도 쉽게 이해할 수 있고, 중급자도 얻어갈 수 있는 내용을 포함하여 설명하겠다.  

## @SpringBootTest  
스프링 부트 애플리케이션의 전체 컨텍스트를 로드하여 통합 테스트를 수행할 수 있게 해주는 애너테이션.  
이는 애플리케이션의 전반적인 동작을 검증하는 데 유용하다.  

```java
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class ApplicationTests {

    @Test
    void contextLoads() {
        // 애플리케이션 컨텍스트가 성공적으로 로드되는지 확인하는 테스트
    }
}
```  

@SpringBootTest를 사용하여 애플리케이션 컨텍스트가 로드되는지 확인한다.  
이 테스트는 애플리케이션의 주요 설정이 제대로 작동하는지 확인하는 데 유용하다.  

## Aircraft Positions 애플리케이션 단위 테스트  
단위 테스트는 애플리케이션의 개별 구성 요소를 독립적으로 테스트 한다.  
Aircraft Positions 애플리케이션을 통해 단위 테스트를 작성해 보자.  

AircraftService  
```java
import static org.mockito.Mockito.*;
import static org.assertj.core.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class AircraftServiceTest {

    @Mock
    private AircraftRepository aircraftRepository; // AircraftRepository의 Mock 객체를 생성

    @InjectMocks
    private AircraftService aircraftService; // Mock 객체가 주입된 AircraftService 객체를 생성

    public AircraftServiceTest() {
        MockitoAnnotations.openMocks(this); // Mock 객체 초기화
    }

    @Test
    void testGetAllAircrafts() {
        // Mock 객체의 동작을 정의
        when(aircraftRepository.findAll()).thenReturn(Arrays.asList(
                new Aircraft(1L, "Boeing 747", "Airliner"),
                new Aircraft(2L, "Cessna 172", "Private")
        ));

        // 서비스 메서드 호출
        List<Aircraft> aircrafts = aircraftService.getAllAircrafts();

        // 결과 검증
        assertThat(aircrafts).hasSize(2); // 리스트 크기가 2인지 검증
        assertThat(aircrafts.get(0).getName()).isEqualTo("Boeing 747"); // 첫 번째 항목의 이름이 "Boeing 747"인지 검증
        assertThat(aircrafts.get(1).getName()).isEqualTo("Cessna 172"); // 두 번째 항목의 이름이 "Cessna 172"인지 검증
    }
}
```  

AircraftService의 getAllAircrafts 메서드를 테스트하는 단위 테스트다.  
@Mock을 사용하여 AircraftRepository의 Mock 객체를 생성, @InjectMocks를 사용하여 해당 Mock 객체가 주입된 AircraftService 객체를 생성한다.  

## 리팩터링  

테스트 코드를 리팩터링하여 유지보수성과 가독성을 높이는 방법이다.  
테스트 데이터 생성과 검증 로직을 분리하고, 반복되는 코드를 줄이는 것이 중요하다.    
테스트 데이터 생성기 에제를 통해 알아보자.  
```java
import java.util.Arrays;
import java.util.List;

public class TestDataGenerator {

    public static List<Aircraft> generateAircrafts() {
        return Arrays.asList(
                new Aircraft(1L, "Boeing 747", "Airliner"),
                new Aircraft(2L, "Cessna 172", "Private")
        );
    }
}
```  

위의 TestDataGenerator 클래스는 테스트 데이터를 생성하는 유틸리티 클래스이다. 
이를 통해 테스트 코드에서 반복되는 데이터 생성 로직을 분리할 수 있다.  

리팩터링된 테스트 코드  
```java
import static org.mockito.Mockito.*;
import static org.assertj.core.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class AircraftServiceTest {

    @Mock
    private AircraftRepository aircraftRepository; // AircraftRepository의 Mock 객체를 생성

    @InjectMocks
    private AircraftService aircraftService; // Mock 객체가 주입된 AircraftService 객체를 생성

    public AircraftServiceTest() {
        MockitoAnnotations.openMocks(this); // Mock 객체 초기화
    }

    @Test
    void testGetAllAircrafts() {
        // Mock 객체의 동작을 정의
        when(aircraftRepository.findAll()).thenReturn(TestDataGenerator.generateAircrafts());

        // 서비스 메서드 호출
        List<Aircraft> aircrafts = aircraftService.getAllAircrafts();

        // 결과 검증
        assertThat(aircrafts).hasSize(2); // 리스트 크기가 2인지 검증
        assertThat(aircrafts.get(0).getName()).isEqualTo("Boeing 747"); // 첫 번째 항목의 이름이 "Boeing 747"인지 검증
        assertThat(aircrafts.get(1).getName()).isEqualTo("Cessna 172"); // 두 번째 항목의 이름이 "Cessna 172"인지 검증
    }
}
```  

리팩터링된 테스트 코드는 테스트 데이터 생성을 TestDataGenerator 클래스로 분리하여 가독성과 유지보수성을 높였다.  

## 동적 분리를 통한 개선  
동적 분리는 테스트에서 Mock 객체를 사용하여 실제 객체 대신 가짜 객체를 사용한다.  
이는 테스트의 독립성을 높이고, 테스트 실행 속도를 향상시킬 수 있다.  
MockMvc를 사용해 컨트롤러 테스트를 해 보자.  
MockMvc는 Spring MVC 테스트 프레임워크로, 컨트롤러를 테스트할 때 유용하다.  
이를 통해 HTTP 요청과 응답을 모킹할 수 있다.  
```java
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(AircraftController.class)
class AircraftControllerTest {

    @Autowired
    private MockMvc mockMvc; // MockMvc 객체를 주입

    @MockBean
    private AircraftService aircraftService; // AircraftService의 Mock 객체를 생성

    @BeforeEach
    void setUp() {
        // Mock 객체의 동작을 정의
        when(aircraftService.getAllAircrafts()).thenReturn(TestDataGenerator.generateAircrafts());
    }

    @Test
    void testGetAllAircrafts() throws Exception {
        // MockMvc를 사용하여 /aircrafts 엔드포인트에 GET 요청을 보냄
        mockMvc.perform(get("/aircrafts"))
                .andExpect(status().isOk()) // 응답 상태가 200 OK인지 검증
                .andExpect(jsonPath("$[0].name").value("Boeing 747")) // 첫 번째 항목의 이름이 "Boeing 747"인지 검증
                .andExpect(jsonPath("$[1].name").value("Cessna 172")); // 두 번째 항목의 이름이 "Cessna 172"인지 검증
    }
}
```  

MockMvc를 사용하여 AircraftController를 테스트 한다.  
@WebMvcTest를 사용하여 컨트롤러만 로드하고, @MockBean를 사용하여 AircraftService의 Mock 객체를 생성한다.  

## 슬라이드 테스트  
스프링 부트 애플리케이션의 특정 레이어만 로드하여 테스트하는 방법이다.  
이는 테스트 속도를 향상시키고, 특정 레이어의 문제를 더 쉽게 디버깅할 수 있게 해준다.  

@DataJpaTest는 JPA 리포지토리를 테스트할 때 유용한 애너테이션이다.  
이는 내장 데이터베이스를 사용하여 빠르고 독립적인 테스트를 가능하게 한다.  
```java
import static org.assertj.core.api.Assertions.*;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

@DataJpaTest
class AircraftRepositoryTest {

    @Autowired
    private AircraftRepository aircraft

Repository; // 실제 AircraftRepository 객체를 주입

    @Test
    void testFindById() {
        // 테스트 데이터를 데이터베이스에 저장
        Aircraft savedAircraft = aircraftRepository.save(new Aircraft(null, "Boeing 747", "Airliner"));

        // 저장된 데이터를 ID로 검색
        Optional<Aircraft> retrievedAircraft = aircraftRepository.findById(savedAircraft.getId());

        // 결과 검증
        assertThat(retrievedAircraft).isPresent(); // 결과가 존재하는지 검증
        assertThat(retrievedAircraft.get().getName()).isEqualTo("Boeing 747"); // 이름이 "Boeing 747"인지 검증
    }
}
```

@DataJpaTest를 사용하여 AircraftRepository를 테스트하는 방법이다.  
내장 데이터베이스를 사용하여 JPA 리포지토리의 동작을 검증한다.  

이와 같이, 스프링 부트는 다양한 테스트 방법을 제공하여 애플리케이션의 품질을 높일 수 있다.  