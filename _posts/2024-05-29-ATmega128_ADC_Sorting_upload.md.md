---
layout: single
title:  "ATmega128dml A/D 변환기에 sorting 추가."
categories: "ANTImega"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
# 평균을 망치는 이레귤러
평균을 구할 때 혼자서 튀어나와 있는 값들이 있죠.   
그런 값들을 배제하고 병균을 구하는 방법은 보다 실제 평균에 가깝게 해 줍니다.  
다음은 전에 포스팅 했던 ATmega128의 A/D 변환기에 Sorting 과정을 거져 잡음을 보다 확실하게 잡아낼 수 있는 알고리즘 입니다.  

# 코드  
다음은 Sorting를 활용해 코딩하여 ADC의 오차를 보다 완벽하게 정리한 코드 입니다.
```cpp
// 버블 알고리즘 적용 함수
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// 이상치 배제 후 평균 계산 함수
double calculateMean(int arr[], int n) {
    if (n == 0) return 0;

    // 이상치를 배제할 범위 설정 (예: 상위 및 하위 10% 배제)
    int excludeCount = n / 10; // 상위 및 하위 10%를 배제
    int start = excludeCount;
    int end = n - excludeCount;

    if (end <= start) return 0; // 유효한 값이 없는 경우

    double sum = 0;
    int count = 0;

    for (int i = start; i < end; i++) {
        sum += arr[i];
        count++;
    }

    return (count > 0) ? (sum / count) : 0;
}

void InitADC(void)
{
    ADMUX = 0x00;   //00 0 00000
    ADCSR = 0xE5;   //ADC enable, ADC start conversion, Free_run mode, Prescaler 32
    // ADMUX:
    // 00 => 기준전압으로 AREF핀을 사용한다.
    // 0 => ADCH, ADCL의 하위비트부터 10비트를 채웁니다. ADCH는 비트 1 ~ 0을, ADCL은 비트 7 ~ 0을 사용한다.
    // 00000 => 단일 입력채널 ADC0 ~ ADC7을 사용한다.

    // ADCSR:
    // ADEN = 1 => ADC 동작을 허용합니다. ADSC = 1 => AD변환을 시작한다.
    // ADFR = 1 => ADC가 프리러닝 모드로 동작한다.
    // ADIE = 0 => ADC 인터럽트는 사용하지 않는다.
    // ADPS2 ~ 0 = 101 => ADC 입력 클럭의 분주비를 32분주로 설정한다.
}

unsigned int ReadADC(unsigned char channel)
{
    // 변수 설정이다.
    unsigned int AdcDet = 0;
    unsigned int getADCresult = 0;
    unsigned int deta[20];  // 20개의 값을 저장할 배열
    unsigned char i, AdcDetL, AdcDetH = 0;

    // 읽어낼 단일 입력 채널을 설정합니다. 트 4 ~ 0은 00000 ~ 00111의 값을 가진다.
    ADMUX = channel;    // Select Channel

    for (i = 0; i < 20; i++) {  // A/D 값을 20회 읽어서 평균값을 낸다.
        // ADCSR 레지스터의 ADIF 비트가 1이 될 때까지 기다린다. 즉, AD 변환이 완료될 때까지 기다린다.
        while (!(ADCSR & 0x10));
        Delay(70);  // 결과값을 읽기 전에 약간의 딜레이를 건다. 20회 읽는 시간을 벌고 값을 갱신하는 시간을 보장하기 위함이다.
        ADCSR |= 0x10;   // ADIF 비트에 1을 써서 강제로 지운다.

        // 결과값을 넣어 20회 축적한다.
        AdcDetL = ADCL;
        AdcDetH = ADCH;
        AdcDet = (((AdcDetH & 0xff) << 8) | (AdcDetL & 0xff));
        deta[i] = AdcDet;
    }

    // 배열을 정렬하여 이상치를 배제하기 쉽게 한다.
    int n = sizeof(deta) / sizeof(deta[0]);
    bubbleSort(deta, n);

    // 앞서 만든 함수를 활용하여 이상치를 배제하여 계산한 값을 얻는다.
    double mean = calculateMean(deta, n);

    return (unsigned int)mean;   // 검출한 변환값을 반환한다.
}


```

위의 코드로 간단하게 sorting 할 수 있습니다.