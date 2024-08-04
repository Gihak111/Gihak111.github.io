---
layout: single
title:  "ANTImega128 타이머 설정하기"
categories: "ANTImega"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# ANTImega 128에서 사용하는 타이머에 대한 코드 입니다.
```cpp
ISP(TIMER0_OVF_vect)
{
	//타이머 0표 9-5 그거 보고 계산하는거다. 몇 미리 주고 그 미리에 맞게 x값 구한걸 TCNT0에 집어넣는다
	TCNT0 = 112;	//TCNT =184 ->(1024/14.7456MHz)*(256-112)=10msec
	//타이머 플래그 10미리 세컨드 마다 걸리게끔 설정한거다
	TimerFlag = 1;	//Set 10mec EVTflag ! (EXE @main routine)

	Timer1Cnt++;
		
	//타이머를 1초마다 카운트 하게끔 설정한다. 앞서 타이머 플래그를 10mec로 설정하였으므로 100이 곱해져야 1이된다.
	//따라서 1초마다 타이머가 걸리게 하기 위해 Timer1Cnt >= 100로 설정한거다
	if (Timer1Cnt >= 100)	//10 * 100 = 1sec
	{
		//타이머가 1초가 지나 인터럽트가 걸리면 마로 타이머를 0으로 초기화 시켜 다음 1초가 되었을때 다시 인터럽트 걸리게 한다.
		Timer1Cnt = 0;
		//타이머 플래그를 true로 반환한다. 이거 bool이랑 같은 타입이다.
		Timer1Flag = 1;	//Set 1sec EVT flag !(EXE @main routine)
	}
}

void InitTimer0(void)
{
	//무슨 크리스탈 사용하는건지다. 표 9-5인가 그거 보고 적는거다. 여기선 1024를 사용하였으므로 111인 0x07을 집어넣는다.
	TCCR0 = 0x07;	//Trigger => Fclock/1024
	//=144
	TCNT0 = 256-112;	//clear Timer Counter Register
	
	TIMSK = 0x01;	//Timer0 Interrupt Enable Mask Register
}
```

위와 같은 코드를 사용하여 timer0 를 초기화하고 사용하는 방법을 알아보았습니다.
