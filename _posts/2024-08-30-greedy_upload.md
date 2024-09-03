---
layout: single
title:  "알고리즘 정리 2. 그리디 알고리즘"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Greedy
탐욕 알고리즘이라고 불린다.  
여러 경우 중 하나를 결정해야 할 때마다, 현재 상황에서 최적이라고 생각되는 경우를 선택하여 정답를 구하는 알고리즘이다.  
하지만, 나중에 미칠 영햘을 고려하지 않기 때문에, 항상 최적의 해를 보장하는건 아니다.  
따라서, 주로 근사치 추정에 활용한다.  

이런 조건에도 상요하는 이유는 간단하다. 소곧가 DB 보다 빠르다. 엄청나게 빨라서 복잡하다는 잔점은 알바가 아니다.  
바로 예제를 통해서 알아보자.  

어던 가게 손님에게 거스름돈을 주어야 한다고 하자.  
주어진 동전들로 거스름돈을 줄 떄, 동전의 개수를 최소한으로 주려면 어덯게 해야 할까?
지폐를 먼저 소모, 이어서 가장 큰 단위의 동전부터 사용하면 된다.  

```java
public class GreedyExample {

    // 동전의 종류를 배열로 정의
    static int[] coins = {500, 100, 50, 10};

    public static int minCoins(int change) {
        int count = 0; // 동전의 개수를 세는 변수

        // 큰 단위의 동전부터 반복
        for (int coin : coins) {
            count += change / coin;  // 현재 동전으로 거슬러줄 수 있는 개수를 더함
            change %= coin;  // 남은 금액 업데이트
        }

        return count; // 필요한 동전의 총 개수 반환
    }

    public static void main(String[] args) {
        int change = 1260; // 거슬러줘야 할 금액

        // 결과 출력
        System.out.println("거스름돈 " + change + "원을 최소한의 동전 개수로 거슬러주려면 " + minCoins(change) + "개의 동전이 필요합니다.");
    }
}

```

위 코드를 보면, 현재 조건에서 최선의 술르 찾는 알고리즘인 걸 알 수 있다.  
그리디 패턴은 다음과 같은 상황에서 사용하면 유리하다.  
- **문제가 단뎨별 최선의 선택으로 풀릴수 있는 경우**
- **한 번의 결정이 이후의 결정에 영향을 미치지 않는 경우**
- **해결할 수 잇는 모든 부분 문제응 독립적으로 풀풀 수 있는 경우**

간단한 만큼 사용할 곳이 많다.  