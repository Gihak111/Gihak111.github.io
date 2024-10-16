---
layout: single  
title:  "알고리즘 정리 7. Prefix Sum (구간 합)"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---

# Prefix Sum
이번에는 Prefix Sum(구간 합) 알고리즘에 대해 알아보자.  
Prefix Sum은 배열의 구간 합을 효율적으로 계산하기 위한 방법으로, 특히 여러 번의 구간 합을 계산해야 할 때 매우 유용하다.  

# Prefix Sum(구간 합) 개요
Prefix Sum 알고리즘은 배열에서 특정 구간의 합을 빠르게 계산하기 위해 사용된다.  
일반적으로, 각 구간의 합을 직접 계산하면 O(n)의 시간이 걸리지만, Prefix Sum 배열을 미리 계산해 두면 O(1) 시간에 구간 합을 계산할 수 있다.  

## Prefix Sum의 동작 원리
Prefix Sum 알고리즘은 다음과 같은 단계로 동작한다:  
1. **Prefix Sum 배열 계산**  
   주어진 배열의 각 원소까지의 합을 저장하는 새로운 배열을 생성한다.  
2. **구간 합 계산**  
   특정 구간 [i, j]의 합은 Prefix Sum 배열에서 `prefix[j] - prefix[i-1]`으로 간단히 계산할 수 있다.  
## Prefix Sum의 시간 복잡도
Prefix Sum 배열을 미리 계산하는 데는 O(n)의 시간이 걸리며, 이후 구간 합을 계산하는 데는 O(1)의 시간이 소요된다.  
여러 번 구간 합을 계산해야 할 때, 매우 효율적이다.  

## Prefix Sum 예제 코드
아래는 Prefix Sum 알고리즘을 구현한 간단한 예제 코드이다.  
이 코드에서는 배열의 구간 합을 효율적으로 계산하기 위해 Prefix Sum 배열을 사용한다.  

```java
public class PrefixSumExample {

    // Prefix Sum 배열을 생성하는 함수
    public static int[] computePrefixSum(int[] arr) {
        int[] prefixSum = new int[arr.length];
        prefixSum[0] = arr[0];

        for (int i = 1; i < arr.length; i++) {
            prefixSum[i] = prefixSum[i - 1] + arr[i];
        }

        return prefixSum;
    }

    // 구간 [i, j]의 합을 계산하는 함수
    public static int rangeSum(int[] prefixSum, int i, int j) {
        if (i == 0) {
            return prefixSum[j];
        } else {
            return prefixSum[j] - prefixSum[i - 1];
        }
    }

    public static void main(String[] args) {
        int[] arr = {2, 4, 6, 8, 10};
        int[] prefixSum = computePrefixSum(arr);

        System.out.println("Sum of elements from index 1 to 3: " + rangeSum(prefixSum, 1, 3));  // 출력: 18 (4 + 6 + 8)
        System.out.println("Sum of elements from index 0 to 4: " + rangeSum(prefixSum, 0, 4));  // 출력: 30 (2 + 4 + 6 + 8 + 10)
    }
}
```

# Prefix Sum 사용 사례
- **구간 합 계산**  
  다수의 구간 합을 빠르게 계산해야 할 때 Prefix Sum은 매우 유용하다. 예를 들어, 시간에 따른 누적 데이터를 처리하는 문제에서 사용할 수 있다.  
- **누적 합 계산**  
  배열의 앞에서부터 특정 위치까지의 누적 합을 계산하는 작업에 Prefix Sum이 자주 사용된다. 예를 들어, 게임에서의 누적 점수 계산이나 통계적 분석에서 활용될 수 있다.  
- **부분 배열 합 문제**  
  주어진 배열에서 특정 구간의 합이 최대가 되는 구간을 찾는 문제 등에서 Prefix Sum이 유용하게 활용된다.  
# Prefix Sum의 장점과 한계
- **장점**  
  다수의 구간 합을 효율적으로 계산할 수 있다. 이로 인해 복잡한 연산을 단순화하고, 실행 시간을 크게 줄일 수 있다.  
- **한계**  
  배열의 요소가 자주 변경되는 경우, Prefix Sum 배열을 다시 계산해야 하므로 효율성이 떨어질 수 있다.  
  따라서, 배열이 자주 업데이트되는 상황에서는 다른 접근법이 필요할 수 있다.  
Prefix Sum은 구간 합 문제를 해결하는 데 매우 강력한 도구이다.  
이 알고리즘을 잘 이해하고 활용하면, 여러 가지 문제에서 효율적으로 구간 합을 계산할 수 있을 것이다.  
이 글을 통해 Prefix Sum에 대해 이해하고, 문제에 적용하는 데 도움이 되길 바란다.  