---
layout: single
title:  "알고리즘 정리 5. 이분탐색 알고리즘"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Binary Search
이번에는 이분탐색(Binary Search)에 대해 알아보자. 이분탐색은 효율적인 검색 알고리즘 중 하나로, 이미 정렬된 배열이나 리스트에서 특정 값을 찾는 데 사용된다.

# 이분탐색(Binary Search)
이분탐색은 배열의 가운데 있는 요소와 비교하여, 찾고자 하는 값이 가운데 값보다 작은지, 큰지를 확인한 후 배열을 반으로 나누어 검색 범위를 줄여가는 방법이다. 이 과정이 반복되면서 검색 범위가 점점 좁아지며, 결국 원하는 값을 찾거나, 배열에 값이 없는 경우 탐색을 종료한다.

## 이분탐색의 동작 원리
이분탐색은 다음과 같은 단계로 동작한다:

1. 배열의 중간 값을 찾는다.
2. 중간 값이 찾고자 하는 값과 같은지 확인한다.
3. 중간 값이 찾고자 하는 값보다 크다면, 배열의 왼쪽 절반에서 검색을 계속한다.
4. 중간 값이 찾고자 하는 값보다 작다면, 배열의 오른쪽 절반에서 검색을 계속한다.
5. 이 과정을 반복하여 값을 찾거나 배열의 범위가 없어질 때까지 진행한다.

## 이분탐색의 시간 복잡도
이분탐색의 시간 복잡도는 O(log n)이다. 이는 검색 범위를 절반씩 줄여 나가기 때문에 매우 효율적이다. 하지만, 배열이 반드시 **정렬되어 있어야** 한다는 전제가 필요하다.

## 예제 코드
아래는 이분탐색을 구현한 간단한 예제 코드이다. 이 코드에서는 정렬된 정수 배열에서 특정 값을 찾는 기능을 구현했다.

1. 이분탐색 구현 예제 (재귀 방식)

```java
public class BinarySearchRecursive {

    public static int binarySearch(int[] arr, int target, int left, int right) {
        if (left > right) {
            return -1;  // 값이 배열에 없음
        }

        int mid = left + (right - left) / 2;

        // 중간 값이 타겟과 같은 경우
        if (arr[mid] == target) {
            return mid;
        }

        // 타겟이 중간 값보다 작은 경우, 왼쪽 반에서 검색
        if (arr[mid] > target) {
            return binarySearch(arr, target, left, mid - 1);
        }

        // 타겟이 중간 값보다 큰 경우, 오른쪽 반에서 검색
        return binarySearch(arr, target, mid + 1, right);
    }

    public static void main(String[] args) {
        int[] sortedArray = {1, 3, 5, 7, 9, 11, 13};
        int target = 7;

        int result = binarySearch(sortedArray, target, 0, sortedArray.length - 1);

        if (result != -1) {
            System.out.println("Element found at index: " + result);  // 출력: Element found at index: 3
        } else {
            System.out.println("Element not found");
        }
    }
}
```

2. 이분탐색 구현 예제 (반복문 방식)

```java
public class BinarySearchIterative {

    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            // 중간 값이 타겟과 같은 경우
            if (arr[mid] == target) {
                return mid;
            }

            // 타겟이 중간 값보다 작은 경우, 왼쪽 반에서 검색
            if (arr[mid] > target) {
                right = mid - 1;
            }
            // 타겟이 중간 값보다 큰 경우, 오른쪽 반에서 검색
            else {
                left = mid + 1;
            }
        }

        return -1;  // 값이 배열에 없음
    }

    public static void main(String[] args) {
        int[] sortedArray = {2, 4, 6, 8, 10, 12, 14};
        int target = 10;

        int result = binarySearch(sortedArray, target);

        if (result != -1) {
            System.out.println("Element found at index: " + result);  // 출력: Element found at index: 4
        } else {
            System.out.println("Element not found");
        }
    }
}
```

## 이분탐색 사용 사례
- **데이터 검색**  
  대규모 데이터베이스나 파일 시스템에서 특정 데이터를 빠르게 검색할 때 이분탐색이 유용하다. 예를 들어, 사전에서 특정 단어를 찾는 작업에 이분탐색을 적용할 수 있다.
  
- **문제 해결**  
  알고리즘 문제에서 이분탐색은 최적화 문제나 결정 문제를 해결하는 데 자주 사용된다. 예를 들어, 특정 조건을 만족하는 최대 또는 최소 값을 찾을 때 사용할 수 있다.

## 이분탐색을 사용하기 위한 조건
- **정렬된 배열 또는 리스트**  
  이분탐색은 배열이 반드시 정렬되어 있어야 한다. 정렬되지 않은 배열에서는 이분탐색을 사용할 수 없다.
  
- **랜덤 접근 가능 구조**  
  배열처럼 인덱스를 통해 직접 접근 가능한 구조에서 이분탐색이 유효하다. 연결 리스트와 같은 구조에서는 이분탐색의 효율이 떨어진다.

## 이분탐색을 활용한 문제들
이분탐색은 매우 중요한 알고리즘이며, 이를 활용한 다양한 문제가 존재한다. 예를 들어, **최소 최대 문제**, **이진 검색 트리 (BST)**, **순위 매기기 문제** 등이 있다. 이분탐색의 기본 원리를 이해하고 응용할 수 있다면, 알고리즘 문제를 해결하는 데 큰 도움이 될 것이다.

이제 이분탐색에 대해 이해가 되었으리라 믿는다. 이 알고리즘을 잘 활용하면, 효율적으로 문제를 해결할 수 있을 것이다.