---
layout: single
title:  "알고리즘 정리 4. 쇼팅 알고리즘"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# sorting  
데이터들을 특정한 순서대로 나열하는 작업이다.  
위 한 문장으로 중요도는 이미 피부로 느껴질 것이다.  
여러 가지 정렬 알고리즘이 있으며, 각기 다른 방법으로 데이터를 정렬한다.  
기초적인 정렬 알고리즘 몇 가지를보자.  

## 버블 정렬 (Bubble Sort)
인접한 두 원소를 비교하여 필요에 따라 자리를 바꾸는 과정을 반복하는 방법이다.  
이 과정이 끝나면 가장 큰 값이 맨 끝에 "버블"처럼 올라가게 된다.  
동작을 정리하면 다음과 같다.  
1. 리스트의 처음부터 끝까지 인접한 원소들을 비교한다.  
2. 앞의 원소가 뒤의 원소보다 크면 두 원소의 위치를 바꾼다.  
3. 리스트의 끝까지 가면, 가장 큰 원소가 리스트의 마지막에 위치하게 된다.  
4. 이 과정을 n-1번 반복한다.  

```java
public class BubbleSortExample {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        // 배열의 크기만큼 반복
        for (int i = 0; i < n - 1; i++) {
            // 인접한 두 원소를 비교
            for (int j = 0; j < n - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    // 앞의 원소가 더 크면 위치를 바꿈
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        System.out.println("Before sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }

        bubbleSort(arr);

        System.out.println("\n\nAfter sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }
    }
}

```

코드를 보면,  
- **bubbleSort**: 배열을 받아서 버블 정렬을 수행
- **for 루프**: 배열의 각 요소를 반복하면서 인접한 두 원소를 비교, 필요하면 자리를 바꿈
- **main**: 함수에서는 정렬하기 전과 후의 배열을 출력


## 선택 정렬 (Selection Sort)
리스트에서 가장 작은 원소를 찾아서 맨 앞으로 보내는 작업을 반복한다.  
동작 원리를 보면,  
1. 리스트에서 가장 작은 원소를 찾아 리스트의 첫 번째 원소와 자리를 바꾼다.  
2. 두 번째 원소부터 나머지 리스트를 반복하면서 다시 가장 작은 원소를 찾아 자리를 바꾼다.  
3. 이 과정을 리스트 끝까지 반복한다.  

```java
public class SelectionSortExample {
    public static void selectionSort(int[] arr) {
        int n = arr.length;

        for (int i = 0; i < n - 1; i++) {
            // 현재 위치에서 최소값을 찾음
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }

            // 최소값을 현재 위치와 교환
            int temp = arr[minIdx];
            arr[minIdx] = arr[i];
            arr[i] = temp;
        }
    }

    public static void main(String[] args) {
        int[] arr = {29, 10, 14, 37, 13};
        System.out.println("Before sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }

        selectionSort(arr);

        System.out.println("\n\nAfter sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }
    }
}

```  
위 코드를 보면,  
- **electionSort**: 함수는 배열을 받아서 선택 정렬을 수행한다.  
- **for**: 루프는 가장 작은 원소를 찾아서 현재 위치의 원소와 교환한다.  
- **main**: 함수에서는 정렬 전과 후의 배열을 출력한다.  

## 삽입 정렬 (Insertion Sort)
리스트를 앞에서부터 차례대로 정렬된 부분과 그렇지 않은 부분으로 나누어, 정렬되지 않은 부분의 첫 번째 원소를 정렬된 부분에 적절한 위치에 삽입한다.  

동작원리는,  
1. 두 번째 원소부터 시작하여 현재 원소를 정렬된 부분의 알맞은 위치에 삽입한다.  
2. 리스트의 끝까지 이 과정을 반복한다.  

```java
public class InsertionSortExample {
    public static void insertionSort(int[] arr) {
        int n = arr.length;

        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;

            // 정렬된 부분에서 현재 원소(key)가 들어갈 위치를 찾음
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }

            // 현재 원소를 올바른 위치에 삽입
            arr[j + 1] = key;
        }
    }

    public static void main(String[] args) {
        int[] arr = {31, 41, 59, 26, 41, 58};
        System.out.println("Before sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }

        insertionSort(arr);

        System.out.println("\n\nAfter sorting:");
        for (int num : arr) {
            System.out.print(num + " ");
        }
    }
}

```
위 코들르 보면,  
- **insertionSort**: 함수는 배열을 받아서 삽입 정렬을 수행한다.  
- **while**: 루프는 현재 원소가 들어갈 위치를 찾기 위해 정렬된 부분을 뒤로 이동시킨다.  
- **main**: 함수에서는 정렬 전과 후의 배열을 출력한다.  

이렇게 해서 버블 정렬, 선택 정렬, 삽입 정렬을 살펴보았다.  
기본적인 정렬 방법이며, 데이터를 순서대로 정렬하는 기본 원리를 알아가면 된다.  