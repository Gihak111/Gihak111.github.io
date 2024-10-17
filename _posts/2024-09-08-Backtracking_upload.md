---
layout: single  
title:  "알고리즘 정리 11. 백트래킹 알고리즘"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---

# Backtracking
이번에는 백트래킹(Backtracking) 알고리즘에 대해 알아보자.  
백트래킹은 탐색 알고리즘 중 하나로, 모든 경우의 수를 탐색하여 문제의 해를 찾는 방법이다.  
이 알고리즘은 주로 **재귀**를 통해 구현되며, 최적의 해를 찾거나 특정 조건을 만족하는 모든 해를 찾는 데 사용된다.  

# 백트래킹 알고리즘 개요
백트래킹은 가능한 모든 경우의 수를 시도해보는 **브루트포스(Brute Force)** 방법의 변형이다.  
하지만 백트래킹은 불필요한 경로를 미리 차단(가지치기)함으로써, 탐색 공간을 줄여 효율성을 높이는 것이 핵심이다.  
이 과정에서 해를 찾지 못하는 경로를 빠르게 포기하고, 다른 경로를 시도하게 된다.  

## 백트래킹 알고리즘의 동작 원리
백트래킹은 다음과 같은 단계로 동작한다:  

1. **결정 트리 탐색**  
   문제를 해결하기 위해 가능한 모든 선택지를 나열한 결정 트리를 구성한다.  
   각 노드는 특정 상태를 나타내며, 가지는 선택지를 나타낸다.  

2. **조건 확인**  
   현재 상태가 문제의 조건을 만족하는지 확인한다.  
   조건을 만족하지 않는다면, 해당 경로는 더 이상 탐색하지 않고 포기(가지치기)한다.  

3. **해 탐색 및 재귀 호출**  
   조건을 만족하는 경우, 다음 단계로 나아가 탐색을 계속한다.  
   이는 재귀 호출을 통해 구현되며, 새로운 선택지를 고려한다.  

4. **백트래킹**  
   특정 경로가 더 이상 유효하지 않거나 해를 찾을 수 없는 경우, 이전 단계로 돌아가 다른 선택지를 시도한다.  

5. **해 찾기 또는 종료**  
   모든 선택지를 시도해 원하는 해를 찾거나, 모든 경로를 탐색한 후 종료한다.  

## 백트래킹 알고리즘의 시간 복잡도
백트래킹의 시간 복잡도는 문제의 종류에 따라 다르지만, 최악의 경우 모든 가능성을 탐색해야 하므로 지수 시간 복잡도(O(2^n))를 가질 수 있다.  
하지만, 가지치기를 통해 탐색 공간을 줄일 수 있기 때문에, 실질적인 실행 시간은 이보다 훨씬 적을 수 있다.  

## 백트래킹 알고리즘 예제 코드
아래는 백트래킹을 사용해 N-Queens 문제를 해결하는 간단한 예제 코드이다.  
이 문제는 N x N 크기의 체스판 위에 N개의 퀸을 서로 공격하지 않도록 배치하는 문제이다.  

```java
public class NQueens {

    private static void solveNQueens(int n) {
        int[] board = new int[n];
        placeQueen(board, 0, n);
    }

    private static void placeQueen(int[] board, int row, int n) {
        if (row == n) {
            printBoard(board, n);
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isSafe(board, row, col, n)) {
                board[row] = col;
                placeQueen(board, row + 1, n);
            }
        }
    }

    private static boolean isSafe(int[] board, int row, int col, int n) {
        for (int i = 0; i < row; i++) {
            int placedCol = board[i];
            if (placedCol == col || Math.abs(placedCol - col) == Math.abs(i - row)) {
                return false;
            }
        }
        return true;
    }

    private static void printBoard(int[] board, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i] == j) {
                    System.out.print("Q ");
                } else {
                    System.out.print(". ");
                }
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {
        int n = 8;  // 예를 들어, 8-Queens 문제
        solveNQueens(n);
    }
}
```

## 백트래킹 알고리즘 사용 사례
- **N-Queens 문제**  
  위에서 설명한 것처럼, 백트래킹을 사용해 N-Queens 문제를 해결할 수 있다.  
  이 문제는 체스판 위에 퀸을 배치하는 문제로, 서로 공격하지 않도록 퀸을 배치해야 한다.  

- **수도쿠(Sudoku)**  
  백트래킹은 수도쿠 퍼즐을 푸는 데에도 사용된다.  
  각 칸에 숫자를 채워가면서, 규칙에 맞지 않는 경우 해당 경로를 포기하고 다른 숫자를 시도한다.  

- **퍼즐 문제**  
  미로 찾기, 단어 퍼즐, 조합 탐색 등에서도 백트래킹이 자주 사용된다.  
  이러한 문제들은 가능한 모든 경우를 탐색하여 해를 찾는 것이 필요하다.  

- **조합 및 순열 생성**  
  백트래킹은 조합이나 순열을 생성하는 문제에서도 유용하다.  
  특정 조건을 만족하는 모든 조합이나 순열을 찾아야 할 때, 백트래킹을 통해 효율적으로 해결할 수 있다.  

## 백트래킹 알고리즘의 장점과 한계
- **장점**  
  백트래킹은 모든 가능한 해를 탐색하기 때문에, 정확한 해를 보장할 수 있다.  
  가지치기를 통해 탐색 공간을 줄여 효율성을 높일 수 있으며, 복잡한 문제를 해결하는 데 매우 유용하다.  

- **한계**  
  최악의 경우, 백트래킹은 모든 가능성을 탐색해야 하므로 시간이 오래 걸릴 수 있다.  
  또한, 큰 문제에서는 탐색 공간이 급격히 증가하여 현실적으로 실행이 어려울 수 있다.  