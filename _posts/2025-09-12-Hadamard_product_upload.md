---
layout: single
title:  "Hadamard Product"
categories: "AI"
tag: "linear algebra"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Hadamard Product
오늘은, 딥러닝 하다보면 수식으로 많이 보게 되는 이 Hadamard Product에 대해 짧게 알아보자.  

## 하드마르 곱
두 행렬의 같은 위치에 있는 원소들끼리 곱하는 연산이다.  
즉, **원소별 곱셈(element-wise product)**인 거다.  
복잡한 행렬 곱셈과는 완전히 다른 방식이다.  


### **하드마르 곱(Hadamard Product) 계산**

$A$와 $B$ 두 행렬의 하드마르 곱은 $A \circ B$로 표기하며, 그 계산 공식은 다음과 같다.
$$(A \circ B)_{ij} = A_{ij} \cdot B_{ij}$$


이제 주어진 행렬들의 하드마르 곱을 계산해 보자.
$A = [[1, 2], [3, 4]]$
$B = [[5, 6], [7, 8]]$

1.  **왼쪽 위 원소**: $1 \times 5 = 5$
2.  **오른쪽 위 원소**: $2 \times 6 = 12$
3.  **왼쪽 아래 원소**: $3 \times 7 = 21$
4.  **오른쪽 아래 원소**: $4 \times 8 = 32$

그렇게 해서 얻은 최종 결과는 새로운 행렬 C가 된다.

$$A \circ B = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$$

간단하니, 꼭 알고 있자.  
딥러닝 하면 무조건 볼 날 온다.  