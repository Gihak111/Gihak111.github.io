---
layout: single  
title:  "알고리즘 정리 10. 분리 집합 알고리즘"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---

# Disjoint Set
이번에는 분리 집합 알고리즘에 대해 알아보자.  
분리 집합 알고리즘은 서로소 집합 자료 구조로도 알려져 있으며, 효율적으로 합집합 연산과 찾기 연산을 수행할 수 있도록 설계된 알고리즘이다.  
이 알고리즘은 주로 그래프에서 연결 요소를 찾거나, 크루스칼 알고리즘에서 최소 신장 트리를 찾는 데 사용된다.  

# 분리 집합 알고리즘 개요
분리 집합 알고리즘은 여러 개의 원소가 있을 때, 이들 원소를 몇 개의 집합으로 분리하여 관리하고, 두 원소가 같은 집합에 속해 있는지를 판단하는 문제를 해결하는 데 사용된다.  
대표적인 연산으로는 **합집합(union)** 연산과 **찾기(find)** 연산이 있다.  

## 분리 집합 알고리즘의 주요 연산
분리 집합 알고리즘은 크게 두 가지 연산을 중심으로 동작한다:  

1. **합집합(Union)**  
   두 개의 집합을 하나로 합치는 연산이다. 두 원소가 속한 집합을 하나로 병합하여, 두 원소가 같은 집합에 속하게 만든다.  

2. **찾기(Find)**  
   특정 원소가 속한 집합의 대표(루트) 원소를 찾는 연산이다. 이 연산을 통해 두 원소가 같은 집합에 속해 있는지를 확인할 수 있다.  

## 최적화 기법: 경로 압축(Path Compression)과 랭크에 의한 합치기(Union by Rank)
분리 집합 알고리즘의 효율성을 높이기 위해 두 가지 최적화 기법이 자주 사용된다:  

- **경로 압축(Path Compression)**  
  찾기 연산을 수행할 때, 모든 원소가 직접 루트 노드를 가리키도록 하여 트리의 높이를 줄이는 방법이다.  
  이로 인해 나중에 찾기 연산이 더욱 빠르게 수행된다.  

- **랭크에 의한 합치기(Union by Rank)**  
  합집합 연산을 수행할 때, 트리의 높이가 더 낮은 집합을 트리의 높이가 더 높은 집합 아래에 연결하여 트리의 높이를 최소화하는 방법이다.  

이 두 가지 최적화 기법을 함께 사용하면, 분리 집합 알고리즘의 시간 복잡도는 거의 상수 시간인 O(α(n))에 근접하게 된다.  
여기서 α(n)은 아커만 함수의 역함수로, 매우 느리게 증가하는 함수이다.  

## 분리 집합 알고리즘 예제 코드
아래는 경로 압축과 랭크에 의한 합치기 최적화 기법을 사용하여 구현한 분리 집합 알고리즘의 예제 코드이다.  

```java
public class DisjointSet {

    private int[] parent;
    private int[] rank;

    public DisjointSet(int size) {
        parent = new int[size];
        rank = new int[size];

        for (int i = 0; i < size; i++) {
            parent[i] = i;  // 초기에는 자기 자신이 부모
            rank[i] = 1;    // 초기 랭크는 1
        }
    }

    public int find(int node) {
        if (parent[node] != node) {
            parent[node] = find(parent[node]);  // 경로 압축
        }
        return parent[node];
    }

    public void union(int node1, int node2) {
        int root1 = find(node1);
        int root2 = find(node2);

        if (root1 != root2) {
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else if (rank[root1] < rank[root2]) {
                parent[root1] = root2;
            } else {
                parent[root2] = root1;
                rank[root1]++;
            }
        }
    }

    public static void main(String[] args) {
        DisjointSet ds = new DisjointSet(7);

        ds.union(1, 2);
        ds.union(2, 3);
        ds.union(4, 5);
        ds.union(6, 7);
        ds.union(5, 6);

        System.out.println(ds.find(1) == ds.find(3));  // true, 같은 집합
        System.out.println(ds.find(1) == ds.find(4));  // false, 다른 집합

        ds.union(3, 7);

        System.out.println(ds.find(1) == ds.find(7));  // true, 같은 집합이 됨
    }
}
```

# 분리 집합 알고리즘 사용 사례
- **그래프 사이클 검출**  
  그래프에서 사이클이 존재하는지 여부를 확인하는 문제에서 분리 집합 알고리즘을 사용할 수 있다.  
  서로 다른 두 노드를 연결하려고 할 때, 이미 같은 집합에 속해 있다면 사이클이 존재함을 의미한다.  

- **최소 신장 트리(MST) 생성**  
  크루스칼 알고리즘에서 최소 신장 트리를 만들기 위해 분리 집합 알고리즘이 사용된다.  
  간선을 추가할 때, 사이클을 피하기 위해 두 노드가 같은 집합에 속해 있는지를 검사할 수 있다.  

- **네트워크 연결성**  
  네트워크 상에서 컴퓨터나 서버 간의 연결성을 분석할 때, 분리 집합 알고리즘을 사용하여 서로 연결된 컴퓨터 그룹을 찾을 수 있다.  

# 분리 집합 알고리즘의 장점과 한계
- **장점**  
  분리 집합 알고리즘은 합집합과 찾기 연산을 매우 효율적으로 수행할 수 있어, 그래프 관련 문제를 해결하는 데 필수적인 도구이다.  
  경로 압축과 랭크에 의한 합치기를 적용하면 거의 상수 시간에 가까운 성능을 제공한다.  

- **한계**  
  분리 집합 알고리즘은 각 집합을 관리하기 위해 트리 구조를 사용하므로, 매우 큰 데이터셋을 처리할 때는 메모리 사용량이 증가할 수 있다.  
