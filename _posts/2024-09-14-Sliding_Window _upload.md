---
layout: single
title:  "알고리즘 정리 15. 슬라이딩 윈도우"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 플로이드-와샬(Floyd-Warshall) 알고리즘
이번에는 플로이드 와샬 알고리즘에 대해 알아보자. 이 알고리즘은 모든 쌍의 최단 경로를 구하는 알고리즘으로, 그래프 이론에서 매우 중요한 역할을 한다. 특히, 그래프의 모든 노드 쌍 사이의 최단 경로를 계산할 수 있다.

# 플로이드-와샬 알고리즘 개요
플로이드-와샬 알고리즘은 **동적 프로그래밍**을 기반으로 하여, 주어진 그래프에서 모든 노드 쌍 간의 최단 경로를 찾는다. 이 알고리즘은 노드 수가 많을 때도 효율적으로 모든 쌍의 최단 경로를 구할 수 있다. 기본 아이디어는 각 노드를 중간 노드로 사용하여, 최단 경로를 갱신하는 것이다.

## 플로이드-와샬 알고리즘의 동작 원리
플로이드-와샬 알고리즘은 다음과 같은 단계로 동작한다:

1. **초기화**  
   - 그래프의 모든 간선에 대해 거리 행렬을 초기화한다. 만약 간선이 존재하지 않으면 무한대(`∞`)로 설정한다.

2. **중간 노드 추가**  
   - 모든 노드를 중간 노드로 고려하여 최단 경로를 갱신한다. 즉, 중간 노드를 추가한 경로가 기존 경로보다 짧은지 확인하고, 짧다면 거리 행렬을 갱신한다.

3. **최단 경로 계산**  
   - 중간 노드를 사용하여 모든 노드 쌍 사이의 최단 경로를 반복적으로 계산한다.

## 플로이드-와샬 알고리즘의 시간 복잡도
플로이드-와샬 알고리즘의 시간 복잡도는 O(n^3)이다. 여기서 `n`은 그래프의 노드 수를 의미한다. 따라서, 노드 수가 많을수록 알고리즘의 실행 시간이 증가한다.

## 예제 코드
아래는 플로이드-와샬 알고리즘을 구현한 간단한 예제 코드이다. 이 코드는 주어진 그래프에서 모든 노드 쌍 간의 최단 경로를 계산한다.

```java
import java.util.Arrays;

public class FloydWarshall {

    final static int INF = 99999;  // 무한대를 나타내는 값

    public static void floydWarshall(int[][] graph) {
        int V = graph.length;
        int[][] dist = new int[V][V];

        // 거리 행렬 초기화
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i == j) {
                    dist[i][j] = 0;
                } else if (graph[i][j] != 0) {
                    dist[i][j] = graph[i][j];
                } else {
                    dist[i][j] = INF;
                }
            }
        }

        // 플로이드-와샬 알고리즘 적용
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF &&
                        dist[i][j] > dist[i][k] + dist[k][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        // 결과 출력
        printSolution(dist);
    }

    public static void printSolution(int[][] dist) {
        int V = dist.length;
        System.out.println("최단 경로 거리 행렬:");
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == INF) {
                    System.out.print("INF\t");
                } else {
                    System.out.print(dist[i][j] + "\t");
                }
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 3, INF, INF, INF, INF},
            {INF, 0, 1, INF, INF, INF},
            {INF, INF, 0, 7, INF, 2},
            {INF, INF, INF, 0, 2, INF},
            {INF, INF, INF, INF, 0, 3},
            {INF, INF, INF, INF, INF, 0}
        };

        floydWarshall(graph);
    }
}
```

## 플로이드-와샬 알고리즘 사용 사례
- **네트워크 최적화**  
  네트워크에서 모든 노드 쌍 간의 최단 경로를 구하여, 최적의 경로를 계산하거나 문제를 해결할 때 유용하다.

- **도로망 분석**  
  도로망의 모든 도시 간의 최단 경로를 계산하여 교통 흐름을 분석하거나 최적의 경로를 찾는 데 사용할 수 있다.

- **그래프 이론 연구**  
  그래프 이론에서 모든 쌍의 최단 경로를 구할 때 사용되며, 다양한 문제에 응용될 수 있다.  

## 플로이드-와샬 알고리즘의 장점과 한계
- **장점**  
  모든 쌍의 최단 경로를 한 번에 계산할 수 있어, 복잡한 그래프 문제를 해결하는 데 유용하다. 구현이 간단하고 직관적이다.

- **한계**  
  노드 수가 많을 경우 O(n^3)의 시간 복잡도로 인해 계산량이 커질 수 있다. 또한, 음수 가중치가 있는 그래프에서 최단 경로를 찾을 때는 주의가 필요하다.

플로이드-와샬 알고리즘은 그래프의 모든 노드 쌍 간의 최단 경로를 효과적으로 계산할 수 있는 강력한 도구이다. 이를 통해 복잡한 네트워크 문제를 해결하고, 최적의 경로를 찾는 데 도움을 줄 수 있을 것이다.