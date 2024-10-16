---
layout: single  
title:  "알고리즘 정리 6. 너비 우선 탐색(BFS) & 깊이 우선 탐색(DFS)"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---

# BFS & DFS
이번에는 너비 우선 탐색(BFS)과 깊이 우선 탐색(DFS) 알고리즘에 대해 알아보자.  
이 두 가지 탐색 방법은 그래프나 트리 구조에서 데이터를 탐색하거나 순회할 때 자주 사용되는 기본 알고리즘이다.  

# 너비 우선 탐색(BFS)
너비 우선 탐색은 주로 그래프나 트리에서 최단 경로를 찾거나, 특정 레벨에 있는 노드들을 탐색할 때 사용된다.  
BFS는 먼저 시작 노드에서 가까운 노드들을 차례로 탐색하며, 큐(queue)를 사용하여 구현된다.  

## BFS의 동작 원리
BFS는 다음과 같은 단계로 동작한다:  
1. 시작 노드를 큐에 삽입하고, 방문 표시를 한다.  
2. 큐에서 노드를 하나씩 꺼내 해당 노드와 인접한 모든 노드를 탐색한다.  
3. 아직 방문하지 않은 인접 노드를 큐에 삽입하고, 방문 표시를 한다.  
4. 큐가 빌 때까지 2번과 3번 과정을 반복한다.  

## BFS의 시간 복잡도
BFS의 시간 복잡도는 O(V + E)이다. 여기서 V는 노드의 수, E는 간선의 수를 의미한다.  
BFS는 노드와 간선의 개수에 비례하여 시간이 소요된다.  

## BFS 예제 코드
아래는 BFS를 구현한 간단한 예제 코드이다.  
이 코드에서는 그래프에서 특정 노드를 시작으로 너비 우선 탐색을 수행한다.  

```java
import java.util.*;

public class BFSExample {

    public static void bfs(int startNode, Map<Integer, List<Integer>> graph) {
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();

        queue.add(startNode);
        visited.add(startNode);

        while (!queue.isEmpty()) {
            int currentNode = queue.poll();
            System.out.print(currentNode + " ");

            for (int neighbor : graph.get(currentNode)) {
                if (!visited.contains(neighbor)) {
                    queue.add(neighbor);
                    visited.add(neighbor);
                }
            }
        }
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        graph.put(1, Arrays.asList(2, 3, 4));
        graph.put(2, Arrays.asList(5, 6));
        graph.put(3, Arrays.asList(7, 8));
        graph.put(4, Arrays.asList(9));
        graph.put(5, new ArrayList<>());
        graph.put(6, new ArrayList<>());
        graph.put(7, new ArrayList<>());
        graph.put(8, new ArrayList<>());
        graph.put(9, new ArrayList<>());

        bfs(1, graph);  // 출력: 1 2 3 4 5 6 7 8 9
    }
}
```

# 깊이 우선 탐색(DFS)
깊이 우선 탐색은 가능한 깊은 노드까지 탐색을 진행한 후, 더 이상 갈 수 없는 경우 다시 이전 노드로 돌아가는 방식으로 탐색을 진행한다.  
DFS는 주로 재귀를 통해 구현되며, 백트래킹(backtracking) 문제 해결에 자주 사용된다.  

## DFS의 동작 원리
DFS는 다음과 같은 단계로 동작한다:  
1. 시작 노드를 방문하고, 방문 표시를 한다.  
2. 현재 노드에서 방문하지 않은 인접 노드가 있으면, 그 노드를 따라가 탐색을 계속한다.  
3. 더 이상 방문할 노드가 없으면, 이전 노드로 돌아가면서 탐색을 이어나간다.  
4. 모든 노드를 방문할 때까지 과정을 반복한다.  

## DFS의 시간 복잡도
DFS의 시간 복잡도 역시 O(V + E)이다.  
노드와 간선을 모두 탐색하기 때문에 BFS와 동일한 시간 복잡도를 가진다.  

## DFS 예제 코드
아래는 DFS를 재귀적으로 구현한 간단한 예제 코드이다.  
이 코드에서는 그래프에서 특정 노드를 시작으로 깊이 우선 탐색을 수행한다.  

```java
import java.util.*;

public class DFSExample {

    public static void dfs(int currentNode, Map<Integer, List<Integer>> graph, Set<Integer> visited) {
        visited.add(currentNode);
        System.out.print(currentNode + " ");

        for (int neighbor : graph.get(currentNode)) {
            if (!visited.contains(neighbor)) {
                dfs(neighbor, graph, visited);
            }
        }
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        graph.put(1, Arrays.asList(2, 3, 4));
        graph.put(2, Arrays.asList(5, 6));
        graph.put(3, Arrays.asList(7, 8));
        graph.put(4, Arrays.asList(9));
        graph.put(5, new ArrayList<>());
        graph.put(6, new ArrayList<>());
        graph.put(7, new ArrayList<>());
        graph.put(8, new ArrayList<>());
        graph.put(9, new ArrayList<>());

        Set<Integer> visited = new HashSet<>();
        dfs(1, graph, visited);  // 출력: 1 2 5 6 3 7 8 4 9
    }
}
```

# BFS & DFS의 사용 사례
- **경로 탐색**  
  BFS는 최단 경로를 찾는 문제에 자주 사용되며, DFS는 특정 경로를 찾거나 모든 가능한 경로를 탐색해야 하는 문제에 유용하다.  

- **그래프 순회**  
  BFS와 DFS는 그래프 순회에서 각각 다른 방식으로 그래프를 탐색하므로, 문제의 특성에 따라 적합한 방법을 선택할 수 있다.  

- **백트래킹**  
  DFS는 백트래킹 알고리즘의 핵심으로, 가능한 모든 해결책을 탐색하는데 사용된다.  
  예를 들어, 퍼즐 문제나 조합 문제에서 활용된다.  

# BFS와 DFS 선택 기준
- **탐색 목적**  
  최단 경로를 찾고자 한다면 BFS를, 모든 경로를 탐색하거나 특정 깊이까지 탐색하고자 한다면 DFS를 사용하는 것이 좋다.  

- **메모리 사용**  
  BFS는 큐에 모든 노드를 저장해야 하므로 메모리 사용이 많다.  
  반면 DFS는 재귀 호출을 사용하므로 상대적으로 적은 메모리를 사용한다.  

BFS와 DFS는 모두 중요한 탐색 알고리즘이며, 상황에 맞게 잘 선택하고 활용하면 효율적으로 문제를 해결할 수 있을 것이다.  
이 글을 통해 BFS와 DFS에 대해 더 깊이 이해할 수 있기를 바란다.  