---
layout: single
title:  "알고리즘 정리 13. 위상정렬"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Topological Sorting
정렬 방법 중 하나인 위상정렬이다.  
순서가 정해져 있지만, 노드가 여러개일 경우 차례로 작업을 수해애 할 때 노드를 옮겨야 할 때 순서를 결정해 주기 위해 사용한다.  
간단하게, 사이클 그래프에서 시작점을 찾지 못하는 경우 해결할 수 있다.  

위상 정렬은 방향성 비순환 그래프(DAG, Directed Acyclic Graph)에서 정점들의 선후 관계를 정렬하는 알고리즘이다.  
주로 작업의 순서를 정하거나, 과목의 수강 순서를 결정할 때 사용된다.  

### 예제 그래프
다음과 같은 그래프를 위상 정렬해보자:

```
A → C
B → C
C → D
```

이 그래프를 위상 정렬하면 `A → B → C → D` 또는 `B → A → C → D`와 같은 순서를 얻을 수 있다.

### 자바 코드 구현

아래는 위상 정렬을 구현한 자바 코드입니다. 이 코드는 Kahn의 알고리즘을 사용하며, 각 단계마다 주석을 달아 설명하였습니다.

```java
import java.util.*;

public class TopologicalSort {
    // 정점의 개수
    private int vertices;
    // 인접 리스트를 사용하여 그래프를 표현
    private List<List<Integer>> adjList;

    // 생성자
    public TopologicalSort(int vertices) {
        this.vertices = vertices;
        adjList = new ArrayList<>();
        // 모든 정점에 대한 리스트 초기화
        for (int i = 0; i < vertices; i++) {
            adjList.add(new ArrayList<>());
        }
    }

    // 간선을 추가하는 메서드
    public void addEdge(int from, int to) {
        adjList.get(from).add(to);
    }

    // 위상 정렬을 수행하는 메서드
    public List<Integer> topologicalSort() {
        // 각 정점의 진입 차수를 저장할 배열
        int[] inDegree = new int[vertices];

        // 모든 간선을 순회하며 진입 차수 계산
        for (int i = 0; i < vertices; i++) {
            for (int neighbor : adjList.get(i)) {
                inDegree[neighbor]++;
            }
        }

        // 진입 차수가 0인 정점을 담을 큐
        Queue<Integer> queue = new LinkedList<>();

        // 초기 큐에 진입 차수가 0인 정점들을 추가
        for (int i = 0; i < vertices; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        // 위상 정렬 결과를 저장할 리스트
        List<Integer> topoOrder = new ArrayList<>();

        // 큐가 빌 때까지 반복
        while (!queue.isEmpty()) {
            // 큐에서 정점을 하나 꺼냄
            int current = queue.poll();
            topoOrder.add(current);

            // 현재 정점과 연결된 모든 정점의 진입 차수 감소
            for (int neighbor : adjList.get(current)) {
                inDegree[neighbor]--;
                // 진입 차수가 0이 된 정점을 큐에 추가
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // 모든 정점을 방문했는지 확인 (DAG 여부)
        if (topoOrder.size() != vertices) {
            throw new IllegalStateException("그래프에 사이클이 존재하여 위상 정렬을 수행할 수 없습니다.");
        }

        return topoOrder;
    }

    public static void main(String[] args) {
        /*
         * 예제 그래프:
         * A → C
         * B → C
         * C → D
         *
         * 정점 번호:
         * 0: A
         * 1: B
         * 2: C
         * 3: D
         */

        // 정점의 개수는 4개 (A, B, C, D)
        TopologicalSort ts = new TopologicalSort(4);

        // 간선 추가 (from -> to)
        ts.addEdge(0, 2); // A → C
        ts.addEdge(1, 2); // B → C
        ts.addEdge(2, 3); // C → D

        // 위상 정렬 수행
        List<Integer> result = ts.topologicalSort();

        // 정점 번호를 문자로 변환하여 출력
        System.out.print("위상 정렬 결과: ");
        for (int vertex : result) {
            // 'A'의 ASCII 값은 65이므로, 0을 더하면 'A', 1을 더하면 'B' 등으로 변환
            System.out.print((char) ('A' + vertex) + " ");
        }
        // 출력 예시: A B C D 또는 B A C D
    }
}
```

### 코드 설명

1. **그래프 표현**:
    - `adjList`: 인접 리스트를 사용하여 그래프를 표현한다. 각 정점마다 연결된 정점들의 리스트를 저장한다.
    - `addEdge(int from, int to)`: 그래프에 간선을 추가하는 메서드이다. `from` 정점에서 `to` 정점으로 향하는 간선을 추가한다.

2. **진입 차수 계산**:
    - `inDegree[]`: 각 정점의 진입 차수를 저장하는 배열이다. 모든 간선을 순회하며 진입 차수를 계산한다.

3. **큐 초기화**:
    - 진입 차수가 0인 정점을 큐에 추가한다. 이러한 정점들은 선행 작업이 필요 없는 작업들 이다.

4. **위상 정렬 수행**:
    - 큐에서 정점을 하나씩 꺼내어 결과 리스트에 추가한다.
    - 해당 정점과 연결된 모든 정점의 진입 차수를 감소시킨다.
    - 진입 차수가 0이 된 정점을 큐에 추가하여 다음 단계에서 처리할 수 있도록 한다.

5. **사이클 검사**:
    - 모든 정점을 방문했는지 확인한다. 만약 방문하지 못한 정점이 있다면 그래프에 사이클이 존재하여 위상 정렬을 수행할 수 없다.

6. **결과 출력**:
    - 위상 정렬의 결과를 문자로 변환하여 출력한다. 정점 번호를 문자로 변환하기 위해 `'A' + vertex`를 사용했다.

### 실행 결과

위의 코드를 실행하면 다음과 같은 결과를 얻을 수 있다:

```
위상 정렬 결과: A B C D 
```

또는

```
위상 정렬 결과: B A C D 
```

두 결과 모두 올바른 위상 정렬이다. 위상 정렬은 여러 가지 가능한 결과가 있을 수 있으며, 이 예제에서는 `A`와 `B`의 순서가 서로 바뀌어도 올바른 결과이다.

### 추가 설명

- **DAG (Directed Acyclic Graph)**:
    - 방향성이 있는 그래프이면서 사이클이 없는 그래프를 의미한다.  
    - 위상 정렬은 반드시 DAG에서만 가능하다.
  
- **Kahn의 알고리즘**:
    - 위상 정렬을 수행하는 대표적인 알고리즘 중 하나로, 진입 차수를 활용하여 순서를 정한다.
  
- **시간 복잡도**:
    - O(V + E), V는 정점의 수, E는 간선의 수 이다. 
    - 모든 정점과 간선을 한 번씩 처리하기 때문이다.

위 과정을 통해 위상 정렬을 알 수 있다.