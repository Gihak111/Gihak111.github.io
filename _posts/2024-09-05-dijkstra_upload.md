---
layout: single  
title:  "알고리즘 정리 8. 다익스트라 알고리즘"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---
  
# Dijkstra
이번에는 다익스트라 알고리즘에 대해 알아보자. 다익스트라 알고리즘은 그래프에서 최단 경로를 찾기 위한 매우 유명한 알고리즘으로, 특히 가중치가 있는 그래프에서 한 노드에서 다른 모든 노드까지의 최단 경로를 구하는 데 사용된다.

# 다익스트라 알고리즘 개요
다익스트라 알고리즘은 그래프 내에서 출발 노드로부터 다른 모든 노드까지의 최단 경로를 찾는 알고리즘이다. 이 알고리즘은 주로 네트워크 경로 최적화, 지도 애플리케이션에서 경로 탐색, 그리고 게임에서 캐릭터의 이동 경로 결정 등에 사용된다.

## 다익스트라 알고리즘의 동작 원리
다익스트라 알고리즘은 다음과 같은 단계로 동작한다:

1. **시작 노드 설정**  
   출발 노드를 선택하고, 그 노드의 거리를 0으로 설정한다. 나머지 노드들은 무한대(∞)로 설정한다.

2. **최단 거리 노드 선택**  
   아직 방문하지 않은 노드 중에서 현재까지의 최단 거리가 가장 짧은 노드를 선택한다.

3. **거리 갱신**  
   선택된 노드를 통해 다른 인접 노드로 가는 경로를 계산하고, 현재 기록된 거리보다 더 짧은 경로가 발견되면 그 거리를 갱신한다.

4. **반복**  
   모든 노드를 방문할 때까지 2번과 3번 과정을 반복한다.

## 다익스트라 알고리즘의 시간 복잡도
다익스트라 알고리즘의 시간 복잡도는 사용된 자료구조에 따라 달라진다. 일반적인 구현에서 우선순위 큐를 사용할 경우 O((V + E) log V)이다. 여기서 V는 노드의 수, E는 간선의 수를 의미한다.

## 다익스트라 알고리즘 예제 코드
아래는 다익스트라 알고리즘을 우선순위 큐(Priority Queue)를 사용해 구현한 예제 코드이다. 이 코드에서는 그래프에서 특정 노드를 시작으로 다른 모든 노드까지의 최단 경로를 계산한다.

```java
import java.util.*;

class Node implements Comparable<Node> {
    int vertex;
    int weight;

    Node(int vertex, int weight) {
        this.vertex = vertex;
        this.weight = weight;
    }

    @Override
    public int compareTo(Node other) {
        return this.weight - other.weight;
    }
}

public class DijkstraExample {

    public static void dijkstra(int start, Map<Integer, List<Node>> graph, int[] distances) {
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.add(new Node(start, 0));
        distances[start] = 0;

        while (!pq.isEmpty()) {
            Node currentNode = pq.poll();
            int currentVertex = currentNode.vertex;

            for (Node neighbor : graph.get(currentVertex)) {
                int newDist = distances[currentVertex] + neighbor.weight;

                if (newDist < distances[neighbor.vertex]) {
                    distances[neighbor.vertex] = newDist;
                    pq.add(new Node(neighbor.vertex, newDist));
                }
            }
        }
    }

    public static void main(String[] args) {
        Map<Integer, List<Node>> graph = new HashMap<>();
        graph.put(1, Arrays.asList(new Node(2, 2), new Node(3, 4)));
        graph.put(2, Arrays.asList(new Node(3, 1), new Node(4, 7)));
        graph.put(3, Arrays.asList(new Node(5, 3)));
        graph.put(4, Arrays.asList(new Node(5, 1)));
        graph.put(5, new ArrayList<>());

        int[] distances = new int[6];
        Arrays.fill(distances, Integer.MAX_VALUE);

        dijkstra(1, graph, distances);

        for (int i = 1; i < distances.length; i++) {
            System.out.println("Distance from 1 to " + i + " is " + distances[i]);
        }
    }
}
```

# 다익스트라 알고리즘 사용 사례
- **네트워크 경로 최적화**  
  인터넷 라우팅에서 네트워크 내에서의 데이터 전송 경로를 최적화하는 데 다익스트라 알고리즘이 사용된다. 가장 빠르고 효율적인 경로를 찾는 데 매우 유용하다.  

- **지도 및 내비게이션 시스템**  
  구글 지도나 내비게이션 앱에서 목적지까지의 최단 경로를 계산할 때, 다익스트라 알고리즘이 사용된다. 도로의 거리, 시간, 교통 상황 등을 반영하여 최적의 경로를 제공한다.  

- **게임 개발**  
  게임에서 캐릭터의 이동 경로를 최적화하거나, AI가 맵에서 목표 지점까지 도달하는 경로를 계산할 때 다익스트라 알고리즘을 사용할 수 있다.  

# 다익스트라 알고리즘의 장점과 한계
- **장점**  
  다익스트라 알고리즘은 가중치가 있는 그래프에서 최단 경로를 매우 효율적으로 계산할 수 있다. 특히, 모든 간선의 가중치가 양수일 때 정확하게 작동한다.  

- **한계**  
  음수 가중치를 가진 간선이 포함된 그래프에서는 다익스트라 알고리즘을 사용할 수 없다. 이런 경우에는 벨만-포드(Bellman-Ford) 알고리즘 같은 다른 방법을 고려해야 한다.  