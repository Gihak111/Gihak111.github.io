---
layout: single  
title:  "알고리즘 정리 12. 최소 스패닝 트리 알고리즘"  
categories: "algorithm"  
tag: "code"  
toc: true  
author_profile: false  
sidebar:  
    nav: "docs"  
---

# Minimum Spanning Tree
이번에는 최소 스패닝 트리(Minimum Spanning Tree, MST) 알고리즘에 대해 알아보자. MST 알고리즘은 주어진 그래프에서 최소 비용으로 모든 노드를 연결하는 트리를 찾는 알고리즘이다. 네트워크 설계, 클러스터링, 이미지 처리 등 다양한 분야에서 활용된다.

# 최소 스패닝 트리(MST) 개요
그래프가 주어졌을 때, 모든 노드를 포함하면서 사이클이 없고, 가장 적은 가중치의 간선들로 이루어진 트리를 **최소 스패닝 트리(MST)**라고 한다. MST 알고리즘은 주로 **크루스칼(Kruskal)**과 **프림(Prim)** 두 가지 방법으로 구현된다.

## 최소 스패닝 트리 알고리즘의 동작 원리
최소 스패닝 트리를 찾기 위한 두 가지 대표적인 알고리즘이 있다:

1. **크루스칼 알고리즘(Kruskal's Algorithm)**  
   - **간선 중심**으로 동작하는 알고리즘이다.
   - 그래프의 모든 간선을 가중치의 오름차순으로 정렬한 뒤, 가장 작은 간선부터 선택해 나간다.
   - 이때, 사이클이 형성되지 않도록 주의하며, 간선을 추가할 때마다 **분리 집합(Disjoint Set)**을 사용해 두 노드가 이미 연결되어 있는지 확인한다.
   
2. **프림 알고리즘(Prim's Algorithm)**  
   - **정점 중심**으로 동작하는 알고리즘이다.
   - 임의의 정점에서 시작하여, 인접한 간선 중에서 가중치가 가장 작은 간선을 선택해 MST에 추가한다.
   - 이미 선택된 노드 집합에서 가장 적은 비용으로 연결되는 간선을 반복해서 추가하며, 모든 노드가 연결될 때까지 이 과정을 반복한다.

## 크루스칼 알고리즘 예제 코드
아래는 크루스칼 알고리즘을 이용해 MST를 구하는 예제 코드이다.

```java
import java.util.*;

class Edge implements Comparable<Edge> {
    int src, dest, weight;

    public Edge(int src, int dest, int weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }

    @Override
    public int compareTo(Edge other) {
        return this.weight - other.weight;
    }
}

class KruskalMST {
    int[] parent, rank;
    int vertices;
    List<Edge> edges;

    public KruskalMST(int vertices) {
        this.vertices = vertices;
        parent = new int[vertices];
        rank = new int[vertices];
        edges = new ArrayList<>();

        for (int i = 0; i < vertices; i++) {
            parent[i] = i;
            rank[i] = 1;
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

    public void addEdge(int src, int dest, int weight) {
        edges.add(new Edge(src, dest, weight));
    }

    public void kruskalMST() {
        Collections.sort(edges);

        List<Edge> mst = new ArrayList<>();
        int mstWeight = 0;

        for (Edge edge : edges) {
            int root1 = find(edge.src);
            int root2 = find(edge.dest);

            if (root1 != root2) {
                mst.add(edge);
                mstWeight += edge.weight;
                union(edge.src, edge.dest);
            }
        }

        System.out.println("Minimum Spanning Tree Weight: " + mstWeight);
        for (Edge edge : mst) {
            System.out.println(edge.src + " - " + edge.dest + " : " + edge.weight);
        }
    }

    public static void main(String[] args) {
        int vertices = 6;
        KruskalMST graph = new KruskalMST(vertices);

        graph.addEdge(0, 1, 4);
        graph.addEdge(0, 2, 4);
        graph.addEdge(1, 2, 2);
        graph.addEdge(1, 3, 5);
        graph.addEdge(2, 3, 5);
        graph.addEdge(2, 4, 6);
        graph.addEdge(3, 4, 8);
        graph.addEdge(3, 5, 10);
        graph.addEdge(4, 5, 7);

        graph.kruskalMST();
    }
}
```

## 프림 알고리즘 예제 코드
다음은 프림 알고리즘을 사용해 MST를 구하는 예제 코드이다.

```java
import java.util.*;

class PrimMST {
    private static class Node implements Comparable<Node> {
        int vertex, weight;

        Node(int vertex, int weight) {
            this.vertex = vertex;
            this.weight = weight;
        }

        @Override
        public int compareTo(Node other) {
            return this.weight - other.weight;
        }
    }

    private List<List<Node>> graph;
    private int vertices;

    public PrimMST(int vertices) {
        this.vertices = vertices;
        graph = new ArrayList<>(vertices);
        for (int i = 0; i < vertices; i++) {
            graph.add(new ArrayList<>());
        }
    }

    public void addEdge(int src, int dest, int weight) {
        graph.get(src).add(new Node(dest, weight));
        graph.get(dest).add(new Node(src, weight));
    }

    public void primMST() {
        PriorityQueue<Node> pq = new PriorityQueue<>();
        boolean[] inMST = new boolean[vertices];
        int[] key = new int[vertices];
        int[] parent = new int[vertices];
        Arrays.fill(key, Integer.MAX_VALUE);

        key[0] = 0;
        pq.add(new Node(0, key[0]));
        parent[0] = -1;

        while (!pq.isEmpty()) {
            int u = pq.poll().vertex;
            inMST[u] = true;

            for (Node node : graph.get(u)) {
                int v = node.vertex;
                int weight = node.weight;

                if (!inMST[v] && key[v] > weight) {
                    key[v] = weight;
                    pq.add(new Node(v, key[v]));
                    parent[v] = u;
                }
            }
        }

        int mstWeight = 0;
        for (int i = 1; i < vertices; i++) {
            System.out.println(parent[i] + " - " + i + " : " + key[i]);
            mstWeight += key[i];
        }
        System.out.println("Minimum Spanning Tree Weight: " + mstWeight);
    }

    public static void main(String[] args) {
        int vertices = 6;
        PrimMST graph = new PrimMST(vertices);

        graph.addEdge(0, 1, 4);
        graph.addEdge(0, 2, 4);
        graph.addEdge(1, 2, 2);
        graph.addEdge(1, 3, 5);
        graph.addEdge(2, 3, 5);
        graph.addEdge(2, 4, 6);
        graph.addEdge(3, 4, 8);
        graph.addEdge(3, 5, 10);
        graph.addEdge(4, 5, 7);

        graph.primMST();
    }
}
```

# 최소 스패닝 트리 알고리즘 사용 사례
- **네트워크 설계**  
  컴퓨터 네트워크, 통신 네트워크 등에서 최소 비용으로 모든 노드를 연결하는 최적의 설계도를 찾을 때 MST 알고리즘이 사용된다.

- **도로망 건설**  
  도시 간 도로망을 최소한의 비용으로 연결할 때, MST 알고리즘을 사용해 최적의 도로망 설계를 도출할 수 있다.

- **클러스터링**  
  데이터를 클러스터로 묶을 때, MST 알고리즘을 이용해 비슷한 데이터들을 연결하는 방식으로 클러스터를 형성할 수 있다.

# 최소 스패닝 트리 알고리즘의 장점과 한계
- **장점**  
  MST 알고리즘은 네트워크 설계, 최적화 문제 등 다양한 분야에서 최소 비용을 구하는 데 유용하다. 특히, 크루스칼과 프림 알고리즘은 각각

의 특징을 살려 다양한 그래프 구조에 적용 가능하다.

- **한계**  
  MST 알고리즘은 가중치가 동일한 간선이 많을 경우 최적의 해를 보장하지 못할 수 있다. 또한, 음수 가중치가 포함된 그래프에서는 MST의 의미가 모호해질 수 있다.

최소 스패닝 트리 알고리즘은 효율적인 네트워크 설계 및 최적화 문제를 해결하는 데 필수적인 도구이다. 이번 글을 통해 MST의 개념과 알고리즘을 이해하고, 이를 다양한 문제에 적용할 수 있기를 바란다.