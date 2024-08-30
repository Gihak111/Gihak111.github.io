---
layout: single
title:  "알고리즘 정리 1. 스택과 큐"
categories: "algorithm"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 알고리즘 하면 딱 떠오르는 스택과 큐에 대해 알아보자.  
먼저, 스택과 큐에 대해 알기 전에, 다음 자료 타입의 차이를 알아야 한다.  
- **Abstract Data Type**  
    ADT:  
        추상 자료형이다.
        개념적으로 어껀 동작이 있는지만 정의한다.  
        구현에 대해서는 다루지 않는다.  
- **Data Structure**  
    DS:  
        자료구조 이다.  
        ADT 에서 정의한 동작을 실제로 정의한 것이다.  

먼저, ADT의 관점에서 스택과 큐를 알아보자.  

## 스택과 큐  
- **스택(stack)**  
    LIFO(Last In First Out) 형태로 데이터를 저장하는 구조  
    주요동작:  
        - push: 스택에 아이템을 집어넣는다.  
        - pop: 스택에서 아이템을 빼낸다.  
        - peek: 아이템을 빼진 않지만, 최상단의 아니템이 무너지 알 수 있다.  
    
    동작 예시:  
    stack을 push(1), push(2), push(3) 을 하면,  
    1이 제일 밑으로, 3이 제일 위로 올라온다.  
    pop() = 3  
    peek() = 2  
    pop() = 2  
    가 실행이 된다.  
    이어서, push(4) 를 ㅎ라면, 1 위에 4가 쌓인다. 가장 나중에 들어온 놈이 가장 먼저 나가는 느낌이다.  
    ATmega128의 스택 포인터가 작동하는 것과 같은 개념이다.  

- **큐(queue)**
    FIFO(First In First Out) 형태로 데이털르 저장하는 구조  
    주요동작  
        - enqueue: 큐에서 아이템을 집어넣는다.  
        - dequeue: 큐에서 아이템을 꺼낸다.  
        - peek: 큐에서 아이템을 빼진 않지만, 곧 꺼내게 될 아이템의 값을 알려준다.  
    동작 예시:  
    enqueue(1)  
    enqueue(2)  
    enqueue(3)  
    을 하면, 1, 2, 3 이 쌓인다.  
    여기서, dequeue()를 하면, 1이 사라진다.  
    한번더 dequeue()를 하면, 2가 사라지고 3만 남는다.  
    여기에 peek을 하면, 3이 풀력된다. 큐에는 남아있다.  
    enqueue(5) 를 하면, 5가 쌓여서 5와 3만 남게 된다.  
    말 그대로, 먼저 들어간 놈이 먼저 나오는, 선입 선출의 구조이다.  

## 사용 사례
둘의 사용사례를 알아보자.  
- **스택 사용 사례**  
    - stack memory & stack frame:  
    함수가 호출 될 때 마다 스택이 쌓이고, 함수가 사라지면 그에 해당하는 함수의 stack frame도 사라진다.  
    def a()가 def()b를 부르고, def(b)가 def()c를 부르면, 스텍에 a가 1층, b가 2층, c가 3층에 쌓이고, c의 발동이 끝나면 3층, 2, 1층 순서대로 스택이 사라지는 느낌이다.  

- **쿠 사용 사례**  
    - producer/consumer architecture:  
    producer에서 consumer 사이로 큐가 존재하고, producer만 든 아이템이 큐에 차곡차곡 쌓이면, 이가 consumer로 차례대로 들어가 처리된다.  
    백엔드에서 전형적으로 많이 사용되는 아키텍펴중 하나이다.  

## 기술 문서에서 큐를 만났을때  
    ### 항상 FIFO 를 의미하진 않는다.  
    1코어인 CPU가 p1, p2, p3의 일 3개를 멀티 캐스팅으로 실행된다고 하면, p1, p2, p3가 번갈아 가며 실행된다.  
    p1이 실행중이면, p2, p3는 ready queue에 있게 된다.  
    이런 경우에는 FIFO가 아닌, Priority Queue 즉 우선숭의 큐 에 해당한다.  
    큐는 대기열로 사용될 때도 있기 때문에, 잘 봐야 한다.  

## 스택/큐 관련 에러와 해결방법
- **StackOverflowErro**
    스택 메모리 공간을 다 썼을 떄 발생하는 에러 이다.  
    보통 재귀함수에서 탈출하지 못해서 발생한다.  
    자기 자신을 함수 안에서 호출하는 제귀함수는, 꼭 탈출 조건이 있어야 하는데, 탈출 조건을 잘못 잡으면 스택이 넘쳐서 생길때가 많다.  

- **OutOfMemortError**
    Java의 힙 메모리를 다 썼을 경우에 발생한다.  
    힙 메모리는 객테가 거주하는 메모리로, 고갈에는 여러 이유가 잇지만, 내부적으로 큐를 사용했을때, 쌓이기만 하고, 컨슘이 느리거나 없을 경우에 발생한다.  
    이럴 경우엔 큐의 싸이즈를 고정하는 것으로, 리미트를 걸어서 해결할 수 있다.  
    큐가 다 찼을때의 해결 방안은,   
    1. 예외 던지기  
    2. 특별한 값을 반환  
    3. 성공할 때 까지 영원히 스레드 블락  
    4. 제한된 시간만 블락되고 그래도 안되면 포기  
    이런걸로 해결할 수 있다.  
    위의 기능이 구현된 API가 있다.  
    LinkedBlockingQueue  
    라는 자바 APi를 활용해서 할 수 있다.  

중간에 Priority Queue가 나왔었다. 이도 알아보자.  
우선순위 큐는 힙과 비교된다.  
# Priority Queue와 Heap  
- **Priority Queue**  
    우선순위이다.  
    큐와 유사하지만, 우선순위가 높은 아이템이 먼저 처리된다.  
    주요동작:  
        - insert: 아이템을 집어넣는다. 이때, 우선순위 정보도 같이 들어간다.  
        - delete: 아이템의 우선순위가 가장 높은 것을 빼닌다.  
        - peek: delete오 ㅏ같지만, 제거하지는 않는다.  
    동작은 큐와 같은 방식으로 쌓이지만, 비워지는건 우선순위 순서이다.  
    인덱스는 변동이 없고, 타공된 것 처럼 중간에 빈 공간이 생긴다.  

- **Heap**
    이진 트리 구조를 기반으로 한다.  
    트리(tree): 부모-자녀 처럼 계층적인 형태를 지닝으 구조 이다.  
    이진 트리는 부모가 자녀를 초대 2개만 가질 수 있는 구조 이다.  
    2개의 힙이 있는데,  
    1. max heap: 부모의 노드의 키(key)가 자식 노드의 키 보자 크거나 같은 트리  
    2. min heap: 부모 노드의 키가 자식 노드의 키 보다 작거나 같은 트리  
    주요 동작  
        - insert: 아이템을 집어넣는다. key 값도 넣는다.  
        - delete: mim이면 키가 가장 작은 노드를, max면 가장 큰 노드의 아이템을 가져오고 지운다.  
        - peek: delete랑 같지만, 지우지 않는다.  
    max heap 동작시에는, 자식이 부모보다 크면 자리를 바꾸는 식으로 작동한다.  
    delete는 가장 위에 잇는 rooe node. 즉 전체 노드 중에서 가장 큰 값을 가진 노드를 가져오고, 빈 공간은 가장 끝에 있는 노드가 들어간다.   이후, max heap의 특징을 유지하기 위해, 공간이 재정렬된다.  

## Priority Queue와 Heap  
    힘의 키를 우선순위로 사용한다면 힙은 우선순위 큐의 구현체가 된다.  
    Priority queue = ADT  
    Heap = data structure  
    그래서 Priority queue 의 구현테가 Heap 이라고들 한다. 효울, 성능 모두 좋기 떄문이다.  

## 사용 사례
- **프로세스 스케줄링**
    여러 프로세스가 워코어 CPU에서 멀티 캐스링으로 돌아간다면, ready queue에 작동중이지 않는 프로세스가 있을 것이다.  
    지금 작업이 끝나거나, 타임 슬라이스가 있어서 작업이 넘거간다면, ready queue에서 가장 우선순위가 높은 작업이 실행된다.  
    이런 상황이 Priority Queue 로 만들 수 있다.  

- **Heap Sort**
    정렬할때 사용한다.  
    n개의 아이템을 힙에 전부 넣고, 차례대로 delete 하면 정렬되어 나온다.  

힙 메모리의 heap은 이 heap과는 다르다.  
heap의 사전적인 의미가 더미 이다. 힙 메모리는 사실 메모리 더미 그런거다.  

예제를 통해서 알아보자.  
# 예제 코드  

1. 스택 구현 예제  
```java
import java.util.Stack;

public class StackExample {

    public static void main(String[] args) {
        // Stack 생성
        Stack<Integer> stack = new Stack<>();
        
        // Stack에 요소 추가 (push)
        stack.push(1);
        stack.push(2);
        stack.push(3);

        // Stack 상태: [1, 2, 3] (3이 가장 위에 있음)

        // 최상단 요소 확인 (peek)
        System.out.println("Top element: " + stack.peek()); // 출력: 3

        // Stack에서 요소 제거 (pop)
        System.out.println("Popped element: " + stack.pop()); // 출력: 3
        System.out.println("Top element after pop: " + stack.peek()); // 출력: 2

        // Stack 상태: [1, 2] (2가 가장 위에 있음)
    }
}
```
    
2. 큐 구현 예제  
```java
import java.util.LinkedList;
import java.util.Queue;

public class QueueExample {

    public static void main(String[] args) {
        // Queue 생성
        Queue<Integer> queue = new LinkedList<>();
        
        // Queue에 요소 추가 (enqueue)
        queue.add(1);
        queue.add(2);
        queue.add(3);

        // Queue 상태: [1, 2, 3] (1이 가장 앞에 있음)

        // 가장 앞의 요소 확인 (peek)
        System.out.println("Front element: " + queue.peek()); // 출력: 1

        // Queue에서 요소 제거 (dequeue)
        System.out.println("Dequeued element: " + queue.poll()); // 출력: 1
        System.out.println("Front element after dequeue: " + queue.peek()); // 출력: 2

        // Queue 상태: [2, 3] (2가 가장 앞에 있음)
    }
}

```

3. 힘 구현 예제
```java
import java.util.PriorityQueue;

public class MinHeapExample {

    public static void main(String[] args) {
        // Min Heap 생성 (PriorityQueue를 사용)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        // Min Heap에 요소 추가
        minHeap.add(3);
        minHeap.add(1);
        minHeap.add(2);

        // Min Heap 상태: [1, 3, 2] (가장 작은 값이 root에 있음)

        // 최소값 확인 (peek)
        System.out.println("Min element: " + minHeap.peek()); // 출력: 1

        // Min Heap에서 최소값 제거 (poll)
        System.out.println("Removed min element: " + minHeap.poll()); // 출력: 1
        System.out.println("Min element after poll: " + minHeap.peek()); // 출력: 2

        // Min Heap 상태: [2, 3] (2가 root에 있음)
    }
}

```

4. 우선순위 큐 구현 예제
```java
import java.util.PriorityQueue;

class Task implements Comparable<Task> {
    private String name;
    private int priority;

    public Task(String name, int priority) {
        this.name = name;
        this.priority = priority;
    }

    public String getName() {
        return name;
    }

    @Override
    public int compareTo(Task other) {
        // 우선순위가 낮은 숫자가 더 높은 우선순위
        return Integer.compare(this.priority, other.priority);
    }

    @Override
    public String toString() {
        return "Task{name='" + name + "', priority=" + priority + '}';
    }
}

public class PriorityQueueExample {

    public static void main(String[] args) {
        // Priority Queue 생성
        PriorityQueue<Task> priorityQueue = new PriorityQueue<>();
        
        // 우선순위 큐에 작업 추가
        priorityQueue.add(new Task("Task 1", 3));
        priorityQueue.add(new Task("Task 2", 1));
        priorityQueue.add(new Task("Task 3", 2));

        // Priority Queue 상태: [Task 2, Task 1, Task 3] (우선순위가 낮은 숫자가 먼저 처리됨)

        // 가장 높은 우선순위 작업 확인 (peek)
        System.out.println("Highest priority task: " + priorityQueue.peek()); // 출력: Task 2

        // 우선순위 큐에서 작업 처리 (poll)
        System.out.println("Processing task: " + priorityQueue.poll()); // 출력: Task 2
        System.out.println("Next highest priority task: " + priorityQueue.peek()); // 출력: Task 3

        // Priority Queue 상태: [Task 3, Task 1] (우선순위에 따라 정렬됨)
    }
}

```

이제 다 이해했으리라 믿는다.  



이 글은, 이 유튜브 영상을 참고하였다.
[https://www.youtube.com/watch?v=-2YpvLCT5F8](https://www.youtube.com/watch?v=-2YpvLCT5F8)
[https://www.youtube.com/watch?v=P-FTb1faxlo](https://www.youtube.com/watch?v=P-FTb1faxlo)