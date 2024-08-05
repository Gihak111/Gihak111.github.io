---
layout: single
title:  "spring boot 2. 어노테이션"
categories: "spring"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---  
# RESTAPI  
RESTAPI는 무상태성을 원칙으로 한다.  
A가 B와 통신할 때, A는 B의 통신 시점의 상태를 가져온다.  
통신 하는 중의 상태가 아닌, 통신 시작 상태를 가져온다는 거다.  
A는 B에 요청할 때 마다 관련 상태의 표현을 제공한다.  
각각의 서비스는 개별적인 상태를 지니기 때문에, 한곳의 상태가 무너지면 다른 곳의 상태도 무너지는 것이 아니라,  
각각의 서비스가 서로 다른 상태를 갖는다.  
이러면, 통신문제가 나와서 재시작이 되어도, 현재 상태를 잃어버리지 않는다.  
따라서 중간에 통신이 두절 되어도, 한곳에 상태가 남아있기 때문에  
A가 다시 통신을 요청하면, 두 애플리케이션이 중단된 곳에서 시작된다.  

이게 RESTAPI의 큰 특징이다.  


# 어노테이션  
지금부터 사용되는, 사용했던 이노베이션들을 알아보자.  
그중, 가장 많이 사용되는 것은 GET 메서드 이다.  


## @RestController  
웹사이트를 운영한다고 치면, 데이터와 데이터를 전송하는 부분과 데이터를 표현하는 부분을 분리해 생성한다.  
이 구간을 연결하는게 @Controller 어노테이션이다.  
@Controller가 붙은 클래스는 Model객체를 받는다.  
@ResponseBody를 클래스나 메서드에 추가해서, Json이나 XML 같은 데이터 형식처럼, 형식화된 응답을 반환하도록 지시할 수 있다.  
이걸 전에 했던 예제에서 이걸 지정 했기 때문에, Json형식으로 반환되고, 다운도 안됬었던거다.  
위의 둘을 합친 것이 @RestController이다.  
클래스에 @RestController를 달아서 RESTAPI를 만들 수 있다.  

느낌을 get set 으로 생각해 보면 get이랑 다를게 없다.  
그냥 생성자 하나라고 생각하고 사용하면 나름 써진다. like  
```java  
@RestController
class RestApiDemoController {
    private List<Coffee> coffees = new ArrayList<>();

    public RestApiDemoController() {
        coffees.addAll(List.of(
            new Coffee("Cafe Cereza")
            new Coffee("Cafe Ganador")
            new Coffee("Cafe Americano")
            new Coffee("Cafe latte")
        ));
    }
}
```  
위의 코드는 커피의 리스트를 생성하는 메서드다.  


## @GetMapping   
이제, @RequestMapping 어노테이션으로 목록을 가져온다.  
코드를 추가하면, 다음과 같다.  
```java
@RestController
class RestApiDemoController {
    private List<Coffee> coffees = new ArrayList<>();

    public RestApiDemoController() {
        coffees.addAll(List.of(
            new Coffee("Cafe Cereza")
            new Coffee("Cafe Ganador")
            new Coffee("Cafe Americano")
            new Coffee("Cafe latte")
        ));
    }

    @RequestMapping(value = "/coffes", method = RequestMethod.GET)
    Iterable<Coffee> getCoffees(){
        return coffees;
    }
}
```
@RequestMapping에 API URL인 /coffes와, HTTP 메서드 타입인 RequestMethod.GET를 가져온다.  
getCoffees 메서드가 GET 요청의 /coffee URL에만 등답하게 제한한 것이다.  
여기서, @RequestMapping에서 @GetMapping으로 바꾸면 가독성이 더 올라간다.  
```java
@GetMapping("/coffees")
Iterable<Cpffee> getCOffees() {
    return coffees;
}
```  
이런 식으로 코드가 엄청 줄게 된다.  
@RequestMapping에는 GetMapping 말고도 이것 저것 있다. 자주 사용하는 것은  
1. @GetMapping  
2. @PostMapping  
3. @PutMapping  
4. @PatchMapping  
5. @DeleteMapping  

이런 것들이 있다.  하나 하나 보게 될 거다.
이런 어노테이션이 어떻게 호출되고, 어떻게 사용되냐면,  

다시한번 예시를 들어보자.
```java
@RestController
public class GreetingController {

    @GetMapping("/greeting")
    public String greeting() {
        return "Hello, Spring!";
    }
}

```  
이런 식으로 Hello, spring!을 출력하는 어노테이션이 있다고 하자.  
애플리케이션이 시작이 되면, Spring은 GreetingController 클래스를 스캔하고, @GetMapping 어노테이션을 찾아서 등록한다.  
이 정보는, HandlerMapping 인터페이스를 구현한 객체에 저장된다.  
(HandlerMapping는 최상위 인터페이스 이다)  
이후, 클라이언트가 GET /greeting 요청을 보내면, Spring은 요청 URL과 HTTP 메서드를 기반으로 매핑된 메서드를 찾는다.  
매핑된 메서드 greeting()이 호출되고, "Hello, Spring!" 문자열이 응답을 반환한다.  
이런 구조로 spring이 돌아가는거다.  


## @PostMapping  
이건 리소스의 세부 정보를 일반적으로 JSON 형식으로 제공한다.  
해당 서비스에 POST 요청을 해서 지정된 URL에 리소스를 생성한다.  
```java
@PostMapping("/coffees")
Coffee postCoddee(@RequestBody Coffee coffee) {
    coffee.add(coffee);
    return coffee;
}
```  
이런 식으로 구현할 수 있다.  
스프링 부트의 자동 마샬링 덕분에 헤당 커피 정볼르 Coffee로부터 받는다.  
이 객체는 언마샬링 되어 기본값인 JSON으로 되어 요청한 애플리케이션이나 서비스로 변환된다.  


## @PutMapping  
이건 파악된 URI를 통해 지곤 리소스의 업데이트에 사용된다.
```java
@PutMapping("/coffees/{id}")
Coffee putCoffee(@PathVariable String id, @RequestBody Coffee coffee) {
    int coffeeIndex = -1;

    for (Coffee c: coffees) {
        if (c.getId().equals(id)) {
            coffeeIndes = coffees.index0f(c);
            coffees.set(coffeeIndex, coffee);
        }
    }

    return (coffeeIndex == -1) ? postCoffrr(coffee) : coffee;
}
```  
이 코드는 특정 식별자로 커피를 검색하고, 찾으면 업데이트 한다. 목록에 없으면 새로 리소스를 만든다.  


## @DeleteMapping
리소스를 삭제한다.  
```java
@DeleteMapping("/coffees/{id}")
void deleteCoffee(@PathVariable String id) {
    coffees.removeIf(c -> c.getId().equals(id));
}
```
@PathVariable로 커피 식별자인 id를 받아서 Collection 메서드의 removeIf를 사용해 목록을 제거한다.  
removeIf는 Predicate 값을 받는다.  
즉, 목록에 제거할 커피가 존재하면, 참 인 불 값 을 반환하는 람다이다.  

암튼, 이런 식으로 작동한다. 앞서 만든 예시에서 나온 오류를 고치면서 잘 이해 했는지 확인하자.