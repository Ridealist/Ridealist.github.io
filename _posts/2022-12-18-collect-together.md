---
published: true
layout: posts
title: Ch7. 함께 모으기
categories: 
  - oop
toc: true
---

## 객체 지향 설계 속 세 가지 상호 연관된 관점

### 1. 개념 관점

#### ✔️ 도메인 안에 존재하는 개념과 개념들 사이의 관계를 표현하는 설계이다.

- 소프트웨어는 결국 도메인 속 문제를 해결하기 위해 개발
- 사용자가 도메인을 바라보는 관점을 반영한다.

#### → 실제 도메인의 규칙과 제약을 최대한 유사하게 반영하는 것이 핵심

<br>

### 2. 명세 관점

- 개발자가 주목하는 **객체들의 책임에 초점**
- **객체들이 협력을 위해 어떤 일을 할 수 있는가**에 초점

#### 중요한 것은 인터페이스와 구현을 분리하는 것이다.

<br>

### 3. 구현 관점

- 구현 관점의 초점은 객체들이 **책임을 수행하는 데 필요한 코드를 작성**하는 것이다

#### 개발자는 객체의 책임을 '어떻게' 수행할 시킬 것인가에 초점을 맞추며 인터페이스를 구현하는 데 필요한 속성과 메서드를 클래스에 추가

<br>

#### 💡개념 → 명세 → 구현 관점의 순서대로 소프트웨어를 개발한다는 것이 아니라, 동일한 클래스를 세 가지 관섬에서 바라보는 것을 의미

- 개념 관점 : 클래스가 은유하는 개념
- 명세 관점 : 공용 인터페이스
- 구현 관점 : 클래스의 속성과 메서드

<br>

## **커피 전문점 예제**

작은 커피전문점 예제를 통해 개념에서 구현 관점까지의 과정을 살펴보자.

### 1. 도메인 파악

- **손님**: 커피전문점의 존재이유이다. 손님은 메뉴판을 보고 바리스타에게 커피를 주문하는 객체이다.
- **메뉴판**: 손님은 메뉴판이 있어야 주문을 할 수 있다. 메뉴판에는 항목들이 존재한다. 예제에서는 아메리카노, 카푸치노, 카라멜 마끼아또, 에스프레소 4 가지 커피 메뉴 항목이 존재한다.
- **메뉴 항목**: 메뉴판에 구성된 항목 역시 객체이다.
- **바리스타**: 손님이 요청한 커피를 제조한다.
- **커피**: 해당 메뉴 항목의 주문을 받은 바리스타가 실제로 제조한 커피객체이다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmsmxS%2FbtrdqLpK36Q%2FnnotTa31uQ7HQNRRsafAF1%2Fimg.png)


### 2. **협력 메시지 추출**

 메시지를 먼저 식별해야 한다. 메시지를 먼저 식별하고 이를 적절한 객체에 할당해야 한다.

- '커피를 주문하라' 는 것이 첫번째로 발생한 메시지이다.
- 커피를 주문하는것은 손님의 책임이므로 첫번째 메시지는 손님 객체가 수신해야 한다.

- 메시지를 수신받은 손님은 메뉴판 객체에 '메뉴를 찾아라'는 메시지를 전달한다.
- 메뉴 항목을 찾으면 바리스타 객체에게 '커피를 제조하라'는 메시지를 전달해야 한다.

- 바리스타는 커피 객체에게 '생성하라'는 메시지를 전달한다.

이런 과정을 협력 다이어그램으로 나타내면 아래와 같이 나타낼 수 있다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOQdt4%2FbtrdqMoEYxe%2FLELSkwKPBk1doc8PjHtKK1%2Fimg.png)

---

### 3. **구현하기**

```java
public class MenuItem {

	private String menuName;
	private long price;

..............

public class MenuBoard {

	private List<MenuItem> menuItems;
	
	public MenuBoard() {
		super();
		
		menuItems = new ArrayList<>();
		
		menuItems.add(new MenuItem("Americano", 1000));
		menuItems.add(new MenuItem("Cappuccino", 1100));
		menuItems.add(new MenuItem("CaramelMacchiato)", 1200));
		menuItems.add(new MenuItem("Espresso", 1300));
	}

	public Optional<MenuItem> findMenu(String menuItemName) {
		
		return menuItems.stream()
			.filter(menuItem -> menuItem.getMenuName().equals(menuItemName))
			.findAny();
	}
}
```
