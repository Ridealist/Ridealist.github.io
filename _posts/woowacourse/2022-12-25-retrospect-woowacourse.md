---
published: true
layout: posts
title: 최종 코딩테스트 및 프리코스 마무리 후기
categories: 
  - woowacourse
toc: true
---

12/17(토) 최종 코딩테스트를 기점으로 프리코스 과정이 종료되었다.

주중에 바빠 미루던 후기를 크리스마스인 오늘 적어본다.

---

## 1. 회고


모든 시험이 그렇겠지만, 최종 코딩테스트는 정말 아쉬움이 많이 남는다.

### 1-1. 빠트린 기능 요구 사항

> 먹지 못하는 메뉴가 없으면 빈 값을 입력한다.

```java
// FoodRepository Class
    public static Food getByName(String name) {
        return foods.stream()
                .filter(food -> food.hasSameName(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(INVALID_FOOD_NAME_MESSAGE));
    }

// Coach Class
    public void updateFoods(List<String> foods) {
        for (String name : foods) {
            cannotEatFoods.add(FoodRepository.getByName(name));
        }
    }
```

- 입력된 음식이 올바른지 확인하는 로직이 있는데, 여기서 빈값에 대한 예외처리를 놓쳐버렸다.
- 빈값은 FoodRepository에 저장되어 있지 않기 때문에 빈값을 입력하면 [ERROR] 를 반환한다...ㅠㅠ

<br>

### 1-2. 프로그래밍 요구 사항 준수 못함

> 함수(또는 메서드)의 길이가 15라인을 넘어가지 않도록 구현한다.

```java
    public static void loadData() {
        String japaneseData = "일식: 규동, 우동, 미소시루, 스시, 가츠동, 오니기리, 하이라이스, 라멘, 오코노미야끼";
        for (String foodName: parsingFoodName(japaneseData)) {
            FoodRepository.addFood(new Food(foodName, japanese));
        }
        String koreanData = "한식: 김밥, 김치찌개, 쌈밥, 된장찌개, 비빔밥, 칼국수, 불고기, 떡볶이, 제육볶음";
        for (String foodName: parsingFoodName(koreanData)) {
            FoodRepository.addFood(new Food(foodName, korean));
        }
        String chineseData = "중식: 깐풍기, 볶음면, 동파육, 짜장면, 짬뽕, 마파두부, 탕수육, 토마토 달걀볶음, 고추잡채";
        for (String foodName: parsingFoodName(chineseData)) {
            FoodRepository.addFood(new Food(foodName, chinese));
        }
        String asianData = "아시안: 팟타이, 카오 팟, 나시고렝, 파인애플 볶음밥, 쌀국수, 똠얌꿍, 반미, 월남쌈, 분짜";
        for (String foodName: parsingFoodName(asianData)) {
            FoodRepository.addFood(new Food(foodName, asian));
        }
        String westernData = "양식: 라자냐, 그라탱, 뇨끼, 끼슈, 프렌치 토스트, 바게트, 스파게티, 피자, 파니니";
        for (String foodName: parsingFoodName(westernData)) {
            FoodRepository.addFood(new Food(foodName, western));
        }
    }
```

- 데이터를 넣는 부분에 대한 코드를 급한대로 작성한 후 리팩터링을 하지 못한채 제출했다...
- TODO 리스트에 적어놓았으나 시간이 부족해 TODO 리스트를 볼 정신도 없이 제출하기에 급급했다.

<br>

### 1-3. 코딩 실수_1

> "[ERROR]"로 시작하는 에러 메시지를 출력 후 그 부분부터 입력을 다시 받는다.

```java
    private static void setMenusOnCoach(Coach coach) {
        try {
            List<String> menus = InputVIew.readCannotEatMenus(coach);
            coach.updateFoods(menus);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            setMenusOnCoach(coach);
        }
    }
```

- 말도 안되는 실수를 해버렸다...
- 멀쩡한 OutputView를 놔두고 System.out을 쓰다니 어떻게 이런 일이...

<br>


### 1-4. 코딩 실수_2

> - 한 주에 같은 카테고리는 최대 2회까지만 고를 수 있다.

```java
    public Category getRandomCategory() {
        Category category = Category.getByNumber(Randoms.pickNumberInRange(1, 5));
        if (CategoryRecommendRepository.hasOverDuplicatedCategory(category)) {
            return getCategory();
        }
        return category;
    }
```

- 충격적인 실수를 다시 발견... `getRandomCategory()`로 재귀 호출을 `getCategory()`를 호출해 버렸다...
- 어떻게 이런 일이...

<br>

### 1-5. 동시성 처리 문제

- 로직에는 이상이 없어 보이는데 메뉴 추천과 카테고리 추천이 조건에 부합하지 않게 생성되는 경우가 생긴다.
- 잘 되는 경우가 대부분이긴 한데 가끔씩 중복 오류가 그대로 출력된다.
- 정확한 원인은 찾지 못했지만, 동시성 처리 문제가 아닐까 추측해보았다...
- 추가로 해결해야 함!


## 2. 반성할 점

### 2-1. 필요 이상의 개발

코드를 살펴보니 문제에서 주어진 요구사항이 아닌 부분도 신경쓰며 개발한 흔적이 보인다.

안쓰이는 메서드 들이 몇 개 보였다...

적어도 시험 환경에서는 살만을 취해야 한다. 지방 덩어리는 필요 없다.

꼭 구현해야 하는 것이 언제나 최우선이고, 나머지는 그 다음이다.

### 2-2. 시간의 부족

이건 아직 절대적인 연습량이 부족해서 발생한 문제가 아닌가 싶다.

항상 구상한 시간보다 실제 구현 시간이 많이 소요된다...

평소 업무 상황이면 모르겠지만, `시험` 환경에서 느린 개발 속도는 치명적인 단점이다...

Practice makes Perfect. 최대한 많이 코딩을 해보는 방법 밖에는 없을 것 같다.

### 2-3. 테스트 코드 작성의 안 익숙함

테스트 코드가 익숙했다면 단위 테스트를 하면서 오류를 줄일 수 있었을텐데...

단위 테스트가 익숙하지 않음이 못내 아쉬움으로 남는다.

앞으로 무엇을 개발하든 단위 테스트를 작성해보는 습관을 들이기로 다짐...

### 2-4. 설계 능력의 부족

모든 정보를 Repository에 저장하도록 하는 구조로 설계를 했었다.

하지만, 카테고리를 고르거나 메뉴를 고르는 것을 하나하나 객체로 관리하는 것이 비효율적이어 보인다.

단순 정보는 `Cache` 메모리에 저장했다가 빠르게 인출하듯이, 객체 내부의 데이터 구조를 두는 것도 좋은 선택으로 보인다.

무조건 DB에 모든 정보를 저장하는 설계를 반성해본다...

## 3. 코드 수정해보기

### 1-1. 빠트린 기능 요구 사항

```java
    public void updateFoods(List<String> foods) {
        if (foods.size() == 1 && foods.get(0).equals("")) {
            return;
        }
        for (String name : foods) {
            cannotEatFoods.add(FoodRepository.getByName(name));
        }
    }
```

- 빈칸이 입력된 경우에 if문으로 조기 반환 처리를 해준다.
- 빈칸이 아닌 경우만 유효성 검사를 한다.


### 1-2. 프로그래밍 요구 사항 준수 못함

```java
public class Data {
    private static final String rawData = "일식: 규동, 우동, 미소시루, 스시, 가츠동, 오니기리, 하이라이스, 라멘, 오코노미야끼\n"
            + "한식: 김밥, 김치찌개, 쌈밥, 된장찌개, 비빔밥, 칼국수, 불고기, 떡볶이, 제육볶음\n"
            + "중식: 깐풍기, 볶음면, 동파육, 짜장면, 짬뽕, 마파두부, 탕수육, 토마토 달걀볶음, 고추잡채\n"
            + "아시안: 팟타이, 카오 팟, 나시고렝, 파인애플 볶음밥, 쌀국수, 똠얌꿍, 반미, 월남쌈, 분짜\n"
            + "양식: 라자냐, 그라탱, 뇨끼, 끼슈, 프렌치 토스트, 바게트, 스파게티, 피자, 파니니";

    private static final int CATEGORY_INDEX = 0;
    private static final int FOODS_INDEX = 1;
    private static final String RAW_DATA_DELIMITER = "\n";
    private static final String CATEGORY_DELIMITER = ":";
    private static final String FOOD_DELIMITER = ",";

    public static void loadData() {
        String[] foods = splitFoodByCategory(rawData);
        for (String categoryFoods : foods) {
            Category category = parsingFoodCategory(categoryFoods);
            for (String foodName : parsingFoodName(categoryFoods)) {
                FoodRepository.addFood(new Food(foodName, category));
            }
        }
    }

    private static String[] splitFoodByCategory(String rawData) {
        return rawData.split(RAW_DATA_DELIMITER);
    }

    private static Category parsingFoodCategory(String rawData) {
        List<String> data = Arrays.asList(rawData.split(CATEGORY_DELIMITER));
        String categoryName = data.get(CATEGORY_INDEX);
        return Category.getByName(categoryName);
    }

    private static List<String> parsingFoodName(String rawData) {
        List<String> data = Arrays.asList(rawData.split(CATEGORY_DELIMITER));
        List<String> foods = Arrays.asList(data.get(FOODS_INDEX).split(FOOD_DELIMITER));
        return foods.stream()
                .map(String::trim)
                .collect(Collectors.toList());
    }
}
```

- 이건 사실 더 좋은 방법이 있어 보이지만 우선은...
- rawData를 필드로 정의해버렸다.
- 이중 for문으로 각각의 값을 가져오고 parsing하고 category를 찾고 그것을 Repository에 넣는다


### 1-3. 코딩 실수_1

```java
    private static void setMenusOnCoach(Coach coach) {
        try {
            List<String> menus = InputVIew.readCannotEatMenus(coach);
            coach.updateFoods(menus);
        } catch (IllegalArgumentException e) {
            OutputView.printError(e.getMessage());
            setMenusOnCoach(coach);
        }
    }
```

- 이건 그냥 할 말이 없다...

### 1-4. 코딩 실수_2

```java
    public Category getRandomCategory() {
        Category category = Category.getByNumber(Randoms.pickNumberInRange(1, 5));
        if (CategoryRecommendRepository.hasOverDuplicatedCategory(category)) {
            return getRandomCategory();
        }
        return category;
    }
```

- 이것도 그냥 할 말이 없다...

### 1-5. 기타

- 추가로 부족한 부분 계속 수정해 나갈 예정
    - [refactoring_PR_링크](https://github.com/Ridealist/java-menu/pull/1)


## 4. 프리코스가 끝난 소감

최종 코딩테스트를 잘 못 본 것이 너무 아쉽지만 어떻게하리... 이것이 현재 내 실력인 것을...

우울해하고만 있을 수는 없다. 그저 한발자국 한발자국 나아가야 한다.

프리코스 2달을 정말 불태웠다. 직장과 병행하며 공부하는 것은 정말 쉬운 일이 아니었다.

그래도 마지막 미션(다리 게임)보다 최종 코딩 테스트에서 더 성장한 모습을 보여준 것은 칭찬하고 싶다.


### [그래도 성장한 점]

1. stream()을 사용하여 collections framework를 좀 더 효율적으로 다루는 방법을 적용해 보았다.

2. DB의 개념인 Repository를 두는 구조를 적용해 보았다.

3. Enum 클래스에서 다양한 메서드를 정의해 유효성 검증 및 값 반환 등 좀 더 `객체스럽게` 사용해 보았다.

4. 데이터를 꺼내지 말고 메세지를 보내는 `객체지향`을 조금 이나마 더 적용해보았다.

5. DTO라는 중간 객체를 두어 Domain과 View 사이에 의존성을 줄이는 효율적 연결 구조를 적용해 보았다.


## 5. 앞으로의 계획

### 1. 우선 Teameet 프로젝트를 끝낸다.

칼을 뽑았으면 무라도 썰어야 한다. 기존에 하던 팀 프로젝트를 질질 끌지 말고 마무리 짓는다.

죽이 되는 밥이 되든 우선 완성한다. `안 돌아가는 프로그램보다 돌아가는 쓰레기`가 낫다.


### 2. 코딩 테스트 연습을 다시 재개한다.

프로그램 설계도 중요하지만, 내가 설계한 것을 빠르게 구현하는 것은 훈련인 것 같다.

결국 알고리즘 공부가 내 생각을 빠르게 코드로 전환하는 `훈련`이지 않나 싶다.

매일 매일 코딩 테스트를 한 문제씩이라도 꾸준히 풀어 나간다.


### 3. 규칙적인 생활을 한다.

우테코를 준비하면서 생활 리듬이 많이 깨졌다.

밤 늦게까지 코딩하다가 새벽에 잠들어 얼마 못자고 출근 하기도 하고.

스터디가 10시에 끝나고서 스트레스 해소를 위해 맥주를 너다섯 캔 먹고 자서 다음날 숙취로 고생도 하고.

운동도 제대로 못해서 체력은 떨어지고.

곧 방학이 시작되면 시간 여유가 많으니, 주어진 시간을 알차게 쓰고 규칙적인 생활을 하자.

<br>

그 외 공부의 방향성은 우테코의 합격 여부에 따라 많이 달라질 것 같다.

나머지는 12/28 (수) 15:00 이후에 생각하기로 하자.


## 6. 소회...

![image](/assets/img/스크린샷-2023-04-12.png)

중요한 것은 꺽이지 않는 마음(중꺾마?!)

개발자가 되기 위해 우테코에 지원한 것이니, 개발자의 꿈을 꺾이지 않는게 가장 중요하다.

한사람의 좋은 개발자가 되기 위해 정진하자.

결국 좋은 개발자로 가는 방향으로 가다보면 시작점이 달라도 언젠가 만나게 되겠지...