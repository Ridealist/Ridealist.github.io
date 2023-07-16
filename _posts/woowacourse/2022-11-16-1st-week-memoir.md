---
published: true
layout: posts
title: 1주차 과제 피드백 정리
categories: 
  - woowacourse
toc: true
---

코드 리뷰 스터디에 참여하고 있다.

많이 늦었지만... 스터디원들의 피드백 중 기억해야할 내용을 정리해본다.

### 1. |, & vs. ||, && 연산자 선택

```java
    if (page1 < 1 | page1 > 400 | page2 < 1 | page2 > 400)
    // 리팩터링
    if (page1 < 1 || page1 > 400 || page2 < 1 || page2 > 400)
```

- | 연산자
    - 피연산자을 int 형으로 변환할 수 있는 경우에는 Bitwise OR 연산
    - 피연산자가 boolean 형일 경우에는 Logical inclusive OR 연산
        - Conditional OR 연산자인 ||를 사용할 때와 결과가 같음
        - 하지만, |는 short-circuit이 발생하지 않음
- 참고자료
    - [오라클 공식 문서](https://docs.oracle.com/javase/specs/jls/se7/html/jls-15.html#jls-15.22.2)


### 2. 변수 지정 최소화

```java
// Case.1
private int getSum(int page) {
    int sum = 0;
    while (page > 0) {
        int q = page / 10;
        int r = page % 10;
        sum += r;
        page = q;
    }
    return sum;
    }
// 리팩터링
private int getSum(int page) {
    int sum = 0;
    while (page > 0) {
        sum += page % 10;
        page = page / 10;
    }
    return sum;
}

// Case2.
String answer = Decoder.decoding(cryptogram);
return answer;
// 리팩터링
return Decoder.decoding(cryptogram);
```

### 3. StringBuilder 활용

- String은 불변 속성 객체(immutablie)
    - +연산을 사용하는 경우 더해진 새로운 문자열이 새로운 메모리 구역에 저장
    - 이때 기존 문자열은 참조되지 않음
        - 이후 가비지 콜렉터에 의해 정리
    - String에 +연산을 많이 사용하는 코드의 경우 heap 메모리 부족이 일어날 수도
- 문자열 추가, 삭제가 잦은 경우 StringBuilder를 사용 권장

- [참고 블로그](https://ifuwanna.tistory.com/221)

### 4. 불변값 변수 처리

```java
    for (int i = 0; i < n_str.length(); i++)
```

- n_str.length()의 값은 변하지 않음
    - 반복문의 시작과 동시에 (혹은 시작하기 전에) 변수로 선언해 주는 게 바람직
    - for loop이 돌 때마다 같은 값을 반복해서 계산하는 비효율성

### 5. 클래스 내 코드 컨벤션 준수

```java
class Withdrawal {
    private static Withdrawal withdrawal = new Withdrawal();

    static Withdrawal getInstance() {
        return withdrawal;
    }

    private List<Integer> money_array = new ArrayList<>(9);
    private Withdrawal() {
        setMoney_array(this.money_array);
    }

// refactoring
class Withdrawal {
    // 클래스 변수
    private static Withdrawal withdrawal = new Withdrawal();
    // 인스턴스 변수
    private List<Integer> money_array = new ArrayList<>(9);
    // 생성자
    private Withdrawal() {
        setMoney_array(this.money_array);
    }
    // 메서드
    static Withdrawal getInstance() {
        return withdrawal;
    }
```


### 6. 객체를 객체답게

- final Map<String, List> friends_map; 을 제외하고는 받은 인자값을 그대로 Recommendation 클래스에 전달하는 모습
    - 객체를 만들었다라는 코드라고 생각되지 않음

- 객체를 생성하고자는 목표로?
    - Friend 객체와 Visitor 객체를 만들기
    - Firend 객체에서 setFriendScore(), makeFriendsMap()를 처리
    - Visitor 객체에서 setVisitorScore부터 나머지를 처리하는 방식으로 권유
    - visitor 객체에서 쓰이는 Friend 관련된 데이터들은 Firend 객체에서 메서드 만들어서 불러오고 방식으로

```java
package onboarding;

import java.util.Collections;
import java.util.List;
import java.util.*;

public class Problem7 {
    public static List<String> solution(String user, List<List<String>> friends, List<String> visitors) {
        List<String> answer = Collections.emptyList();
        Recommendation rec = new Recommendation(user, friends, visitors);
        List<String> answer = rec.returnResult();
        return answer;
    }
}

class Recommendation {
    final String user;
    final List<List<String>> friends;
    final List<String> visitors;
    final Map<String, List<String>> friends_map;

    Recommendation(String user, List<List<String>> friends, List<String> visitors) {
        this.user = user;
        this.friends = friends;
        this.visitors = visitors;
        this.friends_map = makeFriendsMap(friends);
    }

    List<String> returnResult() {
        List<String> result = new ArrayList<>();
        Map<String, Integer> friend_score = setFriendScore();
        Map<String, Integer> visitor_score = setVisitorScore(friend_score);
        Map<String, Integer> user_score = setUserScore(visitor_score);
        Map<String, Integer> fltr_user_score = filterUserScore(user_score);
        List<Map.Entry<String, Integer>> sorted_user_entryList = sortUserScore(fltr_user_score);
        int i = 0;
        while (i < 5 & i < sorted_user_entryList.size()) {
            result.add(sorted_user_entryList.get(i).getKey());
            i += 1;
        }
        return result;
    }

    private List<Map.Entry<String, Integer>> sortUserScore(Map<String, Integer> user_score) {
        List<Map.Entry<String, Integer>> entryList = new LinkedList<>(user_score.entrySet());
        entryList.sort(Map.Entry.comparingByKey());
        entryList.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
        return entryList;
    }

    private Map<String, Integer> filterUserScore(Map<String, Integer> user_score) {
        for (String key : user_score.keySet()) {
            if (user_score.get(key) == 0) {
                user_score.remove(key);
            }
        }
        return user_score;
    }

    private Map<String, Integer> setUserScore(Map<String, Integer> user_score) {
        // user key에 대한 친구 목록이 없는 경우 빈 array 반환
        List<String> array = new ArrayList<>();
        List<String> user_friends = friends_map.getOrDefault(user, array);
        for (String friend : user_friends) {
            user_score.remove(friend);
        }
        return user_score;
    }

    private Map<String, Integer> setVisitorScore(Map<String, Integer> user_score) {
        for (String visitor : visitors) {
            user_score.put(visitor, user_score.getOrDefault(visitor, 0) + 1);
        }
        return user_score;
    }

    private Map<String, Integer> setFriendScore() {
        Map<String, Integer> user_score = new HashMap<>();
        // user key에 대한 친구 목록이 없는 경우 빈 array 반환
        List<String> array = new ArrayList<>();
        List<String> friend_list = friends_map.getOrDefault(user, array);
        for (String friend : friend_list) {
            List<String> cross_friend_list = friends_map.get(friend);
            for (String cross_friend : cross_friend_list) {
                if (cross_friend.equals(user)) {
                    continue;
                }
                user_score.put(cross_friend, user_score.getOrDefault(cross_friend, 0) + 10);
            }
        }
        return user_score;
    }

    private Map<String, List<String>> makeFriendsMap(List<List<String>> friends) {
        Map<String, List<String>> friends_map = new HashMap<>();
        for (List<String> friend_list : friends) {
            for (int i = 0; i <= 1; i++) {
                String friend1 = friend_list.get(i);
                String friend2 = friend_list.get(1 - i);
                List<String> array = new ArrayList<>();
                List<String> friend1_friend_list = friends_map.getOrDefault(friend1, array);
                friend1_friend_list.add(friend2);
                friends_map.put(friend1, friend1_friend_list);
            }
        }
        return friends_map;
    }
}
```

### 기타

- 노션에 팀원들 코드 리뷰하며 배울점 정리해 놓은 링크
    - [Notion_1주차_코드 리뷰](https://extreme-pipe-53e.notion.site/098ee6a8c37045898dfe057eae507222)

<br>
