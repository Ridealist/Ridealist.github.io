---
published: true
layout: posts
title: '[우테코] 숫자 야구 with 자바 기초_강의 내용 정리'
categories: 
  - woowacourse
toc : false
---

2주차 공통 피드백 문서에서 시청을 권고한 Jason 리더님의 강의를 듣고,

구현 과정을 정리해보려고 한다.

---

## 1. 구현할 기능 목록 뼈대 뽑아내기

- [ ] 1부터 9까지의 서로 다른 임의의 수 3개를 선택한다.
- [ ] 같은 수가 같은 자리에 있으면 스트라이크
- [ ] 같은 수가 다른 자리에 있으면 볼
- [ ] 같은 수가 전혀 없으면 낫싱

## 2. 구현할 기능 목록 구체화

- [ ] 1부터 9까지의 서로 다른 임의의 수 3개를 생성한다.
- [ ] `컴퓨터의 숫자 3개`와 `플레이어의 숫가 3개`를 서로 비교한다.
    - [ ] 몇 개의 숫자가 같은지 찾는다.
        - [ ] 같은 수가 같은 자리에 있으면 스트라이크를 출력한다.
        - [ ] 같은 수가 다른 자리에 있으면 볼을 출력한다.
        - [ ] 같은 수가 전혀 없으면 낫싱을 출력한다.

## 3. 프로그램 뼈대 설계

- 메인 클래스는 Application.java로
- **패키지 설정**을 통해 클래스의 정체성 드러내기
    - ex) `package baseball`의 `Application class` -> 야구 게임
- 하위 패키지 구조 잡기 package.model / package.domain
    - 내가 만들 프로그램의 비즈니스 로직이 들어 있는 영역
- 내가 쓸 클래스 import 해오기
    - ex) import baseball.domain.Calculator;


## 4. main 메소드에서 내가 적은 기능들 모아보기

```java
package baseball;

import baseball.Game.Game;

/**
    // 객체 지향 프로그래밍...!
    // 1. 기능을 가지고 있는 클래스를 인스턴스화(=객체)한다.
    // 2. 필요한 기능을 (역할에 맞는) 각 인스턴스가 수행하게 한다.
    // 3. 각 결과를 종합한다.
*/

public class Application {
    public static void main(String[] args) {
        makeRandomNumber
        int[] computer = {1, 2, 3};

        getUserInput
        int[] player = {4, 5, 6};

        getGameScore
        (   ).match(computer, player)

        printResult
        System.out.println("----")
    }
}
```

## 5. 기능 네이밍 - 클래스 네이밍 연결 짓기

- [ ] 1부터 9까지의 서로 다른 임의의 수 3개를 생성한다. - createRandomNumbers() - # NumberGenerator
- [ ] `컴퓨터의 숫자 3개`와 `플레이어의 숫가 3개`를 서로 비교한다. - compare() - # Referee
    - [ ] 몇 개의 숫자가 같은지 찾는다. - countMatchNumber() # Judgement
    - [ ] 특정 자리에 특정 숫자가 있는지 확인한다. - checkIndex() # Judgement
- [ ] 같은 수가 같은 자리에 있으면 스트라이크를 출력한다.
- [ ] 같은 수가 다른 자리에 있으면 볼을 출력한다.
- [ ] 같은 수가 전혀 없으면 낫싱을 출력한다.


## 6. 구상한 클래스-메소드의 뼈대 쌓기
- 클래스 생성
- 자료구조 생성
- 메소드 생성
    - `메소드 시그니처(Method Signature)` 작성
    - void
    - return 값 지정
        - return null; / return 0; / return false;
- skeleton code를 짜면 git commit 하기


## 7. 뼈대 코드 내부 로직 채워넣기
- 구현된 내용을 commit 한다
- 구현하면서 추가된 사항을 README에 반영한다
- 완성된 기능을 CheckList 표시한다


```java
package baseball.domain;

import java.util.List;
import java.util.Random;

public class NumberGenerator{
    // 구현
    public List<Integer> createNumbers() {
        List<Integer> numbers = new ArrayList<>();
        for (int i = 0; i < 3; i++>) {
            int number = new Random().nextInt(9) + 1;
            numbers.add(number);
        }
        return numbers;
    }

    // 리팩터링
        public List<Integer> createNumbers() {
        List<Integer> numbers = new ArrayList<>();
        while (numbers.size() < 3) {
            int number = new Random().nextInt(9) + 1;
            // 만약 이미 존재하는 숫자라면 넣지 않는다
            if (numbers.contains(number)) {
                // 조건 만족하면 이후 문장 실행하지 않고 다음 반복으로 넘어감
                continue;
            }
            numbers.add(number);
        }
        return numbers;
    }
}
```


```java
package baseball.domain;

import java.util.List;

public class Judgement {
    public int correctCount(List<Integer> computer, List<Integer> player) {
        int result = 0;
        for (int i = 0; i < player.size(), i++) {
            int playerNumbers = player.get(i)
            if (computer.contains(playerNumbers)) {
                result++;
            }
        }
        return result;
    }

    public boolean hasPlace(List<Integer> computer, int placeIndex, int number) {
        // if (computer.get(placeIndex) == number) {
        //     return true;
        // }
        // return false;
        return computer.get(placeIndex) == number;
        // 리팩터링 가능!
    }
}
```

```java
package baseball.domain;

import java.util.List;

public class Referee {
    public int compare(List<Integer> computer, List<Integer> player) {
        // 몇 개의 숫자가 같은지 찾는다
        // 자리수까지 같은 개수(스트라이크 수)를 구해 뺀다.
        // 남은 수는 볼의 개수이다.
        
        // 몇 개의 숫자인지 찾는 것은 'Judgement' 클래스 역할
        // 이런 것을 `협력`한다고 이야기 함
        Judgement judge = new Judgement();
        int correctCount = judge.correctCount(computer, player)

        int strikeCount = 0;
        for (int placeIndex = 0; placeIndex < player.size(); i++) {
            if (judge.hasPlace(computer, placeIndex, player.get(placeIndex))) {
                strikeCount++;
            }
        }
        int ballCount = correctCount - strikeCount;
        return ballCount + "볼" + strikeCount + "스트라이크"
    }
}
```

- 기능목록을 계속 update하면서 진행
- 이슈를 쳐내면서 하니씩 구현
    - [x] 1부터 9까지의 서로 다른 임의의 수 3개를 생성한다. - createRandomNumbers() - # NumberGenerator
    - [x] `컴퓨터의 숫자 3개`와 `플레이어의 숫가 3개`를 서로 비교한다. - compare() - # Referee
        - [x] 몇 개의 숫자가 같은지 찾는다. - countMatchNumber() # Judgement
        - [x] 특정 자리에 특정 숫자가 있는지 확인한다. - checkIndex() # Judgement
    - [x] 같은 수가 같은 자리에 있으면 스트라이크를 출력한다.
    - [x] 같은 수가 다른 자리에 있으면 볼을 출력한다.
    - [x] 같은 수가 전혀 없으면 낫싱을 출력한다.

완료!!!


## 8. 클래스들을 Application.java에 최종 연결하기
- 객체를 불러와 참조하고 사용한다
    - 객체지향에서 이를 `협력`한다 라고 함


## 9. 테스트 코드 작성하기

> 테스트코드는 기능 구현과 동시에 작성하는게 바람직!!!

- 실행 결과를 직접 실행시키며 확인하는 것이 아닌, 컴퓨터에게 맡긴다
- 테스트를 통해 실행 검증 자동화
    - 코드의 버그를 가장 먼저 알아차리는 건 `테스트 코드`

```java
package baseball.domain;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.csvSource;

class RefereeTest {

    private static final List<Integer> ANSWER;
    // static(클래스 변수) 필드를 만들어도 좋음
    private Referee referee;

    // BeforeEach 셋업 메소드
    // 매번 인스턴스가 새롭게 생성됨
    @BeforeEach
    public void setUp() {
        referee = new Referee();
    }

    // JUnit5부터 지원하는 기능
    // 반복된 Test에 대해 중복 코드를 줄일 수 있음
    @ParameterizedTest
    @CsvSource({"1,2,3,0볼 3 스트라이크", "7,8,9,아웃", "2,3,1,3볼 0 스트라이크", "1,3,2,2볼 1스트라이크"})
    public void compareTest(int number1, int number2, int number3, String result) {
        String actualResult = referee.compare(ANSWER, Arrays.asList(number1, number2, number3));
        assertThat(actualResult).isEqualTo(result);
    }
}
```

---

이상의 과정을 이번 3주차 미션부터 적용해봐야겠다.

항상 기초부터 튼튼히... 화이팅!