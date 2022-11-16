---
published: true
layout: posts
title: 2주차 회고록 + 과제 피드백 정리
categories: 
  - woowacourse
toc: true
---

2주차 피드백 정리 시작!

### 1. 유틸성 클래스 static 클래스 설계 고려

```java
public class Validator {

    private final String input;

    public Validator(String input) {
        this.input = input;
    }

    public void isLengthThree() {
        if (input.length() != 3) {
            throw new IllegalArgumentException("3자리의 숫자를 입력해주세요(ex. 123)");
        }
    }
}
//리팩터링
public static class Validator {
    
    public static void isLengthThree(String input) {
        if (input.length() != 3) {
            throw new IllegalArgumentException("3자리의 숫자를 입력해주세요(ex. 123)");
        }
    }
}
```

- Validator 클래스를 static으로 설정하는 것 추천
- 객체의 사용자 입장에서 validator의 다른 메서드를 호출할 수 있어 사용에 오류 발생 가능성

#### 정적 클래스(Static Class)

- 정적 클래스는 new 키워드를 사용해서 인스턴스를 만들 수 없습니다.
- 정적 클래스는 class 키워드 앞에 static 키워드를 선언해서 만듭니다.
- 정적 클래스의 모든 멤버는 static으로 선언되어야 합니다.
- 정적 클래스는 생성자를 포함할 수 없습니다.
- 정적 클래스는 객체들이 처음 호출될 때 생성되고 프로그램이 종료될 때 해제되기 때문에 정적 클래스는 어디서든 접근할 수 있습니다.
- 전역적으로 접근해야 하는 `유틸리티 클래스`를 만들 때 정적 클래스로 만들면 유용하게 사용할 수 있습니다.

- [참고 블로그](https://ssabi.tistory.com/27)


### 2. 공통 상수 public 선언 후 프로젝트 전체 활용

- 모든 클래스에 3이라는 공통된 숫자가 들어감
- public으로 선언해서 다른곳에서도 사용하는 방법

```java
// settings/CorrectEnum.java
package baseball.setting;

public enum CorrectEnum {

    STRIKE("스트라이크"),
    BALL("볼"),
    NOTHING("낫싱");

    private final String correct;

    CorrectEnum(String correct) {
        this.correct = correct;
    }

    public String valueOf(){
        return correct;
    }
}

// settings/Settings.java
package baseball.setting;

public class Setting {
    // N 자리 수 맞추기 게임
    public static int INPUT_NUMBER = 3;

    // 게임 시작 코드
    public static int START_GAME = 1;
    // 게임 종료 코드
    public static int END_GAME = 2;
}
```

#### My Opinion

저도 진현님 말에 동감합니다. 변수를 상수화하여 관리하는거 아이디어 얻어갑니다:)
그런데 Enum의 본질이 결국 '연관된' 상수를 클래스로 관리하는 것이니
출력문의 else를 없애기 위해 아래처럼 연관된 상수 전체를 관리해보면 어떨까 생각해봤습니다.

```java
STRIKE("스트라이크", true, false)
BALL("볼", false, true),
STRIKE_BALL("스트라이크, 볼", true, true)
NOTHING("낫싱", false, false);

private final String correct;
private final boolean hasStrikeCount
pirvate final boolean hasBallCount
```

### 3. UI 로직 프린트문 통합 관리 고려

```java
// ui/OutputText.java
package baseball.ui;

import baseball.setting.CorrectEnum;
import baseball.setting.Setting;

public class OutputText {

    public static void printStartGame(){
        System.out.println("숫자 야구 게임을 시작합니다.");
    }

    public static void printEndGame(){
        System.out.println(Setting.INPUT_NUMBER + "개의 숫자를 모두 맞히셨습니다! 게임 종료");
        System.out.println("게임을 새로 시작하려면 "+Setting.START_GAME+", 종료하려면 "+Setting.END_GAME+"를 입력하세요.");
    }

    public static void printInputNumber(){
        System.out.println("숫자를 입력해주세요 : ");
    }

    public static void printJudgeStrike(int count){
        System.out.println(String.valueOf(count)+CorrectEnum.STRIKE.valueOf());
    }

    public static void printJudgeBall(int count){
        System.out.println(String.valueOf(count)+CorrectEnum.BALL.valueOf());
    }

    public static void printJudgeStrikeAndBall(int strike, int ball){
        System.out.println(String.valueOf(ball)+CorrectEnum.BALL.valueOf()+" "+String.valueOf(strike)+CorrectEnum.STRIKE.valueOf());
    }

    public static void printJudgeNothing(){
        System.out.println(CorrectEnum.NOTHING.valueOf());
    }
}
```

### 4. toString 메소드 오버라이딩 고려하기

```java
class GameResult {
    int strike;
    int ball;

    GameResult(int strike, int ball) {
        this.strike = strike;
        this.ball = ball;
    }

    boolean isNothing() {
        return strike == 0 && ball == 0;
    }

    @Override
    public String toString() {
        if (isNothing()) {
            return "낫싱";
        }
        List<String> result = new ArrayList<>();
        if (ball != 0) {
            result.add(String.format("%d볼", ball));
        }
        if (strike != 0) {
            result.add(String.format("%d스트라이크", strike));
        }
        return String.join(" ", result);
    }
}
```

### 5. ParameterizedTest 적절히 활용해보기

```java
public class BaseballGameTest {

    BaseballGame game;

    @BeforeEach
    public void beforeEach() {
        game = new BaseballGame();
    }

    @Test
    void checkUserNumber_메서드가_올바르지않은_입력에_예외_발생() {
        assertThatThrownBy(() -> game.checkUserNumber("3456")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> game.checkUserNumber("61")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> game.checkUserNumber("ab2")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> game.checkUserNumber("224")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> game.checkUserNumber("204")).isInstanceOf(IllegalArgumentException.class);
    }
    // 리팩터링
    @DisplayName("메서드가 올바르지 않은 입력에 예외 처리")
    @ValueSource(strings = {"3456", "61", "ab2", "224", "204"})
    @ParameterizedTest
    void checkUserNumber_invalidInput_exceptionThrown(String input) {
        assertThatThrownBy(() -> game.checkUserNumber(input)).isInstanceOf(IllegalArgumentException.class);
    }
}
```

- 블로그 참고하며 공부해보기
  - [JUnit5 사용법 - Parameterized Tests](https://gmlwjd9405.github.io/2019/11/27/junit5-guide-parameterized-test.html)


### 6. Enum 필드로 UI Output 관리를 고려해 본다

```java
package baseball;

enum Message {
    GREETING("숫자 야구 게임을 시작합니다."),
    PROMPT_FOR_NUMBER("숫자를 입력해주세요 : "),
    CONGRATULATIONS("개의 숫자를 모두 맞히셨습니다! 게임 종료"),
    PROMPT_PLAY_ON("게임을 새로 시작하려면 1, 종료하려면 2를 입력하세요.");

    private final String message;

    Message(String message) {
        this.message = message;
    }

    @Override
    public String toString() {
        return this.message;
    }
}
```

### 7. 정규식을 적절히 활용해본다

```java
class Play {
    private static final int COUNT;
    private static final Pattern PATTERN;
    {% raw %}
    static {
        COUNT = 3;
        String regex = String.format("^(?:([1-9])(?!.*\\1)){%d}$", COUNT);
        PATTERN = Pattern.compile(regex);
    }
    {% endraw %}
    static List<Integer> getNumberFrom(String input) {
        if (input == null || !PATTERN.matcher(input).matches()) {
            throw new IllegalArgumentException();
        }
        List<Integer> givenNumber = new ArrayList<>();
        for (char digitChar : input.toCharArray()) {
            int digitInt = Character.getNumericValue(digitChar);
            givenNumber.add(digitInt);
        }
        return givenNumber;
    }
}
```

하루하루 조금씩 성장하자!

<br>