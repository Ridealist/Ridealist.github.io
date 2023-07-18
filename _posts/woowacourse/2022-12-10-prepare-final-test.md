---
published: true
layout: posts
title: '[우테코] 최종 코딩테스트 대비 정리 자료'
categories: 
  - woowacourse
toc: true
---

`greeng00se`님의 다음 글에 감명받아, 최종 코딩테스트를 위한 나만의 기준을 정리해본다.
- [[BE] 자신만의 기준 정립하여 생각 시간 단축하기 ](https://github.com/orgs/woowacourse-precourse/discussions/1706)

## 1. 예외

### 예외 발생시 재호출 제네릭 예제

- view에서 입력 받아오는 것만 재요청할 경우

```java
// 사용
repeat(inputView::readMoving)

// 정의
private <T> T repeat(Supplier<T> inputReader) {
    try {
        return inputReader.get();
    } catch (IllegalArgumentException e) {
        outputView.printError(e.getMessage());
        return repeat(inputReader);
    }
}
```

- view 입력과 도메인로직 validate까지 묶어서 재요청할 경우
-`twoosky`님 댓글 참고

```java
// 사용
BridgeSize bridgeSize = read(BridgeSize::new, inputView::readBridgeSize);

// 정의
private <T, R> R repeat(Function<T, R> object, Supplier<T> input) {
    try {
         return object.apply(input.get());
    } catch (IllegalArgumentException e) {
         outputView.printErrorMessage(e.getMessage());
         return repeat(object, input);
    }
}
```

### 예외 처리 위치

- (기본 원칙) 예외 처리는 view 와 model 모두의 책임이다
- 객체가 생성될 때 내 상태값이 올바른지 검증하는것 또한 객체의 자율성을 보장하는 일이라 생각
    - 입력 요구사항을 만족시키기 위해 → 예) InputView에서 예외 던지기
    - 도메인의 요구사항을 만족시키기 위해 → 예) 클래스 생성할 때 validation

> 객체의 행동을 유발하는 것은 외부로부터 전달된 메시지지만 객체의 상태를 변경할지 여부는 객체 스스로 결정한다.

### 예외 메시지

- 해당 예외를 던지는 클래스에 상수로 정의한다
- 추후 Enum을 활용한 Error 관리 공부
    - [Best way to define error codes/strings in Java?](https://stackoverflow.com/questions/446663/best-way-to-define-error-codes-strings-in-java)
    - [[Spring] Enum을 사용하여 예외처리 해보기](https://fenderist.tistory.com/116)

- 표준 예외를 사용한다

### 정규표현식


## 2. 클래스

### 클래스 나누기

- 최대한으로 클래스를 분리하는 연습을 한다. 클래스가 하나의 책임을 가지도록 연습한다.
    - 분리를 했으면 직접 사용하지 않고 메시지를 전달한다.
- 예) 로또 번호 리스트(일급 컬렉션) → 로또(일급 컬렉션) → 로또 번호(원시값 포장한 클래스)
- 참고 자료
    - [일급 컬렉션](https://jojoldu.tistory.com/412)
    - [객체지향 생활체조의 원시값 포장](https://jamie95.tistory.com/99)

### Enum
- 내부에 상태를 확인하는 메서드를 만들어둔다면 더욱 객체스럽게 사용할 수 있다.
- Enum을 사용하는 Map이 필요한 경우 EnumMap을 사용한다.(더 빠름)

```java
public enum BridgeGameStatus {
    PLAY("R"),
    STOP("Q");

    private static final String INVALID_COMMAND_MESSAGE = "R과 Q 중 하나의 값을 입력해주세요.";

    private final String command;

    BridgeGameStatus(String command) {
        this.command = command;
    }

    public boolean isPlayable() {
        return this.equals(PLAY);
    }

    public boolean isNotPlayable() {
        return this.equals(STOP);
    }

    public static BridgeGameStatus gameStart() {
        return PLAY;
    }

    public static BridgeGameStatus from(String command) {
        return Arrays.stream(values())
                .filter(status -> status.command.equals(command))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(INVALID_COMMAND_MESSAGE));
    }
}
```


## 3. 메서드

### 네이밍 시간 단축하기

- 결과를 boolean으로 받는경우 메서드명을 is로 시작한다. + (can, should, will, has 등)
    - [boolean 네이밍 컨벤션](https://soojin.ro/blog/naming-boolean-variables)
- 출력을 위해 값을 가져오는 경우 메서드명을 get으로 시작한다.
- 변환하는 경우 메서드명을 to로 시작한다.

### toString()

- 디버깅 용도로만 toString()을 재정의한다. (Effective Java Item 12)

### 메스드 길이 줄이기

- 인라인을 적절히 사용한다면 메서드의 길이를 줄일 수 있다.

### Stream

- 스트림을 적절히 사용한다면 메서드 길이를 줄일 수 있고 가독성을 개선할 수 있다.
- 스트림 생성
    - `Stream.generate()` → 예) `Stream.generate(bridgeNumberGenerator::generate)`
    - `IntStream.range()`, `IntStream.rangeClosed()`
    - `Enum → Arrays.stream(values())`
    - 컬렉션을 이용한 스트림 생성
- 중간 연산
    - limit → 크기 제한
    - map, mapToObj → 변환
    - filter → 필터링
    - findFirst, findAny → 필터링 후 일치하는 요소 하나만 가져오기
    - distinct → 중복 제거
- 최종 연산
    - collect(toList()) → 가장 많이 사용
    - collectingAndThen → collect로 수집 후 객체 생성할 때


## 4. 출력

- 포맷팅한 문자열을 상수로 정의한 후 MessageFormat.format을 이용하면 출력에 필요한 라인 수를 줄일 수 있다.

```java
// 4주차 다리 건너기 출력 예시
private static final String MAP_MESSAGE_FORMAT = "[ {0} ]";
private static final String MAP_MESSAGE_DELIMITER = " | ";

private String getResultMessage(List<MoveResult> moveResults) {
    return format(MAP_MESSAGE_FORMAT, moveResults.stream()
            .map(출력값으로변환하는메서드())
            .collect(Collectors.joining(MAP_MESSAGE_DELIMITER))
    );
}
```

## 5. 테스트

### 테스트 대상

- public 메서드의 경우에만 테스트한다.
- 생성자도 validation이 있어 예외를 던지는 경우 테스트한다.

### 반복

- 테스트가 반복되는 경우 `@ParameterizedTest`를 활용한다.
    - 입력이 하나인 경우 `@ValueSource`
    - 여러개인 경우 `@CsvSource`, `@MethodSource`를 사용한다.

### 가독성

- 테스트 메서드명은 한글을 사용한다. (가독성, 생각할 시간 줄이기)
    - intellij → build → gradle → Run test using을 intellij로 수정한다.
- 테스트 메서드명은 메서드명_입력_결과 형식으로 작성한다.(테스트 대상의 행동으로 작성한다)
    - getEnum_메서드는_올바르지_않은_값을_입력받으면_IllegalArgumentException을_던진다.

### 준비, 실행, 검증

- given, when, then을 사용하면 테스트를 이해하기 쉽게 하지만 더욱 간결하게 빈 줄로 구분할 수 있다.
- 짧은 테스트일 때 주석을 제거한다면 간결하고 가독성이 좋아진다.

```java
@Test
void reset_메서드는_게임_결과를_초기화_시킨다() {
    BridgeGameResult bridgeGameResult = generateGameResult(1, List.of(UP_SUCCESS));

    bridgeGameResult.reset();

    assertThat(bridgeGameResult.getResult()).isEmpty();
}
```

## 6. IntelliJ

### 라이브 템플릿 활용하기

- preferences -> live templates에 들어가면 직접 라이브 템플릿을 추가할 수 있다.
- File -> Manage IDE Settings -> Export Settings를 통해 설정을 파일 형태로 저장할 수 있다.
- 기존에 있는 라이브 템플릿 + 커스텀 라이브 템플릿을 이용하면 반복되는 코드를 작성하는 시간을 단축할 수 있다.
- 커스텀 라이브 템플릿 예시
    - psfs → `private static final String`

    - pf → `private final`

    - at → `Assertions.assertThat(`

    - atb → `assertThatThrownBy(() →`

    - ptest, test → parameterized test, 일반적인 test 생성

