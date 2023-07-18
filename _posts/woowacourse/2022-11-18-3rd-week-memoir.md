---
published: true
layout: posts
title: '[우테코] 3주차 회고록 + 코드 리뷰 스터디 피드백 정리'
categories: 
  - woowacourse
toc: true
---

스터디원 코드에서 배울 점

### 1. 반복되는 상수들을 별도 클래스 관리 고려해보기

```java
package lotto.setting;

public class Setting {

    public static int LOTTO_MIN_NUMBER = 1;
    public static int LOTTO_MAX_NUMBER = 45;
    public static int LOTTO_PICK_NUMBER = 6;
    public static int LOTTO_PRICE_PER_ONE = 1_000;
}
    // 활용 모습
    private void validateInputNumberIsBetween(int number){
        if(number < Setting.LOTTO_MIN_NUMBER || number > Setting.LOTTO_MAX_NUMBER){
            throw new LottoException("번호는 "+Setting.LOTTO_MIN_NUMBER+"부터 "+ Setting.LOTTO_MAX_NUMBER +" 사이의 숫자여야 합니다.");
        }
    }
```

### 2. Error 접두사 처리

#### 2-1. IllegalArgumentException 상속

```java
package lotto.exception;

public class LottoException extends IllegalArgumentException{
    private static final String ERROR_PREFIX = "[ERROR] ";

    public LottoException(String message){
        super(ERROR_PREFIX + message);
    }
}
    // 활용 예시
    private void validateInputMoneyIsRightUnit(String money){
        if(Math.toIntExact(Long.parseLong(money) % Setting.LOTTO_PRICE_PER_ONE) != 0){
            throw new LottoException("구입 금액은 "+ Setting.LOTTO_PRICE_PER_ONE +"원 단위입니다.");
        }
    }
```

#### 2-2. ErrorUI 클래스 정의

```java
package lotto.ui;

public class ErrorUI {
    private static final String ERROR_HEADER = "[ERROR]";

    public void printError(String errorMessage) {
        System.out.println(String.format("%s %s", ERROR_HEADER, errorMessage));
    }
}

// 활용 예시
public class LottoController {
    public void run() {
        try {
            Ticket ticket = purchase();
            List<Lotto> lottos = exchange(ticket);
            printPurchaseLotto(lottos);
            Ranker ranker = inputNumber();
            Map<WinningResult, Integer> result = ranker.rankTotal(lottos);
            printResult(result);
            printRateOfReturn(result, ticket);
        } catch (IllegalArgumentException e) {
            ErrorUI ui = new ErrorUI();
            ui.printError(e.getMessage());
        }
    }
```
