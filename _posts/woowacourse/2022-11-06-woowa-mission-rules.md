---
published: true
layout: posts
title: '[우테코] 우테코 프리코스 미션 규칙'
categories: 
  - woowacourse
toc: true
---

## 여는 글

이번 글에서는 우테코 프리코스 미션을 진행하면서 내가 지켜야 할 규칙들을 정리해보려고한다.

1. 포수타(포비와수다타임)
2. 1주 차 공통 피드백
3. 2주차 미션 프로그래밍 요구사항 / 과제 진행 요구사항

위 3가지 내용을 총 정리하여 내가 지켜야 할 체크리스트 목록을 만들 것이다.

지금 적은 글을 바탕으로 매 미션을 할 때마다 아래 내용들을 잘 지키고 있는지 확인하자!

(사실 우테코 프리코스 뿐만 아니라 프로그래머로서 꼭 지켜야할 기본 규칙이니 프로그래머의 삶을 사는 동안 앞으로도 계속...!)

<br>

## 1. 커밋 메시지

### 1-1. 의미 잇는 커밋 메시지
-  커밋 메시지를 의미 있게 작성한다
    - **보는 사람이 읽고 이해하기 좋은 커밋 메세지**가 좋은 커밋 메세지
    - 커밋 메시지에서 해당 커밋에서 작업한 내용에 대한 이해가 가능하도록 작성

커밋 메세지도 하나의 `문서`라는 표현이 인상적이었다. 잘 쓰여진 문서처럼 올바르게 커밋 메세지를 쓰자.

### 1-2. 커밋 메시지 컨벤션

- 우테코에서 제시한 기준
    - [AngularJS Commit Message Comventions](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)
- 번역본 블로그들
    - [outstandingb.log](https://velog.io/@outstandingboy/Git-%EC%BB%A4%EB%B0%8B-%EB%A9%94%EC%8B%9C%EC%A7%80-%EA%B7%9C%EC%95%BD-%EC%A0%95%EB%A6%AC-the-AngularJS-commit-conventions)
    - [prefer2 Log](https://prefer2.tistory.com/entry/git-%EC%BB%A4%EB%B0%8B-%EC%BB%A8%EB%B2%A4%EC%85%98-AngularJS-Git-Commit-Message-Conventions)


찾아보니 위 문서를 번역해서 올려준 친절한 블로그들이 많이 있었다. 나도 시간을 내서 한 번 번역해보겠다 마음먹으며...

TODO `AngularJS 커밋 메시지 컨벤션` 번역해보기


## 2. git 사용법 숙지

### 2-1. git을 통해 관리할 자원에 대해서도 고려한다

- 다음의 파일들은 git을 통해 관리하지 않아도 됨
    - .class 파일은 java 코드가 있으면 생성 가능
    - IntelliJ IDEA의 .idea  폴더, Eclipse의 .metadata 폴더 또한 개발 도구가 자동으로 생성하는 것
- git에 코드를 추가할 때는 git을 통해 관리할 필요가 있는지를 고려


적절하게 `.gitignore 파일`에 git으로 관리할 필요 없는 파일들을 명시해놓자.

관련 자료들을 친절하게도 많이 추가해주셨다... (코치님 사랑합니다)

- [[10분 테코톡] 오리&코린의 Merge, Rebase, Cherry pick](https://www.youtube.com/watch?v=b72mDco4g78&ab_channel=%EC%9A%B0%EC%95%84%ED%95%9CTech)
- [[10분 테코톡] 🎲 와일더의 Git Commands](https://www.youtube.com/watch?v=JsRD2AWxxFg&ab_channel=%EC%9A%B0%EC%95%84%ED%95%9CTech)
- [git - 간편 안내서](https://rogerdudler.github.io/git-guide/index.ko.html)
- [git과 github_inflearn강의](https://www.inflearn.com/course/git-and-github#curriculum)


## 3. Java 코딩 컨벤션

- [Java 코드 컨벤션 가이드](https://github.com/woowacourse/woowacourse-docs/tree/main/styleguide/java)

결국 코드는 내가 작성하는 시간보다 남들이 읽는 시간이 더 많이 소요된다.

좋은 코드란 다른 사람이 읽기 쉬운 코드이고, 가독성을 위해 모두가 합의한 규칙(문법)인 코딩 컨벤션에 맞춰 작성해야 한다.

아래는 코치님이 1주차 공통 피드백에서 알려주신 내용. 숙지하고 항상 체크하자!

- 공백도 코딩 컨벤션이다
    - if, for, while문 사이의 공백도 코딩 컨벤션이다
- 공백 라인을 의미 있게 사용한다
    - 공백은 문맥을 분리하는 부분에 사용하는 것이 좋다
    - 과도한 공백은 다른 개발자에게 의문을 줄 수 있다
- space와 tab을 혼용하지 않는다
    - 들여쓰기에 space와 tab을 혼용하지 않는다. 둘 중에 하나만 사용한다
- IDE의 `코드 자동 정렬 기능`을 활용한다
    - IDE의 코드 자동 정렬 기능을 사용하면 더 깔끔한 코드를 볼 수 있다
        - IntelliJ IDEA: `⌥⌘L`, `Ctrl+Alt+L`
        - Eclipse: `⇧⌘F`, `Ctrl+Shift+F`


## 4. 클린 코드를 위한 원칙들

### 4-1. 객체지향 생활 체조 원칙 9가지

`The ThoughtWorks Anthology`라는 책에 나오는 내용이라고 한다

- [woowacourse_pr_checklist](https://github.com/woowacourse/woowacourse-docs/blob/main/cleancode/pr_checklist.md)
- [[Refactoring] 객체지향 생활 체조 원칙_블로그](https://blogshine.tistory.com/241)


### 4-2. 좋은 이름 짓기

> 이름을 통해 의도를 드러낸다!

- 변수 이름, 함수(메서드) 이름, 클래스 이름을 짓는데 시간을 투자하라
- 이름을 통해 변수의 역할, 함수의 역할, 클래스의 역할에 대한 의도를 드러내기 위해 노력하라
    - 연속된 숫자를 덧붙이거나(a1, a2, ..., aN) 방식
    - 불용어(Info, Data, a, an, the)를 추가하는 방식등은 적절하지 못하다

### 4-3. 축약하지 않기

- 의도를 드러낼 수 있다면 이름이 길어져도 괜찮다.

- 객체 지향 생활 체조 원칙 5: 줄여쓰지 않는다 (축약 금지)

> 누구나 실은 클래스, 메서드, 또는 변수의 이름을 줄이려는 유혹에 곧잘 빠지곤 한다. 그런 유혹을 뿌리쳐라. 축약은 혼란을 야기하며, 더 큰 문제를 숨기는 경향이 있다. 클래스와 메서드 이름을 한 두 단어로 유지하려고 노력하고 문맥을 중복하는 이름을 자제하자. 클래스 이름이 Order라면 shipOrder라고 메서드 이름을 지을 필요가 없다. 짧게 ship()이라고 하면 클라이언트에서는 order.ship()라고 호출하며, 간결한 호출의 표현이 된다.


### 4-4. 의미 없는 주석 달지 않기
- `좋은 이름 짓기`가 선행된다면 주석은 대부분 불필요하다
    - 변수 이름, 함수(메서드) 이름을 통해 어떤 의도인지가 드러난다면 굳이 주석을 달지 않는다
- 가능하면 `이름`을 통해 의도를 드러낸다
    - 의도를 드러내기 힘든 경우 주석을 단다



## 5. Java 언어에 대한 이해

- Java에서 제공하는 API를 적극 활용한다
    - 함수(메서드)를 직접 구현하기 전에 Java API에서 제공하는 기능인지 검색을 먼저 해본다
    - Java API에서 제공하지 않을 경우 직접 구현한다

```java
//사용자가 2명 이상이면 쉼표(,) 기준으로 출력을 위한 문자열 만들기

List<String> members = Arrays.asList("pobi", "jason");
String result = String.join(",", members); // "pobi,jason"
```

- 배열 대신 Java Collection Framework(JCF)을 사용한다
    - Java Collection 자료구조(List, Set, Map 등)를 사용하면 데이터를 조작할 때 다양한 API를 사용할 수 있다


---

## 닫는 글

이상의 것들을 지금 모두 내것으로 만들기는 어려울 것 같다.

다만, 위 내용들은 지식보다 실천이 중요한 내용들인 만큼 계속 이 글을 확인하면서 위 규칙들을 지키려고 노력해야겠다.

그렇게 **습관처럼 몸에 배어 나도 모르게 이미 지키고 있는 수준**이 되는게 목표다.

프로그래밍을 할 때 마다 위 원칙들을 항상 확인하고 되뇌자!