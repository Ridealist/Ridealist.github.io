---
published: true
layout: posts
title: 우테코 프리코스 미션 규칙
categories: 
  - woowacourse
---

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