# 1. Your Deep Learning Journey

## Deep Learning If For Everyone

- World-Class 딥러닝을 배우는데 그렇게 많은 것이 필요하지 않다!
    - 고등학교 수준의 수학
    - 50개 이하의 데이터로도 기록 갱신
    - 무료로 최신의 컴퓨팅 작업 가능

- 현재 딥러닝에 의존한 방법이나 딥러닝을 활용하여 세계 최고의 성과를 내고 있는 분야
    1. Natural Language Processing(NLP)
        - 챗봇, 발화 인식, 문서 요약, 문서 분류
    2. Computer Vision(CV)
        - 위성/드론 사진 해석, 얼굴 인식, 이미지 캡셔닝
    3. Medicine
    4. Bilogy
    5. Image generation
        - 이미지 색칠, 해상도 늘리기, 노이즈 제거하기, 특정 화풍으로 바꾸기
    6. Recommendation System
    7. Playing games
    8. Robotics


## Neural Network: A Brief History

### 1. “A Logical Calculus of the Ideas Immanent in Nervous Activity"

- 인공 신경망의 최초의 수학적 모델
- 실제 신경계의 단순화된 모델은 덧셈과 역치값을 활용해 나타낼 수 있음

>신경계는 역치를 넘어서면 신호가 전달되는 방식. 즉 'all-or-none' 특징 이므로 이는 결국 수학적 논리식(Propositional Logic)으로 나타낼 수 있음.


### 2. "The Design of an Intelligent Automaton"
- Perceptron 개념 구상
    - https://needjarvis.tistory.com/181
- 인공 신경계 + 학습 / 이를 적용한 기계 개발
    
    **-> single layer perceptron은 XOR 문제를 해결하지 못한다**


### 3. PDP(Parallel Destributed Processing)
- 실제 뇌가 작동하는 방식을 최대한 본떠서 컴퓨터를 작동하게 하자
- PDP의 요구 조건
    - A set of processing units
    - A state of activation
    - An output function for each unit
    - A pattern of connectivity among units
    - A propagation rule for propagating patterns of activities through the network of connectivities
    - An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
    - A learning rule whereby patterns of connectivity are modified by experience
    - An environment within which the system must operate


### 4. 2단의 layer로 구성된 뉴런
- XOR 문제 해결
- 수학적으로 모든 함수형태에 근사 가능

    **-> 실제로 너무 크거나 너무 느려서 사용하기에 부적합**

### 5. 지금의 Deep Learning
- 여러단의 layers + 컴퓨터 하드웨어 + 데이터 + 알고리즘의 조합으로 신경망이 더욱 빠르고 쉽게 학습 가능

![image](https://www.researchgate.net/publication/346219836/figure/fig1/AS:980115399913472@1610689129209/Schematic-of-shallow-neural-network-and-deep-neural-network.ppm)


## How to Learn Deep Learning

1. 전체로서 배운다
- 실제 문제를 해결할 수 있는 딥러닝 모델을 먼저 만들어 본다
- 더 깊게 파 들어가며 도구들이 어떻게 만들어졌는지 이해한다
- 도구들을 만드는 도구들이 어떻게 만들어졌는지 이해한다. 이것의 반복

2. 예시를 통해 배운다
- 직관적 이해가 먼저. 수학적 기호로 표현은 나중.

3. 최대한 간결화한다


> 딥러닝을 잘하려면 장인정신이 필요하다
- 데이터가 충분한지?
- 그것이 올바른 형식인지?
- 학습이 잘 되지 않는다면 무엇을 해야 하는지?

결국, Learning By Doing!

‘The key is to just code and try to
solve problems: the theory can come later, when you have context and
motivation.’


### Your Mindset
- 이해가 안되면 다시 그 부분부터 다시 천천히 읽어라!
- 코드를 작성해보며 실험해보라!
- 그 부분에 대해서 Google 검색이나 tutorial을 찾아봐라!
- 이해가 안되면 너무 붙잡지 말고 때론 넘어가라!

### Your Projects
- 내가 정말 관심있고 열정을 갖고 할 수 있는 '작은' 프로젝트를 '여러 개' 해봐라!
    - 큰 문제를 해결하려고 하다가 지쳐 나가떨어질 수 있다
    - 기본을 마스터했다 판단이 들면, 그때 정말 해결해보고 싶은 문제에 도전하라

>이 책을 통해 배운 것을 개인 프로젝트에 적용하라. 항상 끈기를 갖고 노력하라


### PyTorch, fastai, Jupyter

### 1. PyTorch vs. TensorFlow
    - https://youtu.be/z7F91vilnDc

### 2. fastai : PyTorch 기반으로 고수준 기능을 추가한 Library

    - 고수준 개념: fastai
    - 저수준 개념: PyTroch, Python

> ‘You should assume that whatever specific libraries and software you learn today will be obsolete in a year or two’

- 학습의 초점
    - 기반 지식에 대한 이해
    - 그것을 어떻게 실제적으로 활용하는지
    - 새로운 툴과 기술들에 대한 전문지식을 어떻게 빠르게 쌓을지

### 3. Jupyter : Intereaction 형식으로 python 코드를 작성할 수 있는 플랫폼
- 딥러닝은 코드를 치고 실험해보는 과정이 무엇보다도 중요
- 코드 실험이 편한 플랫폼 중 하나

