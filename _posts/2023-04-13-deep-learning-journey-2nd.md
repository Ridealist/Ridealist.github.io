---
published: true
layout: posts
title: Ch1. Your Deep Learning Journey(2)
categories: 
  - dl4coders
---


## Your First Model

### 1. GPU 사용가능한 딥러닝 서버 얻기

- GPU?
    - 한 가지일을 동시에 처리할 수 있는 processor : 병렬 처리에 최적화된 processor
    - neural network 학습시 CPU보다 수백배 이상 빠른 처리 가능
    - DL에 맞는 GPU는 소수(NVIDIA GPU 등)

- Cloud환경 GPU 서버 목록
  - ['GPU Servers' 탭](https://course.fast.ai/Lessons/lesson9.html#links-from-the-lesson)


### 2. 첫번째 노트북 실행해보기

```python
from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

1. Dataset은 'Oxford-IIIT Pet Dataset' 활용 (fast.ai 데이터베이스에서 다운로드)
2. pretrained model은 'resnet34'를 사용 (1.3M 이미지들로 학습된 모델)
3. 전이학습(transfer learning)하기 위해 fine-tuning해서 개-고양이 분류 모델 생성


### 3. 머신러닝 이란? Machine Learning

전동적 프로그래밍(연역적 방법) vs. 머신러닝 프로그래밍(귀납적 방법)

![image](/assets/img/스크린샷 2023-04-13 오후 10.12.52.png)

"Artificial Intelligence: A Frontier of Automation" - 'Arthur Samuel' at IBM

- 문제를 풀기 위해 모든 단계를 정확하게 알려주지 말고
  - 풀어야 하는 문제의 예시를 모여준다
    - 그것을 어떻게 풀어야 할지 스스로 찾도록 한다

1. "weight assignment 가중치 할당" 아이디어
2. 모든 가중치 할당은 "실제 수행 능력"를 가짐
3. 결과를 테스트 하기위해 "automatic means 자동화된 방법"이 요구됨
4. 가중치 할당을 바꿔가면서 결과를 향상시키기 위한 "mechanism 방법론"이 필요함

![image](/assets/img/스크린샷 2023-04-13 오후 10.18.13.png)

- weight
  - 가중치도 `변수`
  - 가중치 할당이란 input 변수 외에 프로그램이 어떻게 작동하는지를 정의하는 결국 **또다른 변수**
  - 가중치는 'parameter'라고도 불림
    - 가중치란 model의 특정한 parameter 종류 중 하나

- model
  - 많은 일을 할 수 있는 특별한 종류의 프로그램
  - 가중치에 의존함

- actual performance
  - 얼마나 해당 task를 잘 수행했는지

- mechanism
  - 가중치에 대한 최적화를 자동적으로 하게 하는 방법론
  - 실제 수행능력에 근거해 자동화된 방법으로 최적화 반복 = Learning 학습

![image](/assets/img/스크린샷 2023-04-13 오후 10.28.17.png)

- 결과(result)와 모델의 수행능력(performance)의 차이
- 모델이 학습 되었다면 weight 가중치는 모델의 일부로써 취급 가능
    - 가중치 할당이 완벽하게 최적화가 된 경우
    - 더이상 가중치 변경 없을 경우

결론적으로...
  - **<u>학습된 모델은 마치 전통적 프로그램과 똑같이 취급될 수 있다!</u>**

![image](/assets/img/스크린샷 2023-04-13 오후 10.34.07.png)

### 4. 신경망이란? Neural Network

> 매우매우 유연해서 가중치만 변경할 경우 어떤 문제에도 적용할 수 있는 `함수`가 없을까?

- 이것이 바로 신경망!
  - neural network = 하나의 수학적 함수
  - 가중치에 따라서 아주 유연하게 적용할 수 있는 함수
  - `universal approximation theorem`
    - 이론적으로 모든 수준의 정확도인 어떤 문제도 모두 다 풀 수 있음이 증명됨
  - 우리는 좋은 '가중치 할당'을 찾기만 하면 어떤 모델에도 맞출 수 있다!

- 어떤 과정으로 가중치를 찾지?
  - `stochastic gradient descent (SGD)`
    - 신경망의 가중치를 업데이트하기 위한 가장 일반적인 방법
    - **Chapter 4**에서 자세히 다룰 예정

- 신경망 -> 머신러닝 모델의 일종
  - 매우 유연하다는 점에서 특별함
  - 올바른 가중치만 찾는다면 엄청나게 넓은 범위의 문제에 적용될 수 있다!
  - feat. SGD로 가중치를 찾는 과정을 자동화

---

결론!

> 개-고양이 분류기 === 머신러닝 모델

- 구조
  - input = 이미지들
  - weight = 신경망의 가중치
  - model = resnet34
  - result = predict 결과값

- 평가
  - "actual performance"를 결정한다?
    - 모델의 수행 능력을 `정의`하면 된다
      - ex. 올바르게 답을 예측한 비율 = 정확도(accuracy)
  
- 자동화
  - 가중치 할당을 자동 업데이트하는 mechanism = SGD


