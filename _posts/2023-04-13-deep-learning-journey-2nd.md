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

![image](https://user-images.githubusercontent.com/89024993/232101507-bfe93f34-3ca2-40c7-b33c-5e240e2ee158.png)

"Artificial Intelligence: A Frontier of Automation" - 'Arthur Samuel' at IBM

- 문제를 풀기 위해 모든 단계를 정확하게 알려주지 말고
  - 풀어야 하는 문제의 예시를 모여준다
    - 그것을 어떻게 풀어야 할지 스스로 찾도록 한다

1. "weight assignment 가중치 할당" 아이디어
2. 모든 가중치 할당은 "실제 수행 능력"를 가짐
3. 결과를 테스트 하기위해 "automatic means 자동화된 방법"이 요구됨
4. 가중치 할당을 바꿔가면서 결과를 향상시키기 위한 "mechanism 방법론"이 필요함

![image](https://user-images.githubusercontent.com/89024993/232101504-30f0f5cc-4f0a-4bb9-89b0-775ec794a65f.png)

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

![image](https://user-images.githubusercontent.com/89024993/232101500-3323ed2d-3918-4f0c-8c63-c10ba9228e10.png)

- 결과(result)와 모델의 수행능력(performance)의 차이
- 모델이 학습 되었다면 weight 가중치는 모델의 일부로써 취급 가능
    - 가중치 할당이 완벽하게 최적화가 된 경우
    - 더이상 가중치 변경 없을 경우

결론적으로...
  - **<u>학습된 모델은 마치 전통적 프로그램과 똑같이 취급될 수 있다!</u>**

![image](https://user-images.githubusercontent.com/89024993/232101538-d7fef220-58c5-401e-afdb-8269334ecc69.png)

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

### 5. 딥러닝 전문용어 맛보기

- 'model'의 함수적 형태 = 아키텍쳐 architecture
- 'weight' = 'parameter'
- 에측값(prediction) = labels가 없는 데이터인 독립적인 변수들(independent variable)로부터 계산되는 값
- labels = targets
- 모델의 결과(result) = 예측값
- 성능(performance)의 측정 = 손실(loss)
- 손실은 '예측값'과 '올바른 labels'에 둘다 영향을 받음

![image](/assets/img/스크린샷 2023-04-14 오후 9.55.52 1.png)

### 6. 머신 러닝의 한계

- 데이터가 항상 필요하다
- 학습에 사용된 데이터에서 발견할 수 있는 패턴만 배울 수 있다
- 머신러닝은 오닉 예측(predictions)만 할 수 있을뿐, 추천된 행동(recommended actions)을 할 수 없다
  - 추천시스템은 사용자가 관심을 가질만한 **새로운 상품** 보다는 써봤거나 알고 있는 상품은 추천해주게 된다
- 단순히 데이터가 아닌 labeled 데이터가 필요하다
  - labeling approach가 실제적으로 매우 중요한 이슈이다
- `feedback loops`문제
  - 모델이 더 사용될수록 더 편향된 데이터가 생성되고 이것에 의해 더 편향된 모델이 만들어지는 과정을 반복하게 됨

### 7.이미지 분류기의 작동 원리

#### Classification vs. Regression
- 분류: 클래스나 카테고리를 예측하기 위한 모델
  - 숫자나 이산적인 수치를 예상하기 위한 것
  - ex. 개?고양이?
- 회귀: 수치적인 양을 예측하기 위한 모델
  - ex. 기온/위치

#### Overfitting 과적합
- 머신 러닝 학습에서 항상 고려해야하는 중요한 이슈
- 학습된 데이터에서 잘 예측하는 것보다, **새로운 데이터**에서 잘 예측하는 것이 훨씬 어렵고 또 중요!
- The data will matter in practice
  - ex. MNIST dataset
  - 어떤 데이터도 같은게 없음(조금씩 다름)

> BUT, 과적합이 일어났을 때 과적합을 방지하기!

- validation 정확도가 학습에 따라 떨어짐 -> 과적합
- 모델이 도달할 수 있는 높은 정확도 수준에 못미치게 학습을 마치는 경우 -> underfitting

#### metrics 평가 지표
- validation set을 이용해 모델이 예측한 값의 품질을 측정하는 함수
  - 매 epoch가 끝날때마다 출력됨

loss 손실 함수 vs. metrics 평가 지표
- loss = 학습 시스템이 가중치를 자동으로 갱신하는데 사용되는 "수행 능력의 측정"을 정의하는 목적
  - 좋은 손실 함수란 SGD를 적용하기 좋은 선택이어야 함
- metric = 사람이 이해하기 위한 목적
  - 좋은 평가 지표는 이해하기 쉬운 선택지
  - 내가 모델에게 기대하는 task를 최대한 잘 반영해야 함

#### pretrained model 사전 학습 모델
- 이미 다른 데이터셋에서 학습된 가중치를 가진 모델
  - ex. 1.3M 이미지로 천개의 다를 카테고리를 인식하는 학습된 모델 
- pretrained = 모델의 가중치를 이미 사전 훈련된 모델의 가중치로 맞추는 것
- 대부분의 경우 이런 모델을 사용
  - 이미 성능이 훌륭함

`head`
- pretrained model의 마지막 부분
- 사전 학습 모델을 사용할 경우 마지막 layer를 삭제함
  - 기존의 학습 task에 맞춰져 있기 때문
- 1개 이상의 랜덤 가중치를 가진 새로운 layer로 대체함
  - 작업하려는 데이터셋의 사이즈에 맞춤

- 사전 학습 모델을 사용하는 것은 매우 매우 중요
  - 더 빠르고, 적은 시간과 돈을 들려 더 정확한 모델은 만들 수 있음
- <a>사전 학습 모델을 활용하면 적은 자원을 가지고서도 Deep Learning을 이용한 많은 작업을 할 수 있음</a>

#### transfer learning 전이 학습
- 처음 학습시킨 task와 다른 일을 하도록 pretrained model을 활용하는 것
- 아직 많이 연구되지 못한 분야...
  - 사용가능한 prtrained model이 있는 domain이 한정적
  - 시계열 분석에서 transfer learning이 어떻게 솽용되어야 하는지 아직 잘 모르고 있는 상황

#### fine tuning 파인 튜닝
- pretrained model을 새로운 데이터셋에 적용하기 위한 중요한 'trick' 중 하나
- 사전 학습과 다른 task를 추가적인 epochs만큼 학습하여 pretrained model의 파라미터값을 갱신하는 전이 학습 방법론(transfer learning technique)

#### Jargon : 몇가지 용어
- 모델의 head : 새로운 데이터셋에 맞춰져서 새롭게 추가된 부분
- epoch : 데이터셋 전체를 완전히 한번 훑는 것
(fit이 수행된 이후에)
- training set losses & validation set losses = 각 데이터셋 별 수행 능력의 측정
