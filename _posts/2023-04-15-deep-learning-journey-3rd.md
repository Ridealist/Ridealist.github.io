---
published: true
layout: posts
title: Ch1. Your Deep Learning Journey(3)
categories: 
  - dl4coders
---

## Deep Learning Is Not Just for Image Classification

### Segmentation
- 이미지 안에 있는 모든 개별적인 픽셀의 내용을 인식할 수 있는 모델을 만드는 분야
    - [segment-anything(facebook github repo)](https://github.com/facebookresearch/segment-anything)
    - [Semantic Object Classes in Video: A High-Definition Ground Truth Database](http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf)

### NLP
- 글을 생성하고, 언어를 자동 번역하고, 댓글을 분석하는 등 다양한 task를 수행할 수 있음

### Tabular
- 표 형태로된 데이터, 스프레드시트, 데이터베이스, CSV(comma-separated-values) 파일
- `tabular model`은 표의 한 column(피처)를 예측하기 위해 다른 column들의 정보를 활용하는 형태
    - [Scaling up the accuracy of Naive-Bayes classifiers: a decision-tree hybrid](https://dl.acm.org/doi/10.5555/3001460.3001502)

### Recommendation
- 영화 평점 : **연속된 값**을 예측해야 함
    - `y_range` 파라미터를 사용해 target이 어떤 범위 안에 있는지 fastai library에게 명시해 주어야 함
- tabular 모델을 사용하므로, pretrained model을 활용할 수 없음
    - BUT, fastai에서 fine_tune을 활용할 수 있는 방법을 보여줌
    - **fine_tune** vs. **fit_one_cycle** 을 비교하여 어떻 것이 더 특정 데이터셋에 최고의 성능을 보이는지 확인(experiment)해보는 것이 중요!

### Datasets
- 데이터셋을 본다면 그것이 어디서 왔는지, 어떻게 큐레이팅 됐을지 생각해 보기
- <u>나만의 프로젝트를 위한 데이터셋을 어떻게 만들 수 있을지 생각해보기</u>
- fastai는 빠른 prototyping과(더 학습하기 쉬운) 실험을 돕기 위해 유명한 데이터셋의 가벼운 version들을 제공함
    - cut-down ver.을 먼저 사용하고 그 다음 scale-up해서 full-size ver.을 사용하는 경우도 있음
    - 실제 현업에서 많이 활용되는 방법!

> 데이터의 일부분(subsets of data)에서 prototyping과 expreriment를 대부분 수행한다!
> 무엇을 해야할지 충분히 이해되었다면 그 때 비로소 full dataset을 이용한다!


## Validation Sets & Test Sets

- 데이터셋을 training set과 validation set으로 나눠야 한다는 건 이제 당연...
    - 학습 데이터로 부터 얻은 교훈(lesson)을 새로운 데이터인 검증 데이터로 일반화 할 수 있는지 확인하는 과정

<u>그런데 말입니다... validation data만 준비하면 끝일까? NO!</u>

- 실제 상황에서 한 번의 파라미터 학습으로 모델을 만들지 않고, **여러 모델링 선택지들을 탐색한다**
    - network architecture
    - learning rate
    - data augmentation strategy
    - etc...
- 이런 선택지들을 `hyperparameter`라고 한다
    - 파라미터에 대한 파라미터라는 의미
    - 가중치 파라미터들의 의미를 결정하는 고수준의 선택지(high-level choices)

- 모델을 만드는 사람들(modeler)은 모델을 계속 평가할 수 밖에 없다
- validation data에 대해 예측값을 보면서 우리가 최적의 hyperparameter 값을 설정하도록 탐색하는 과정에서
    - 따라서 모델들은 간접적으로 validation data에 영향을 받은채로 계속 만들어지게 된다

> 자동화된 학습 과정이 training data에 대한 과적합 위험에 노출되듯이, 다양한 시도로 모델을 수정하는 과정에서 validation data에 대한 과적합 위험에 노출된다

- 이러한 이유로 `test set`이 필요하다
    - 학습 과정에서 validation data를 빼놓은 것처럼, 우리 스스로로 부터 test data를 빼 놓아야 한다
    - 모델을 향상시키기 위해서 사용되어서는 안된다!
    - 오직 가장 마지막 단계에서 모델을 평가하기 위해서 사용되어야 한다!

- 결국 데이터는 3개의 계층으로 나눠진다, 학습과 모델링 과정에서 얼마나 노출시킬지 여부에 따라서
    - training data는 완전히 노출
    - validation data는 일부 노출
    - test data는 완전히 노출 차단
- 계층은 모델링과 평가 과정에서의 차이들과 평행 이론을 이룬다
    1. 역전파(backpropagation)를 통한 자동화된 학습 과정
    2. training session 마다 최적의 하이퍼파라미터를 찾는 더 수작업이 필요한 과정
    3. 마지막 결과에 대해서 모델 평가를 하는 과정

- test, validation sets은 데이터 양의 기준
    - **고양이 분류기**에서 적어도 30 종류의 고양이가 validation set에 있어야 한다
    - 천개 내외(thousands of item)의 데이터가 있는 경우
        - 기본값 20%의 validation set 비율은 실제 필요보다 많을 수도 있다
    - 데이터가 많다면 그것의 일부를 validation에 사용하는 것은 어떠한 나쁜점도 없다 (비율은 유동적이다?)

- 두 단계의 예약 데이터(reserved data)를 갖는 것이 힘들어 보일 수 있다
    - 모델은 **기억**함으로써 좋은 예측을 수행하는 쉬운 길을 택할 수 있다
    - 틀리기 쉬운 인간은 모델이 성능이 아주 좋다고 착각하는 오류에 빠질 수 있다

> test set의 원칙은 우리가 지적 겸손을 유지하도록 도와준다

- '항상' 별도의 test data가 필요하다는 것은 아니다
    - 너무 데이터 양이 적다면, validation set만 필요할 수도 있다
    - 하지만 가능하다면 test data를 준비하는 것이 좋다

- 특히 third party를 통해 대신 모델링을 수행할 경우 이런 원칙이 더욱 중요해진다!
    - 내 요구사항을 정확히 파악하지 못하거나, 성능을 과장할 수 있다.
    - 좋은 test set은 그들의 작업이 실제 문제를 해결했는지 평가함으로써 이런 위험을 매우 완화시켜 준다
- 외부 업체를 통해 AI를 도입한다면
    - 업체가 절대 접근할 수 없는 test data를 준비한다
    - 스스로 test data를 모델에 적용해 본다
    - 실제적으로 가장 중요한 부분을 평가할 수 있는 metric을 스스로 설정한다
    - 어떤 수준의 성능이 적절한지 스스로 결정한다
- 스스로 간단한 baseline 모델을 만들어 보는 것도 좋은 생각이다!

> 때때로 내가 만든 simple model이 외부의 소위 "전문가"가 만든 것보다 나쁘지 않은 경우도 있을 수 있다!


### Use Judgment in Defining Test Sets

- validation set(가능하면 test set까지)을 잘 정의하기 위해서, 단순히 무작위로 데이터의 일부를 선택하는 것 이상이 필요하다

> 기억하라! - validation, test set들의 핵심적인 속성은 미래에 마주치게 될 new data를 대표(representative)할 수 있어야 한다

- 정의해 입각하면, 새로운 데이터를 아직 못봤는데 어떻게 대표하는지 알 것인가?
    - 하지만 몇가지 사실을 알 수 있다!
    - [Kaggle](https://www.kaggle.com/)을 많이 참조하셔라

- **time series data**를 사용한다면?
    - [Kaggle 예시_식료품 판매 예측](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
    - 무작위로 데이터의 일부를 추출하는 것은 매우 나쁜 선택이다
        - 1. 너무 쉽다: 예측하고 싶은 날짜 전과 후 데이터를 모두 살펴보게 된다(답을 알려주고 학습하는 것)
        - 2. 대표성이 없다: 과거 데이터를 토대로 미래에 사용할 수 있는 모델을 만들어야 하는데 그렇지 못하다
    - validation set을 가장 마지막 날짜 일부를 연속된 데이터로 추출하는 것이 좋다
        - ex. 활용 가능한 데이터 중 마지막 월 / 마지막 2주 

![time series_example_1](https://user-images.githubusercontent.com/89024993/232263729-4e06e65a-53f9-4a90-8bb8-9c637dada427.png)
![time series_example_2](https://user-images.githubusercontent.com/89024993/232263726-3fa4b405-84f4-4b63-bb98-38df7d74231e.png)
![time series_example_3](https://user-images.githubusercontent.com/89024993/232263724-08a31d27-e518-4389-9dd3-cec6b6cf63cf.png)

- **image data**를 사용한다면?
    - 모델이 학습했던 데이터와 질적으로 다른(qualitatively different) 데이터를 실제 production 단계에서 예측을 위해 사용한다면?
    - [Kaggle 예시_분산된 행동하는 운전자 탐지](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
        - 많은 사진들이 '같은 운전자'의 다른 위치를 보여줌
            - 중요한 것은 **<u>본적 없는 운전자에게도</u>** 모델이 작 적용되는지!
            - training data는 오직 작은 범위의 사람들만 대상으로 구성되어 있기 때문
        - test data는 학습 데이터에서 등장하지 않는 사람들로 구성된 이미지들로 이루어져야 한다!
        - 모든 사람들이 등장하는 training dataset을 사용한다면?
            - 모델은 사람들에게 집중되어 잘못된 부분으로 과적합될 수 있다
            - 학습의 초점이 texting, eating, etc... '행동'으로 연결 되지 않고...

- 또 다른 예시!
    - [Kaggle 예시_물고기 종류 모니터링](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)
    - 멸종 위기종에 대한 불법 포획을 줄이기 위해 배에서 잡히는 물고기의 종류를 탐지하는 모델 만들기
        - test set은 training set에서 등장하지 않는 배들의 사진으로 구성되어 있다
        - **<u>마찬가지로 validation set도 training set에서 나오지 않은 것들로 구성해야 한다!</u>**

- 때때로 validation set이 어떻게 달라야 될지 확실히지 않을 수 있다
    - satellite imagery 문제
        - 특정 지리적 장소의 정보를 공유하는 training set?
        - 흩어져 있는 여러 지리적 장소의 정보를 담은 training set?