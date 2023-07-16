---
published: true
layout: posts
title: '[DL4C] Ch5. Image Classification'
categories: 
  - dl4coders
---

---

Answer reference:

- [Fastbook Chapter 5 questionnaire solutions (wiki)](https://forums.fast.ai/t/fastbook-chapter-5-questionnaire-solutions-wiki/69301)
    

## CheckPoint

Deep Dive into the DL

- What is the architecture of a computer vision model, an NLP model, a tabular model, and so on?
- How do you create an architecture that matches the needs of your particular domain?
- How do you get the best possible results from the training process?
- How do you make things faster?
- What do you have to change as your datasets change?

Deep Learning Puzzle

- different types of layers
- regularization methods
- optimizers
- how to put layers together into architectures
- labeling techniques
    - and much more…

---

## Questionnaire

1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU? (x)
    - Presize Strategy in fastai
    - 보통은 GPU에서 수행되나, 계산→보간의 반복으로 데이터 손실 및 품질 저하
        - augmentation → 이미지의 최대 크기에서 수행됨 (CPU)
        - RandomSizeCrop → 최종 이미지 사이즈에서 축소되어 수행됨 (GPU)

1. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
2. What are the two ways in which data is most commonly provided, for most deep learning datasets?
    - 데이터 목록으로서의 파일 묶음
        - 문서, 이미지 등
        - 폴더 / 파일 이름으로 목록에 대한 정보 제공
    - 테이블 데이터
        - 각 행은 하나의 데이터
            - 데이터 ↔ 테이블/다른 형태 데이터 사이 관계 제공
3. Look up the documentation for `L` and try using a few of the new methods that it adds.
4. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.

1. Give two examples of ways that image transformations can degrade the quality of the data.
    - rotation - empty zone - interpolation → bad quality
    - zoom in - interpolation → bad quality

1. What method does fastai provide to view the data in a `DataLoaders`?
    - DataLoaders.show_batch(nrows=<>, ncols=<>)
2. What method does fastai provide to help you debug a `DataBlock`?
    - DataBlock.sumarry(<path>)

1. Should you hold off on training a model until you have thoroughly cleaned your data?
    
    <aside>
    💡 Nope! We should create a baseline model as soon as possible.
    
    </aside>
    
2. What are the two pieces that are combined into cross-entropy loss in PyTorch?
    - activation_fuction : softmax
    - negative_log_likelihood
        - (log_softmax + nll_loss) = cross_entrophy
3. What are the two properties of activations that softmax ensures? Why is this important?
    - 어떤 값도 0, 1 사이에
    - 모든 값  더해서 1
        - 확률분포 / act의 차이를 지수를 통해 amplify해 줌
4. When might you want your activations to not have these two properties? (x)
    - multi-label classifiction problem
5. Calculate the `exp` and `softmax` columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
    
    ```python
    acts
    
    def softmax(x): return np.exp(x) / exp(x).sum(dim=1, keepdim=True) 
    
    def neg_log_likelihood(x): return -np.log(tensor(x))
    
    def cross_entrophy_loss(x): return neg_log_likelihood(softmax(x)).sum()
        
    ```
    

1. Why can't we use `torch.where` to create a loss function for datasets where our label can have more than two categories?
    - it uses as list comprehension with if-else statements.
    - binary case only
2. What is the value of log(-2)? Why?
    - not defined. 로그는 음수를 진수로 취할 수 없다.
    - y는 지수 계산을 통해 나온 값. 음수가 불가능.
    
    $$
    e^x = y\\
    x = log{y}
    
    $$
    

---

## Improve Our Model

### 1. Learning Rate Finder

- 아주 작은 학습률에서 시작
- 1개의 mini-batch 돌려서 loss 계산
- 학습률 일정 비율로 증가 (ex.매번 *2)
- 위 과정을 계속 반복
    - loss가 나아지지 않고 더 나빠질 때 까지!
1. What are two good rules of thumb for picking a learning rate from the learning rate finder?
    - minimum loss 달성된 부분에서 한 단계 줄이기
        - min / 10
    - loss가 확실히 줄어들기 시작하는 마지막 부분

### 2. Unfreezing and Transfer Learning

#### Transfer Learning

- 기존의 final linear layer를 제거
- 현재 상황에 맞는 output으로 맞춘 layer 추가
    
    → 전체 가중치를 조정하는 것이 아님!
    

> 💡 마지막 layer의 random weights를 ‘사전 학습 모델의 가중치’를 손상시키지 않으면서 어떻게 학습 시킬 것인가?!

- optimzer 를 조정해 마지막 layer만 가중치 조정하도록 설정 가능 (나머지 부분은 고정)
    - `freezing` pretrained layers

1. What two steps does the `fine_tune` method do? (freeze, unfreeze)
    - 모든 다른 layers들을 freezing한 채로 마지막 layer만 1 epoch 학습을 수행
    - 모든 layer를 unfreeze하고 (n) epoch만큼 학습시킴
2. In Jupyter Notebook, how do you get the source code for a method or function? → method??

### 3. Discriminative Learning Rate

- pretrained weights have been trained for hundreds of epochs, on millions of images
    - let the layer layers fine-tune more quickly than earlier layers
- nn의 초기 layer는 낮은 학습률로, 나중의 layer는 높은 학습률로

>💡 loss의 변화도 중요하지만 (learn.recorder.plot_loss)
> 가장 중요한건 metrics! not loss…

1. What are discriminative learning rates?
    - lower layer, higher layer different learning rate
2. How is a Python `slice` object interpreted when passed as a learning rate to fastai?
    - first : the earliest layers
    - second : the final layer
        - in the middle → multiplicatively equidistant throughout the range

### 4. Selecting the Number of Epochs

- 처음에는 기다릴 수 있는 적당한 시간의 epoch를 골라서 수행한다
    - In practice, 항상 중요한건 metrics!!!
        - loss가 나빠지는거에 너무 신경쓰지 말기
    - 초반에는 과적합으로 loss 나빠질 수 있음
        - 후반에 데이터를 잘못 기억해서 발생하는 loss 저하만 신경쓰면 됨

1. Why is early stopping a poor choice when using 1cycle training? → (x)
- learning rate이 더 낮게 되기 까지 충분한 시간이 되기 전에 종료되버릴 위험
    - 모델을 향상시킬 수 있는 데 종료되어 버림

#### 해결
##### Retrain
- 다시 바닥부터 모델을 재훈련한다
    - 직전의 최고 성능을 보여주었을 때를 기준으로 epoch를 선택한다

### 5. Deeper Architectures

#### 장단점 비교
- 더 큰(깊은) 모델 → 더 나은 학습 손실 제공
    - 과적합의 위험성도 커짐 (파라미터가 많아서)
    - more layers and parameters = larger model = the capacity of model
- 더 큰(깊은) 모델 → 데이터의 숨겨진 관계도 잘 찾음
    - 개별 이미지의 특수한 부분도 잘 기억함

깊은 학습 → GPU 메모리 많이 잡아먹음 → 뻗을수도…

#### 해결
##### mixed-precision training

- less-precise number (half-precision floating point : fp16) 사용

[fastai - Mixed precision training](https://docs.fast.ai/callback.fp16.html)

1. What is the difference between `resnet50` and `resnet101`?
    - how deep nn is (num of layers)
2. What does `to_fp16` do?
    - float point drop
    - 텐서의 정보량 줄여서 연산 속도 빠르게


> 💡 큰 모델이 항상 더 좋은건 아니다! (특정 부분에서는 더욱)
> 작은 모델에서 시작해서 Scaling-up 해가기


---

## Further Research - Why?

That’s because the gradient of cross-entropy loss is proportional to the
difference between the activation and the target, so SGD always gets a
nicely scaled step for the weights.

An interesting feature about cross-entropy loss appears when we consider its gradient. The gradient of cross_entropy(a,b) is softmax(a)-b. Since softmax(a) is the final activation of the model, that means that the gradient is proportional to the difference between the prediction and the target. This is the same as mean squared error in regression (assuming there’s no final activation function such as that added by y_range), since the gradient of (a-b)**2 is 2*(a-b). Because the gradient is linear, we won’t see sudden jumps or exponential increases in gradients, which should lead to smoother training of models.