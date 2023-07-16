---
published: true
layout: posts
title: '[DL4C] Ch4. Under the Hood, Training a Digit Classifier(1)'
categories: 
  - dl4coders
---

## Pixels: The Foundations of Computer Vision

- [딥러닝에 시간을 더하다: 위르겐 슈미후버](https://brunch.co.kr/@hvnpoet/55)
- [인공지능은 어떻게 학습할까? BackPropagation](https://brunch.co.kr/@hvnpoet/48)

---

# Questionnaire

1. How is a grayscale image represented on a computer? How about a color image?
- 흑백
  - 2차원 배열 : 0 ~ 255까지의 숫자로 표현됨
    - 0은 흰색, 255는 검정색
    - shades of greyscale in between
- 칼라
  - 3 color channels (red, blue, green)
    - 각각은 2차원 배열(흑백과 원리는 동일)

2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
- valid / train : folder
  - subfolder for "3" and "7"
- label.csv : file

3. Explain how the "pixel similarity" approach to classifying digits works.
- 모든 데이터들의 픽셀-wise 평균을 구한다 <-> 그것이 각 사진과 얼마나 비슷한지 구한다 (얼마나 비슷한지 - MAE or MSE로 차이 계산)

4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
- 리스트 각 원소에 대해서 조건문 필터링이나 함수 적용을 간락하게 하도록 하는 Python 문법
  - [i * 2 for i in range(100) if i % 2 == 1]

5. What is a "rank-3 tensor"?
  - 3차원 텐서 - 행렬 테이블 형태 / ex.(1010, 28, 28)

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
  - tensor rank : 텐서의 축 / 차원의 수
  - shape : 텐서 각 축의 크기

7. What are RMSE and L1 norm?
  - RMSE = np.sqrt((y_preds - y_target) ** 2).mean() -> L2 norm
  - L1 norm : MAE 절대값 차 평균을 적용하는 규제 방법

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
  - numPy 또는 PyTorch는 C 언어 기반으로 계산해서 더 빠름. PyTorch는 CUDA 언어로 GPU까지 사용 가능해서 더욱 좋음.

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
  - a = torch.Tensor(list(range(1, 10))).view(3, 3)
  - a[1:, 1:]

10. What is broadcasting?
  - 작은 rank의 tensor/array를 자동적으로 큰 rank로 변환해 주는 것

11. Are metrics generally calculated using the training set, or the validation set? Why?
  - validation set / 성능 평가 지표는 모델이 올바르게 학습됐는지 평가하기 위함 -> 새로운 데이터로 평가하는게 더 적절(과적합 방지)

12. What is SGD?
  - 최적화 알고리즘. 주어진 예측값과 실제 값의 차이에 대한 손실 함수 값을 최소화 하도록 모델의 파라미터를 업데이트 하는 알고리즘

#TODO 논의해보기!!!
13. Why does SGD use mini-batches?
  - 전체 데이터셋에서 위 과정을 하는 것은 너무 시간이 오래걸림
    - 모든 데이터 포인트에서 SGD를 반복하면 이는 불안정하고 정확도가 떨어짐
  - GPU에게 한번에 처리할 작업 단위를 지정해주기에도 용이

14. What are the seven steps in SGD for machine learning?

1. 파라미터 초기화
2. 예측값 계산하기
3. 손실 함수값 계산하기
4. 파라미터값 갱신하기
6. 위 2번 과정 부터를 반복하기
7. 학습 종료하기 

#TODO 논의해보기!!!
15. How do we initialize the weights in a model?
  - 무작위 초기화 사용
  - 방법은 다양. 자비에르 초기화 등 초기화 전략 추후에 나올듯?

16. What is "loss"?
  - 실제 값과 예측 값의 차이
    - 값이 작을 수록 차이가 적은 것

17. Why can't we always use a high learning rate?
  - 손실 값이 너무 크게 변동할 수 있으므로
    - 손실 값이 발산해버릴 가능성
    - 너무 학습이 빨리 끝나 버림

18. What is a "gradient"?
  - 모델의 파라미터 값을 바꿨을 때 손실 함수가 얼마나 바뀌는지 알려주는 값
    - 각 파라미터를 얼마나 바꾸면 더 모델이 나아지는지 알려주는 지표

19. Do you need to know how to calculate gradients yourself?
  - 당연하지. 근데 아직 완벽히 이해 못해서 더 공부해야 함...ㅠㅠㅠㅠ

20. Why can't we use accuracy as a loss function?
  - 손실 함수는 가중치가 조금이라도 바뀌면 값이 바뀌도록 설계되어야 함. (그래야 어떤 방향으로 학습해야 할지 알 수 있음)
  - accuray는 모델이 예측하는 것이 바뀔 때만 변하므로 부적합 함.

21. Draw the sigmoid function. What is special about its shape?
  - 어떤 값에서도 0부터 1까지 값을 매핑시켜 주는 함수
    - 확률 분표를 나타내기에 적합
    - 모든 부분에서 미분가능하여 (급작스런 변화가 없는 완만한 곡선) gradient를 구하기 쉬움

22. What is the difference between a loss function and a metric?
  - 손실 함수 : 학습이 올바르게 진행되고 있는지 알려 주는 지표 - 기계 학습에서 중요
  - 평가 지표 : 전체 학습이 잘 진행되었는지 학습이 끝난 후 알려 주는 지표 - 사람의 확인에서 중요

23. What is the function to calculate new weights using a learning rate?
  - optimization step function / 최적화

24. What does the `DataLoader` class do?
  - Python list 원소들을 특정 사이즈의 여러 batch 들로 iterate 하기 편하게 잘라 줌

25. Write pseudocode showing the basic steps taken in each epoch for SGD.


```python
for x, y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    paramtetrs -= parameter.grad * lr
```

---

26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
  - def func(a, b): return list(zip(a,b))


27. What does `view` do in PyTorch?
  - 텐서의 내용은 지하면서 shape 변환을 시켜주는 메서드

28. What are the "bias" parameters in a neural network? Why do we need them?
  - 선형 함수의 y 절편 값
  - 0에서의 함수 값이 0이 되지 않도록 조정해 줌
    - 함수에 더 유연성 부여

29. What does the `@` operator do in Python?
  - 행렬 곱셈 연산자

30. What does the `backward` method do?
  - backpropagation 역전파를 뜻함
  - 현재의 gradients를 반환해줌

31. Why do we have to zero the gradients?
  - PyTorch는 이미 저장된 gradients에 새로 구한 gradients를 더해 버림
  - 새로 구한 gradient값을 올바르게 유지하기 위해 기존의 값을 0으로 초기화 시켜 주는 것

32. What information do we have to pass to `Learner`?
  - DataLoader, model, optimization funcion, loss function, (선택적으로 metrics)

33. Show Python or pseudocode for the basic steps of a training loop.

```python
def train_epoch(model, lr, params):
    for xb, yb in dl:
      calc_grad(xb, yb, model)
      for p in params:
          p.data -= p.grad*lr
          p.grad.zero_()
    
for i in range(20):
    train_epoch(model, lr, parmas)
```

34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
  - rectified linear unit
  - 모든 음수값 0으로 양수값 해당 값으로


35. What is an "activation function"?
  - 선형 변환 layer 사이에 비선형 변환을 가능하게 해주는 함수
    - 선형 변환의 결합은 결국 선형 변환
    - 더 유연하게 모든 함수에 근사되기 위해서는 선형 변환 사이에 비선형 변환이 들어가야 함 -> 이걸 해주는 함수

36. What's the difference between `F.relu` and `nn.ReLU`?
  - F.relu : Python 함수
  - nn.ReLU : PyTorch module
    - PyTorch에서 layer를 만들 때 relu 클래스 처럼 활용 가능

37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
  - 더 성능이 좋아서. 유용하니까!
    - 더 적은 파라미터로 더 깊은 layer를 쌓는 것이 효율적!
      - 성능이 더 좋음
      - 학습도 더 빠름
      - 계산량과 메모리 소모도 적음
