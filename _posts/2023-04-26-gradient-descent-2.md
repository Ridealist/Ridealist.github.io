---
published: true
layout: posts
title: 배치 경사하강법(BGD), 확률적 경사하강법(SGD), 미니배치 경사하강법(MBGD)
categories: 
  - deeplearning
---



## 0. 용어 정리

- **배치(batch)**란 머신러닝의 모델에서 모델의 가중치들을 한 번 업데이트시킬 때 사용하는 데이터들의 집합

- **배치 사이즈(batch size)**는 말 그대로 배치의 크기, 즉 모델의 가중치들을 한 번 업데이트 시킬 때 사용하는 데이터의 개수

- **에포크(epoch)**란 모델이 전체 데이터를 모두 사용하여 학습을 한 횟수

- **반복(iteration)**이란 1 epoch에 필요한 batch의 갯수



(예시)

- 2500의 dataset을 크기가 100인 dataset 25개로 나누어 학습을 진행한다
	- batch size가 100
	- batch 25개가 생성
	- 1 epoch당 25번의 iteration이 생김



- **배치는 GPU가 한번에 처리하는 데이터의 묶음**



## 1. 배치 경사 하강법 (Batch Gradient Descent: BGD)

### 개념

- 경사 하강법 = 배치 경사 하강법

- **<u>전체 학습 데이터를 하나의 배치로(배치 크기가 n)</u>** 묶어 학습시킴
- 보통 딥러닝 라이브러리에서 배치를 지정하지 않으면 이 방법을 쓰고 있음



### 계산 방법

- 전체 데이터에 대한 모델의 오차의 평균을 구함
- 이를 이용하여 미분을 통해 경사를 산출, 최적화를 진행



### 특징

- 전체 데이터를 통해 학습시키기 때문에, 가장 업데이트 횟수가 적다. (1 Epoch 당 1회 업데이트)
- 전체 데이터를 모두 한 번에 처리하기 때문에, 메모리가 가장 많이 필요하다.
- 항상 같은 데이터 (전체 데이터)에 대해 경사를 구하기 때문에, 수렴이 안정적이다. (아래 그림 참고)
- 단점은 이전 포스팅에서 언급한 4가지
	- [경사하강법 파헤치기](https://ridealist.github.io/deeplearning/gradient-descent-1/)


![img](https://blog.kakaocdn.net/dn/xzcf3/btqEuYhYtuF/jfMqBQlOKTq94H15yADKrK/img.png)



## 2. 확률적 경사 하강법 (Stochastic Gradient Descent: SGD)

### 개념

- 전체 데이터 중 **단 하나의 데이터를 이용하여 경사 하강법을 1회 진행(배치 크기가 1)**
- 전체 학습 데이터 중 랜덤하게 선택된 하나의 데이터로 학습을 하기 때문에 확률적 이라 부름



### 특징

- **수렴에 Shooting이 발생**한다
	- 각 데이터에 대한 손실값의 기울기는 약간씩 다름
		- 손실값의 평균이 아닌 개별 데이터에 대해 미분을 수행하면 기울기의 방향이 매번 크게 바뀐다.
- 배치 경사 하강법에 비해 적은 데이터로 빠르게 학습할 수 있다

![img](https://blog.kakaocdn.net/dn/7hWCJ/btqEuYhZtI9/YXwknUwaKcMhJOdhTUyRR1/img.png)



- 그러나 결국 학습 데이터 전체에 대해 보편적으로 좋은 값을 내는 방향으로 수렴한다.
	- **다만, 최저점(Global Minimum)에 안착하기는 어렵다.**

- 또한, Shooting은 최적화가 지역 최저점Local Minima에 빠질 확률을 줄여준다.



#### 효율성 측면

- 한 번에 하나의 데이터를 이용하므로 GPU의 병렬 처리를 잘 활용하지는 못한다.

- 1회 학습할 때 계산량이 줄어든다.

	

## 3. 미니 배치 확률적 경사 하강법 (Mini-Batch Stochastic Gradient Descent: MSGD)

### 개념

- SGD와 BGD의 절충안
- **전체 데이터를 batch_size개씩 나눠 배치로 학습**시키는 것
	- 배치 크기는 사용자가 지정
	- 딥러닝 라이브러리 등에서 SGD를 얘기하면 최근에는 대부분 이 방법을 의미



(예시)

- 전체 데이터가 1000개인 데이터를 학습
	- batch_size가 100이라면
		- 전체를 100개씩 총 10 묶음의 배치로 나누어
		- 1 Epoch당 10번 경사하강법을 진행



![img](https://blog.kakaocdn.net/dn/bkVbjU/btqEtOUJD9H/L9KHdOnSukjHnhlRwRERy1/img.png)

### 특징

- BGD보다 계산량이 적다. (Batch Size에 따라 계산량 조절 가능)
- Shooting이 적당히 발생한다. (Local Minima를 어느정도 회피할 수 있다.)
	- Shooting이 발생하기는 하지만, 한 배치의 손실값의 평균으로 경사하강을 진행하기 때문에, Shooting이 심하지는 않다.



### Batch Size 정하기



- Batch Size는 보통 2의 n승으로 지정
	- GPU의 VRAM 용량에 따라 Out of memory가 발생하지 않도록 정해줘야

- 가능하면 학습데이터 갯수에 나누어 떨어지도록 지정하는 것이 좋다
	- 마지막 남은 배치가 다른 사이즈이면 해당 배치의 데이터가 학습에 더 큰 비중을 갖게 되기 때문



(예시)

- 530 개의 데이터를 100개의 배치로 나눈다
	- 각 배치 속 데이터는 1/100 만큼의 영향력을 갖게 됨
	- BUT 마지막 배치(30개)의 데이터는 1/30의 영향력을 갖게 됨
		- -> 마지막이 과평가되는 경향
- 그렇기 때문에 보통 마지막 배치의 사이즈가 다를 경우 이는 버리는 방법을 사용



---



reference

- [머신러닝/딥러닝 공부](https://yhyun225.tistory.com/7)

- [컴퓨터와 수학, 몽상 조금](https://skyil.tistory.com/68)
