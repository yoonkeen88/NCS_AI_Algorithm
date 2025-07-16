"""
2. Fashion MNIST 의류 이미지 분류 모델을 만들어 보시오
 아래 조건을 만족하는 딥러닝 모델을 TensorFlow/Keras로 구현하시오. 
 ● 조건
- 데이터 셋 : Fashion MNIST, 신경망 구조는 은닉층 2개
- 데이터 셋 : tensorflow.keras.datasets.fashion_mnist
- 신경망 구조 : Flatten → Dense(128) + ReLU → Dense(64) + ReLU → Dense(10) + Softmax
- 손실 함수 : categorical_crossentropy 이를 위해 라벨을 One-hot encoding할 것
- Optimizer : Adam
- epochs = 5번만 실행
- evaluate test 정확도 : 00.00 %"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


# Fashion MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 데이터 전처리
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # cross entropy 사용을 위한 원핫 코딩
y_test=tf.keras.utils.to_categorical(y_test,10) 


# hyperparameter
n_hidden1 = 128
n_hidden2 = 64
n_output = 10
n_epochs = 5 
n_batch = 32 # 배치 사이즈는 임의?
# 모델링
model = Sequential()
model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(n_hidden1, activation = 'relu'))
model.add(Dense(n_hidden2, activation = 'relu'))
model.add(Dense(n_output, activation = 'softmax'))

# 모델 평가
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # 크로스 엔트로피 설정
# 모델 학습
model.fit(x_train, y_train,
                epochs = n_epochs,
                batch_size = n_batch,
                validation_data=(x_test,y_test),
                verbose = 2 )

res = model.evaluate(x_test, y_test, verbose=0)
print(f"evaluate Test 정확도: {res[1]* 100:.4f} %")
print(f"==========문제 2의 모델 구조===========")
model.summary()