"""
4. Fashion MNIST 데이터셋을 사용하여 CNN 기반 이미지 분류 모델을 설계하고 학습·평가하시오. CNN의 핵심 구성요소인 Conv2D, MaxPooling2D, Dropout을 포함하시오
● 조건
- 데이터셋: tensorflow.keras.datasets.fashion_mnist 
- 신경망 구조 : C → P → C → P → FC → FC 구조
 Conv2D(32, (3,3), 
 activation='relu'),
 MaxPooling2D((2,2)), 
 Conv2D(64, (3,3), 
 activation='relu'), 
 MaxPooling2D((2,2)), 
 Flatten,
 Dense(64, activation='relu'), 
 Dropout(0.3), 
 Dense(10, activation='softmax'),

 손실 함수 : sparse_categorical_crossentropy
 Optimizer : Adam
 Epoch = 5, Batch size = 64
 evaluate test 정확도 : 00.00 %
 데이터 전처리 시 CNN에 입력할 수 있도록 채널 차원 추가 (reshape) 할 것
 
 """

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Fashion MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# 데이터 전처리
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1) # CNN에 입력할 수 있도록 채널 차원 추가

x_train=x_train.astype(np.float32)/255.0 # 정규화
x_test=x_test.astype(np.float32)/255.0


# hyperparameter
n_epochs = 5
n_batch = 64

# 모델링
model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu')) # C
model.add(MaxPooling2D((2,2))) # P
model.add(Conv2D(64, (3,3), activation='relu')) # C
model.add(MaxPooling2D((2,2))) # P
model.add(Flatten())
model.add(Dense(64, activation='relu')) # FC
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) # FC

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
                epochs = n_epochs,
                batch_size = n_batch,
                validation_data=(x_test,y_test),
                verbose = 2 )

res = model.evaluate(x_test, y_test, verbose=0)
print(f"evaluate Test 정확도: {res[1]* 100:.4f} %")

print(f"==========문제 4의 모델 구조===========")

model.summary()
