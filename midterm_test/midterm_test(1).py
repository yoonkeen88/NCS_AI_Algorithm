"""
1번 문제 
MNIST 숫자 이미지 분류 모델을 만들어 보시오
 아래 조건을 만족하는 딥러닝 모델을 TensorFlow/Keras로 구현하시오
● 조건
- 데이터 셋 : MNIST, 신경망 구조는 은닉층 1개
- 데이터셋 : tensorflow.keras.datasets.mnist 사용
- 신경망 구조 : Flatten → Dense(128) + ReLU → Dense(10) + Softmax
- 손실 함수: sparse_categorical_crossentropy
- Optimizer : Adam
- epochs = 5번만 실행
- evaluate test 정확도 : 00.00 %
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 데이터 전처리
x_train = x_train / 255.0  # 정규화
x_test = x_test / 255.0    # 정규화

# 하이퍼 파라미터 설정
epo = 5
batch_si = 32
n_hidden1 = 128
n_output = 10 

# 모델 정의
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # reshape 메소드를 사용하지 않고 입력층 28*28을 입력으로 받아 평탄화
model.add(Dense(n_hidden1, activation='relu'))  # 은닉층
model.add(Dense(n_output, activation='softmax'))  # 출력층

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 모델 학습
model.fit(x_train, y_train, 
          epochs=epo, 
          batch_size= batch_si,
          validation_data = (x_test, y_test),
          verbose=1) # 보기편하게 verbose=1로 설정

# model 평가
res = model.evaluate(x_test, y_test, verbose=0)
print(f"evaluate Test 정확도: {res[1] * 100:.4f} %")

print(f"==========문제 3의 모델 구조===========")
model.summary()
