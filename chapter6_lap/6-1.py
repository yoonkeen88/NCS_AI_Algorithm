import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)= mnist.load_data() # 트레인 테스트를 어떻게 나누는가?
x_train=x_train.reshape(60000,28,28,1) # MNIST 데이터는 28x28 크기의 흑백 이미지로 구성되어 있으므로, 채널 수를 1로 설정
x_test=x_test.reshape(10000,28,28,1) # 테스트 데이터도 동일하게 변환
x_train=x_train.astype(np.float32)/255.0 # 픽셀 값을 0~1 사이로 정규화
x_test=x_test.astype(np.float32)/255.0 # 픽셀 값을 0~1 사이로 정규화
y_train=tf.keras.utils.to_categorical(y_train,10) # 레이블을 원-핫 인코딩으로 변환
y_test=tf.keras.utils.to_categorical(y_test,10) # 레이블을 원-핫 인코딩으로 변환

# LeNet-5 신경망 모델 설계
cnn=Sequential()
cnn.add(Conv2D(6,(5,5),padding='same',activation='relu',input_shape=(28,28,1))) #Convolutional layer
cnn.add(MaxPooling2D(pool_size=(2,2))) #Pooling layer
cnn.add(Conv2D(16,(5,5),padding='same',activation='relu')) #Convolutional layer
cnn.add(MaxPooling2D(pool_size=(2,2))) #Pooling layer
cnn.add(Conv2D(120,(5,5),padding='same',activation='relu')) #Convolutional layer
cnn.add(Flatten()) # Flatten layer은 2차원 데이터를 1차원으로 변환
cnn.add(Dense(84,activation='relu'))
cnn.add(Dense(10,activation='softmax'))
 # FC 는 Fully Connected layer로, 입력과 출력이 모두 연결된 층 분류를 위한 층
# 신경망 모델 학습
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()