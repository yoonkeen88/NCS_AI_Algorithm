import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# 코인데스크 사이트에서 1년치 비트코인 가격 데이터 읽기
f=open("chapter8_lap/BTC_USD_2019-02-28_2020-02-27-CoinDesk.csv","r")
coindesk_data=pd.read_csv(f,header=0)
seq=coindesk_data[['Closing Price (USD)']].to_numpy() # 종가만 취함

def set_device(use_gpu=True):
    if use_gpu:
        # GPU가 있는지 확인하고 설정
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # GPU 메모리 늘리지 않고 필요한 만큼만 사용하게 설정 (optional)
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Using GPU")
            except RuntimeError as e:
                print("⚠️ GPU 설정 실패:", e)
        else:
            print("❌ GPU 없음, CPU 사용")
    else:
        # GPU 사용 안 하고 CPU만 사용
        tf.config.set_visible_devices([], 'GPU')
        print("✅ Forced to use CPU")

# 예시: 여기서 선택
set_device(use_gpu=False)  # GPU 사용
# set_device(use_gpu=False)  # CPU 사용

# 시계열 데이터를 윈도우 단위로 자르는 함수
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=9 # 윈도우 크기
h=1 # 수평선 계수

X,Y=seq2dataset(seq,w,h)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 훈련 집합과 테스트 집합으로 분할
split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

# LSTM 모델 설계와 학습
model=Sequential()
model.add(LSTM(units=128,
               activation='relu',
               input_shape=x_train[0].shape)) # LSTM 적층 unit 은 은닉 W 의 크기
model.add(Dense(1))
model.compile(loss='mae',
              optimizer='adam',
              metrics=['mae']) # 평균 절댓값 오차(MAE) 사용
hist=model.fit(x_train,y_train,epochs=200,batch_size=1,validation_data=(x_test,y_test),verbose=2)

# LSTM 모델 평가
ev=model.evaluate(x_test,y_test,verbose=0)
print("손실 함수:",ev[0],"MAE:",ev[1])

# LSTM 모델로 예측 수행
pred=model.predict(x_test)
print("평균절댓값백분율오차(MAPE):",sum(abs(y_test-pred)/y_test)/len(x_test))

# 학습 곡선
plt.figure(figsize=(12, 6))
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.ylim([120,800])
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# 예측 결과 시각화
x_range=range(len(y_test))
plt.figure(figsize=(12, 6))
plt.plot(x_range,y_test[x_range], color='red')
plt.plot(x_range,pred[x_range], color='blue')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()

# 일부 구간을 확대하여 시각화
x_range=range(50,64)
plt.figure(figsize=(12, 6))
plt.plot(x_range,y_test[x_range], color='red')
plt.plot(x_range,pred[x_range], color='blue')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()