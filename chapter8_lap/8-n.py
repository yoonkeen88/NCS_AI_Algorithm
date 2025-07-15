import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 디바이스 설정 함수
def set_device(use_gpu=True):
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Using GPU")
            except RuntimeError as e:
                print("⚠️ GPU 설정 실패:", e)
        else:
            print("❌ GPU 없음, CPU 사용")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("✅ Forced to use CPU")

# 설정: GPU 또는 CPU
set_device(use_gpu=False)  # 필요하면 False로 변경

# 데이터 불러오기
f = open("chapter8_lap/BTC_USD_2019-02-28_2020-02-27-CoinDesk.csv", "r")
coindesk_data = pd.read_csv(f, header=0)
seq = coindesk_data[['Closing Price (USD)', '24h Open (USD)', '24h High (USD)', '24h Low (USD)']].to_numpy()

# 정규화
scaler = StandardScaler()
seq = scaler.fit_transform(seq)

# 시계열 윈도우 함수
def seq2dataset(seq, window, horizon):
    X, Y = [], []
    for i in range(len(seq) - (window + horizon) + 1):
        x = seq[i:(i + window)]
        y = seq[i + window + horizon - 1]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

w = 7  # 윈도우 크기
h = 1  # 수평선 계수
X, Y = seq2dataset(seq, w, h)

# 훈련/테스트 분할
split = int(len(X) * 0.7)
x_train, y_train = X[:split], Y[:split]
x_test, y_test = X[split:], Y[split:]

# LSTM 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=256, input_shape=x_train[0].shape))  # tanh는 default
model.add(Dense(4))  # 4개 출력 (Closing, Open, High, Low)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=16,
                 validation_data=(x_test, y_test), verbose=2)

# 평가
ev = model.evaluate(x_test, y_test, verbose=0)
print("MSE:", ev[0], "MAE:", ev[1])

# 예측 및 MAPE 계산
pred = model.predict(x_test)
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
print("LSTM 평균절댓값백분율오차(MAPE):", mape)

# 학습 곡선
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.ylim([0, 1.5])  # 정규화되어서 MSE/MAE도 작음
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()
# 예측 결과 시각화
x_range = range(len(y_test))
plt.figure(figsize=(12, 6))
plt.plot(x_range, y_test[x_range], color='red', label='True Prices')
plt.plot(x_range, pred[x_range], color='blue', label='Predicted Prices')
plt.legend(loc='best')
plt.title('Predicted vs True Prices')
plt.xlabel('Time Steps')
plt.ylabel('Prices (Normalized)')
plt.grid()
plt.show()