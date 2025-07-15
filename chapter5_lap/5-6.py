from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

# OR 데이터 구축
x = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=tf.float32)
y = tf.constant([[-1.0], [1.0], [1.0], [1.0]], dtype=tf.float32)
n_input=2
n_output=1

perceptron=Sequential()
perceptron.add(
    Dense(units=n_output,activation='tanh',
                     input_shape=(n_input,),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros'))

perceptron.compile(loss='mse',
                   optimizer=SGD(learning_rate=0.1),
                   metrics=['mse'])

perceptron.fit(x,y,epochs=500,verbose=2) # verbose=2는 에포크마다 손실을 출력합니다.

res=perceptron.predict(x)
print(res)