import tensorflow as tf

# OR 데이터 구축
x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[-1],[1],[1],[1]]

# 가중치 초기화
w=tf.Variable(tf.random.uniform([2,1],-0.5,0.5))
b=tf.Variable(tf.zeros([1]))

# 옵티마이저
opt=tf.keras.optimizers.SGD(learning_rate=0.1)
# matmul: 행렬 곱셈, add: 덧셈, tanh: 하이퍼볼릭 탄젠트 함수
# 전방 계산(식 (4.3))
def forward():
    s=tf.add(tf.matmul(x,w),b) # 행렬 곱셈과 편향 추가
    # 활성화 함수 적용
    o=tf.tanh(s) # 하이퍼볼릭 탄젠트 함수
    return o

# 손실 함수 정의
def loss():
    o=forward()
    return tf.reduce_mean((y-o)**2)

# 500세대까지 학습(100세대마다 학습 정보 출력)

# 학습 루프
for i in range(500): # 에포크를 500으로 설정
    with tf.GradientTape() as tape: # 함수를 인자로 받아서 해당 손실함수를 기준으로 최소화하는 방향으로 최적화
        current_loss = loss()
    grads = tape.gradient(current_loss, [w, b])
    opt.apply_gradients(zip(grads, [w, b]))
    if i % 100 == 0:
        print(f"loss at epoch {i} = {current_loss.numpy()}")


# 학습된 퍼셉트론으로 OR 데이터를 예측
o=forward()
print(o)