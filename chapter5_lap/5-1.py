import tensorflow as tf

print(tf.__version__)
a=tf.random.uniform([2,3],0,1) # 유니폼 분포에서 2 x 3 행렬을 만듬
print(a)
print(type(a))

b=tf.random.normal([2,3],0,1) # 정규 분포에서 2 x 3 행렬을 만듬
print(b)



print(tf.reduce_mean(a)) # a의 평균 
print(tf.reduce_mean(b)) # b의 평균