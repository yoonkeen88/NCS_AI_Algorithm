import tensorflow as tf
import numpy as np

t=tf.random.uniform([2,3],0,1)
n=np.random.uniform(0,1,[2,3])
print("tensorflow로 생성한 텐서:\n",t,"\n")
print("numpy로 생성한 ndarray:\n",n,"\n") # n 차원 dimension array

res=t+n # 텐서 t와 ndarray n의 덧셈
print("덧셈 결과:\n",res)

# 배열의 평균을 구해서 출력
print("텐서 t의 평균:",tf.reduce_mean(t))
print("ndarray n의 평균:",np.mean(n))