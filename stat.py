import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# 데이터 불러오기
df_all = pd.read_csv('new_data.csv')
df_all = df_all.drop(columns=['Unnamed: 0'])  # 불필요한

print(df_all.describe())

# print('각 직원별 정보 평균 및 표준편차 및 근속연수: ', df_all.mean(), df_all.std(), df_all['근속연수'].mean())

# 직원별 특성 정보 평균 정성 데이터와 정량 데이터 구분하여 출력
# 행평균 출력
print("직원별 특성 정보 평균과 근속연수")
a= df_all.mean(axis=1)
b = df_all['근속연수']
c = pd.concat([a, b], axis=1)
print(c)

print(c.corr())  # 상관계수 행렬 출력