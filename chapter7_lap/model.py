import numpy as np
import tensorflow as tf
from PIL import Image
import os

cnn=tf.keras.models.load_model("chapter7_lap/my_cnn_for_deploy.h5") # 학습된 모델 불러오기
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck'] # CIFAR-10의 부류 이름
cnn.summary() # 모델 구조 출력

print(cnn.optimizer.get_config()) # 모델의 최적화기 설정 정보 출력
print(cnn.layers[-1].get_config()) # 마지막 레이어의 설정 정보 출력