import tensorflow as tf
import numpy as np

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