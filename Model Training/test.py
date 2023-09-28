import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
from PIL import Image


# 모델 불러오기
model_path = './Model Training/mnist_model.h100'
model = tf.keras.models.load_model(model_path)

# PNG 이미지 읽어오기 및 전처리
image_path = './Model Training/image.png' # 이미지 파일 경로 지정

image = Image.open(image_path).convert('L') # 그레이스케일로 변환
# image = Image.eval(image, lambda x: 255 - x) # 이미지 색상 반전
# image.show() # 이미지 출력
image = image.resize((28, 28)) # MNIST 데이터셋과 동일한 크기로 리사이즈
image_array = np.array(image) / 255.0 # 이미지 배열 생성 및 전처리 (0~1 범위로 스케일링)
input_image = np.expand_dims(image_array, axis=0) # 차원 변경 (1개 샘플)

# 예측 수행
predictions = model.predict(input_image)
predicted_label = np.argmax(predictions[0])

print("Predicted label:", predicted_label)