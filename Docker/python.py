# TensorFlow 버전 확인
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# 손글씨 정보 데이터 세트 (x:손글씨 이미지(28*28), y:숫자)
# Train 6만개, Test 1만개
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x:0~255 이미지
x_train, x_test = x_train / 255.0, x_test / 255.0 # x:0~1 이미지 변환

# 신경망 구축
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
# 784 > 128 (relu) > Dropout(0.2) > 10 (확률분포)
# Dropout(0.2) : 노드 20% 무작위로 0 (과적합 방지) 
# softmax : 0~9 각 확률들을 나타냄

# 모델 설정
# optimizer adam : 최적화 'adam' 알고리즘
# loss 형 유형을 선택, 훈련데이터의 label 값이 정수일 때
# metrics : 모델 평가, accuracy: 빈도계산
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련, train으로 훈련, 5회
model.fit(x_train, y_train, epochs=5)

# 모델 평가, 몇 % 정확도인지 보여준다.
model.evaluate(x_test,  y_test, verbose=2)

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x:0~255 이미지
x_train, x_test = x_train / 255.0, x_test / 255.0 # x:0~1 이미지 변환

def MNIST_AI (h, t) :
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(h, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=t)
  return model.evaluate(x_test,  y_test, verbose=2)

def MNIST_AI_Save (h, t) :
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(h, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=t)
  model.save('mnist_testv0.128')
  return model.evaluate(x_test,  y_test, verbose=2)


MNIST_AI_Save(128, 5)