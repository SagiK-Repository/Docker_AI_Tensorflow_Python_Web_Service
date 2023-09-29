from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # 클라이언트로부터의 요청에 대해 모든 도메인에서 접근 허용(CORS 허용)

# 모델 불러오기
model_path = 'mnist_model.h100'
model = tf.keras.models.load_model(model_path)

# np 이미지 회전
def rotate_image(image_array):
    angle = np.random.uniform(low=-30.0, high=30.0)  # -30도에서 30도 사이의 각도 랜덤 선택
    image = Image.fromarray(image_array)
    rotated_image = image.rotate(angle)
    rotated_image_array = np.array(rotated_image)
    return rotated_image_array

# np 이미지 기울기
def shear_image(image_array):
    shear = np.random.uniform(low=-0.2, high=0.2)  # -0.2에서 0.2 사이의 기울기 랜덤 선택
    image = Image.fromarray(image_array)
    sheared_image = image.transform(image.size, method=Image.AFFINE, data=(1, shear, 0, 0, 1, 0))
    sheared_image_array = np.array(sheared_image)
    return sheared_image_array

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 데이터 받아오기 (JSON 형태)
    image_data_json=request.get_json()
    image_data = image_data_json['image_data']

    # base64 디코딩하여 이미지 바이너리 데이터 추출
    image_binary = base64.b64decode(image_data.split(',')[1])

    # 바이너리 데이터를 Pillow(PIL) Image 객체로 변환
    image = Image.open(io.BytesIO(image_binary)).convert('L') # 그레이 스케일로 변환

    # 이미지 크기 조정 및 전처리 수행 (MNIST 모델 기준)
    image = Image.eval(image, lambda x: 255 - x) # 이미지 색상 반전
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0  # 정규화

    # 예측 수행 (10회 진행)
    predictions_list = []
    for _ in range(10):
        change_image = rotate_image(image_array)
        change_image = shear_image(change_image)
        input_image = np.expand_dims(change_image, axis=0)  # 차원 변경 (1개 샘플)
        predictions = model.predict(input_image)
        prediction_result = np.argmax(predictions[0])  # 가장 높은 확률의 클래스 선택
		
        predictions_list.append(prediction_result)

	# 가장 많은 빈도로 등장한 레이블 선택하기	
    result_label_counts = np.bincount(predictions_list)
    most_frequent_label_index = np.argmax(result_label_counts)
    most_frequent_label_prediction_result= result_label_counts[most_frequent_label_index]

    return jsonify({'prediction': int(most_frequent_label_prediction_result)})

if __name__ == '__main__':
	app.run(host='0.0.0.0') # 포트 외부 접근 허용