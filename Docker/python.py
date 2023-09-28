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

	# 예측 수행 (예시)
    input_image = np.expand_dims(image_array, axis=0)  # 차원 변경 (1개 샘플)
    predictions = model.predict(input_image)
    prediction_result = np.argmax(predictions[0])  # 가장 높은 확률의 클래스 선택

    return jsonify({'prediction': int(prediction_result)})

if __name__ == '__main__':
	app.run(host='0.0.0.0') # 포트 외부 접근 허용