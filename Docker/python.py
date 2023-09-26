from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 데이터 받아오기 (JSON 형태)
    image_data_json=request.get_json()
    
	# 필요한 모듈 import 및 전처리 등 필요한 작업 진행
    
	# 모델 불러오기 (예시)
	model_path='mnist_model.h5'
	model=tf.keras.models.load_model(model_path)
    
	# 예측 수행 (예시)
	predictions=model.predict(image_data)

	# 예측 결과 반환 (예시)
	prediction=int(tf.argmax(predictions[0]).numpy())
	
	return jsonify({'prediction': prediction})

if __name__ == '__main__':
	app.run()