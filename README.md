문서정보 : 2023.03.13.~09.29. 작성, 작성자 [@SAgiKPJH](https://github.com/SAgiKPJH)

<br>

# Number_Python_Docker_API
파이썬 기반 숫자 인식, 도커를 활용한 웹사이트 활용 프로그램 API

### 목표
- [x] 0. 프로젝트 설계
- [x] 1. Dockerfile 구성
- [x] 2. Python 구성
- [x] 3. index.html 구성
- [x] 4. Run

### 제작자
[@SAgiKPJH](https://github.com/SAgiKPJH)

<br>
---
<br>

# 0. 프로젝트 설계

- Number를 인식하는 AI는 Python의 Tensorflow를 활용합니다.
- docker를 통해 html를 여는데, 이를 Apache2를 통해 구현합니다.
- html의 script를 통해 Java 언어로 Python코드를 호출합니다. 이 방법을 flask를 통해 구현합니다.
- html에서는 마우스 또는 터치로 화면에 숫자를 그립니다.
- 이 숫자 한자리를 AI를 통해 인식하여 값을 추출합니다.
- 이를 docker로 구성합니다.


### 필요 환경

- Port 80
  - Apache2의 서비스 포트
- Port 5000
  - flask의 서비스 포트 
- 필요한 pip install 내용
  - tensorflow
  - numpy
  - flask
  - flask-cors
  - Pillow  

<br>

# 1. Dockerfile 구성

- dockerfile을 다음과 같이 구성한다.  
  ```dockerfile
  # 베이스 이미지 지정
  FROM ubuntu:latest
  
  # 필요한 패키지 설치
  RUN apt-get update && \
      apt-get install -y apache2 python3-pip && \
      rm -rf /var/lib/apt/lists/*
  
  RUN pip3 install tensorflow numpy flask flask-cors Pillow
  
  # 작업 디렉터리를 지정
  WORKDIR /var/www/html/
  
  # COPY는 호스트의 파일이나 디렉터리를 컨테이너의 파일이나 디렉터리로 복사합니다.
  # 예를 들어, 현재 디렉터리의 모든 파일을 컨테이너의 /app 디렉터리로 복사합니다.
  COPY . /var/www/html/
  
  # 웹 서버 포트 열기
  EXPOSE 80
  EXPOSE 5000
  
  # 웹 서버 실행
  CMD ["bash", "-c", "apachectl -D FOREGROUND & python3 /var/www/html/python.py"]
  # CMD ["apachectl", "-D", "FOREGROUND"]
  ```

<br>

# 2. Python 구성

- python을 다음과 같이 구성합니다.  
  ```python
    # python.py
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
  ```
- Training Model을 만드는 Python코드를 다음과 같이 구성합니다.  
  ```python
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)

    # MNIST 데이터셋 로드 및 전처리
    mnist = tf.keras.datasets.mnist 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 신경망 구축
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 모델 설정
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 모델 훈련
    model.fit(x_train, y_train, epochs=1000)

    # 모델 저장
    model.save('mnist_model.h1000')
    print("모델이 저장되었습니다.")
  ```
- 이를 Test하는 코드를 구성합니다. 이떄, Test 이미지는 배경이 까맣고 숫자는 흰 글자입니다.
  ```python
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
  ```

<br>

# 3. index.html 구성

- html을 다음과 같이 구성합니다.  
  ```html
  <!DOCTYPE html>
  <html>
  <head>
    <style>
      #canvas {
        border: 1px solid black;
      }

      button {
        margin: 10px;
        padding: 10px;
        font-size: 16px;
      }

      label {
        margin: 10px;
        padding: 10px;
        font-size: 16px;
      }
    </style>
  </head>
  <body>
  <canvas id="canvas" width="500" height="500"></canvas>

  <button id="clear">Clear</button>
  <button id="recognize">Recognize</button>

  <label id="result">Result:</label>

  <script>
  // 캔버스, 컨텍스트, 버튼, 레이블 가져오기
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const clear = document.getElementById("clear");
  const recognize = document.getElementById("recognize");
  const resultLabel = document.getElementById("result");

  const serverAddress = window.location.hostname + ":10181";

  // 초기화 함수
  function clearCanvas() {
      // 캔버스의 전체 영역을 흰색으로 지우기
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // 인식 함수
  function recognizeNumber() {  
    resultLabel.textContent = "Result : ...";
    
    // 캔버스 이미지 데이터를 base64 형식으로 가져오기
    const imageData = canvas.toDataURL();

    // AJAX 요청을 통해 서버로 이미지 데이터 전송 및 결과 수신하기
    fetch('http://' + serverAddress + '/predict', {
        method: 'POST',
        body: JSON.stringify({ image_data: imageData }),
        headers:{
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        resultLabel.textContent = "Result : " + data.prediction;

        // 예측 결과 출력 후 캔버스 초기화
        clearCanvas();
    })
    .catch(error => console.error('Error:', error));
  }

  // 마우스 이벤트 리스너 추가하여 선 그리기 구현하기
  let isDrawing = false;

  function startDrawing(e) {
      isDrawing = true;

      const rect = e.target.getBoundingClientRect();
      const x = getEventX(e);
      const y= getEventY(e);

      ctx.beginPath();
      ctx.moveTo(x, y);
  }

  function draw(e) {
      if (!isDrawing) return;

      const rect= e.target.getBoundingClientRect();
      const x= getEventX(e);
      const y= getEventY(e);

      ctx.lineWidth = 10; // 선 굵기 설정 (4로 변경)
      ctx.lineTo(x,y);
      ctx.stroke();
  }

  function stopDrawing() {
          isDrawing=false; 
  }

  // 공통된 좌표값 반환 함수 (마우스 및 터치)
  function getEventX(event) {
    return event.type.includes('mouse') ? event.clientX - event.target.getBoundingClientRect().left : event.touches[0].clientX - event.target.getBoundingClientRect().left;
  }

  function getEventY(event) {
    return event.type.includes('mouse') ? event.clientY - event.target.getBoundingClientRect().top : event.touches[0].clientY -event.target.getBoundingClientRect().top;
  }

  // 버튼에 클릭 이벤트 리스너 추가
  clear.addEventListener("click", clearCanvas);
  recognize.addEventListener("click", recognizeNumber);

  // 캔버스에 마우스 이벤트 리스너 추가하여 선 그리기 기능 활성화하기
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDrawing);

  canvas.addEventListener('touchstart', startDrawing);
  canvas.addEventListener('touchmove', draw);
  window.addEventListener('touchend', stopDrawing);

  </script>

  </body>
  </html>
  ```
- 다음과 같은 화면이 나타납니다.  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/3dd867f7-acef-49f0-a5c2-b0097c67e092)


<br>

# 4. Run

- docker build  
  ```bash
  docker build -t juhyung1021/number_python_docker_api_image .
  ```
- docker run  
  ```bash
  docker run -it --name number_python_docker_api -p 10180:80 -p 10181:5000 --network="host" -d juhyung1021/number_python_docker_api_image:latest
  ```
- Docker 확인  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/982ca4e2-f10f-4c1c-bb82-f8d9491aaffd)
- falsk 대기화면 확인  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/c1436edb-7f29-4e1a-a731-ef81c9e292d8)
- 숫자 인식 화면  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/32813135-986e-4f96-9b34-9145e050ce1d)
- Console 및 flask 확인  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/eac50a39-b3e8-4aae-8b5b-4c494d414629)  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/fad92bf2-47d1-47c8-9854-3fbabc7ad86c)  
- 외부 기기 접속 화면  
  ![image](https://github.com/SagiK-Repository/Number_Python_Docker_API/assets/66783849/a4b97755-956e-4e14-ab8f-d8ae2fd13122)




<br>
