문서정보 : 2023.03.13. 작성, 작성자 [@SAgiKPJH](https://github.com/SAgiKPJH)

<br>

# Number_Python_Docker_API
파이썬 기반 숫자 인식, 도커를 활용한 웹사이트 활용 프로그램 API

### 목표
- [x] : GitHub 기본 환경 구축
- [x] : GitFork를 통한 Local - Web GitHub 연결
- [ ] : 도커를 활용한 API에 대한 힌트 얻기

### 제작자
[@SAgiKPJH](https://github.com/SAgiKPJH)

<br>
---
<br>

# 1. GitHub 기본 환경 구축

- Git Hub를 통해 기본적인 환경을 갖춘다.

<br>

# 2. GitFork를 통한 Local - Web GitHub 연결

- GitFork를 활용하여 Git Branch 및 프로젝트를 관리하도록 한다.

<br>


# 3. 도커를 활용한 API에 대한 힌트 얻기

- GPT Chat, BingAI를 통해 힌트를 획득한다.


<br>

# 4. dockerfile 작성하기

- Dockerfile을 다음과 같이 작성한다.
  ```dockerfile
  # 베이스 이미지 지정
  FROM ubuntu:latest
  
  # 필요한 패키지 설치
  RUN apt-get update && apt-get install -y apache2 python3-pip
  
  # 작업 디렉터리를 지정
  WORKDIR /app
  
  # COPY는 호스트의 파일이나 디렉터리를 컨테이너의 파일이나 디렉터리로 복사합니다.
  # 예를 들어, 현재 디렉터리의 모든 파일을 컨테이너의 /app 디렉터리로 복사합니다.
  COPY . /app
  
  # Tensorflow 설치
  RUN pip3 install tensorflow
  
  # numpy 설치
  RUN pip3 install numpy
  
  # 웹 서버 포트 열기
  EXPOSE 80
  
  # 웹 페이지 파일 복사
  COPY index.html /var/www/html/
  
  # 웹 서버 실행
  CMD ["/usr/sbin/apache2ctl", "-D", "FOREGROUND"]
  ```