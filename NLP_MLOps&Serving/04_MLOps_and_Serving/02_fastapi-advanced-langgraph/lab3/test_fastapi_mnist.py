"""
🎯 FastAPI 테스트: MNIST 모델 API 클라이언트
1. 실제 MNIST 테스트 데이터셋에서 이미지를 가져와 API 테스트
2. PyTorch 텐서를 API 요청 형식(1D 리스트)으로 변환하는 방법
3. requests 라이브러리를 사용한 POST 요청 전송

📌 사전 준비:
1. pip install requests torch torchvision
2. MNIST 모델 서버가 실행 중이어야 함 (localhost:8000)
3. 먼저 서버 실행: python mnist_fastapi.py

📌 실행 방법:
새 터미널에서: python test_fastapi_mnist.py


📌 예상 출력:
실제 정답 숫자: 7
-------- API 서버에 전송 중...
결과 받음: {'prediction': 7, 'confidence': '98.32%'}

💡 데이터 변환 과정:
- MNIST 원본: [1, 28, 28] 텐서 (채널, 높이, 너비)
- view(-1): [784] 텐서 (1차원으로 펼침)
- tolist(): [0.0, 0.1, ...] 파이썬 리스트 (API 전송용)
"""

import requests
import torch
from torchvision import datasets, transforms
import random

# 1. 실제 MNIST 테스트 데이터 하나 가져오기
dataset = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

# 랜덤하게 이미지 하나 선택
idx = random.randint(0, len(dataset)-1)
image_tensor, label = dataset[idx]  

# 2. 텐서를 1차원 리스트로 변환 (API 요청 규격에 맞춤)
pixel_list = image_tensor.view(-1).tolist() # 784개의 float 리스트

# # ------------------------
# print(pixel_list)
# # ------------------------
print(f"실제 정답 숫자: {label}")

# 3. API 요청 보내기
print("-------- API 서버에 전송 중...")
try:
    #  API 요청 전송
    # 1. requests.post():
    #    - 새로운 TCP/IP 소켓 생성 (Keep-Alive 없으면 매번 연결)
    #    - HTTP 헤더 자동 추가 ("Content-Type": "application/json")
    #    - json 인자: Python dict를 JSON 문자열로 직렬화하여 Body에 작성
    # 2. 서버 응답 대기 (동기 블로킹)
    # 3. response.json(): 
    #    - 응답 바이트 스트림을 디코딩(UTF-8)하고 JSON 파싱하여 dict로 반환
    response = requests.post(
        "http://127.0.0.1:8000/predict/digit",
        json={"pixels": pixel_list}
    )
    print("✅ 결과 받음:", response.json())
except Exception as e:
    print("❌ 연결 실패!!!", e)