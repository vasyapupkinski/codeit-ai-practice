"""
🎯 FastAPI Lab 3 과제: PyTorch MNIST 모델 서빙 (혼자해보기)

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **Lab 3의 과제 1, 2, 3을 구현하기 위한 템플릿**입니다.
PyTorch로 학습한 MNIST CNN 모델을 서빙하며, 다음 3가지 기능을 추가합니다:

**과제 1: Top-3 예측 반환** (⭐ 기본)
- 가장 확률이 높은 숫자 3개를 반환
- torch.topk() 함수 활용

**과제 2: 이미지 파일 업로드** (⭐⭐ 심화)
- PNG/JPG 이미지 파일을 직접 업로드
- PIL로 28x28 그레이스케일 변환

**과제 3: 배치 추론 API** (⭐⭐⭐ 도전)
- 여러 이미지를 한 번에 처리
- 배치 텐서 생성 및 추론

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **Top-3 예측의 유용성 (과제 1)**:
   - 단일 예측은 틀릴 수 있음
   - 상위 3개를 보여주면 사용자가 선택 가능
   - 모델의 확신도 파악 가능

2. **이미지 파일 업로드의 편의성 (과제 2)**:
   - 픽셀 리스트 변환이 번거로움
   - 파일 업로드가 더 직관적
   - 실제 서비스에서 필수

3. **배치 처리의 효율성 (과제 3)**:
   - 이미지 10개를 10번 호출 vs 1번 호출
   - GPU 활용 극대화
   - 처리 속도 10배 이상 향상

===============================================
💡 핵심 학습 포인트
===============================================
- **torch.topk()**: 텐서에서 상위 k개 값 추출
- **PIL (Pillow)**: 이미지 로드 및 전처리
- **torchvision.transforms**: 이미지 정규화 파이프라인
- **배치 텐서**: [N, C, H, W] 형태 이해

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install torch pillow python-multipart
   - mnist_cnn.pth 파일 필요 (train_mnist.py 실행)

2. 실행 방법:
   python ./lab3/mnist_fastapi_practice.py

⚠️ 참고: LLM/NLP 취업 목표라면 이 Lab은 선택사항입니다.
대신 Lab 4 (RAG), Lab 5 (AWS Bedrock)에 집중하세요!
"""

#  필수 모듈 임포트
# 1. FastAPI 관련:
#    - FastAPI: 웹 프레임워크 클래스
#    - HTTPException: HTTP 에러 응답 생성
#    - UploadFile, File: 파일 업로드 처리
# 2. Pydantic:
#    - BaseModel: 요청 데이터 검증 기반 클래스
#    - Field: 필드 메타데이터 (min/max length 등)
# 3. PyTorch:
#    - torch: 텐서 연산 및 모델 로드
#    - nn: 신경망 레이어 정의
#    - F: 함수형 API (softmax 등)
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

#  과제 2를 위한 추가 임포트 (이미지 처리)
# 1. PIL.Image: 이미지 파일 로드 및 변환
# 2. torchvision.transforms: 이미지 전처리 파이프라인
# 3. io: 바이트 스트림 처리 (필요 시)
from PIL import Image
import torchvision.transforms as transforms
import io

#  전역 변수
# - 빈 딕셔너리를 생성하여 모델을 담을 준비
ml_models = {}

# [개념] 왜 PyTorch 모델을 직접 정의하는가?
# - .pth 파일은 "가중치(숫자)"만 저장
# - 모델 구조(클래스)는 코드로 정의해야 함
# - 학습 시 구조 != 서빙 시 구조 → 에러 발생!
# - 따라서 train_mnist.py와 100% 동일한 클래스 필요

#  CNN 모델 클래스 정의
# 1. nn.Module 상속:
#    - PyTorch 모델의 기본 클래스
#    - forward() 메서드 구현 필수
class MNISTModel(nn.Module):
    def __init__(self):
        #  부모 클래스 초기화
        # - nn.Module의 __init__ 호출 (필수!)
        super(MNISTModel, self).__init__()
        
        #  레이어 정의
        # 1. Conv2d(1, 32, kernel_size=3):
        #    - 입력 채널 1 (그레이스케일), 출력 채널 32
        #    - 3x3 커널로 특징 추출
        # 2. MaxPool2d(2): 2x2 영역에서 최댓값만 선택 (크기 1/2)
        # 3. Linear: 완전 연결 레이어 (Fully Connected)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #  순전파 (Forward Pass)
        # 1. 입력 x: [batch, 1, 28, 28]
        # 2. Conv → ReLU → Pool: [batch, 32, 13, 13]
        # 3. Flatten: [batch, 32*13*13]
        # 4. FC1 → ReLU: [batch, 128]
        # 5. FC2: [batch, 10] (각 숫자의 점수)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# [개념] 왜 Lifespan 패턴을 쓰는가?
# - 모델 로딩은 느림 (수 초)
# - 매 요청마다 로딩하면 성능 폭발
# - 서버 시작 시 한 번만 로딩 → 모든 요청이 공유
# - 메모리 절약 + 속도 향상

#  Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("====== 모델 로딩중 ...")
    
    #  모델 로딩
    # 1. MNISTModel() 인스턴스 생성 (껍데기)
    # 2. load_state_dict(): .pth 파일에서 가중치 로드
    #    - map_location='cpu': GPU 없이 CPU에서 로드
    # 3. model.eval(): 추론 모드 (Dropout/BatchNorm 비활성화)
    model = MNISTModel()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    ml_models["mnist"] = model
    
    print("✅ 모델 로딩 완료!")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

#  요청 스키마
class PredictRequest(BaseModel):
    pixels: List[float] = Field(..., min_length=784, max_length=784)

# 과제 3을 위한 배치 요청 스키마
class BatchRequest(BaseModel):
    images: List[List[float]]  # 여러 개의 784 픽셀 리스트

# ========================================
# 기본 엔드포인트 (이미 구현됨)
# ========================================
@app.post("/predict")
async def predict(request: PredictRequest):
    """기본 예측 API (단일 숫자 반환)"""
    try:
        model = ml_models["mnist"]
        tensor = torch.tensor(request.pixels).view(1, 1, 28, 28)
        
        with torch.no_grad():
            logits = model(tensor)
            prob = F.softmax(logits, dim=1)
            
            # 기본: 가장 확률 높은 숫자 1개만 반환
            max_prob, predicted_class = torch.max(prob, 1)
            
            return {
                "prediction": predicted_class.item(),
                "confidence": f"{max_prob.item() * 100:.2f}%"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# 과제 1: Top-3 예측 반환 (⭐ 기본)
# ========================================
@app.post("/predict/top3")
async def predict_top3(request: PredictRequest):
    """
    [과제 1] Top-3 예측을 반환하는 API
    
    요구사항:
    - torch.topk(3)을 사용하여 상위 3개 확률 추출
    - 응답 형식:
      {
        "prediction": 7,
        "confidence": "98.32%",
        "top3": [
          {"digit": 7, "confidence": "98.32%"},
          {"digit": 1, "confidence": "1.20%"},
          {"digit": 9, "confidence": "0.35%"}
        ]
      }
    """
    try:
        model = ml_models["mnist"]
        tensor = torch.tensor(request.pixels).view(1, 1, 28, 28)
        
        with torch.no_grad():
            logits = model(tensor)
            prob = F.softmax(logits, dim=1)
            
            # [개념] 왜 torch.topk()를 쓰는가? (과제 1)
            # - torch.max()는 1등만 반환
            # - topk(3)는 상위 3개를 한 번에 반환
            # - 사용자에게 여러 후보를 보여주면 더 유용
            
            #  topk로 상위 3개 추출
            # 1. prob.topk(3):
            #    - prob: [1, 10] 텐서 (배치 1, 클래스 10)
            #    - 반환: (values, indices)
            #    - values: [1, 3] (상위 3개 확률)
            #    - indices: [1, 3] (상위 3개 클래스 번호)
            values, indices = prob.topk(3)
            
            #  top3 리스트 구성
            # 1. for 루프로 3번 반복
            # 2. indices[0][i]: 배치 차원 [0] 필수!
            #    - indices는 [1, 3] 형태 → 첫 번째 배치의 i번째 값
            # 3. .item(): 텐서 → Python 숫자 변환
            top3 = []
            for i in range(3):
                top3.append({
                    "digit": indices[0][i].item(),  # [0] 중요! (배치 차원)
                    "confidence": f"{values[0][i].item() * 100:.2f}%"
                })
            
            return {
                "prediction": indices[0][0].item(),
                "confidence": f"{values[0][0].item() * 100:.2f}%",
                "top3": top3
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# 과제 2: 이미지 파일 업로드 (⭐⭐ 심화)
# ========================================
@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    [과제 2] 이미지 파일을 직접 업로드하여 예측
    
    요구사항:
    - PIL.Image로 이미지 로드
    - 28x28 그레이스케일로 변환
    - torchvision.transforms로 정규화
    - 텐서 형태: [1, 1, 28, 28]
    """
    try:
        # [개념] 왜 이미지 파일 업로드를 지원하는가? (과제 2)
        # - 픽셀 리스트 변환이 번거로움
        # - 사용자가 PNG/JPG 파일을 직접 올리는 게 편함
        # - 실제 서비스에서 필수 기능
        
        #  1. 이미지 로드 및 전처리
        # 1. Image.open(file.file):
        #    - UploadFile의 .file 속성 = 바이트 스트림
        #    - PIL로 이미지 로드
        # 2. .convert("L"): RGB → Grayscale (1채널)
        # 3. .resize((28, 28)): MNIST 크기로 변환
        image = Image.open(file.file).convert("L")  # Grayscale
        image = image.resize((28, 28))  # 28x28로 리사이즈
        
        #  2. 텐서 변환 및 정규화
        # 1. transforms.ToTensor():
        #    - PIL Image → PyTorch 텐서
        #    - [0, 255] → [0.0, 1.0] 자동 정규화
        # 2. transforms.Normalize((0.1307,), (0.3081,)):
        #    - MNIST 데이터셋의 평균/표준편차
        #    - (pixel - 0.1307) / 0.3081 연산
        # 3. .unsqueeze(0): [1, 28, 28] → [1, 1, 28, 28]
        #    - 배치 차원 추가 (모델은 4D 텐서 요구)
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차
        ])
        tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]
        
        #  3. 모델 추론
        # - 위의 기본 엔드포인트와 동일한 로직
        model = ml_models["mnist"]
        with torch.no_grad():
            logits = model(tensor)
            prob = F.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(prob, 1)
        
        return {
            "filename": file.filename,
            "prediction": predicted_class.item(),
            "confidence": f"{max_prob.item() * 100:.2f}%"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# 과제 3: 배치 추론 API (⭐⭐⭐ 도전)
# ========================================
@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """
    [과제 3] 여러 이미지를 한 번에 추론
    
    요구사항:
    - 입력: 여러 개의 784 픽셀 리스트
    - 배치 텐서 생성: [N, 1, 28, 28]
    - 한 번의 모델 호출로 모든 이미지 추론
    - 응답: 각 이미지의 예측 결과 리스트
    """
    try:
        # [개념] 왜 배치 처리가 중요한가? (과제 3)
        # - 이미지 10개를 10번 호출 vs 1번 호출
        # - GPU 사용 시 병렬 처리로 10배 이상 빨라짐
        # - 실무에서 필수 (대량 데이터 처리)
        
        #  1. 배치 텐서 생성
        # 1. torch.tensor(request.images):
        #    - Python 리스트 → PyTorch 텐서
        #    - [[784개], [784개], ...] → [N, 784] 텐서
        # 2. .view(-1, 1, 28, 28):
        #    - -1: 자동 계산 (N = 이미지 개수)
        #    - [N, 784] → [N, 1, 28, 28]
        batch_tensor = torch.tensor(request.images)  # [N, 784]
        batch_tensor = batch_tensor.view(-1, 1, 28, 28)  # [N, 1, 28, 28]
        
        #  2. 한 번에 모든 이미지 추론
        # - model(batch_tensor): [N, 1, 28, 28] → [N, 10]
        # - 내부적으로 N개 이미지를 동시에 처리 (병렬)
        model = ml_models["mnist"]
        with torch.no_grad():
            logits = model(batch_tensor)  # [N, 10]
            prob = F.softmax(logits, dim=1)  # [N, 10]
        
        #  3. 각 이미지의 결과를 리스트에 담기
        # 1. for 루프로 N번 반복
        # 2. prob[i]: i번째 이미지의 확률 분포 [10]
        # 3. torch.max(prob[i], 0): 최대 확률 및 클래스 번호
        results = []
        for i in range(len(request.images)):
            max_prob, predicted_class = torch.max(prob[i], 0)
            results.append({
                "image_index": i,
                "prediction": predicted_class.item(),
                "confidence": f"{max_prob.item() * 100:.2f}%"
            })
        
        return {
            "batch_size": len(request.images),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  메인 블록
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)