"""
🎯 FastAPI 실습: PyTorch MNIST CNN 모델 서빙

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **직접 학습한 PyTorch 모델(.pth)을 로드하여 API로 서빙**하는 실습입니다.
`train_mnist.py`에서 학습시킨 모델 파일을 불러와서, 손글씨 숫자 이미지를 예측합니다.

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **학습과 서빙의 분리**:
   - 학습: 오래 걸림 (수 분~수 시간)
   - 서빙: 빨라야 함 (밀리초 단위)
   - 학습은 한 번, 서빙은 계속

2. **모델 클래스가 동일해야 하는 이유**:
   - .pth 파일은 가중치(숫자)만 저장
   - 모델 구조는 코드로 정의
   - 학습 시 구조 != 서빙 시 구조 → 에러!

3. **PyTorch의 특징**:
   - define-by-run (동적 그래프)
   - 명시적 전처리 필요
   - GPU/CPU 선택 가능

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. MNISTModel 클래스 정의 (껍데기)
2. mnist_cnn.pth에서 가중치 로드
3. 껍데기에 가중치 적용
4. model.eval() (추론 모드)

[요청 처리]
1. 784개 float 리스트 수신 (28x28 픽셀 평탄화)
2. 전처리:
   리스트 → 텐서 → reshape[1,1,28,28]
3. 추론 (no_grad 모드)
4. Softmax로 확률 변환
5. argmax로 최종 숫자 선택

===============================================
💡 핵심 학습 포인트
===============================================
- 학습/서빙 분리: 역할의 명확한 구분
- state_dict: 가중치만 저장하는 포맷
- view/reshape: 텐서 차원 변환
- Softmax: 로짓 → 확률 변환

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install torch
   - 학습된 모델 파일 준비: mnist_cnn.pth (같은 디렉토리에 위치)

2. 실행 방법:
   python mnist_fastapi.py

💡 입력 데이터 형태 변환 과정:
- 입력: [784] - 1차원 리스트 (28×28 펼친 것)
- 변환: [1, 1, 28, 28] - [배치, 채널, 높이, 너비]
- CNN은 반드시 4차원 텐서를 입력으로 받음!

⚠️ 주의사항:
- 모델 클래스(MNISTModel)는 학습 시 사용한 코드와 100% 동일해야 함
- 픽셀값은 0.0~1.0 사이로 정규화된 값이어야 함
"""

#  필수 모듈 임포트
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import torch.nn as nn

#  모델 클래스 정의 (학습 코드와 완전히 동일해야 함!)
# --- 1. 모델 클래스 정의 (유의: 학습한 코드와 완전히 동일해야 함) ---
class MNISTModel(nn.Module):
    #  생성자: 레이어 정의
    def __init__(self):
        #  부모 클래스 초기화
        super(MNISTModel, self).__init__()
        #  레이어 정의
        # - Conv2d(1, 32, kernel_size=3): 입력 1채널, 출력 32채널, 3x3 필터
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # - ReLU: 활성화 함수
        self.relu = nn.ReLU()
        # - MaxPool2d(2): 2x2 영역에서 최댓값 선택 (크기 절반)
        self.pool = nn.MaxPool2d(2)
        # - Flatten: 다차원 → 1차원
        self.flatten = nn.Flatten()
        # - Linear(32*13*13, 128): 완전 연결층
        #   * 28x28 입력 → Conv(3x3) → 26x26 → Pool(2) → 13x13
        #   * 32 채널 × 13 × 13 = 5408
        self.fc1 = nn.Linear(32*13*13, 128)
        # - Linear(128, 10): 출력층 (0-9 숫자 분류)
        self.fc2 = nn.Linear(128, 10)

    #  순전파(Forward) 함수: 데이터가 통과하는 경로
    def forward(self, x):
        #  연산 순서
        # x: [배치, 1, 28, 28]
        # → conv1 → relu → pool
        x = self.pool(self.relu(self.conv1(x)))
        # → flatten: [배치, 32*13*13]
        x = self.flatten(x)
        # → fc1 → relu
        x = self.relu(self.fc1(x))
        # → fc2: [배치, 10] (로짓)
        x = self.fc2(x)
        return x
    
#  전역 변수
# --- 2. 전역 변수 ---
ml_models = {}

#  Lifespan 컨텍스트 매니저
# --- 3. Lifespan (모델 로드) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
# [개념] 왜 state_dict로 모델을 로드하는가?
# - PyTorch의 모델 저장 방식:
#   1) 전체 모델 저장: torch.save(model, "model.pth") - 크고 비효율적
#   2) state_dict 저장: torch.save(model.state_dict(), "model.pth") - 가볍고 효율적 (추천)
# - state_dict란?
#   * 모델의 "가중치"만 저장한 딕셔너리
#   * "구조"는 코드로 정의 (class MNISTModel)
# - 로드 과정:
#   1) 모델 구조 생성: model = MNISTModel() (랜덤 가중치)
#   2) 가중치 로드: model.load_state_dict(torch.load("model.pth"))
#   3) 엄격擔 클라스 이름도 일치해야 함!
# - 실무 장점:
#   * 메모리 절감 (100MB → 10MB)
#   * 환경 이식 쉽음 (CPU 훈련 → GPU 서빙)
#   * 버전 관리 용이

    # [Startup] 서버 시작 시 실행
    print("===== 서버 시작: MNIST 모델 로딩 중 ...")
    
    try:
        #  모델 인스턴스 생성
        # - MNISTModel(): 랜덤 가중치로 초기화된 빈 껍데기
        model = MNISTModel()

        #  모델 파일 경로 계산
        from pathlib import Path
        # - __file__: 현재 파일의 절대 경로
        # - .parent: 부모 디렉토리
        # - / "mnist_cnn.pth": 경로 결합
        MODEL_PATH = Path(__file__).parent / "mnist_cnn.pth"
        
        #  가중치 로드 및 적용
        # 1. torch.load(MODEL_PATH):
        #    - 바이너리 파일(.pth)을 역직렬화(deserialize)
        #    - 가중치 텐서들을 CPU 메모리에 적재 (OrderedDict 형태)
        # 2. model.load_state_dict(...):
        #    - 모델의 파라미터(이름 기준)와 로드된 가중치를 1:1 매핑
        #    - 파라미터 텐서의 데이터 포인터를 로드된 데이터로 업데이트 (또는 복사)
        #    - strict=True(기본값): 키 이름이 완벽히 일치해야 함
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        
        #  평가 모드 전환
        # 1. model.eval():
        #    - Dropout layer: 비활성화 (모든 뉴런 사용)
        #    - BatchNorm layer: 학습된 moving average/variance 사용 (배치 통계 무시)
        #    - self.training 속성을 False로 설정
        model.eval()
        
        #  전역 딕셔너리에 저장
        # - 모델 객체는 힙에 존재하며, ml_models가 그 참조를 유지
        ml_models["mnist"] = model
        print("✅ MNIST 모델 로드 성공!")

    except Exception as e:
        #  에러 처리
        print(f"!!! 모델 로드 실피: {e}")
        ml_models["mnist"] = None

    #  yield - 서버 실행 대기
    yield
    
    # [Shutdown] 서버 종료 시 메모리 정리
    ml_models.clear()

#  FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

#  입력 스키마 정의
# --- 4. 입력 스키마 ---
class ImageRequest(BaseModel):
    #  Field로 리스트 길이 제약
    # 28x28 = 784개의 픽셀 값 (0.0 ~ 1.0 사이의 흑백 강도)
    # - min_length=784, max_length=784: 정확히 784개만 허용
    pixels: list[float] = Field(..., min_length=784, max_length=784)

#  API 엔드포인트
# --- 5. 추론 API ---
@app.post("/predict/digit")
async def predict_digit(request: ImageRequest):
    #  모델 조회
    model = ml_models.get("mnist")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    #  전처리 단계
    # [전처리]
    # 1. 리스트 -> 텐서 변환
    # - torch.tensor(): Python 리스트를 PyTorch 텐서로 변환
    # - dtype=torch.float32: 32비트 부동소수점
    input_tensor = torch.tensor(request.pixels, dtype=torch.float32)

    #  차원 변환 (Reshape)
    # 2. View/Reshape 동작:
    #    - Tensor는 데이터(storage)와 메타데이터(stride, shape)로 구성
    #    - .view()는 데이터를 복사하지 않고 메타데이터(stride)만 변경하여 차원 재해석
    #    - 메모리 효율적 (O(1) 연산)
    #    - [784] -> [1, 1, 28, 28] (배치, 채널, 높이, 너비)
    input_tensor = input_tensor.view(1,1,28,28)

    #  추론 단계
    # [추론]
    # 1. torch.no_grad():
    #    - Autograd 엔진 비활성화
    #    - 중간 연산 결과(activation)를 저장하지 않음 → 메모리 사용량 대폭 감소
    #    - 연산 그래프 생성 오버헤드 제거 → 속도 향상
    with torch.no_grad():
        #  모델 통과 (Forward Pass)
        # 1. model(input_tensor) 호출:
        #    - __call__ -> forward 메서드 실행
        #    - Conv2d -> MaxPool -> Flatten -> Linear 연산 수행
        #    - 이 과정은 순수 행렬/텐서 연산으로 진행 (GPU였다면 CUDA 커널 실행)
        logits = model(input_tensor)
        
        #  Softmax 적용
        # 1. Softmax 함수:
        #    - exp(x_i) / sum(exp(x_j)) 계산
        #    - 로짓값을 0~1 사이의 확률값으로 변환
        #    - dim=1: 클래스 축(10개)을 기준으로 계산
        prob = torch.nn.functional.softmax(logits, dim=1)

    #  후처리 단계
    # [후처리]
    # 가장 높은 확률을 가진 숫자의 인덱스(.argmax())와 그 확률값(.max()) 가져오기
    # - .argmax(): 가장 큰 값의 인덱스 (0-9)
    # - .max(): 가장 큰 확률값 (0.0-1.0)
    # - .item(): 텐서 → Python 숫자 변환
    predicted_class = prob.argmax().item()
    confidence = prob.max().item()

    #  결과 반환
    return {
        "prediction": predicted_class,              # 예측된 숫자(0-9)
        "confidence": f"{confidence*100:.2f}%"      # 확신도
    }

#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)