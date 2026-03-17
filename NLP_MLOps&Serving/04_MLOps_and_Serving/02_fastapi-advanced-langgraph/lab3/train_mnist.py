"""
🎯 PyTorch 실습: MNIST CNN 모델 학습 및 저장
1. CNN(Convolutional Neural Network) 모델 구조 이해
2. MNIST 손글씨 숫자 데이터셋으로 모델 학습
3. 학습된 모델 가중치를 .pth 파일로 저장
4. 저장된 모델을 FastAPI에서 로드하여 서빙하는 워크플로우의 첫 단계

📌 사전 준비:
pip install torch torchvision

📌 실행 방법(모델저장 위치를 위해서 디렉터리 이동후에: "cd lab3") 
python train_mnist.py

📌 생성되는 파일:
- mnist_cnn.pth (학습된 모델 가중치)
- ./data/MNIST/ (다운로드된 데이터셋)

💡 CNN 모델 구조:
입력 [1,28,28] → Conv2d → ReLU → MaxPool → Flatten → FC → FC → 출력 [10]
- Conv2d: 이미지에서 특징(엣지, 패턴) 추출
- MaxPool: 특징 맵 크기 축소, 중요 정보만 유지
- Flatten: 2D → 1D 변환 (FC 레이어 입력용)
- FC(Linear): 최종 분류 (0~9 숫자)

⚠️ 주의사항:
- 이 모델 클래스는 서빙 코드에서도 동일하게 정의해야 함!
- 1 Epoch만 학습 (실습용)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. CNN 모델 정의 (이미지 처리에 적합한 모델)
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 이미지 특징 추출 (Convolution)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # 분류 (Linear)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10) # 0~9 숫자 분류

    #  순전파(Forward) 함수
    # - 데이터가 네트워크를 통과하는 경로 정의
    # - PyTorch의 Autograd 엔진이 이 메서드를 추적하여 연산 그래프 생성
    def forward(self, x):
        # 1. Conv2d + ReLU + MaxPool
        #    - 합성곱 연산 (Winograd 알고리즘 등 사용)
        #    - 활성화 함수 (음수값 0으로 제거)
        #    - 풀링 (특징맵 축소)
        x = self.pool(self.relu(self.conv1(x)))
        # 2. Flatten
        #    - 텐서 메모리 구조 변경 없이 stride만 조작하여 차원 변경 (View)
        x = self.flatten(x)
        # 3. FC Layers
        #    - 행렬 곱셈 (GEMM: General Matrix Multiply)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# [개념] 왜 학습(Train)과 서빙(Serve)을 분리하는가?
# - 학습의 특성:
#   * 시간: 수 분 ~ 수 시간 소요 (MNIST는 짧지만 실제론 매우 김)
#   * 리소스: GPU, 대용량 메모리, 학습 데이터셋 필요
#   * 빈도: 한 번 또는 주기적 (일주일에 1회 등)
# - 서빙의 특성:
#   * 시간: 밀리초 단위 (실시간 응답 필수)
#   * 리소스: CPU만으로도 가능, 가벼움
#   * 빈도: 초당 수백~수천 번 요청
# - 분리의 장점:
#   * 학습: 강력한 GPU 서버 (비쌈, 1대)
#   * 서빙: 여러 대의 저렴한 CPU 서버 (스케일 아웃)
#   * 비용 절감 + 성능 최적화
# - 실무 패턴:
#   * 학습: Jupyter, Python 스크립트
#   * 저장: .pth, .h5, .onnx 파일
#   * 서빙: FastAPI, Flask로 API 제공
# - state_dict의 역할:
#   * 모델 "구조"는 코드로 정의 (class MNISTModel)
#   * 모델 "가중치"만 파일로 저장 (.pth)
#   * 서빙 시: 구조 생성 + 가중치 로드

def train():
    # 데이터셋 준비 (없으면 다운로드함)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("🧠 학습 시작 (데이터가 많아 1~2분 걸릴 수 있습니다)...")
    
    # 빠르게 1 Epoch만 학습 (실습용)
    # 빠르게 1 Epoch만 학습 (실습용)
    for batch_idx, (data, target) in enumerate(loader):
        #  그래디언트 초기화
        # 1. optimizer.zero_grad():
        #    - 이전 배치의 그래디언트(grad 속성)를 0으로 초기화
        #    - PyTorch는 기본적으로 그래디언트를 누적(accumulate)하므로 매 배치마다 초기화 필수
        optimizer.zero_grad()
        
        #  순전파 (Forward Pass)
        # 1. output = model(data):
        #    - model.__call__ 실행 (내부적으로 forward 호출)
        #    - 입력 텐서가 각 레이어를 통과하며 연산 그래프(Computation Graph) 생성
        #    - 각 연산 노드는 역전파를 위한 미분 함수(grad_fn)를 기억
        output = model(data)
        
        #  손실 계산
        # 1. criterion(output, target):
        #    - CrossEntropyLoss 계산 (LogSoftmax + NLLLoss)
        #    - 스칼라 값(loss) 반환 및 연산 그래프의 끝점 형성
        loss = criterion(output, target)
        
        #  역전파 (Backward Pass)
        # 1. loss.backward():
        #    - 연산 그래프를 역방향으로 탐색 (Chain Rule 적용)
        #    - 각 파라미터(w, b)에 대한 손실의 기울기(gradient) 계산
        #    - 계산된 기울기를 각 파라미터 텐서의 .grad 속성에 저장
        loss.backward()
        
        #  파라미터 업데이트
        # 1. optimizer.step():
        #    - 계산된 .grad를 사용하여 가중치 수정
        #    - w = w - learning_rate * grad (Adam 알고리즘 적용)
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"진행률: {batch_idx}/{len(loader)}")

    # 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("✅ 모델 저장 완료: mnist_cnn.pth")

if __name__ == "__main__":
    train()