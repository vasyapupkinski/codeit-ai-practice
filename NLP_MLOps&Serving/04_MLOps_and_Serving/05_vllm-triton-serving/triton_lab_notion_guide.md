
### 각 항목 설명

| 항목 | 값 | 설명 |
|------|-----|------|
| `name` | "simple_onnx" | 모델 이름 (폴더 이름과 일치해야 함) |
| `platform` | "onnxruntime_onnx" | 사용할 추론 엔진 |
| `max_batch_size` | 0 | 0 = 배칭 비활성화 (모델에 따라 다름) |
| `input.name` | "Input3" | 모델의 입력 텐서 이름 (모델마다 다름!) |
| `input.data_type` | TYPE_FP32 | 32비트 부동소수점 |
| `input.dims` | [1, 1, 28, 28] | [배치, 채널, 높이, 너비] |
| `output.name` | "Plus214_Output_0" | 모델의 출력 텐서 이름 |
| `output.dims` | [1, 10] | 10개 클래스의 확률 |

### 🔍 입력/출력 이름은 어떻게 알아내나?

```bash
# Python에서 ONNX 모델 분석
pip install onnx
python -c "
import onnx
model = onnx.load('model.onnx')
print('=== 입력 ===')
for input in model.graph.input:
    print(f'이름: {input.name}')
    print(f'shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}')
print('=== 출력 ===')
for output in model.graph.output:
    print(f'이름: {output.name}')
"
```

### 📏 dims 이해하기: MNIST 예시

```
dims: [1, 1, 28, 28]
       │  │   │   │
       │  │   │   └── 가로 28픽셀
       │  │   └────── 세로 28픽셀
       │  └────────── 1채널 (흑백, RGB면 3)
       └───────────── 1개 이미지 (배치 크기)

최종: 28x28 흑백 이미지 1장
```

---

# 8. Step 5: Triton 서버 실행 - 완전 해부

## 🎯 이 단계에서 하는 것

**준비한 모델을 가지고 Triton Inference Server를 실행**합니다.
서버가 성공적으로 뜨면, API로 추론 요청을 받을 수 있게 됩니다.

```
docker run ... tritonserver
        ↓
Triton이 /models 폴더를 읽음
        ↓
simple_onnx 모델 로딩 성공
        ↓
포트 8000/8001/8002 열림
        ↓
curl로 헬스체크 → "200 OK"
```

## ❓ 왜 필요한가?

| 단계 | 이전까지 한 것 | 이번에 하는 것 |
|------|---------------|---------------|
| 준비 | 모델 파일, 설정 파일 작성 | - |
| 실행 | - | **실제로 서버를 띄워서 동작 확인** |
| 검증 | - | curl로 API가 응답하는지 테스트 |

**아무리 모델을 잘 준비해도, 실제로 서버를 띄워보기 전까지는 제대로 설정됐는지 알 수 없습니다.**
이 단계에서 처음으로 "내가 만든 AI 서버가 살아있다!"를 확인할 수 있습니다.

## 🚀 실행 명령어 완전 분해

```bash
docker run -d --gpus all --name triton-server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton-lab/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
```

### 한 줄씩 분석

**줄 1: 기본 설정**
```bash
docker run -d --gpus all --name triton-server \
```

| 옵션 | 설명 |
|------|------|
| `docker run` | 컨테이너 생성 및 실행 |
| `-d` | detached mode (백그라운드 실행) |
| `--gpus all` | 모든 GPU 사용 |
| `--name triton-server` | 컨테이너 이름 지정 |
| `\` | 명령어 줄 바꿈 (가독성) |

**줄 2: 포트 매핑**
```bash
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
```

```
-p 호스트포트:컨테이너포트

┌────────────────────────────────────────────────┐
│                 EC2 (호스트)                   │
│                                                │
│   외부 요청 ──► 포트 8000 ──┐                  │
│   외부 요청 ──► 포트 8001 ──┼──┐               │
│   외부 요청 ──► 포트 8002 ──┼──┼──┐            │
│                             │  │  │            │
│   ┌─────────────────────────┼──┼──┼────────┐  │
│   │  Docker 컨테이너        │  │  │        │  │
│   │                         ▼  ▼  ▼        │  │
│   │  Triton ◄── 8000:HTTP   ──────────────►│  │
│   │          ◄── 8001:gRPC  ──────────────►│  │
│   │          ◄── 8002:Metrics ──────────►  │  │
│   └────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

**줄 3: 볼륨 마운트**
```bash
-v ~/triton-lab/model_repository:/models \
```

```
-v 호스트경로:컨테이너경로

┌────────────────────────────────────────────────┐
│  EC2 (호스트)                                   │
│  ~/triton-lab/model_repository/                │
│  ├── simple_onnx/                              │
│  │   ├── config.pbtxt                          │
│  │   └── 1/model.onnx                          │
│                    │                            │
│                    ▼ 연결됨 (실시간 동기화)     │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Docker 컨테이너                          │  │
│  │  /models/                                 │  │
│  │  ├── simple_onnx/                         │  │
│  │  │   ├── config.pbtxt  ◄── 읽을 수 있음! │  │
│  │  │   └── 1/model.onnx  ◄── 읽을 수 있음! │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

**줄 4: 이미지**
```bash
nvcr.io/nvidia/tritonserver:24.10-py3 \
```

| 부분 | 의미 |
|------|------|
| `nvcr.io` | NVIDIA Container Registry |
| `nvidia` | 네임스페이스 |
| `tritonserver` | 이미지 이름 |
| `24.10-py3` | 태그 (2024년 10월 버전, Python 3) |

**줄 5: 실행 명령어**
```bash
tritonserver --model-repository=/models
```

컨테이너가 시작되면 실행할 명령어입니다.
- `tritonserver`: Triton 서버 실행 프로그램
- `--model-repository=/models`: 모델을 여기서 찾아라

## ✅ 상태 확인 명령어

### 로그 확인
```bash
docker logs -f triton-server
```
- `-f`: follow (실시간으로 계속 보기)
- `Ctrl+C`로 나가기 (서버는 계속 돌아감)

### 헬스 체크
```bash
curl -i http://localhost:8000/v2/health/ready
```

| 부분 | 의미 |
|------|------|
| `-i` | include (HTTP 헤더도 출력) |
| `localhost` | 같은 서버 내에서 접속 |
| `/v2/health/ready` | Triton의 준비 상태 확인 API |

**성공 응답:**
```
HTTP/1.1 200 OK
Content-Length: 0
Content-Type: text/plain
```

---

# 9. Step 6: AWS CLI & ECR - 완전 해부

## 🎯 이 단계에서 하는 것

**AWS 명령어 도구(CLI)를 설정하고, ECR(이미지 저장소)에 로그인**합니다.

```
aws configure
    ↓
Access Key, Secret Key 입력
    ↓
aws ecr get-login-password | docker login
    ↓
ECR에 이미지 push 권한 획득!
```

## ❓ 왜 필요한가?

| 문제 상황 | 설명 |
|----------|------|
| EC2 #1에서만 쓸 거면? | 이 단계 필요 없음 |
| **다른 서버에서도 쓰려면?** | 이미지를 "공용 창고"에 올려야 함 |
| 공용 창고가 뭔데? | AWS ECR(Elastic Container Registry) |
| 왜 로그인이 필요해? | 아무나 올리고 받으면 안 되니까 (보안) |

**ECR의 역할**: Docker Hub의 AWS 버전. 프라이빗 이미지 저장소입니다.

```
지금 EC2 #1 ──► 이미지 빌드
       │
       └──► ECR에 업로드 (push)
                 │
                 ▼
       다른 EC2들이 다운로드 (pull) 가능!
```

## 🔐 AWS CLI 인증 원리

```
┌─────────────────────────────────────────────────────────────┐
│                        인증 흐름                            │
│                                                             │
│  1. aws configure                                          │
│     ┌─────────────────────────────┐                        │
│     │ Access Key ID: AKIA...      │ ← 신분증 번호          │
│     │ Secret Access Key: xxxx...  │ ← 신분증 비밀번호      │
│     │ Region: ap-northeast-2      │ ← 서울 리전            │
│     └─────────────────────────────┘                        │
│                    │                                        │
│                    ▼                                        │
│  2. 설정 파일 저장                                          │
│     ~/.aws/credentials                                      │
│     ~/.aws/config                                           │
│                    │                                        │
│                    ▼                                        │
│  3. AWS API 호출 시                                         │
│     모든 요청에 서명 첨부 (HMAC-SHA256)                     │
│                    │                                        │
│                    ▼                                        │
│  4. AWS 서버에서 검증                                       │
│     "이 Key ID의 주인이 맞는지 서명 확인"                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📦 ECR 로그인 명령어 분해

```bash
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin 075176197136.dkr.ecr.ap-northeast-2.amazonaws.com
```

### 파이프라인 흐름

```
┌──────────────────────────────────┐
│ aws ecr get-login-password       │
│ --region ap-northeast-2          │
│                                  │
│ 출력: eyJhbGciOiJ....(토큰)      │
└─────────────┬────────────────────┘
              │
              │  파이프 (|)
              │  앞 명령어의 출력을
              │  뒤 명령어의 입력으로
              ▼
┌──────────────────────────────────┐
│ docker login                     │
│ --username AWS                   │
│ --password-stdin                 │
│ 075176197136.dkr.ecr...          │
│                                  │
│ "토큰을 비밀번호로 사용해서      │
│  ECR에 로그인해라"               │
└──────────────────────────────────┘
```

### ECR 주소 구조

```
075176197136.dkr.ecr.ap-northeast-2.amazonaws.com
│            │   │   │               │
│            │   │   │               └── AWS 도메인
│            │   │   └────────────────── 리전 (서울)
│            │   └────────────────────── ECR 서비스
│            └────────────────────────── Docker Registry 표시
└─────────────────────────────────────── AWS 계정 ID (12자리)
```

---

# 10. Step 7: 커스텀 이미지 빌드 - 완전 해부

## 🎯 이 단계에서 하는 것

**모델이 포함된 커스텀 Docker 이미지를 만들고, ECR에 업로드**합니다.

```
Dockerfile 작성
    ↓
docker build (이미지 생성)
    ↓
docker tag (ECR 주소 붙이기)
    ↓
docker push (ECR로 업로드)
```

## ❓ 왜 필요한가?

**Step 5에서 한 방식 vs 이번 방식 비교:**

| 항목 | Step 5 방식 | Step 7 방식 |
|------|------------|------------|
| 모델 위치 | EC2 로컬 폴더 | 이미지 안에 포함 |
| 서버 실행 | `-v`로 폴더 연결 필요 | 이미지만 있으면 바로 실행 |
| 다른 서버 배포 | 모델 파일도 복사해야 함 | 이미지 pull만 하면 끝 |
| 실무 적합성 | 개발/테스트용 | **프로덕션 배포용** |

**핵심 차이:**
- Step 5: 모델이 "밖에" 있고 연결해서 씀
- Step 7: 모델이 "안에" 들어있어서 어디서든 바로 실행

이미지 안에 모델이 포함되어 있으면, 새로운 서버에서 `docker pull` 한 번으로 바로 서비스를 시작할 수 있습니다.

## 📝 Dockerfile 분석

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Triton model repository copy
COPY model_repository /models

CMD ["tritonserver","--model-repository=/models"]
```

### 각 명령어 설명

| 명령어 | 의미 | 상세 |
|--------|------|------|
| `FROM` | 베이스 이미지 | 이 이미지 위에 추가 작업 |
| `COPY` | 파일 복사 | 호스트 → 이미지 내부 |
| `CMD` | 기본 실행 명령 | 컨테이너 시작 시 실행할 것 |

### 이미지 레이어 개념

```
┌─────────────────────────────────────────────┐
│ 최종 이미지: triton-lab:latest              │
│                                             │
│  Layer 3: CMD ["tritonserver", ...]         │ ← 새로 추가
│  ─────────────────────────────────────────  │
│  Layer 2: COPY model_repository /models     │ ← 새로 추가
│  ─────────────────────────────────────────  │
│  Layer 1: FROM tritonserver:24.10-py3       │ ← 기존 (재사용)
│           (수십 개의 레이어)                │
│  ─────────────────────────────────────────  │
│  Base: ubuntu                               │
└─────────────────────────────────────────────┘
```

## 🏗️ 빌드 과정

```bash
docker build -t triton-lab:latest .
```

```
┌─────────────────────────────────────────────────────────────┐
│ docker build -t triton-lab:latest .                         │
│                                                             │
│ 1. Dockerfile 읽기                                          │
│    ↓                                                        │
│ 2. FROM: 베이스 이미지 확인 (있으면 재사용)                  │
│    ↓                                                        │
│ 3. COPY: model_repository 복사                              │
│    ↓                                                        │
│ 4. CMD: 메타데이터 기록                                      │
│    ↓                                                        │
│ 5. 이미지 생성: triton-lab:latest                           │
│    ↓                                                        │
└─────────────────────────────────────────────────────────────┘
```

| 옵션 | 의미 |
|------|------|
| `-t triton-lab:latest` | tag 이름 지정 (이미지명:태그) |
| `.` | 빌드 컨텍스트 (현재 디렉토리) |

## 🏷️ 태그 & 푸시

```bash
# 태그: 로컬 이미지에 ECR 주소를 붙임
docker tag triton-lab:latest 075176197136.dkr.ecr.ap-northeast-2.amazonaws.com/triton-lab:latest

# 푸시: ECR로 업로드
docker push 075176197136.dkr.ecr.ap-northeast-2.amazonaws.com/triton-lab:latest
```

> [!WARNING]
> **⏰ 이 명령어는 시간이 오래 걸립니다! (5~15분)**
> 
> Triton 베이스 이미지가 약 4~5GB에 달하기 때문에 업로드에 상당한 시간이 소요됩니다.
> 진행 막대(Bar)가 천천히 올라가도 정상이니 끝까지 기다려 주세요!

```
태그 과정:

triton-lab:latest
        │
        │ docker tag
        ▼
075176197136.dkr.ecr.ap-northeast-2.amazonaws.com/triton-lab:latest
│                                                │
└── 이 주소로 푸시하면 ECR에 저장됨! ──────────────┘
```

### ⏰ 실습 중 가장 오래 걸리는 두 단계

| 순위 | 단계 | 명령어 | 소요 시간 | 이유 |
|:---:|------|--------|:-------:|------|
| **1위** 🥇 | **Step 7: Push** | `docker push ECR주소/triton-lab:latest` | **5~15분** | EC2 → ECR로 4GB+ 업로드 |
| **2위** 🥈 | **Step 8: Pull** | `docker pull ECR주소/triton-lab:latest` | **5~15분** | ECR → EC2 #2로 4GB+ 다운로드 |

> [!TIP]
> 이 두 단계에서 시간이 오래 걸려도 당황하지 마세요!
> 실무에서도 대용량 AI 이미지를 전송할 때는 항상 이 정도 시간이 소요됩니다.

---

## 🏬 가게 안 vs 가게 밖 (최종 정리)

강사님의 비유를 실무 관점에서 다시 정리해 드리겠습니다. 사용자님이 방금 하신 방식이 왜 맞는지 명확해질 겁니다!

| 구분 | **가게 안 (Inside)** | **가게 밖 (Outside)** |
|------|-----------------------|-----------------------|
| 터미널 | EC2 SSH 터미널 (`ubuntu@ip-...`) | **내 컴퓨터 터미널 (`vasya@Lee`)** |
| 주소 | `localhost` 또는 `127.0.0.1` | **`52.78.232.142` (Public IP)** |
| 테스트 대상 | **주방 시설 (Triton 소프트웨어)** | **가게 전체 (네트워크 + 보안 그룹)** |
| 비유 | 요리사가 주방 불이 켜졌인지 확인 | **손님이 길거리에서 가게 문을 열어봄** |
| **결론** | 작동 여부만 확인 (기초 테스트) | **진짜 서비스 가능 여부 확인 (최종 테스트)** |

### 💡 왜 "가게 밖" 테스트가 진짜인가요?
가게 안(`localhost`)에서 테스트하면 **AWS 보안 그룹(방화벽)**이 막혀 있어도 성공으로 나옵니다. 하지만 손님은 밖에서 오기 때문에, 반드시 **내 컴퓨터에서 서버 IP로** 요청을 날려봐야 "진짜로 배포가 됐다"라고 말할 수 있습니다.

> [!IMPORTANT]
> **사용자님은 이미 "가게 밖"에서 Public IP로 테스트를 잘 하셨습니다!** 
> 8002번 포트(Metrics)든 8000번 포트(Health)든, 내 컴퓨터에서 서버 IP를 쳐서 응답이 왔다면 이미 **네트워크 대문이 완벽하게 열려있다**는 것을 증명하신 겁니다.

---

---

## 🚀 Part 2: EC2 #2 - 프로덕션 서버 작업 (Step 8~10)

> ✅ **이 섹션(Step 8~10)은 모두 EC2 #2 (프로덕션 서버)에서 실행합니다.**
> 개발 서버에서 ECR에 Push한 이미지를 여기서 Pull 받아 실제 서비스를 시작합니다.

---

# 11. Step 8-10: ECR Pull & 성능 측정 `EC2 #2`

## 🎯 이 단계에서 하는 것

**새로운 EC2 서버에서 ECR의 이미지를 받아 Triton을 실행하고, 외부에서 API를 호출하고, 성능을 측정**합니다.

```
EC2 #2 (새 서버)
    │
    ├── 1) aws configure + ECR 로그인
    ├── 2) docker pull (ECR에서 이미지 다운로드)
    ├── 3) docker run (Triton 실행)
    │
    └── 클라이언트에서 접속
            │
            ├── curl로 추론 요청
            └── ab로 성능 측정 (Throughput, Latency)
```

## ❓ 왜 필요한가?

이 단계가 **실무에서 가장 중요한 단계**입니다!

| 단계 | EC2 #1에서 한 것 | EC2 #2에서 하는 것 |
|------|-----------------|-------------------|
| 개발 | 모델 준비, 테스트 | - |
| 빌드 | 이미지 생성, ECR 푸시 | - |
| **배포** | - | **이미지 다운로드, 서버 실행** |
| **검증** | - | **외부에서 API 호출 테스트** |
| **운영** | - | **성능 측정, 모니터링** |

**EC2 #1 → ECR → EC2 #2 흐름이 바로 CI/CD 파이프라인의 기본 패턴**입니다.
실무에서는 이 과정이 자동화되어, 코드 푸시만 하면 자동으로 배포됩니다.

---

## 🏭 개발 서버 vs 프로덕션 서버

### 왜 서버를 두 개 쓸까?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   EC2 #1 (개발 서버, "serving")          EC2 #2 (프로덕션 서버, "serving-p") │
│   ────────────────────────────           ──────────────────────────────────  │
│                                                                              │
│   • 모델 실험, 테스트                    • 실제 사용자에게 서비스            │
│   • 자주 껐다 켰다 함                    • 24시간 안정적으로 가동            │
│   • 실수해도 괜찮음                      • 실수하면 서비스 장애!             │
│   • 개발자만 접근                        • 사용자가 접근                     │
│                                                                              │
│   여기서 빌드 완료 후 삭제 가능          이 서버가 실제로 돈을 버는 서버     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 실무에서의 예시

| 회사 | 개발 서버 | 프로덕션 서버 |
|------|----------|--------------|
| 네이버 | 사내망에서만 접속 가능 | naver.com에서 서비스 |
| 카카오 | 개발자 테스트용 | kakao.com에서 서비스 |
| 쿠팡 | 신규 기능 테스트 | 실제 주문 처리 |

**오늘 실습에서:**
- EC2 #1 (`serving`): 개발 완료 후 **삭제**했습니다.
- EC2 #2 (`serving-p`): 프로덕션 서버로 지금 배포 중입니다.

---

## 🌍 ECR에서 이미지 가져오기 (지역 개념)

### "서울에서 부산으로 택배 보내기" 비유

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   서울 (EC2 #1)                    택배 창고 (ECR)                           │
│   ┌───────────┐                   ┌───────────┐                             │
│   │ 개발 서버 │ ───docker push──► │ 이미지    │                             │
│   │           │   "택배 보내기"   │ 저장됨    │                             │
│   └───────────┘                   └─────┬─────┘                             │
│                                         │                                    │
│                                         │ docker pull                        │
│                                         │ "택배 받기"                        │
│                                         ▼                                    │
│   부산 (EC2 #2)                   ┌───────────┐                             │
│   ┌───────────┐                   │ 프로덕션  │                             │
│   │ 프로덕션  │ ◄─────────────────│ 서버      │                             │
│   │ 서버      │   이미지 도착!    │           │                             │
│   └───────────┘                   └───────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**핵심 포인트:**
- EC2 #1과 EC2 #2는 **전혀 다른 컴퓨터**입니다.
- 두 서버가 같은 이미지를 쓸 수 있는 이유는 **ECR(창고)에서 공유**하기 때문입니다.
- 실무에서는 서울, 도쿄, 미국 등 **전 세계 어디서든** ECR에서 이미지를 가져올 수 있습니다.

---

## 🔐 Step 8-1: EC2 #2에서 ECR 로그인

### AWS 자격 증명 설정

새로운 서버에서도 "저 AWS 고객입니다"라고 신분증을 보여줘야 합니다.

```bash
aws configure
```

```
AWS Access Key ID [None]: AKIA...         ← 신분증 번호
AWS Secret Access Key [None]: xxxx...     ← 신분증 비밀번호
Default region name [None]: ap-northeast-2  ← 서울 리전
Default output format [None]:             ← 그냥 엔터
```

### ECR 로그인

```bash
aws ecr get-login-password --region ap-northeast-2 | \
docker login --username AWS --password-stdin 075176197136.dkr.ecr.ap-northeast-2.amazonaws.com
```

**이 명령어가 하는 일:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  aws ecr get-login-password                                                  │
│  └── AWS에게 "ECR 출입증 주세요" 요청                                        │
│              │                                                               │
│              ▼                                                               │
│  임시 비밀번호 발급 (12시간 유효)                                            │
│              │                                                               │
│              ▼ (파이프로 전달)                                               │
│                                                                              │
│  docker login --username AWS --password-stdin                                │
│  └── Docker에게 "이 비밀번호로 ECR에 로그인해줘"                             │
│              │                                                               │
│              ▼                                                               │
│  Login Succeeded ← 성공!                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📥 Step 8-2: 이미지 Pull (다운로드)

```bash
docker pull 075176197136.dkr.ecr.ap-northeast-2.amazonaws.com/triton-lab:latest
```

### 이 명령어가 하는 일

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ECR (창고)                          EC2 #2 (프로덕션 서버)                  │
│  ┌───────────────┐                  ┌───────────────┐                       │
│  │ triton-lab    │                  │               │                       │
│  │ :latest       │ ──── 다운로드 ──►│ /var/lib/     │                       │
│  │               │     (약 4GB)     │ docker/images │                       │
│  │ • 베이스이미지│                  │               │                       │
│  │ • 모델 파일   │                  │ 이미지 저장됨 │                       │
│  │ • 설정 파일   │                  │               │                       │
│  └───────────────┘                  └───────────────┘                       │
│                                                                              │
│  💡 시간이 좀 걸립니다 (네트워크 속도에 따라 5~15분)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**진행 상태 예시:**
```
latest: Pulling from triton-lab
a2abf6c4d29d: Downloading [====>                      ] 123.4MB/1.2GB
5f2dbab2b3fd: Download complete
68e92d11b04f: Downloading [========>                  ] 456.7MB/2.1GB
...
```

---

## 🚀 Step 8-3: Triton 서버 실행

```bash
docker run -d --gpus all --name triton-server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  075176197136.dkr.ecr.ap-northeast-2.amazonaws.com/triton-lab:latest
```

### EC2 #1과 다른 점

| 항목 | EC2 #1 (개발) | EC2 #2 (프로덕션) |
|------|--------------|------------------|
| 이미지 소스 | `nvcr.io/nvidia/tritonserver:24.10-py3` | `ECR/triton-lab:latest` |
| 모델 연결 | `-v ~/model_repository:/models` 필요 | **필요 없음** (이미지 안에 포함) |
| 모델 위치 | EC2 로컬 폴더 | 이미지 내부 `/models` |

**왜 -v 옵션이 없나요?**

이번에는 모델이 **이미지 안에 포함**되어 있기 때문입니다!

```
EC2 #1에서:
  -v 로컬폴더:/models  ← 외부에서 모델 연결

EC2 #2에서:
  이미지 안에 /models 폴더가 이미 있음  ← 연결 불필요
```

### 실행 확인

```bash
# 컨테이너 상태 확인
docker ps

# 로그 확인 (모델 로딩 성공 여부)
docker logs triton-server
```

**정상 로그 예시:**
```
I0127 04:30:00.123456 1 server.cc:592] 
+------------------+------+
|       Model      |Ready |
+------------------+------+
|    simple_onnx   |  OK  |
+------------------+------+

Started GRPCInferenceService at 0.0.0.0:8001
Started HTTPService at 0.0.0.0:8000
Started Metrics Service at 0.0.0.0:8002
```

---

## 📡 Step 9: 외부에서 API 호출 테스트

### 헬스체크 (서버 상태 확인)

**본인 컴퓨터 터미널 또는 다른 서버에서:**

```bash
curl -i http://52.78.232.142:8000/v2/health/ready
```

**성공 응답:**
```
HTTP/1.1 200 OK
Content-Length: 0
Content-Type: text/plain
```

### 실제 추론 요청 (완전한 예시)

MNIST 모델에 28x28 픽셀 이미지 데이터를 보내려면 **784개의 숫자**가 필요합니다.

```bash
# 간단한 테스트용 데이터 (모두 0으로 채운 검은 이미지)
curl -X POST http://52.78.232.142:8000/v2/models/simple_onnx/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "Input3",
      "shape": [1, 1, 28, 28],
      "datatype": "FP32",
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }]
  }'
```

**응답 예시:**
```json
{
  "model_name": "simple_onnx",
  "outputs": [{
    "name": "Plus214_Output_0",
    "datatype": "FP32",
    "shape": [1, 10],
    "data": [-0.12, 0.05, 0.23, -0.45, 0.67, -0.89, 1.23, 0.34, -0.56, 0.78]
  }]
}
```

> [!TIP]
> **결과 해석**: `data`의 10개 숫자는 0~9 각 숫자일 확률입니다. 가장 큰 값의 위치가 예측된 숫자입니다.
> 위 예시에서 `1.23`(인덱스 6)이 가장 크므로 → 모델이 **"6"**이라고 예측한 것!

---

## 📊 Step 10: 성능 측정

### ApacheBench로 부하 테스트

```bash
# 먼저 AB 도구 설치 (EC2에서)
sudo apt-get update && sudo apt-get install -y apache2-utils

# 성능 측정 시작 (총 200번, 동시 20개씩)
ab -n 200 -c 20 http://52.78.232.142:8000/v2/health/ready
```

| 옵션 | 의미 |
|------|------|
| `-n 200` | 총 200번 요청 |
| `-c 20` | 동시에 20개씩 |

### 오늘 실제 측정 결과 (2026-01-27)

```
Server Software:        
Server Hostname:        52.78.232.142
Server Port:            8000

Document Path:          /v2/health/ready
Document Length:        0 bytes

Concurrency Level:      20
Time taken for tests:   0.023 seconds
Complete requests:      200
Failed requests:        0
Total transferred:      9000 bytes
HTML transferred:       0 bytes
Requests per second:    8887.70 [#/sec] (mean)      ← 🔥 Throughput!
Time per request:       2.250 [ms] (mean)            ← 🔥 Latency!
Time per request:       0.113 [ms] (mean, across all concurrent requests)
Transfer rate:          390.57 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    1   0.4      1       2
Processing:     0    1   0.5      1       3
Waiting:        0    1   0.4      1       3
Total:          1    2   0.7      2       3

Percentage of the requests served within a certain time (ms)
  50%      2
  66%      3
  75%      3
  80%      3
  90%      3
  95%      3
  98%      3
  99%      3
 100%      3 (longest request)
```

### 결과 해석

| 항목 | 오늘 측정값 | 의미 | 판정 |
|------|-----------|------|:---:|
| **Requests per second** | **8887.70 [#/sec]** | 초당 약 8,900번 처리 | **최상! 🔥** |
| **Time per request** | **2.250 [ms]** | 응답까지 약 2ms | **초고속! ⚡** |
| **Failed requests** | **0** | 실패 없음 | **안정적 ✅** |
| **Max response** | **3 [ms]** | 가장 느린 건도 3ms | **일관성 ✅** |

### 좋은 성능의 기준

| 지표 | 🔥 최상 | ✅ 좋음 | ⚠️ 보통 | ❌ 나쁨 |
|------|--------|--------|--------|--------|
| Throughput | 5000+ req/sec | 1000-5000 req/sec | 100-1000 req/sec | <100 req/sec |
| Latency | <5ms | 5-20ms | 20-100ms | >100ms |

> [!NOTE]
> 오늘 테스트한 것은 간단한 Health Check 요청입니다.
> 실제 복잡한 딥러닝 추론(예: 이미지 분류) 시에는 Throughput이 낮아지고 Latency가 늘어납니다.
> 하지만 Triton 엔진 자체의 성능이 이 정도라면, 실제 모델 추론도 충분히 빠를 것입니다!

---

# 12. 트러블슈팅 완전 가이드

## 🚨 문제 1: SSH 접속 실패

**증상:**
```
Permission denied (publickey)
```

**원인과 해결:**
| 가능한 원인 | 해결 방법 |
|------------|----------|
| 키 파일 권한 문제 | `chmod 400 mykey.pem` |
| **WSL 권한 문제** | Windows 폴더(`/mnt/c/`)에서는 `chmod`가 안 먹힐 수 있음. 키를 WSL 내부(`~/`)로 복사 후 권한 설정 (아래 설명 참고) |
| 잘못된 사용자 이름 | Ubuntu AMI는 `ubuntu`, Amazon Linux는 `ec2-user` |
| 잘못된 키 파일 | EC2 생성 시 선택한 키와 동일한지 확인 |

### 💡 WSL 사용자라면? (Permissions 0555 에러 해결)

WSL 환경에서 `/mnt/c/` 경로에 있는 키 파일은 윈도우 파일 시스템 특성상 권한 변경이 되지 않을 수 있습니다. 이럴 때는 **키 파일을 WSL 전용 경로로 복사**해서 사용해야 합니다.

```bash
# 1. 키 파일을 WSL 홈 디렉토리로 복사
cp "q11-key.pem" ~/

# 2. 이동한 파일에 대해 권한 설정
chmod 400 ~/q11-key.pem

# 3. 새로운 경로의 키로 접속 시도
ssh -i ~/q11-key.pem ubuntu@<EC2-IP>
```

## 🚨 문제 2: Docker 권한 오류

**증상:**
```
Got permission denied while trying to connect to the Docker daemon socket
```

**해결:**
```bash
sudo usermod -aG docker $USER
# 그 다음 SSH 재접속 필요!
exit
ssh -i key.pem ubuntu@IP
```

## 🚨 문제 3: GPU 인식 안 됨

**증상:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**해결:**
```bash
# NVIDIA Container Toolkit 재설치
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 🚨 문제 4: AWS 인증 오류

**증상:**
```
InvalidClientTokenId: The security token included in the request is invalid
```

**원인:** Access Key가 잘못되었거나 비활성화됨

**해결:**
1. IAM 콘솔에서 새 Access Key 생성
2. 기존 키 삭제 (혼동 방지)
3. `aws configure`로 새 키 입력

## 🚨 문제 5: Triton 모델 로딩 실패

**증상:**
```
model 'simple_onnx' is not ready
```

**원인과 해결:**
| 가능한 원인 | 해결 방법 |
|------------|----------|
| config.pbtxt의 input/output 이름 불일치 | 모델 분석해서 정확한 이름 사용 |
| 폴더 구조 오류 | `model_repository/모델명/버전/model.onnx` 확인 |
| config.pbtxt 문법 오류 | 콤마, 따옴표, 대괄호 확인 |

## 🚨 문제 6: curl에서 "connection refused" 또는 타임아웃

**증상:**
```
curl: (7) Failed to connect to 52.78.232.142 port 8000: Connection refused
```
또는 명령어가 한참 멈췄다가 타임아웃

**원인:** **보안 그룹(Security Group)**에서 해당 포트가 열려 있지 않음

**해결:**
1. AWS 콘솔 → EC2 → 인스턴스 선택 → **보안** 탭
2. 보안 그룹 클릭 → **인바운드 규칙 편집**
3. 필요한 포트 추가:

| 유형 | 포트 범위 | 소스 |
|------|----------|------|
| 사용자 지정 TCP | 8000 | 0.0.0.0/0 |
| 사용자 지정 TCP | 8001 | 0.0.0.0/0 |
| 사용자 지정 TCP | 8002 | 0.0.0.0/0 |

## 🚨 문제 7: SSH 접속 시 "Connection refused"

**증상:**
```
ssh: connect to host 52.78.xxx.xxx port 22: Connection refused
```

**원인과 해결:**
| 가능한 원인 | 해결 방법 |
|------------|----------|
| 보안 그룹에 SSH(22) 미오픈 | 인바운드 규칙에 22번 포트 추가 |
| EC2가 아직 부팅 중 | 몇 분 기다린 후 재시도 |
| EC2가 중지 상태 | AWS 콘솔에서 "인스턴스 시작" 클릭 |

## 🚨 문제 8: curl 명령 후 "curl: (23) Failed writing body"

**증상:**
```
curl: (23) Failed writing body
```

**원인:** 이건 **에러가 아닙니다!**

`curl ... | head -n 20` 처럼 데이터 일부만 읽고 파이프를 닫아버리면, curl이 "아직 보내야 할 데이터가 남았는데 상대방이 문을 닫았다"고 불평하는 것입니다.

**해결:** 무시해도 됩니다. 또는 `| head` 없이 전체 결과를 받으세요.

```bash
# 에러 없이 전체 결과 보기
curl -i http://52.78.232.142:8002/metrics
```

## 🚨 문제 9: docker run 후 컨테이너가 바로 종료됨

**증상:**
```bash
docker ps  # 아무것도 안 보임
docker ps -a  # STATUS가 "Exited (1)" 등으로 표시
```

**원인과 해결:**
1. 로그 확인: `docker logs 컨테이너이름`
2. 흔한 원인:
   - GPU 드라이버 문제: NVIDIA Toolkit 재설치
   - 모델 경로 오류: `-v` 옵션으로 마운트한 경로 확인
   - 포트 충돌: 이미 8000/8001/8002를 쓰는 프로세스가 있으면 충돌

# 13. 실무에서는 어떻게 쓰이나?

## 🏢 실제 프로덕션 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        실제 프로덕션 환경                           │
│                                                                      │
│   사용자 → Load Balancer → [EC2 #1] ──┐                             │
│                            [EC2 #2] ──┼── 오토스케일링              │
│                            [EC2 #3] ──┘   (부하에 따라 자동 확장)   │
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  ECR        │    │  S3         │    │  CloudWatch │             │
│   │  이미지저장 │    │  모델저장   │    │  모니터링   │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
│   ┌─────────────────────────────────────────────────────┐           │
│   │  GitHub Actions / Jenkins                            │           │
│   │  코드 푸시 → 자동 빌드 → 자동 배포 (CI/CD)          │           │
│   └─────────────────────────────────────────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 📈 오늘 배운 것의 가치

| 오늘 실습 | 실무 적용 |
|----------|----------|
| EC2 수동 생성 | Terraform/CloudFormation으로 자동화 |
| 수동 SSH 접속 | Bastion Host + SSM Session Manager |
| 수동 docker build | CI/CD 파이프라인 (GitHub Actions, Jenkins) |
| 단일 EC2 | Kubernetes (EKS) + Auto Scaling |
| curl 테스트 | 자동화된 통합 테스트 |

---

# 14. 심화: 멀티 모델 & 파이프라인 구성

실무에서는 하나의 Triton 서버에서 **여러 모델을 동시에 서빙**하거나, **모델을 연결해서 파이프라인**을 구성하는 경우가 많습니다.

## 📁 멀티 모델 폴더 구조

```
model_repository/
├── image_classifier/           ← 이미지 분류 모델
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
│
├── speech_to_text/             ← 음성 인식 모델
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
│
└── llm_model/                  ← LLM 모델
    ├── config.pbtxt
    └── 1/
        └── model.plan          # TensorRT 형식
```

**핵심: 각 모델마다 자기만의 `config.pbtxt`가 있습니다!**

## 📝 각 모델별 config.pbtxt 예시

### 1. 이미지 분류 모델
```protobuf
name: "image_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]  # RGB, 224x224
  }
]

output [
  {
    name: "class_probs"
    data_type: TYPE_FP32
    dims: [ 1000 ]  # ImageNet 1000 클래스
  }
]
```

### 2. 음성 인식 모델
```protobuf
name: "speech_to_text"
platform: "onnxruntime_onnx"
max_batch_size: 1

input [
  {
    name: "audio_signal"
    data_type: TYPE_FP32
    dims: [ -1 ]  # -1 = 가변 길이
  }
]

output [
  {
    name: "transcription"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
```

### 3. LLM 모델
```protobuf
name: "llm_model"
platform: "tensorrt_llm"
max_batch_size: 4

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]  # 가변 토큰 길이
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

# LLM 특화 설정: GPU 인스턴스 지정
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

## 🔗 모델 파이프라인 (Ensemble)

**"이미지 → 캡션 생성 → LLM으로 답변"** 같이 **모델을 연결**하려면 **Ensemble Model**을 사용합니다.

```
model_repository/
├── image_encoder/              ← 실제 모델 1
├── caption_generator/          ← 실제 모델 2
├── llm_model/                  ← 실제 모델 3
└── full_pipeline/              ← Ensemble (가상 모델, 연결만 정의)
    └── config.pbtxt
```

### Ensemble config.pbtxt 예시

```protobuf
name: "full_pipeline"
platform: "ensemble"
max_batch_size: 1

ensemble_scheduling {
  step [
    {
      model_name: "image_encoder"
      model_version: -1
      input_map {
        key: "image"           # Ensemble 입력 이름
        value: "INPUT_IMAGE"   # 실제 모델 입력 이름
      }
      output_map {
        key: "features"        # 실제 모델 출력 이름
        value: "image_features" # 다음 단계에서 쓸 이름
      }
    },
    {
      model_name: "caption_generator"
      model_version: -1
      input_map {
        key: "image_features"  # 이전 단계 출력
        value: "image_features"
      }
      output_map {
        key: "caption"
        value: "generated_caption"
      }
    },
    {
      model_name: "llm_model"
      model_version: -1
      input_map {
        key: "prompt"
        value: "generated_caption"
      }
      output_map {
        key: "response"
        value: "FINAL_RESPONSE"  # 최종 출력
      }
    }
  ]
}
```

### 📊 Ensemble 작동 원리

```
┌─────────────────────────────────────────────────────────────┐
│  클라이언트 요청: /v2/models/full_pipeline/infer            │
│  입력: 이미지                                                │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: image_encoder                                       │
│  이미지 → 특징 벡터 (512차원)                                │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: caption_generator                                   │
│  특징 벡터 → "A cat sitting on a couch"                      │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: llm_model                                           │
│  캡션 → "이 이미지는 소파에 앉아있는 고양이를 보여줍니다..."  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  클라이언트 응답: 최종 텍스트                                │
└─────────────────────────────────────────────────────────────┘
```

## 📡 호출 방식

```bash
# 개별 모델 호출 (각각 따로)
curl http://localhost:8000/v2/models/image_classifier/infer -d '...'
curl http://localhost:8000/v2/models/speech_to_text/infer -d '...'
curl http://localhost:8000/v2/models/llm_model/infer -d '...'

# 파이프라인(Ensemble) 호출 - 한 번에 전체 흐름 실행!
curl http://localhost:8000/v2/models/full_pipeline/infer -d '...'
```

## 💡 멀티 모델 요약

| 상황 | 구성 방법 |
|------|----------|
| 여러 모델을 **개별적**으로 서빙 | 각 폴더에 각자의 `config.pbtxt` |
| 모델을 **순차적으로 연결** | Ensemble Model (`platform: "ensemble"`) |
| GPU 메모리 분배 | `instance_group`에서 count/kind 설정 |
| 모델 버전 관리 | 폴더 `1/`, `2/`, `3/`로 버전 구분 |

---

# 📌 마무리: 핵심 정리

## 🎯 오늘 배운 핵심 기술

1. **Docker**: 환경을 패키징해서 일관된 실행 환경 보장
2. **NVIDIA Container Toolkit**: Docker에서 GPU 사용 가능하게
3. **Triton**: 전문적인 AI 추론 서버
4. **Model Repository**: Triton이 모델을 인식하는 규칙
5. **AWS ECR**: Docker 이미지를 클라우드에 저장
6. **AWS CLI**: 터미널에서 AWS 서비스 제어

## 🔄 전체 흐름 한 눈에

```
모델 학습 → ONNX 변환 → Model Repository 구성 → Triton 실행 테스트
                                                        ↓
                                                   커스텀 이미지 빌드
                                                        ↓
                                                   ECR에 Push
                                                        ↓
                                               다른 EC2에서 Pull & Run
                                                        ↓
                                               외부에서 API 호출 테스트
                                                        ↓
                                                   성능 측정 & 최적화
```

---

# 15. 비용 방지: 리소스 삭제 (!!!!!!!!중요!!!!!!!!)

> [!CAUTION]
> **EC2, ECR을 삭제하지 않으면 계속 비용이 나갑니다!**
> 실습이 끝나면 **반드시** 아래 순서대로 리소스를 삭제하세요.

## 💸 왜 삭제해야 하나요?

AWS는 **사용한 만큼 비용을 청구**합니다. 실습이 끝나고 EC2가 계속 켜져 있거나, ECR에 이미지가 저장되어 있으면 **하루에 수천 원씩** 나갈 수 있습니다.

| 리소스 | 상태 | 예상 일일 비용 |
|--------|------|---------------|
| **EC2 g4dn.xlarge** | 켜져 있음 | **약 $12~15 (15,000원~20,000원)** |
| **EC2 g4dn.xlarge** | 중지(Stop) | 약 $0.5~1 (EBS 스토리지만) |
| **ECR 이미지** | 저장됨 | 약 $0.1~0.5 (용량에 따라 다름) |

> [!WARNING]
> **EC2를 "중지(Stop)"해도 스토리지 비용은 나갑니다!**
> 완전히 비용을 피하려면 **"종료(Terminate)"**를 해야 합니다.

---

## 🗑️ 1단계: EC2 인스턴스 종료 (Terminate)

### AWS 콘솔에서 삭제하기

1. **AWS 콘솔** 접속: [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)
2. 왼쪽 메뉴에서 **"인스턴스"** 클릭
3. 삭제할 인스턴스 선택 (`serving-p` 또는 `serving`)
4. 상단 메뉴에서 **"인스턴스 상태"** → **"인스턴스 종료"** 클릭
5. 확인 팝업에서 **"종료"** 클릭

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   AWS EC2 콘솔                                                               │
│   ─────────────                                                              │
│                                                                              │
│   ☑ serving-p        g4dn.xlarge    running    52.78.232.142                │
│                                                                              │
│   [인스턴스 상태 ▼]                                                          │
│        │                                                                     │
│        ├── 인스턴스 중지 (Stop) ← 잠시 멈춤, 비용 일부 발생                   │
│        │                                                                     │
│        └── 인스턴스 종료 (Terminate) ← 완전 삭제, 비용 0원 ✅                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **"종료(Terminate)"와 "중지(Stop)"는 다릅니다!**
> - **중지(Stop)**: 컴퓨터를 끈 것. 언제든 다시 켤 수 있지만, 하드디스크(EBS) 비용은 계속 나갑니다.
> - **종료(Terminate)**: 컴퓨터를 반납한 것. 완전히 사라지고, 비용이 0원이 됩니다.

---

## 🗑️ 2단계: ECR 이미지 삭제

### AWS 콘솔에서 삭제하기

1. **AWS 콘솔** 접속: [https://console.aws.amazon.com/ecr](https://console.aws.amazon.com/ecr)
2. 왼쪽 메뉴에서 **"리포지토리"** 클릭
3. `triton-lab` 레포지토리 선택
4. 안에 있는 **이미지 전체 선택** 후 **"삭제"** 클릭
5. (선택) 레포지토리 자체도 삭제하려면, 레포지토리 이름 옆 체크박스 선택 후 **"삭제"**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   AWS ECR 콘솔                                                               │
│   ────────────                                                               │
│                                                                              │
│   레포지토리: triton-lab                                                     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  이미지 태그    │  크기      │  푸시 시간                       │      │
│   │  latest         │  4.2 GB    │  2026-01-27 12:30                │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   [☑ 전체 선택]  [삭제] ← 이미지 삭제                                       │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│   [☑ triton-lab]  [삭제] ← 레포지토리 자체 삭제                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗑️ 3단계: (선택) IAM 사용자 & Access Key 삭제

실습에서 만든 IAM 사용자와 Access Key도 삭제하면 보안상 더 안전합니다.

1. **AWS 콘솔** 접속: [https://console.aws.amazon.com/iam](https://console.aws.amazon.com/iam)
2. 왼쪽 메뉴에서 **"사용자"** 클릭
3. 실습용으로 만든 사용자 선택 후 **"삭제"**

---

## ✅ 삭제 체크리스트

실습이 끝난 후 아래 항목을 모두 확인하세요:

| 리소스 | 삭제 완료 |
|--------|:--------:|
| EC2 인스턴스 (`serving-p`) 종료 | ☐ |
| EC2 인스턴스 (`serving`) 종료 (이미 했다면 ✓) | ☐ |
| ECR 이미지 (`triton-lab:latest`) 삭제 | ☐ |
| ECR 레포지토리 (`triton-lab`) 삭제 | ☐ |
| (선택) IAM 사용자 삭제 | ☐ |

> [!TIP]
> **나중에 다시 실습하고 싶다면?**
> ECR 레포지토리는 비어있으면 거의 비용이 안 나니, 레포지토리만 남겨두고 이미지만 삭제해도 됩니다.
> EC2는 필요할 때 다시 만들면 되니 종료(Terminate)하는 것이 가장 경제적입니다.

---

> 💡 **마지막으로 한번 더!**
> AWS 비용은 **켜놓고 잊어버리면** 청구서가 날아옵니다. 
> 반드시 **실습이 끝난 직후 삭제**하시고, 다음 날 AWS 콘솔에 들어가서 리소스가 남아있지 않은지 한 번 더 확인하세요!
> 
