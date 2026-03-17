"""
🎯 FastAPI Lab 0 과제: Lifespan 패턴 마스터 (혼자해보기)

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **Lab 0의 과제 1, 2, 3을 모두 구현한 완성본**입니다.
FastAPI Lifespan 패턴을 활용하여 다음 3가지 개선사항을 포함합니다:

**과제 1: 다중 모델 로딩**
- 두 개의 가짜 모델(sentiment, translator)을 동시에 로딩
- 각 모델별로 독립적인 엔드포인트 제공

**과제 2: 모델 로딩 시간 측정**
- time 모듈로 로딩 시작/종료 시간 측정
- /health 엔드포인트에서 로딩 시간 정보 제공

**과제 3: 환경 설정 분리**
- .env 파일에서 MODEL_LOAD_TIME 환경변수 로드
- 환경변수가 없으면 기본값(3초) 사용

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **다중 모델 관리의 필요성 (과제 1)**:
   - 실무에서는 여러 AI 모델을 동시에 서빙
   - 각 모델을 독립적으로 관리하면서도 효율적으로 공유
   - 딕셔너리 패턴으로 확장 가능한 구조 제공

2. **로딩 시간 측정의 중요성 (과제 2)**:
   - 서버 시작 시간 모니터링
   - 성능 튜닝의 기준점 제공
   - Health Check로 서비스 상태 확인

3. **환경변수 분리의 장점 (과제 3)**:
   - 개발/운영 환경 분리
   - 코드 수정 없이 설정 변경
   - 민감 정보(API 키 등) 보호 패턴 학습

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. .env 파일에서 MODEL_LOAD_TIME 로드 (없으면 3.0초)
2. Lifespan 시작: 시간 측정 시작
3. sentiment 모델 로딩 (MODEL_LOAD_TIME초 소요)
4. translator 모델 로딩 (MODEL_LOAD_TIME초 소요)
5. 총 로딩 시간 계산 및 저장
6. 로딩 완료 메시지 출력
7. yield로 서버 대기

[요청 처리]
1. /predict/sentiment: sentiment 모델만 사용
2. /predict/translate: translator 모델만 사용
3. /health: 로딩 시간 정보 반환

[서버 종료]
4. ml_models 딕셔너리 정리

===============================================
💡 핵심 학습 포인트
===============================================
- **딕셔너리 패턴**: 여러 모델을 하나의 저장소에서 관리
- **시간 측정**: time.time()으로 성능 모니터링
- **환경변수**: python-dotenv로 설정 외부화
- **Health Check**: 서비스 상태 확인 API 패턴

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install python-dotenv
   - .env 파일 생성 (선택):
     MODEL_LOAD_TIME=5

2. 실행 방법:
   python ./lab0/lifespan_practice.py

3. 테스트 예시 (curl):
   # Sentiment 예측
   curl "http://localhost:8000/predict/sentiment?text=좋아요"
   
   # 번역 예측
   curl "http://localhost:8000/predict/translate?text=안녕하세요"
   
   # Health Check
   curl "http://localhost:8000/health"
"""

#  모듈 임포트
# 1. from fastapi import FastAPI:
#    - sys.modules에서 'fastapi' 패키지 검색 (이미 로드되었다면 재사용)
#    - 없으면 sys.path에서 찾아 __init__.py 실행
#    - FastAPI 클래스를 현재 네임스페이스에 바인딩
# 2. from contextlib import asynccontextmanager:
#    - 표준 라이브러리 contextlib에서 asynccontextmanager 가져오기
# 3. import time:
#    - time 모듈 전체를 현재 네임스페이스에 바인딩
# 3. import time:
#    - time 모듈 전체를 현재 네임스페이스에 바인딩
# 4. import os, from dotenv import load_dotenv:
#    - 환경변수 접근 및 .env 파일 로드를 위한 모듈
from fastapi import FastAPI
from contextlib import asynccontextmanager
import time
import os
from dotenv import load_dotenv

#  환경변수 로드
# 1. load_dotenv():
#    - .env 파일을 찾아 그 내용을 os.environ에 로드
#    - 파싱된 KEY=VALUE 쌍을 프로세스 환경변수로 설정
load_dotenv()

#  전역 상수 설정
# 1. os.getenv("MODEL_LOAD_TIME", 3):
#    - 환경변수에서 "MODEL_LOAD_TIME" 값을 문자열로 가져옴 (없으면 기본값 3)
# 2. float(...):
#    - 문자열 "5"를 부동소수점 5.0으로 변환
#    - 프로그램 전체에서 사용할 상수로 바인딩
MODEL_LOAD_TIME = float(os.getenv("MODEL_LOAD_TIME", 3))

# [이 코드의 역할] 가짜 모델 로드 함수
# - 실제 ML 모델 로딩을 시뮬레이션하기 위해 3초 대기
#  함수 정의
# 1. def 키워드:
#    - 함수 객체 생성 (코드, 기본값, 클로저 포함)
#    - 전역 네임스페이스에 'load_model' 이름으로 바인딩
def load_model():
    #  시스템 호출
    # 1. time.sleep(3):
    #    - C 구현 sleep() 시스템 콜 호출
    #    - 현재 스레드를 OS 대기 큐로 이동
    #    - 3초 후 스레드 깨우기 (인터럽트)
    time.sleep(MODEL_LOAD_TIME)
    #  딕셔너리 반환
    # 1. {"model": "fake-ml-model"} 평가:
    #    - 힙 메모리에 dict 객체 할당 (약 240 bytes + 엔트리)
    #    - 해시 테이블 초기화 (8개 버킷)
    #    - ("model", "fake-ml-model") 엔트리 저장
    # 2. return:
    #    - dict 객체의 메모리 주소를 호출자에게 반환
    return {"model": "fake-ml-model"}

# [이 코드의 역할] 전역 모델 저장소
# - 서버가 로드한 모델들을 저장하는 딕셔너리
# - 모든 요청이 이 딕셔너리에서 모델을 가져다 쓴
#  전역 변수 초기화
# 1. {} 리터럴:
#    - 빈 dict 객체를 힙에 할당
#    - 해시 테이블 초기화 (8버킷)
# 2. ml_models 바인딩:
#    - 전역 네임스페이스(__main__.__dict__)에 엔트리 추가
#    - 프로그램 종료까지 메모리에 유지
ml_models = {}

# [이 코드의 역할] 서버 수명 주기 관리
# - 서버 시작 시: 모델들을 로드
# - 서버 종료 시: 메모리 정리
#  데코레이터 및 함수 정의
# 1. @asynccontextmanager:
#    - asynccontextmanager(lifespan) 호출
#    - lifespan 함수를 async context manager로 래핑
#    - __aenter__, __aexit__ 메서드 자동 생성
# 2. async def:
#    - 코루틴 함수 정의 (호출 시 코루틴 객체 반환)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시
    start_time = time.time()
    print("Model Loading...")
    
    # [이 코드의 역할] 다중 모델 로드 (과제 핀트!)
    # - sentiment 모델과 translator 모델 2개를 로드
    #  두 모델 로딩
    # 1. load_model() 호출 (#1):
    #    - 새 스택 프레임 생성 → 3초 블로킹 → dict 반환
    # 2. ml_models["sentiment"] 할당:
    #    - "sentiment" 해시값 계산 → 버킷 찾기 → 엔트리 저장
    #  sentiment 모델 로딩
    # 3. load_model() 호출 (#1):
    #    - 새 스택 프레임 생성 → 5초 블로킹(.env 설정값) → dict 반환
    # 4. ml_models["sentiment"] 할당:
    #    - "sentiment" 문자열의 해시값 계산
    #    - 해시 충돌 처리 후 해시 테이블의 빈 버킷에 (key, value) 참조 저장
    ml_models["sentiment"] = load_model()
    #  translator 모델 로딩
    # 5. load_model() 호출 (#2):
    #    - 또 5초 블로킹 (총 10초 소요 예상)
    # 6. ml_models["translator"] 할당:
    #    - "translator" 해시값 계산 → 버킷 찾기 → 엔트리 저장
    ml_models["translator"] = load_model()
    #  모델 로딩 시간 측정 완료
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ 모델 로딩 완료! (소요시간: {elapsed_time:.2f}초)")
    
    #  로딩 시간 기록
    # - 공용 딕셔너리에 "loading_time" 키로 저장
    ml_models["loading_time"] = elapsed_time

    #  제어권 반환
    # 1. yield:
    #    - 코루틴 실행 일시 정지
    #    - 스택 프레임(로컬 변수) 저장
    #    - FastAPI에게 제어권 전달
    # 2. 서버 Running:
    #    - uvicorn이 HTTP 요청 수신 시작
    # 3. Ctrl+C 시:
    #    - yield 다음 줄부터 실행 재개
    yield
    
    # [Shutdown] 서버 종료 시
    #  메모리 정리
    # 1. ml_models.clear():
    #    - 딕셔너리의 모든 키-값 엔트리 제거
    #    - 각 값에 대한 참조 카운트 감소
    #    - 가비지 컬렉터가 메모리 회수
    ml_models.clear()

# [이 코드의 역할] FastAPI 애플리케이션 생성
# - lifespan 관리자를 등록한 FastAPI 앱 인스턴스 생성
#  앱 인스턴스화
# 1. FastAPI() 생성자:
#    - FastAPI 클래스 __init__ 메서드 실행
#    - Starlette 상속 및 초기화
#    - 라우팅 테이블 초기화
# 2. lifespan=lifespan:
#    - lifespan 함수를 app._lifespan 속성에 저장
#    - uvicorn 실행 시 자동 호출
# 3. app 변수 바인딩:
#    - FastAPI 인스턴스를 전역 변수 app에 저장
app = FastAPI(lifespan=lifespan)

# [이 코드의 역할] API 엔드포인트 등록
# - GET /predict 경로로 요청을 받을 함수
#  라우트 등록
# 1. @app.get("/predict"):
#    - app.get 데코레이터 평가
#    - FastAPI 라우팅 테이블에 등록
#    - 경로: "/predict", 메서드: GET
@app.get("/predict/sentiment")
def predict(text: str):
    # [이 코드의 역할] sentiment 모델 사용
    #  모델 조회 및 응답 생성
    # 1. ml_models에서 "sentiment" 키로 모델 접근
    # 2. 간단한 응답 딕셔너리 생성 및 반환
    model = ml_models["sentiment"]
    return {
        "input": text,
        "prediction": "positive",
        "sentiment_model": model["model"]
    }

#  번역 엔드포인트 정의
# 1. @app.get(...): 라우팅 테이블에 "/predict/translate" 등록
# 2. def translate(text: str):
#    - 요청 쿼리 파라미터 "?text=..."를 파싱하여 인자로 전달
@app.get("/predict/translate")
def translate(text: str):
    #  모델 조회
    # 1. ml_models["translator"]:
    #    - 전역 딕셔너리에서 "translator" 키 해시 조회
    #    - 힙 메모리에 있는 모델 객체의 참조를 스택(model 변수)에 복사
    model = ml_models["translator"]
    
    #  응답 데이터 생성
    # 1. 리터럴 {...}:
    #    - 새로운 dict 객체 생성 및 반환
    #    - FastAPI가 이를 JSON으로 직렬화(Serialization)하여 HTTP Body에 작성
    return {
        "input": text,
        "translated_text": "Hello (Dummy Translation)",
        "translator_model": model["model"]
    }

@app.get("/health")
def health_check():
    #  로딩 시간 조회
    return {
        "status": 200,
        "message": "OK",
        "loading_time": f"{ml_models.get('loading_time', 0.0):.2f}초"
    }