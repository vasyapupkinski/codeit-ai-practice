"""
🎯 FastAPI 실습: Lifespan 

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **FastAPI의 Lifespan 이벤트를 사용하여 서버 시작/종료 시점에 코드를 실행하는** 핵심 패턴을 보여줍니다.
AI 모델 같은 무거운 리소스를 "딱 한 번만" 로딩하고, 모든 요청이 그것을 공유하게 만드는 "Singleton 패턴"의 실전 예시입니다.

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **문제: 요청마다 모델을 로딩하면?**
   - 모델 로딩: 3초 (실제로는 수십 초 ~ 수 분)
   - 사용자가 요청할 때마다 매번 3초씩 기다려야 함
   - 동시에 100명이 요청하면? → 서버 폭발

2. **해결: 서버 시작 시 한 번만 로딩**
   - 서버 시작 시 lifespan 함수가 자동 실행
   - 모델을 전역 변수(ml_models)에 저장
   - 이후 모든 요청은 이미 로딩된 모델을 즉시 사용 (0초)

3. **Singleton 패턴이란?**
   - 프로그램 전체에서 "단 하나의 인스턴스"만 존재
   - 모든 곳에서 그 하나를 공유
   - 메모리 절약 + 성능 향상

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. uvicorn 실행
2. lifespan 함수의 yield 윗부분 실행
   → load_model() 호출 (3초 소요)
   → ml_models["sentiment"]에 저장
3. yield에서 대기 (서버 Running 상태)

[요청 처리: /predict 엔드포인트]
1. 클라이언트 요청
2. ml_models["sentiment"] 조회 (0초, O(1))
3. 즉시 결과 반환

[요청 처리: /bad 엔드포인트 (안티 패턴)]
1. 클라이언트 요청
2. load_model() 다시 호출 (3초 대기!)
3. 결과 반환

[서버 종료]
1. Ctrl+C
2. lifespan 함수의 yield 아래 실행
   → ml_models.clear() (메모리 정리)

===============================================
💡 핵심 학습 포인트
===============================================
- Lifespan: 서버 생명주기 관리의 핵심
- 전역 변수: 모든 요청이 공유하는 데이터 저장소
- yield: 함수 실행을 "일시 정지"하는 마법
- O(1) 조회: 딕셔너리 해시 테이블의 성능
- 안티 패턴 비교: "/bad" 엔드포인트로 잘못된 방식 체험

===============================================
📌 실행 방법
===============================================
uvicorn lab0.lifespan_test:app --reload
"""

#  필수 모듈 임포트
# https://fastapi.tiangolo.com/advanced/events/
# 1. from fastapi import FastAPI:
#    - 파이썬 인터프리터가 sys.modules에서 'fastapi' 패키지를 검색
#    - 없으면 sys.path에서 fastapi 폴더를 찾아 __init__.py 실행
#    - FastAPI 클래스 정의를 현재 네임스페이스(__main__)에 바인딩
# 2. from contextlib import asynccontextmanager:
#    - 표준 라이브러리 contextlib에서 asynccontextmanager 함수를 가져옴
#    - 이 데코레이터는 async generator를 컨텍스트 매니저로 변환
# 3. import time:
#    - time 모듈 전체를 현재 네임스페이스에 바인딩
#    - time.sleep 등의 함수에 접근 가능
from fastapi import FastAPI
from contextlib import asynccontextmanager
import time

# ===============================
# Fake model loader
# ===============================
#  모델 로딩 시뮬레이션 함수
# - def 키워드: 함수 객체를 생성하고 현재 네임스페이스에 'load_model' 이름으로 바인딩
# - 함수 객체에는 코드 객체(__code__), 클로저, 기본값 등이 포함됨
def load_model():
    #  3초 대기 (실제 모델 로딩 시뮬레이션)
    # 1. time.sleep(3) 호출:
    #    - C 구현된 sleep 시스템 콜 호출 (Windows: Sleep, Linux: nanosleep)
    #    - 현재 스레드를 OS 스케줄러의 대기 큐로 이동
    #    - 3초 후 타이머 인터럽트 발생 → 스레드를 실행 가능 상태로 전환
    # 2. CPU는 이 시간 동안 다른 작업 처리 가능 (비어있지 않음)
    # - 실제로는 수십 MB~수 GB 모델 파일을 디스크에서 읽어 메모리에 적재하는 시간
    time.sleep(3)
    #  가짜 모델 객체 반환
    # 1. 딕셔너리 리터럴 {"model": "fake-ml-model"} 평가:
    #    - 힙 메모리에 dict 객체 할당 (약 240 bytes + 엔트리 크기)
    #    - 해시 테이블 구조 초기화 (기본 8개 버킷)
    #    - "model" 문자열의 해시값 계산 (hash("model"))
    #    - 해시값을 버킷 인덱스로 변환 (hash % 8)
    #    - 해당 버킷에 ("model", "fake-ml-model") 엔트리 저장
    # 2. return: dict 객체의 메모리 주소를 호출자에게 반환
    return {"model": "fake-ml-model"}

# ===============================
# 전역 모델 저장소
# ===============================
#  전역 딕셔너리 초기화
# 1. {} 리터럴 평가:
#    - 힙 메모리에 빈 dict 객체 생성 (초기 크기: 약 240 bytes)
#    - 빈 해시 테이블 초기화 (8개 버킷, 로드 팩터 2/3)
# 2. 'ml_models' 이름에 dict 객체 참조 바인딩:
#    - 전역 네임스페이스(__main__.__dict__)에 엔트리 추가
#    - 모든 함수에서 global 키워드 없이 읽기 가능, 쓰기는 global 필요
# 3. 이 객체는 프로그램 종료까지 메모리에 유지됨 (가비지 컬렉션 대상 아님)
# - 빈 딕셔너리를 생성하여 모든 요청이 공유할 수 있는 저장소 마련
ml_models = {}  # global dict

# ===============================
# Lifespan: 서버 시작/종료 시 리소스 관리
# ===============================
# [개념] 왜 Lifespan 패턴을 쓰는가? (RAG와 동일!)
# - 문제: 요청마다 모델/DB를 로딩하면?
#   * 한 명 요청 → 3초 대기 (load_model 호출)
#   * 10명 동시 요청 → 10번 로딩 → 30초 대기
#   * CPU/메모리 낭비, 서버 폭발
# - 해결: Lifespan 패턴 (싱글턴)
#   * 서버 시작 시 한 번만 로딩
#   * ml_models 전역 변수에 저장
#   * 모든 요청이 동일한 모델 인스턴스 공유
#   * 요청 응답 시간: 0초 (O(1) 해시 타부 조회)
# - 실무 필수:
#   * DB 커넥션 풀 (SQLAlchemy, MongoDB)
#   * AI 모델 (트랜스포머, OpenAI 클라이언트)
#   * 레디스 커넥션, 캐시 등
# - 이 파일의 목적:
#   * /predict: 올바른 패턴 (lifespan 사용)
#   * /bad: 안티 패턴 (매번 로딩) → 생생한 뵄교 체험

#  데코레이터 적용
# 1. @asynccontextmanager:
#    - 데코레이터 평가: asynccontextmanager(lifespan) 호출
#    - asynccontextmanager는 GeneratorBasedContextManager 객체를 반환
#    - 이 객체는 __aenter__, __aexit__ 메서드를 가짐 (async with 지원)
# 2. async def:
#    - 코루틴 함수로 정의 (호출 시 코루틴 객체 반환)
#    - 함수 내부에서 await, async with 사용 가능
@asynccontextmanager                # ① 비동기 컨텍스트 매니저로 만들어주는 데코레이터
async def lifespan(app: FastAPI):
    #  Startup 단계
    # ② 서버 시작 시 실행되는 부분 (startup)
    print("====== 모델로딩중...")
    #  모델 로딩 및 저장
    # 1. load_model() 호출:
    #    - 함수 호출 → 새로운 스택 프레임 생성
    #    - time.sleep(3) 실행 → 3초 블로킹
    #    - dict 객체 생성 및 반환
    # 2. ml_models["sentiment"] = ...:
    #    - "sentiment" 해시값 계산
    #    - ml_models의 해시 테이블에서 버킷 찾기
    #    - load_model()이 반환한 dict 객체의 참조를 저장
    #    - 딕셔너리가 커지면 자동으로 리해싱 (2x 확장)
    # 3. 메모리 상태:
    #    - ml_models: 1개의 엔트리 보유
    #    - 참조 카운트: load_model() 반환 dict의 refcount = 1
    # - load_model() 호출: 3초간 블로킹 (실제 모델 로딩 시뮬레이션)
    # - 반환된 딕셔너리를 ml_models["sentiment"]에 저장
    # - 이제 모든 요청이 이 모델을 O(1) 시간에 조회 가능
    ml_models["sentiment"] = load_model()   # 모델을 메모리에 올림
    print("✅ 모델 로딩 완료")

    #  yield - 제어권 반납
    # ③ 여기서 "일시정지" → 서버가 요청을 받기 시작
    # - 코루틴의 실행이 여기서 멈춥니다.
    # - FastAPI가 HTTP 요청을 처리할 준비가 됩니다.
    # - 서버 종료 신호(Ctrl+C)가 올 때까지 여기서 대기
    yield

    #  Shutdown 단계
    # ④ 서버 종료 시 실행되는 부분 (shutdown)
    print("🧹 모델 메모리 정리")
    #  메모리 정리
    # - 딕셔너리를 비워서 모델 객체에 대한 참조를 제거
    # - 가비지 컬렉터가 나중에 메모리를 회수
    ml_models.clear()
    print("✅ 모델 메모리 정리 완료")

#  FastAPI 앱 생성
# 1. FastAPI() 생성자 호출:
#    - FastAPI 클래스의 __init__ 메서드 실행
#    - 내부에서 Starlette 앱 상속/초기화
#    - 라우팅 테이블 초기화 (APIRouter 객체 생성)
#    - 미들웨어 스택 초기화
# 2. lifespan=lifespan 파라미터:
#    - lifespan 함수를 FastAPI.용 _lifespan 속성에 저장
#    - uvicorn 시작 시에 이 함수 호출
# 3. 'app' 변수에 FastAPI 인스턴스 참조 바인딩
# - FastAPI(lifespan=lifespan): lifespan 함수를 서버 생명주기 관리자로 등록
app = FastAPI(lifespan=lifespan)


# ===============================
# API Endpoints
# ===============================
#  엔드포인트 정의 (올바른 패턴)
# 1. @app.get("/predict") 데코레이터:
#    - app.get 메서드 호출: FastAPI 라우팅 데코레이터 반환
#    - 이 데코레이터가 predict 함수를 감쌌
# 2. 라우팅 테이블 등록:
#    - FastAPI 내부 routes 리스트에 새 APIRoute 객체 추가
#    - 경로: "/predict", HTTP 메서드: GET, 핸들러: predict 함수
# 3. 파라미터 검사:
#    - text: str 타입 힌트 분석
#    - FastAPI가 쿼리 파라미터로 자동 인식 (?text=...)
@app.get("/predict")
def predict(text: str):
    #  전역 딕셔너리에서 모델 조회
    # 1. ml_models["sentiment"] 표현식 평가:
    #    - "sentiment" 문자열의 해시값 계산 (hash("sentiment"))
    #    - 해시값 % 버킷_수 → 버킷 인덱스
    #    - 해당 버킷에서 키 비교 (동등성 검사)
    #    - 일치하면 값(모델 dict) 반환
    # 2. 시간 복잡도: O(1) 평균 상수 시간 (충돌 없으면 나노초 단위)
    # 3. model 변수에 모델 dict 객체 참조 저장 (참조 카운트 +1)
    # - O(1) 해시 테이블 조회: 매우 빠름 (나노초 단위)
    # - 이미 로딩된 모델을 즉시 사용
    model = ml_models["sentiment"]
    
    #  응답 반환
    # 1. 딕셔너리 리터럴 생성:
    #    - 힙에 새 dict 객체 할당
    #    - 3개의 키-값 엔트리 저장
    # 2. FastAPI 자동 직렬화:
    #    - response_class가 JSONResponse로 기본 설정
    #    - dict를 json.dumps()로 JSON 문자열로 변환
    #    - Content-Type: application/json 헤더 추가
    # 3. HTTP 응답 젼:
    #    - 상태 코드: 200 OK
    #    - 바디: JSON 문자열
    #    - uvicorn이 TCP 소켓으로 클라이언트에 전송
    # - 딕셔너리를 반환하면 FastAPI가 자동으로 JSON으로 변환
    return {
        "input": text,
        "prediction": "positive",
        "model": model["model"]
    }

#  엔드포인트 정의 (안티 패턴 - 나쁜 예시)
@app.get("/bad")
def bad_example(text: str):
    print("- 모델 매번 로딩 X")
    
    #  매 요청마다 모델 로딩 (❌ 매우 비효율적!)
    # 1. 요청이 올 때마다:
    #    - load_model() 호출 → 3초 블로킹
    #    - 새 dict 객체 생성 → 메모리 낭비
    # 2. 동시 요청 10개:
    #    - 총 30초 대기
    #    - 10개의 동일한 모델이 메모리에 유지
    # 3. 왕 사용 후:
    #    - model 변수가 함수 종료와 함께 사라짐
    #    - 가비지 컬렉션이 메모리 회수
    # - 요청이 올 때마다 load_model() 호출
    # - 매번 3초씩 대기 (실제로는 수십 초)
    # - 동시 요청이 많으면 서버 폭발
    # - 메모리 낭비 (같은 모델을 여러 번 로딩)
    model = load_model()
    
    return {
        "input": text,
        "result": "느림",
        "model": model["model"]
    }
