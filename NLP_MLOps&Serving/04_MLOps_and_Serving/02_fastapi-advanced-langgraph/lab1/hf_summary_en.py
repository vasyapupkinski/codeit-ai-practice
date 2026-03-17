"""
🎯 FastAPI 실습: HuggingFace 요약 모델 서빙

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **HuggingFace Transformers 라이브러리의 요약 모델을 FastAPI로 서빙**하는 실습입니다.
로컬에서 AI 모델을 직접 다운로드하고 실행하여, 영어 텍스트를 짧게 요약해주는 API를 만듭니다.

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **Lifespan 패턴 사용**:
   - 모델을 서버가 시작할 때 딱 한 번만 로딩
   - 매 요청마다 로딩하면 시간 낭비 (수 초~수 분)
   - 모든 요청이 하나의 모델 인스턴스를 공유

2. **HuggingFace Pipeline**:
   - 복잡한 ML 코드 없이 pipeline() 함수 하나로 간단하게 사용
   - 모델 다운로드, 토크나이저, 전처리를 자동으로 처리

3. **DistilBART 모델**:
   - 경량화된 BART (Bidirectional and Auto-Regressive Transformers)
   - CNN/DailyMail 데이터셋으로 학습
   - 영어 뉴스 요약에 최적화

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. HuggingFace Hub에서 "sshleifer/distilbart-cnn-12-6" 모델 다운로드
2. 메모리에 모델 + 토크나이저 로드
3. ml_models 딕셔너리에 저장

[요청 처리]
1. POST /summarize로 텍스트 수신
2. Pydantic으로 데이터 검증
3. 텍스트 길이 체크 (50자 미만이면 에러)
4. Pipeline으로 추론 실행
5. 요약 결과 반환

===============================================
💡 핵심 학습 포인트
===============================================
- Lifespan: 모델을 한 번만 로딩하는 효율적인 패턴
- HuggingFace Pipeline: 복잡한 ML을 간단하게
- pipeline("summarization"): 요약 전용 파이프라인

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   pip install torch transformers

2. 로컬 개발용 실행:
   python ./lab1/hf_summary_en.py

3. 🚀 운영 환경 실행 (Uvicorn):
   uvicorn lab1.hf_summary_en:app --host 0.0.0.0 --port 8000
"""

#  필수 모듈 임포트
# 1. from fastapi import FastAPI, HTTPException:
#    - fastapi 패키지의 __init__.py에서 FastAPI 클래스, HTTPException 클래스를 가져옴
#    - 현재 네임스페이스에 바인딩
# 2. from pydantic import BaseModel:
#    - pydantic 패키지의 BaseModel 클래스 가져오기
#    - 데이터 검증의 기반 클래스
# 3. from contextlib import asynccontextmanager:
#    - 비동기 컨텍스트 매니저 데코레이터 가져오기
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
#  HuggingFace Transformers 임포트
# 1. from transformers import pipeline:
#    - transformers 패키지는 수천 개의 모듈로 구성되어 있음 (로드 시간 1~2초)
#    - pipeline 함수는 고수준 API로 모델 로딩, 토크나이저, 전처리를 자동화
# 2. 내부적으로 pytorch 의존성도 함께 로드됨
# - pipeline: 사전 학습된 모델을 간단하게 사용할 수 있는 고수준 API
from transformers import pipeline

#  전역 모델 저장소
# - 빈 딕셔너리를 생성하여 모델을 담을 준비
ml_models = {}

#  Lifespan 컨텍스트 매니저
# - @asynccontextmanager: 비동기 컨텍스트 매니저로 변환하는 데코레이터
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시 실행
    print("======= 요약모델 로딩중 ...")
    
    # [개념] 왜 HuggingFace를 쓰는가? (vs OpenAI)
    # - HuggingFace 장점:
    #   * 비용 무료: API 키 불필요, 회사 서버에서 실행
    #   * 데이터 보안: 데이터가 외부로 나가지 않음 (금융·의료계 필수)
    #   * 오프라인 가능: 인터넷 없이도 동작
    #   * 커스터마이징: 모델을 회사 데이터로 Fine-tuning 가능
    # - HuggingFace 단점:
    #   * 모델 다운로드 필요: ~300MB (최초 1회)
    #   * 서버 리소스 소모: CPU/GPU, RAM 사용
    #   * 성능: GPT보다 떨어질 수 있음 (GPT-4 vs DistilBART)
    # - 언제 HuggingFace?
    #   * 비용 제약, 보안 중요, 오프라인 필요
    # - 언제 OpenAI?
    #   * 성능 우선, 비용 감당 가능, 최신 모델 필요

    #  HuggingFace 파이프라인 생성
    # 1. pipeline("summarization", ...): 요약 task를 위한 파이프라인 생성
    # 2. model="sshleifer/distilbart-cnn-12-6": HuggingFace Hub에서 모델 다운로드
    #    - 처음 실행 시: 인터넷에서 다운로드 (~300MB)
    #    - 이후 실행: 캐시(~/.cache/huggingface)에서 로드
    # 3. 내부 동작:
    #    a) 모델 파일 다운로드 (pytorch_model.bin 등)
    #    b) 토크나이저 다운로드 (vocab.json, merges.txt 등)
    #    c) 설정 파일 다운로드 (config.json)
    #    d) 메모리에 모델 가중치 로드 (수백 MB)
    ml_models["summarizer"] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    print("✅ 모델 로딩 완료!")
    
    #  yield - 제어권 반납
    # 1. 여기서 코드 실행이 일시 정지됩니다.
    # 2. FastAPI 서버가 요청을 받을 준비가 됩니다.
    # 3. 서버 종료 신호가 오면 yield 다음 줄부터 실행 재개
    yield
    
    # [Shutdown] 서버 종료 시 실행
    # - 딕셔너리를 비워서 메모리 해제 준비
    ml_models.clear()

#  FastAPI 앱 생성
# - FastAPI(lifespan=lifespan): lifespan 함수를 이벤트 핸들러로 등록
app = FastAPI(lifespan=lifespan)

# [개념] 왜 Pydantic BaseModel을 쓰는가? (RAG와 동일!)
# - 수동 검증의 문제점:
#   if "text" not in data: return error
#   if not isinstance(data["text"], str): return error
#   if len(data["text"]) < 50: return error
#   → 메바 3줄 코드, 휴먼에러 가능성
# - Pydantic 자동화:
#   class ArticleRequest(BaseModel): text: str
#   → FastAPI가 자동으로 검증, 타입 틀리면 422 에러
# - 장점:
#   * Swagger UI 자동 문서화
#   * IDE 자동완성 (request.text)
#   * 타입 안전성 (실행 전 발견)
# - 실무: dict는 버그 온상, Pydantic 필수

# [이 코드의 역할] 요청 데이터 스키마 정의
# - POST /summarize 엔드포인트로 들어오는 JSON 데이터 구조
# - Pydantic이 자동으로 타입 검증 및 변환 수행
#  Pydantic 클래스 정의
# 1. class ArticleRequest(BaseModel):
#    - Python 메타클래스 시스템 호출
#    - Pydantic의 ModelMetaclass가 __new__ 메서드로 클래스 생성 처리
# 2. 필드 분석:
#    - text: str -> 필수 필드로 등록
#    - min_length: int = 30 -> 선택 필드, 기본값 30
#    - max_length: int = 200 -> 선택 필드, 기본값 200
# 3. 자동 생성되는 내부 메서드:
#    - __init__: 초기화 메서드 (데이터 검증 + 할당)
#    - __pydantic_validator__: 각 필드의 타입 검증 로직
#    - dict, json 메서드: export 기능
# 4. 실행 시점에 FastAPI가 ArticleRequest를 사용하는 방법:
#    - HTTP 요청 바디 JSON 파싱 -> Python dict
#    - ArticleRequest(**dict) 호출
#    - dict의 각 키-값을 필드와 매칭
#    - 타입이 맞지 않으면 ValidationError 발생 -> 422 응답
#  요청 스키마 정의
class ArticleRequest(BaseModel):
    text: str
    min_length: int = 30        # 최소 길이 설정 가능
    max_length: int = 200       # 최대 길이 설정 가능

# [개념] 왜 이 엔드포인트는 def인가? (async def 아님)
# - HuggingFace 모델 추론은 "CPU 작업"
#   * 네트워크 I/O 없음 (모델은 이미 메모리에 로드됨)
#   * 로컬 CPU/GPU에서 즉시 계산
#   * 대기 시간이 극히 짧음 (밀리초 단위)
# - async를 쓰면? → 오히려 오버헤드 발생
# - 규칙:
#   * I/O 대기 (OpenAI, DB, 파일) → async def
#   * CPU 작업 (로컬 모델, 계산) → def
# - 단, FastAPI가 내부적으로 def도 비동기 처리 가능 (스레드 풀 사용)

# [이 코드의 역할] 텍스트 요약 API 엔드포인트
# - POST /summarize 경로로 텍스트 요약 요청 처리
#  API 엔드포인트
@app.post("/summarize")
def summarize_text(request: ArticleRequest):
    #  전역 딕셔너리에서 모델 가져오기
    # 1. ml_models["summarizer"] 조회:
    #    - "summarizer" 해시값 계산
    #    - 해시 테이블에서 O(1) 조회
    #    - Pipeline 객체 참조 반환
    # - O(1) 시간 복잡도로 해시 테이블 조회
    summarizer = ml_models["summarizer"]

    # [이 코드의 역할] 최소 길이 검증
    #  입력 검증
    # 1. len(request.text) 계산: O(1) (Python string은 길이 캐시)
    # 2. 50 미만이면:
    #    - HTTPException 인스턴스 생성
    #    - raise로 예외 발생
    #    - FastAPI가 422 응답으로 자동 변환
    # - 텍스트가 50자 미만이면 에러 발생
    if len(request.text) < 50 :
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다.")
    
    #  try-except 블록
    try:
        # [이 코드의 역할] HuggingFace 모델로 텍스트 요약 실행
        #  요약 실행
        # 1. summarizer(텍스트, ...) 호출:
        #    a) Tokenizer가 텍스트를 토큰 ID로 변환
        #       - BPE 알고리즘으로 subword 분리
        #       - vocab.json에서 ID 매핑
        #       - 결과: [101, 2054, 2003, ...] (토큰 ID 리스트)
        #    b) 토큰 ID를 PyTorch 텐서로 변환
        #       - torch.tensor([...]).unsqueeze(0)
        #       - shape: [1, seq_len]
        #    c) DistilBART 모델 forward pass:
        #       - Encoder: 입력 시퀀스 인코딩
        #       - Decoder: 요약 시퀀스 디코딩 (자기회귀적)
        #       - Attention 메커니즘으로 중요 정보 추출
        #    d) 출력 토큰 ID를 텍스트로 디코딩
        #       - ID -> vocab -> subword -> 텍스트
        # 2. max_length, min_length 파라미터:
        #    - 디코딩 시 생성할 토큰 수 제한
# 1. summarizer(텍스트, ...): HuggingFace 파이프라인 실행
        # 2. 내부 동작:
        #    a) 토크나이저로 텍스트를 토큰 ID로 변환
        #    b) 모델이 토큰 시퀀스를 압축하여 요약 생성
        #    c) 토크나이저로 토큰 ID를 다시 텍스트로 변환
        # 3. max_length, min_length: 요약문의 길이 제약
        result = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        # [이 코드의 역할] 요약 결과 추출
        #  결과 추출 및 반환
        # 1. result 구조: [{"summary_text": "..."}]
        # 2. result[0]: 첫 번째 요소 (리스트 인덱싱)
        # 3. ['summary_text']: 딕셔너리 키 접근
        # 4. 딕셔너리 생성 및 반환 -> FastAPI가 JSON으로 직렬화
        # - result[0]['summary_text']: 리스트의 첫 요소에서 요약 텍스트 추출
        return {"summary": result[0]['summary_text']}
        
    except Exception as e:
        #  에러 처리
        # 1. 모든 예외 캐치
        # 2. HTTPException 생성 (500 Internal Server Error)
        # 3. FastAPI가 JSON 에러 응답 생성
        # - 모든 예외를 500 에러로 변환하여 반환
        raise HTTPException(status_code=500, detail=str(e))
    
# [이 코드의 역할] 직접 실행 시 서버 시작
#  메인 블록
# 1. if __name__ == "__main__":
#    - __name__ 변수 확인
#    - 직접 실행: "__main__", import: 모듈명
# 2. uvicorn.run():
#    - ASGI 서버 시작
#    - host="0.0.0.0": 모든 인터페이스 허용
#    - port=8000: TCP 포트 바인딩
if __name__ == "__main__":      # 이 파일을 직접 실행할 때만 아래 코드를 실행하고, import될 때는 실행하지 않음
    import uvicorn
    #  웹 서버 시작
    # - host="0.0.0.0": 모든 네트워크 인터페이스에서 접속 허용
    # - port=8000: TCP 포트 8000번 사용
    uvicorn.run(app, host="0.0.0.0", port=8000)
