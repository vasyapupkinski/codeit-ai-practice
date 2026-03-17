"""
🎯 FastAPI 실습: HuggingFace 한국어 요약 모델 서빙

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **HuggingFace의 한국어 전용 요약 모델(KoBART)을 FastAPI로 서빙**하는 실습입니다.
영어 모델(`hf_summary_en.py`)과 구조는 동일하지만, 한국어 텍스트 처리에 특화된 모델을 사용합니다.

===============================================
🤔 왜 별도 파일인가? (설계 의도)
===============================================
1. **언어별 모델 분리**:
   - 영어 모델로는 한글 처리 불가능 (토크나이저가 다름)
   - KoBART는 한국어 문법, 어순, 조사 등을 이해
   - 각 언어에 최적화된 성능 제공

2. **포트 8001 사용**:
   - 영어 모델이 8000 포트 사용 중
   - 두 서비스를 동시에 실행하여 비교 가능
   - 실무에서는 로드밸런서로 통합

3. **KoBART 모델**:
   - SKT가 공개한 한국어 BART 모델
   - 뉴스/문서 요약에 강점
   - gogamza팀이 fine-tuning한 버전 사용

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. HuggingFace Hub에서 "gogamza/kobart-summarization" 모델 다운로드
2. 메모리에 한국어 모델 + 토크나이저 로드
3. ml_models 딕셔너리에 저장

[요청 처리]
1. POST /summarize-korean-simple로 한글 텍스트 수신
2. Pydantic으로 데이터 검증
3. 텍스트 길이가 50자 미만이면 에러
4. KoBART로 한국어 추론 실행
5. 요약 결과 반환

===============================================
💡 핵심 학습 포인트
===============================================
- 다국어 모델: 언어마다 다른 모델 필요
- 멀티 서비스: 같은 패턴으로 여러 서비스 운영
- 포트 관리: 충돌 방지

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   pip install torch transformers

2. 로컬 개발용 실행:
   python ./lab1/hf_summary_kr.py

3. 🚀 운영 환경 실행:
   uvicorn lab1.hf_summary_kr:app --host 0.0.0.0 --port 8001
"""

#  필수 모듈 임포트
# 1. from fastapi import FastAPI, HTTPException:
#    - sys.modules에서 fastapi 패키지 검색
#    - 없으면 sys.path에서 찾아 __init__.py 실행
#    - FastAPI 클래스, HTTPException 클래스를 현재 네임스페이스에 바인딩
# 2. from transformers import pipeline:
#    - transformers 패키지 로드 (수천 개 모듈, 로드 시간 1~2초)
#    - pipeline 함수는 모델 로딩/토크나이저/전처리 자동화 API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline

# [이 코드의 역할] 전역 모델 저장소
#  전역 변수 초기화
# 1. {} 리터럴: 빈 dict 객체를 힙 메모리에 생성
# 2. ml_models 바인딩: 전역 네임스페이스에 참조 저장
# [개념] 왜 한국어 모델을 따로 만드는가?
# - 언어모델의 특수성:
#   * 영어 모델로는 한글 처리 불가: 토크나이저가 한글을 모름
#   * 한글의 문법: 조사(은/는/이/가), 어순, 존댓말
#   * 한글 특화 모델 필요: KoBART, KoGPT 등
# - 포트 8001 사용 이유:
#   * 영어 모델(8000)과 동시 실행 가능
#   * 실무에서는 로드밸런서로 통합
# - 실무 패턴:
#   * 다국어 지원 서비스: 언어별 모델 분리 운영
#   * API Gateway로 언어 감지 → 해당 모델로 라우팅

ml_models = {}

#  Lifespan 컨텍스트 매니저
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시 실행
    print("한국어 요약 모델(KoBART) 로딩 중...")
    
    # [이 코드의 역할] KoBART 한국어 요약 모델 로드
    #  한국어 요약 파이프라인 생성
    # 1. pipeline("summarization", model="gogamza/kobart-summarization"):
    #    - HuggingFace Hub API 호출하여 모델 메타데이터 확인
    #    - ~/.cache/huggingface/hub/ 확인
    #    - 없으면 HTTP 다운로드 (~500MB)
    # 2. 한국어 특화 토크나이저:
    #    - 한글 Unicode 처리 (U+AC00-U+D7A3)
    #    - 형태소 인식 서브워드 토큰화
    #    - vocab.json에는 한국어 어휘 포함
    # 3. KoBART 모델:
    #    - SKT가 학습한 한국어 BART 아키텍처
    #    - Encoder-Decoder 구조
    #    - RAM에 약 400~600MB 적재
    # 파이프라인 생성 (모델 이름만 한국어 모델로 교체)
    # gogamza/kobart-summarization: 한국어 뉴스/문서 요약에 특화된 유명한 모델
    # 1. 처음 실행: HuggingFace Hub에서 다운로드 (~500MB)
    # 2. 이후 실행: 로컬 캐시에서 로드
    # 3. 한국어 토크나이저: 한글 자모 분리, 서브워드 토큰화 등 처리
    ml_models["ko_summarizer"] = pipeline("summarization", model="gogamza/kobart-summarization")
    
    print("✅ 모델 로딩 완료!")
    
    #  yield - 서버 실행 대기
    yield
    
    # [Shutdown] 서버 종료 시 메모리 정리
    ml_models.clear()

#  FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# [이 코드의 역할] 요청 데이터 스키마
#  Pydantic 클래스 정의
# 1. class ArticleRequest(BaseModel):
#    - Pydantic ModelMetaclass가 __new__로 클래스 생성
#    - 필드 분석하여 자동 검증 로직 생성
#    - FastAPI가 HTTP 요청 바디를 ArticleRequest(**dict)로 변환
#  요청 스키마 정의
class ArticleRequest(BaseModel):
    text: str
    min_length: int = 30        # 최소 길이 설정 가능
    max_length: int = 200       # 최대 길이 설정 가능

#  API 엔드포인트
# - 경로: /summarize-korean-simple (한국어 전용 엔드포인트)
@app.post("/summarize-korean-simple")
def summarize_korean_simple(request: ArticleRequest):
    #  모델 조회
    # - .get(): 딕셔너리에서 안전하게 값 가져오기 (없으면 None 반환)
    summarizer = ml_models.get("ko_summarizer")
    
    #  입력 검증
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다.")

    #  try-except 블록
    try:
        #  한국어 요약 실행
        # 파이프라인 실행
        # 1. 한글 텍스트를 유니코드로 처리
        # 2. 한국어 토크나이저가 음절/서브워드 단위로 분리
        # 3. KoBART 모델이 한국어 문법에 맞는 요약 생성
        # 4. max_length=128, min_length=32: 한국어는 영어보다 압축률이 높음
        result = summarizer(
            request.text, 
            max_length=128,  # 요약문의 최대 길이
            min_length=32   # 요약문의 최소 길이
        )
        
        #  결과 반환
        # - result[0]['summary_text']: 요약된 한글 텍스트
        return {"summary": result[0]['summary_text']}
        
    except Exception as e:
        #  에러 로깅 및 처리
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="요약 실패")

#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    # - port=8001: 영어 모델(8000)과 충돌 방지
    uvicorn.run(app, host="0.0.0.0", port=8001)