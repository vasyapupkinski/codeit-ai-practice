#  필수 라이브러리 임포트
# 1. transformers.pipeline:
#    - HuggingFace의 고수준 API로, 모델/토크나이저/전처리 코드를 캡슐화
#    - 내부적으로 PyTorch(torch)를 로드하여 텐서 연산 수행
# 2. openai.AsyncOpenAI:
#    - 비동기 HTTP 요청을 위한 클라이언트 클래스
#    - 내부적으로 httpx.AsyncClient를 사용하여 Non-blocking I/O 구현
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline
from openai import AsyncOpenAI

import os
import json
from dotenv import load_dotenv

#  환경변수 로드
# - 프로세스 시작 시 .env 파일을 읽어 os.environ 딕셔너리에 병합
load_dotenv()

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Summary Model Loading...")
    
    #  HuggingFace 파이프라인 로딩
    # 1. pipeline("summarization", ...):
    #    - 지정된 모델("sshleifer/distilbart-cnn-12-6")을 로드
    #    - ~/.cache/huggingface에서 가중치 파일을 찾아 메모리(RAM)에 적재
    #    - 약 1.2GB 정도의 메모리 공간 확보 및 텐서 초기화
    ml_models["summarizer"] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    #  추가 모델 로딩 (과제 2)
    # - "sentiment-analysis" (default: distilbert-base-uncased-finetuned-sst-2-english)
    # - 별도의 파이프라인 객체를 생성하여 "sentiment_analyzer" 키에 바인딩
    ml_models["sentiment_analyzer"] = pipeline("sentiment-analysis")
    
    #  OpenAI 클라이언트 생성 (과제 3)
    # - AsyncOpenAI 인스턴스 생성 (API 키 검증은 요청 시점에 수행)
    # - 커넥션 풀(Connection Pool) 초기화 준비
    ml_models["openai"] = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Summary Model Loaded")
    yield
    #  리소스 정리
    # - 딕셔너리를 비워서 참조 카운트 감소 -> GC(가비지 컬렉터)가 메모리 회수 유도
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

#  설정 딕셔너리 (과제 1)
# - 스타일별 설정을 해시 테이블에 저장하여 O(1) 조회 속도 보장
CONF_STYLES = {
    "short": {"min_length": 20, "max_length": 50},
    "medium": {"min_length": 50, "max_length": 150},
    "long": {"min_length": 100, "max_length": 300}
}

class ArticleRequest(BaseModel):
    text: str
    style: str = "medium"

#  요약 엔드포인트 (과제 1 구현)
# 1. def summarize(...): async가 없는 동기 함수
#    - ML 추론은 CPU-bound 작업이므로, FastAPI가 별도의 스레드 풀에서 실행하여
#      이벤트 루프가 차단(Block)되는 것을 방지함
@app.post("/summarize")
def summarize(request: ArticleRequest):
    #  스타일 검증
    # - 해시 테이블 조회(in 연산자)로 유효성 검사
    if request.style not in CONF_STYLES:
        raise HTTPException(status_code=400, detail="Invalid style")
    
    #  설정 로드
    config = CONF_STYLES[request.style]

    try:
        #  전역 모델 참조
        # - 스택 프레임에 전역 딕셔너리 참조를 가져옴
        summerizer = ml_models["summarizer"]
        
        #  추론 실행
        # 1. Tokenizer: 텍스트 -> 토큰 ID 리스트 변환 (Encdoing)
        # 2. Model: 인코더-디코더 연산 수행 (Matrix Multiplication)
        # 3. Parameter 적용: min_length/max_length 제약조건을 Beam Search에 적용
        result = summerizer(
            request.text,
            min_length=config["min_length"],
            max_length=config["max_length"],
        )
        return {"summary": result[0]["summary_text"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#  감성 분석 엔드포인트 (과제 2 구현)
# 1. 동기 함수(def) 사용: 위와 마찬가지로 CPU 사용량 높은 작업 처리
@app.post("/analyze-sentiment")
def analyze_sentiment(request: ArticleRequest):
    try:
        sentiment_analyzer = ml_models["sentiment_analyzer"]
        
        #  추론 실행
        # - 입력 텍스트 -> BERT 모델 -> Classification Head (Linear Layer) -> Softmax
        # - 결과: [{'label': 'POSITIVE', 'score': 0.99...}] 리스트 반환
        result = sentiment_analyzer(request.text)
        
        #  결과 추출
        # - 리스트 첫 번째 요소의 라벨과 점수를 매핑하여 반환
        return {"sentiment": result[0]["label"], "score": result[0]["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  키워드 추출 엔드포인트 (과제 3 구현)
# 1. async def 사용: 
#    - OpenAI API 호출은 외부 네트워크 통신(I/O-bound) 작업
#    - await 키워드로 제어권을 이벤트 루프에 반환하여, 응답 대기 중 다른 요청 처리 가능
@app.post("/extract-keywords")
async def extract_keywords(request: ArticleRequest):
    try:
        client = ml_models["openai"]
        
        #  비동기 API 호출
        # 1. await client.chat.completions.create(...):
        #    - HTTP POST 요청 생성 및 전송 (to https://api.openai.com/v1/chat/completions)
        #    - 소켓에 데이터를 쓰고, 응답이 올 때까지 현재 코루틴 일시 정지(Suspend)
        #    - 응답이 오면 이벤트 루프가 코루틴을 깨우고(Resume) 실행 재개
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts keywords from the given text. The name of the key is keyword. Pick 5 keywords from the given text and return it as a JSON object."
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ],
            #  JSON 모드 활성화
            # - 모델이 유효한 JSON 문자열만 생성하도록 제약
            response_format={
                "type": "json_object"
            }
        )
        
        #  응답 파싱
        # 1. response.choices[0].message.content: JSON 형식의 문자열("String") 추출
        content = response.choices[0].message.content
        
        # 2. json.loads(content):
        #    - JSON 문자열을 파싱하여 파이썬 딕셔너리(Dict) 객체로 변환
        #    - 메모리 힙에 새로운 dict 객체 생성
        result = json.loads(content)
        
        return {"keywords": result["keywords"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        