"""
🎯 FastAPI 실습3: 나만의 AI 서버 만들기 (LLM Wrapper)
============================================
이 파일은 OpenAI 같은 거대 언어 모델(LLM)을 내 서버 뒤에 숨겨서
안전하고 편리하게 사용하는 방법을 배웁니다.

[왜 직접 OpenAI를 안 부르고 이렇게 감싸나요?]
1. 보안: 내 API 키(비밀번호)를 프론트엔드(브라우저)에 노출하면 해킹당합니다.
2. 비용: 사용자가 너무 많이 쓰면 "그만해!" 라고 막을 수 있습니다 (Rate Limiting).
3. 포장: OpenAI가 주는 복잡한 결과에서 딱 필요한 말만 골라서 줄 수 있습니다.

📌 준비물:
1. OpenAI API 키 발급받기
2. .env 파일 만들어서 키 넣어두기 (OPENAI_API_KEY=sk-...)

📌 실행 방법:
uvicorn lab3_llm_api_wrapper:app --reload
"""

# ============================================
# 0단계: 필수 라이브러리 가져오기 (Import)
# ============================================
# [각 라인의 역할]

# os: 운영체제와 상호작용하는 모듈. 환경변수(os.getenv)를 읽을 때 사용
import os

# FastAPI: 웹 서버 프레임워크
# HTTPException: API에서 에러 발생 시 HTTP 상태 코드와 함께 에러를 반환할 때 사용
from fastapi import FastAPI, HTTPException

# BaseModel: Pydantic의 데이터 검증 도구
# Field: 필드에 상세 규칙(최소값, 최댓값, 기본값 등)을 지정할 때 사용
from pydantic import BaseModel, Field

# List: 리스트 타입을 표현 (예: messages는 여러 개의 메시지 리스트)
# Optional: 값이 있을 수도 있고 None일 수도 있음을 표현
from typing import List, Optional

# load_dotenv: .env 파일에서 환경변수를 읽어와 os.environ에 등록하는 함수
from dotenv import load_dotenv

# OpenAI: OpenAI API를 호출하기 위한 공식 클라이언트 라이브러리
from openai import OpenAI

# ============================================
# 1단계: 환경변수 로드
# ============================================
# [기계적 구동 원리 - 아주 중요!]
# 1. load_dotenv() 함수가 실행되면, 현재 디렉토리에서 `.env` 파일을 찾습니다.
# 2. `.env` 파일을 열어서 한 줄씩 읽습니다. (예: OPENAI_API_KEY=sk-abc123)
# 3. 각 줄을 파싱(분석)해서 "키=값" 형태로 분리합니다.
# 4. 파이썬의 os.environ 딕셔너리에 {"OPENAI_API_KEY": "sk-abc123"} 형태로 저장합니다.
# 5. 이제부터 os.getenv("OPENAI_API_KEY")를 호출하면 "sk-abc123"을 얻을 수 있습니다.
load_dotenv()

# ============================================
# 2단계: FastAPI 앱 생성
# ============================================
# 메모리에 FastAPI 객체를 생성하고 app 변수에 저장합니다.
app = FastAPI(title="나만의 LLM API 서버")

# ============================================
# 3단계: OpenAI 클라이언트 초기화
# ============================================
# [기계적 동작]
# 1. os.getenv("OPENAI_API_KEY")를 호출하여 환경변수에서 API 키를 가져옵니다.
# 2. OpenAI 클래스의 생성자에 api_key를 전달하여 client 객체를 생성합니다.
# 3. 이 client 객체는 OpenAI API와 통신할 수 있는 "리모컨" 역할을 합니댠.
# 4. 앞으로 client.chat.completions.create() 같은 메서드로 AI에게 요청을 보낼 수 있습니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# 1단계: 기본 채팅 완성 API
# ============================================
class Message(BaseModel):
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    # 사용자가 보낼 데이터의 모양을 정의합니다.
    messages: List[Message]
    model: str = "gpt-4o-mini" # 모델을 안 고르면 기본적으로 싸고 빠른 녀석 사용
    
    # 0.0(로봇) ~ 2.0(창의적) 사이의 값만 허용 (Pydantic 검증)
    temperature: float = Field(default=0.7, ge=0, le=2)
    
    # 답변 길이 제한 (1토큰 ~= 0.7단어)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

    model_config = {
        "json_schema_extra":{
            "examples": [{
                "messages": [
                    {"role": "user", "content": "안녕하세요!"}
                ]
            }]
        }
    }

class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict

@app.get("/")
def home():
    """서버 상태 확인"""
    return {"message": "LLM API 서버가 실행중입니다..."}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    기본 채팅 API
    
    [동작 원리]
    1. 사용자가 질문(JSON)을 보냄
    2. Pydantic이 'ChatRequest' 검문소에서 내용물 검사
    3. 통과하면 이 함수가 실행됨
    4. 내 서버(FastAPI)가 OpenAI 서버에게 대신 전화를 걺 (client.chat.completions.create)
    5. OpenAI가 답변을 주면, 필요한 정보만 골라서 'ChatResponse' 상자에 담아 리턴
    """
    try:
        # 여기가 핵심! OpenAI에게 실제로 말을 거는 부분
        # [외부 서버 통신 시작]
        # 1. `client.chat...create()` 함수가 실행되면 내 코드는 잠시 멈춥니다 (Blocking).
        # 2. 파이썬이 인터넷을 통해 OpenAI 서버(api.openai.com)로 요청 데이터를 보냅니다.
        # 3. OpenAI 서버가 열심히 생각을 하고 답변을 만들어서 다시 보내줍니다.
        # 4. 답변이 도착하면 멈춰있던 내 코드가 다시 움직이기 시작하고, `response` 변수에 결과가 담깁니다.
        response = client.chat.completions.create(
            model = request.model,
            messages=[m.model_dump() for m in request.messages], # Pydantic 객체를 딕셔너리로 변환해서 전달
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # [응답 데이터 처리]
        # OpenAI가 준 데이터는 엄청나게 큽니다. 그중에서 우리가 필요한 건 딱 3가지입니다.
        # 1. 답변 내용 (choices[0].message.content)
        # 2. 사용된 모델 이름 (model)
        # 3. 토큰 사용량 (usage - 이걸로 요금 계산)
        return ChatResponse(
            response=response.choices[0].message.content, # 실제 AI의 답변 텍스트
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,      # 내가 보낸 질문 길이
                "completion_tokens": response.usage.completion_tokens, # AI가 답변한 길이
                "total_tokens": response.usage.total_tokens         # 총합 (이걸로 돈 나감)
            }
        )
    
    except Exception as e:
        # 만약 OpenAI 서버가 터지거나, 키가 틀리면 여기가 실행됨
        # 500 에러(서버 잘못)로 바꿔서 사용자에게 이유를 알려줌
        raise HTTPException(status_code=500, detail=str(e))
    

# ============================================
# 혼자해보기 3: 질문 답변 API 
# ============================================
"""
아래 엔드포인트를 직접 만들어보기:

POST /ask
- 요청 바디:
  {
    "question": "파이썬이 뭐야?"
  }
- 응답:
  {
    "answer": "파이썬은 프로그래밍 언어입니다..."
  }

힌트:
- 요청 모델: question 필드 1개만
- 응답 모델: answer 필드 1개만
- messages에 user role로 question 넣기
"""

class AskRequest(BaseModel):
    """질문 요청 모델"""
    # [입력 검증]
    # 사용자가 보낸 JSON에 "question" 필드가 있는지, 문자열인지 검사합니다.
    question: str

class AskResponse(BaseModel):
    """질문 응답 모델"""
    # [출력 검증]
    # 우리가 내보낼 응답이 "answer"라는 키를 가진 문자열인지 확인합니다.
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    [구동 과정]
    1. 요청: POST /ask {"question": "..."}
    2. 검증: Pydantic이 AskRequest 모델로 검사
    3. 실행: OpenAI API 호출 후 결과 파싱
    4. 응답: AskResponse 모델로 포장해서 반환
    """
    try:
        # [외부 API 호출]
        # 1. client.chat.completions.create 함수를 호출합니다.
        # 2. 파이썬 코드가 잠시 멈추고(Blocking), OpenAI 서버로 인터넷 요청을 보냅니다.
        # 3. OpenAI가 답을 줄 때까지 기다립니다. (네트워크 대기)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # [데이터 포장]
                # 사용자 질문(request.question)을 OpenAI가 이해하는
                # 대화 형식([{"role": "user", ...}])으로 포장합니다.
                {"role": "user", "content": request.question}
            ]
        )

        # [결과 추출 및 리턴]
        # 1. response(거대한 객체)에서 choices 리스트의 0번째(첫 번째 답변)를 꺼냅니다.
        # 2. 그 안의 message 객체에서 content(실제 답변 텍스트)를 꺼냅니다.
        # 3. AskResponse 객체에 담아서 리턴합니다. -> FastAPI가 JSON으로 변환해 줍니다.
        return AskResponse(answer=response.choices[0].message.content)

    except Exception as e:
        # [예외 처리]
        # 인터넷 끊김, API 키 오류 등 문제가 생기면 여기가 실행됩니다.
        # 500(서버 잘못) 코드로 에러 내용을 클라이언트에게 알려줍니다.
        raise HTTPException(status_code=500, detail=str(e))