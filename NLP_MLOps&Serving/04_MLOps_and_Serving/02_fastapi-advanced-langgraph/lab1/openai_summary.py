"""
🎯 FastAPI 실습: OpenAI GPT API를 활용한 텍스트 요약

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **외부 AI 서비스(OpenAI GPT)를 FastAPI로 감싸서 웹 API로 만드는** 실습입니다.
로컬에 모델을 다운로드하지 않고, OpenAI의 클라우드 서비스를 API 호출로 활용하는 패턴을 배웁니다.

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **간단한 시작 (전역 변수 패턴)**:
   - 파일이 실행되면 즉시 OpenAI 클라이언트 생성
   - Lifespan 같은 복잡한 개념 없이 바로 사용 가능
   - 초보자가 이해하기 쉬운 직관적인 구조

2. **외부 API 활용의 장점**:
   - 로컬에 거대한 모델 파일이 필요 없음
   - OpenAI 서버가 연산을 대신 처리
   - 최신 모델(GPT-4o-mini)을 즉시 사용

3. **비동기(Async) 사용**:
   - OpenAI 응답을 기다리는 동안 서버가 멈추지 않음
   - 다른 사용자 요청을 동시에 처리 가능

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. .env 파일에서 OPENAI_API_KEY 로드
2. 전역 변수 영역에서 AsyncOpenAI 클라이언트 생성
3. FastAPI 앱 생성 및 서버 시작 (포트 8002)

[요청 처리]
1. POST /summarize-gpt로 텍스트 수신
2. Pydantic으로 데이터 검증
3. 텍스트 길이 체크 (50자 미만이면 에러)
4. "신문사 편집장" 프롬프트 구성
5. OpenAI API 비동기 호출 (gpt-4o-mini)
6. 요약 결과를 JSON으로 반환

===============================================
💡 핵심 학습 포인트
===============================================
- Global Client: 간단하지만 실무에서는 Lifespan 권장
- AsyncOpenAI: 비동기 OpenAI API 클라이언트
- 프롬프트 엔지니어링: 역할(신문사 편집장) 부여
- 환경변수 관리: dotenv로 API 키 보호

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   pip install openai python-dotenv
   (.env 파일 생성 후 OPENAI_API_KEY=sk-xxx 추가)

2. 로컬 개발용 실행:
   python ./lab1/openai_summary.py

3. 🚀 운영 환경 실행 (배포용, Gunicorn):
   pip install gunicorn
   gunicorn lab1.openai_summary:app \\
     -k uvicorn.workers.UvicornWorker \\
     -w 2 \\
     -b 0.0.0.0:8002
"""

#  필수 모듈 임포트
# - Python 인터프리터가 각 모듈을 찾아 메모리(Method Area)에 로드합니다.
#  필수 모듈 임포트
# 1. from fastapi import FastAPI, HTTPException:
#    - FastAPI, HTTPException 클래스를 현재 네임스페이스에 로드
# 2. from openai import AsyncOpenAI:
#    - openai 패키지의 비동기 클라이언트 클래스 로드
#    - httpx 라이브러리 기반의 비동기 HTTP 클라이언트
# 3. from dotenv import load_dotenv:
#    - 환경변수 관리 도구 로드
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

#  환경변수 로드
# 1. load_dotenv() 함수가 현재 디렉토리에서 .env 파일을 찾습니다.
# 2. 파일을 읽어 KEY=VALUE 형식을 파싱합니다.
# 3. 파싱된 값을 os.environ 딕셔너리(프로세스 환경변수)에 주입합니다.
load_dotenv()

#  FastAPI 앱 인스턴스 생성
# - 힙 메모리에 FastAPI 객체를 할당하고 'app' 변수에 바인딩합니다.
app = FastAPI()

#  OpenAI 클라이언트 생성 (전역 변수)
# 1. os.getenv("OPENAI_API_KEY"):
#    - os.environ 딕셔너리에서 키 검색 (해시 테이블 조회 O(1))
# 2. AsyncOpenAI(api_key=...):
#    - 클라이언트 인스턴스 생성 (힙 메모리 할당)
#    - 내부적으로 비동기 HTTP 커넥션 풀 초기화 (httpx.AsyncClient)
#    - API 키를 HTTP 요청 헤더("Authorization: Bearer sk-...")에 저장
# 3. client 변수에 바인딩:
#    - 프로그램 실행 동안 유지 (전역 스코프)
# 4. 왜 비동기 클라이언트인가?
#    - 동기 클라이언트는 socket.recv()에서 블로킹됨
#    - AsyncOpenAI는 await 시 제어권을 이벤트 루프에 반환
# ⚠️ 주의: 이 방식은 간단하지만, 실무에서는 Lifespan 패턴이 더 안전합니다.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

#  요청 스키마 정의 (Pydantic)
# - BaseModel을 상속하여 자동 검증 기능을 얻습니다.
class ArticleRequest(BaseModel):
    #  필드 정의
    # - text: 필수 문자열 필드
    # - min_length, max_length: 기본값이 있는 선택적 정수 필드
    text: str
    min_length: int = 30        # 최소 길이 설정 가능
    max_length: int = 200       # 최대 길이 설정 가능

#  API 엔드포인트 정의
# - @app.post(): POST /summarize-gpt 경로를 아래 함수에 연결합니다.
@app.post("/summarize-gpt")
#  비동기 함수 정의
# - async def: 이 함수는 비동기적으로 실행됩니다.
# - request: ArticleRequest: FastAPI가 요청 body를 자동으로 파싱하여 이 타입의 객체로 만듭니다.
# [개념] 왜 이 엔드포인트는 async def인가? (def 아님)
# - OpenAI API 호출은 "네트워크 I/O 작업"
#   * 요청 보내고 응답 기다림 (3~10초)
#   * 이 시간 동안 CPU는 놀고 있음
# - async/await로 하면?
#   * OpenAI 응답 기다리는 동안 다른 요청 처리
#   * 동시에 10명이 요청해도 모두 병렬 처리
# - def로 하면? → 한 번에 한 명만 처리 (나머지는 대기)
# - 규칙: 외부 API, DB, 파일 I/O → 반드시 async def + await

# [개념] 왜 AsyncOpenAI 클라이언트를 쓰는가?
# - OpenAI API 호출의 특성:
#   * 네트워크 I/O: 요청 보내고 응답 기다림 (3~10초)
#   * 동기 방식이면? 이 시간 동안 서버 불록
# - AsyncOpenAI의 장점:
#   * 비동기 처리: await 중 다른 요청 처리 가능
#   * 동시 10명 요청 → 10개 병렬 처리
#   * 서버 자원 활용 극대화
# - Lifespan에 AsyncOpenAI 저장하는 이유:
#   * 클라이언트 객체 생성 비용 절감
#   * HTTP 커넥션 풀 재사용 (TCP 핸드셰이크 반복 방지)
# - 실무 패턴: 모든 외부 API 클라이언트는 Lifespan에서 초기화

async def summarize_with_gpt(request: ArticleRequest):
    #  입력 길이 검증
    # - len() 함수는 O(1) 시간에 문자열 길이를 반환합니다.
    # - 50자 미만이면 HTTPException을 발생시켜 400 에러를 반환합니다.
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="요약할 텍스트가 없습니다.")
    
    #  프롬프트 구성 (메모리 할당)
    # - 멀티라인 문자열 리터럴을 'system_instruction' 변수에 할당합니다.
    # - 이것은 GPT에게 "역할"을 부여하는 프롬프트 엔지니어링 기법입니다.
    # 모델을 로딩하지 않고 API 사용하여 GPT에게 역할을 부여하는 프롬프트 (가장 큰 차이점!)
    system_instruction = """
    너는 신문사 편집장이야. 
    사용자가 입력한 기사를 읽고, 가장 중요한 핵심 내용을 '3줄 요약' 형태로 깔끔하게 정리해줘.
    """

    #  try-except 블록 (예외 처리)
    try:
        #  OpenAI API 비동기 호출
        # 1. client.chat.completions.create(): OpenAI API에 요청을 보냅니다.
        # 2. await 키워드:
        #    - 현재 코루틴의 실행을 일시 정지합니다.
        #    - HTTP 요청을 OpenAI 서버로 전송합니다 (TCP/IP 네트워크 통신).
        #    - 응답을 기다리는 동안 이벤트 루프가 다른 작업을 처리할 수 있습니다.
        #    - 응답이 도착하면 이 지점부터 실행이 재개됩니다.
        # 3. 네트워크 통신 상세:
        #    a) DNS 조회 (api.openai.com → IP 주소)
        #    b) TCP 3-way handshake (SYN, SYN-ACK, ACK)
        #    c) TLS/SSL 핸드셰이크 (암호화 연결 설정)
        #    d) HTTP POST 요청 전송 (헤더 + JSON body)
        #    e) 서버 응답 대기
        #    f) HTTP 응답 수신
        response = await client.chat.completions.create(
            #  모델 지정
            # - "gpt-4o-mini": OpenAI의 소형 고효율 모델 (비용 효율적)
            model="gpt-4o-mini",
            #  메시지 리스트 구성
            # - 리스트 리터럴 생성: 2개의 딕셔너리를 포함
            messages = [
                #  시스템 메시지
                # - "role": "system": GPT에게 전반적인 행동 방식을 지시
                {"role":"system", "content": system_instruction},
                #  사용자 메시지
                # - "role": "user": 실제 사용자의 질문/요청
                # - request.text: ArticleRequest 객체의 text 필드 값
                {"role": "user", "content": request.text}
            ],
            #  Temperature 설정
            # - 0.3: 낮은 값 = 더 일관되고 보수적인 출력 (요약에 적합)
            # - 1.0: 높은 값 = 더 창의적이고 다양한 출력
            temperature=0.3
        )
        
        #  결과 추출
        # 1. response.choices[0]: 응답 객체의 choices 리스트에서 첫 번째 요소 가져오기
        # 2. .message.content: 해당 choice의 message 객체의 content 속성 (실제 GPT가 생성한 텍스트)
        summary = response.choices[0].message.content
        
        #  응답 반환
        # 1. 딕셔너리 리터럴 {"summary": summary}를 생성합니다.
        # 2. FastAPI가 이 딕셔너리를 JSON으로 직렬화합니다 (json.dumps 호출).
        # 3. HTTP 응답 body에 JSON 문자열을 씁니다.
        # 4. Content-Type: application/json 헤더를 자동으로 추가합니다.
        return {"summary": summary}
        
    #  예외 처리
    # - except Exception as e: 모든 예외를 catch합니다.
    except Exception as e:
        #  에러 로깅
        # - print(): 에러 메시지를 stdout에 출력합니다.
        print(f"Error: {e}")
        #  HTTP 에러 응답
        # - HTTPException을 raise하여 500 Internal Server Error를 반환합니다.
        raise HTTPException(status_code=500, detail="OpenAI API 호출중 오류 발생")
    
#  메인 블록
# - if __name__ == "__main__": 이 파일이 직접 실행될 때만 True
if __name__ == "__main__":
    #  uvicorn 모듈 임포트
    # - ASGI 서버인 uvicorn을 메모리에 로드합니다.
    import uvicorn
    #  웹 서버 실행
    # 1. uvicorn.run() 함수 호출
    # 2. 내부 동작:
    #    a) TCP/IP 소켓 생성 (socket.socket())
    #    b) 0.0.0.0:8002에 바인딩 (socket.bind())
    #    c) listen 상태로 전환 (socket.listen())
    #    d) 비동기 이벤트 루프 시작 (asyncio.run())
    #    e) 클라이언트 연결 대기 및 처리
    # 3. 포트 8002: 다른 서비스와 충돌을 피하기 위해 8000, 8001이 아닌 8002 사용
    uvicorn.run(app, host="0.0.0.0", port=8002)