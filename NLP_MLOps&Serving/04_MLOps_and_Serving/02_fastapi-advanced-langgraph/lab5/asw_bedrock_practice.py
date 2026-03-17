"""
🎯 FastAPI 실습: AWS Bedrock + Claude 3 광고 문구 생성기 (실무 교육용 완전판)

===============================================
📋 이 파일이 하는 일 (실무 프로덕션 레벨)
===============================================
이 파일은 **AWS Bedrock을 통해 Claude 3 무려 '비동기(Non-blocking)'로 호출**하는 API입니다.
이전의 학습용(동기) 버전과 달리, **실제 회사에서 서비스할 때 사용하는 진짜 아키텍처**입니다.

1. **aioboto3 (Async SDK)**
   - `boto3`는 동기식이라 서버를 멈추게 하지만, `aioboto3`는 `await`를 사용하여 서버가 멈추지 않습니다.
   - 동시 접속자가 100명이 와도 느려지지 않고 매끄럽게 처리합니다.

2. **Resource Management (Lifespan + Context Manager)**
   - `lifespan` 함수를 사용하여 서버 시작/종료 시 리소스를 안전하게 관리합니다.
   - `async with` 구문을 사용하여 AWS 연결을 안전하게 열고, 사용 후 확실하게 닫습니다.
   - 메모리 누수나 연결 끊김 문제를 방지하는 가장 안전한 패턴입니다.

3. **Multi-cloud & Production Ready**
   - OpenAI 의존성을 제거하고 AWS 생태계를 활용합니다.
   - IAM 인증 방식을 사용하여 보안성을 극대화합니다.

===============================================
📚 핵심 용어 정리 (이게 다 뭔가요?)
===============================================
1. **AWS Bedrock (AI 모델 백화점)**
   - "OpenAI"는 GPT만 팔지만, **Bedrock**은 Claude(Anthropic), Llama(Meta), Titan(Amazon) 
   - 등 **여러 회사의 AI 모델을 모아놓고 파는 AWS 서비스**입니다.
   - 우리는 여기서 "Claude 3" 모델을 빌려 쓸 것입니다.

2. **Boto3 (AWS 리모컨 - 동기)**
   - 파이썬 코드로 AWS(Bedrock, S3 등)를 조종하기 위한 **공식 도구(SDK)**입니다.
   - "리모컨 버튼을 누르면(요청), TV가 켜질 때까지(응답) 꼼짝 않고 기다리는 방식"입니다. (Blocking)

3. **Aioboto3 (AWS 리모컨 - 비동기)**
   - Boto3의 업그레이드 버전으로, **동시 처리**가 가능합니다.
   - "버튼을 누르고, TV가 켜지는 동안 설거지도 하고 청소도 하는 방식"입니다. (Non-blocking)
   - 웹 서버(FastAPI)에서는 무조건 이걸 써야 합니다.

===============================================
🤔 왜 AWS Bedrock인가? (설계 의도)
===============================================
1. **멀티 클라우드 전략**:
   - OpenAI에만 의존하지 않음
   - 가격/성능 비교 가능
   - 리스크 분산

2. **AWS 생태계 활용**:
   - 이미 AWS 쓰는 회사에 유리
   - IAM으로 보안 관리
   - Lambda, ECS 등과 통합 쉬움

3. **Claude의 특징**:
   - Anthropic이 개발
   - 긴 컨텍스트 처리 강점
   - 윤리적 AI 강조

===============================================
🔄 전체 실행 흐름 (async 워크플로우)
===============================================
[서버 시작 - Lifespan]
1. .env에서 AWS 키 로드 및 검증
2. 설정(Config) 초기화
3. 로깅 시스템 설정
4. aioboto3 세션 생성 (미리 준비)
5. 서버 대기

[요청 처리]
1. 제품명, 키워드 수신
2. Claude API 형식으로 JSON 구성
3. **await invoke_model()** 비동기 호출 (서버 안 멈춤)
4. **await StreamingBody.read()** 비동기 읽기
5. JSON 파싱 → 텍스트 추출
6. 로그 기록 후 반환

[서버 종료 - Lifespan]
1. 세션 정리 (필요시)
2. 종료 로그 기록

===============================================
💡 핵심 학습 포인트
===============================================
- aioboto3: AWS Python Async SDK
- Async Context Manager: 자원 관리의 정석 (`async with`)
- Lifespan: 서버 시작/종료 시 리소스 관리 패턴
- Logging: print() 대신 사용하는 실무 로그 시스템
- Pydantic Settings: 환경변수를 클래스로 관리하는 방법
- IAM 인증: API 키 대신 역할 기반
- 멀티 클라우드: 종속성 탈피

===============================================
🤔 Boto3 vs Aioboto3 (언제 무엇을 쓰는가?)
===============================================
1. **Boto3 (동기 - Blocking)** -> "혼자 일할 때"
   - **특징**: 요청하면 응답 올 때까지 멈춥니다.
   - **사용처**: 배치 스크립트, 데이터 분석, AWS Lambda(서버리스)
   - **장점**: 코드가 직관적이고 설정이 쉽습니다.

2. **Aioboto3 (비동기 - Non-blocking)** -> "여럿이 일할 때"
   - **특징**: `await`로 요청해놓고 다른 일을 처리합니다.
   - **사용처**: **FastAPI**, Django 같은 고성능 웹 서버
   - **장점**: 동시 접속자가 많아도 서버가 멈추지 않습니다.
   
   > [결론] 혼자 쓰는 스크립트는 `boto3`, 남들과 같이 쓰는 서버는 `aioboto3`가 정답입니다.

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install fastapi uvicorn aioboto3 python-dotenv pydantic-settings
   - AWS 계정에서 Bedrock 모델 액세스 활성화 (Claude 3 Haiku)
   - .env 파일 설정:
     AWS_ACCESS_KEY=AKIA...
     AWS_SECRET_KEY=...
     (AWS IAM에서 AmazonBedrockFullAccess 권한 필요)

2. 실행 방법:
   python ./lab5/asw_bedrock_practice.py

💡 OpenAI vs AWS Bedrock 비교:
┌─────────────┬──────────────────┬────────────────────┐
│             │ OpenAI API       │ AWS Bedrock        │
├─────────────┼──────────────────┼────────────────────┤
│ 인증         │ API Key          │ IAM (Access/Secret)│
│ 클라이언트    │ openai 라이브러리 │ aioboto3 (Async)    │
│ 모델         │ gpt-4o-mini 등   │ Claude, Titan 등    │
│ 과금         │ OpenAI 직접 결제  │ AWS 청구서 통합      │
│ 보안         │ API Key 관리     │ IAM 정책 관리        │
└─────────────┴──────────────────┴────────────────────┘

💡 Bedrock 모델 ID 예시:
- anthropic.claude-3-haiku-20240307-v1:0 (빠름, 저렴)
- anthropic.claude-3-sonnet-20240229-v1:0 (균형)
- anthropic.claude-3-opus-20240229-v1:0 (고성능)

===============================================
🏭 실무 패턴 설명 (이 파일에서 배우는 것들)
===============================================
1. **Lifespan 패턴**: 서버가 켜지고 꺼질 때 해야 할 일을 정의합니다.
   - 왜 필요?: DB 연결, 캐시 초기화 등 "준비 작업"을 서버 시작 시 한 번만 수행합니다.
   - 왜 중요?: 매 요청마다 연결하면 느리고 리소스 낭비입니다.

2. **Logging 패턴**: print() 대신 logging 모듈을 사용합니다.
   - 왜 필요?: print()는 터미널에만 나오지만, logging은 파일/클라우드에 저장 가능합니다.
   - 왜 중요?: 서버가 크래시 나면 print()는 다 날아갑니다. 로그 파일은 남습니다.

3. **Pydantic Settings 패턴**: 환경변수를 클래스로 관리합니다.
   - 왜 필요?: 환경변수가 없으면 서버 시작 시점에 바로 에러를 냅니다.
   - 왜 중요?: 나중에 "왜 안 되지?" 하고 헤매는 시간을 줄여줍니다.

4. **에러 핸들링 패턴**: 예외를 세분화해서 처리합니다.
   - 왜 필요?: "500 에러" 하나만 내면 어디서 문제인지 알 수 없습니다.
   - 왜 중요?: AWS 에러인지, 우리 코드 에러인지 구분해야 고칠 수 있습니다.
"""

# ============================================================
# ZONE 1: 필수 모듈 임포트
# ============================================================
# [실무 포인트] 임포트 순서: 1. 표준 라이브러리 -> 2. 서드파티 -> 3. 내 코드
# 이 순서를 지키면 PEP 8 스타일 가이드를 따르는 것입니다.

import json #  json 모듈을 메모리에 로드합니다. (JSON 데이터 처리용)
import logging #  logging 모듈을 메모리에 로드합니다. (실무 로깅 시스템)
from contextlib import asynccontextmanager #  Lifespan 정의를 위한 데코레이터를 가져옵니다.

#  서드파티 라이브러리 임포트 (외부 패키지)
import aioboto3 #  aioboto3 모듈 로드 → AWS 비동기 통신 가능해짐
from fastapi import FastAPI, HTTPException #  FastAPI 클래스와 예외 클래스를 가져옵니다.
from pydantic import BaseModel #  Pydantic의 기본 모델 클래스를 가져옵니다.
from pydantic_settings import BaseSettings #  환경변수 관리용 클래스를 가져옵니다.


# ============================================================
# ZONE 2: 설정 관리 (Configuration)
# ============================================================
# [실무 패턴] 왜 설정을 클래스로 관리하나요?
# 1. 환경변수가 없으면 서버 시작 시점에 바로 에러를 냅니다. (나중에 헤매지 않음)
# 2. 타입 힌트가 있어서 IDE가 자동완성을 해줍니다.
# 3. 모든 설정이 한 곳에 모여있어서 관리가 쉽습니다.

class Settings(BaseSettings):
    """
    [실무 패턴] Pydantic Settings 클래스 (이게 바로 Pydantic입니다!)

    Q: "지금 이 클래스 자체가 Pydantic인가요?"
    A: 네, 맞습니다! BaseSettings는 Pydantic의 핵심 기능인 '데이터 검증'을
       '환경변수 관리'에 적용한 특수한 형태의 Pydantic 모델입니다.

    [작동 원리]
    1. BaseSettings는 Pydantic의 BaseModel을 상속받습니다.
    2. 따라서 str, int 같은 타입 힌트를 주면 자동으로 검증(Validation)을 수행합니다.
    3. 차이점은 '입력 데이터'가 아닌 '.env 파일'이나 '시스템 환경변수'에서 값을 읽어온다는 점입니다.
    
    -> 필수 값(예: AWS 키)이 없으면 서버 시작하자마자 에러를 냅니다. (실무 필수 패턴)
    """
    # [필수] AWS 자격 증명 - 없으면 서버가 시작되지 않습니다.
    AWS_ACCESS_KEY: str  # .env에서 AWS_ACCESS_KEY=... 형태로 설정
    AWS_SECRET_KEY: str  # .env에서 AWS_SECRET_KEY=... 형태로 설정
    
    # [선택] 기본값이 있는 설정 - 없어도 기본값으로 동작합니다.
    AWS_REGION: str = "us-east-1"  # Bedrock 모델이 있는 리전 (미국 버지니아 북부)
    MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"  # 사용할 AI 모델
    MAX_TOKENS: int = 2048  # AI가 생성할 최대 글자 수
    
    class Config:
        # .env 파일에서 환경변수를 자동으로 읽어옵니다.
        env_file = ".env"
        # 대소문자를 구분하지 않습니다. (AWS_ACCESS_KEY = aws_access_key)
        case_sensitive = False

#  설정 클래스를 인스턴스화합니다.
# 이 순간, Pydantic이 .env 파일을 읽고 값을 검증합니다.
# 필수 값이 없으면 여기서 ValidationError가 발생합니다.
try:
    settings = Settings() #  Settings() 생성자 호출 → .env 파싱 → 검증 → 객체 생성
except Exception as e:
    #  예외 발생 시 에러 메시지 출력 후 프로그램 종료
    print(f"❌ [설정 오류] .env 파일을 확인하세요: {e}")
    print("   필수 환경변수: AWS_ACCESS_KEY, AWS_SECRET_KEY")
    raise SystemExit(1)  #  시스템 종료 코드 1 반환 (비정상 종료)


# ============================================================
# ZONE 3: 로깅 설정 (Logging)
# ============================================================
# [실무 패턴] 왜 print() 대신 logging을 쓰나요?
# 1. print()는 터미널에만 출력됩니다. 서버가 꺼지면 다 사라집니다.
# 2. logging은 파일, 클라우드(CloudWatch 등)에 저장할 수 있습니다.
# 3. 로그 레벨(DEBUG, INFO, WARNING, ERROR)로 중요도를 구분할 수 있습니다.

#  로거 객체 생성
# __name__은 현재 모듈 이름 (예: "asw_bedrock_practice")입니다.
logger = logging.getLogger(__name__)

#  로깅 시스템 기본 설정
# 이 설정은 프로그램 전체에 적용됩니다.
# Q: "%(asctime)s 같은게 뭔가요? 이걸 왜 이렇게 복잡하게 쓰나요?"
# A: 이것은 "로그의 출력 양식(Format)"을 정의하는 약속 기호입니다.
#    그냥 글자만 출력하면 "언제", "얼마나 심각한지"를 알 수 없기 때문입니다.
#
#    - %(asctime)s   : 현재 시간 (예: 2026-01-16 12:00:00) -> "언제 일어난 일인가?"
#    - %(levelname)s : 로그 레벨 (예: INFO, ERROR)       -> "얼마나 중요한가?"
#    - %(name)s      : 모듈 이름 (예: asw_bedrock)       -> "어디서 발생했나?"
#    - %(message)s   : 실제 메시지 (예: 서버 시작됨)      -> "무슨 내용인가?"
#
#    결국: [시간] | [중요도] | [위치] | [내용] 순서로 깔끔하게 정리해서 보여달라는 뜻입니다.
logging.basicConfig(
    level=logging.INFO,  #  INFO 레벨 이상만 출력 (DEBUG는 무시)
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    # 출력 예: 2026-01-15 22:59:50 | INFO | asw_bedrock_practice | 서버 시작됨
)


# ============================================================
# ZONE 4: 전역 상태 관리 (Global State)
# ============================================================
# [실무 패턴] 왜 전역 딕셔너리를 쓰나요?
# 1. aioboto3 세션은 생성 비용이 큽니다. 매 요청마다 만들면 느립니다.
# 2. Lifespan에서 생성하고, 엔드포인트에서 재사용합니다.
# 3. 딕셔너리로 감싸면 "아직 안 만들어짐" 상태를 표현할 수 있습니다.
#
# Q: "세션이 여러 개일 수 있다는 뜻인가요? AWS 세션이 2개, 3개 들어갈 수도 있나요?"
# A: 네, 맞습니다! 필요하다면 '세션 A', '세션 B' 이렇게 여러 개를 담을 수 있습니다.
#    예를 들어, 회사의 '마케팅팀 AWS 계정'과 '개발팀 AWS 계정'을 동시에 써야 한다면
#    아래처럼 여러 개의 세션을 이 딕셔너리에 담아서 관리하면 됩니다.
#
# Q: "AWS 세션 말고 다른 것도 여기 넣나요? 막 섞여도 되나요?"
# A: 네, 상관없습니다! 여기서 말하는 state는 "앱이 살아있는 동안 계속 쥐고 있어야 하는 물건들"입니다.
#    AWS는 'Session'이라는 이름을 쓰고, DB는 'Connection', AI 모델은 'Model'이라고 부르겠지만,
#    결국 "무겁고 중요한 짐"이라는 점은 똑같습니다. 그래서 이 가방(딕셔너리)에 다 같이 넣습니다.
#
#    app_state = {
#        "marketing_session": aioboto3.Session(...),  # 마케팅팀 계정
#        "dev_session": aioboto3.Session(...),        # 개발팀 계정
#        "db_connection": ...,
#        "ai_model": ...,
#    }
#    -> 딕셔너리(Dictionary)니까 키(Key)만 다르면 몇 개든 넣을 수 있습니다.
#    -> 종류가 달라도 "전역적으로 관리해야 하는 리소스"라는 점에서 한 곳에 모아둡니다.

#  전역 상태 딕셔너리 생성
# 프로그램 시작 시 빈 딕셔너리가 메모리에 생성됩니다.
# Lifespan에서 이 딕셔너리에 세션을 저장하고, 엔드포인트에서 꺼내 씁니다.
app_state = {
    "session": None,  #  초기값은 None (아직 세션 없음)
}


# ============================================================
# ZONE 5: Lifespan (서버 시작/종료 관리)
# ============================================================
# [실무 패턴] Lifespan이 뭔가요?
# - 서버가 **켜질 때** 한 번 실행되는 코드 (yield 이전)
# - 서버가 **꺼질 때** 한 번 실행되는 코드 (yield 이후)
# 
# 왜 필요한가요?
# - DB 연결, 캐시 초기화, AI 모델 로드 등 "준비 작업"을 서버 시작 시 한 번만 수행합니다.
# - 매 요청마다 연결하면 느리고 리소스 낭비입니다.
# 
# 비유:
# - 레스토랑 오픈 전에 재료 손질, 테이블 세팅을 해두는 것과 같습니다.
# - 손님(요청)이 올 때마다 재료 사러 가면 안 되겠죠?

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [실무 패턴] FastAPI Lifespan 함수
    
    서버가 시작될 때 (yield 이전):
    - aioboto3 세션을 미리 생성해둡니다.
    - 로그를 남겨서 "서버가 정상적으로 시작되었다"는 것을 기록합니다.
    
    서버가 종료될 때 (yield 이후):
    - 필요한 정리 작업을 수행합니다.
    - 로그를 남겨서 "서버가 정상적으로 종료되었다"는 것을 기록합니다.
    """
    # ========== 서버 시작 시 실행되는 코드 ==========
    #  uvicorn이 시작되면 이 함수가 자동으로 호출됨
    logger.info("🚀 서버 시작 중... AWS Bedrock 세션을 초기화합니다.")
    
    #  aioboto3.Session() 호출 → AWS 연결 설정 객체 생성
    # 이 세션은 전역 딕셔너리에 저장되어 모든 엔드포인트에서 재사용됩니다.
    app_state["session"] = aioboto3.Session()
    
    #  로그 기록 (디버깅용)
    logger.info(f"✅ AWS 리전: {settings.AWS_REGION}")
    logger.info(f"✅ AI 모델: {settings.MODEL_ID}")
    logger.info("✅ 서버 준비 완료! 요청을 기다리는 중...")
    
    #  yield 키워드 → 여기서 함수가 "일시 정지"됩니다.
    # 서버가 요청을 받을 준비가 되고, 종료 신호가 올 때까지 대기합니다.
    yield
    
    # ========== 서버 종료 시 실행되는 코드 ==========
    #  Ctrl+C 또는 종료 신호 수신 시, yield 이후 코드가 실행됨
    logger.info("🛑 서버 종료 중... 리소스를 정리합니다.")
    
    #  전역 딕셔너리에서 세션 제거 (메모리 정리)
    app_state["session"] = None
    
    logger.info("✅ 서버가 정상적으로 종료되었습니다.")


# ============================================================
# ZONE 6: FastAPI 앱 생성
# ============================================================
# [실무 포인트] lifespan 파라미터를 전달하여 시작/종료 로직을 연결합니다.
# 이렇게 하면 서버 시작 시 aioboto3 세션이 자동으로 생성됩니다.

#  FastAPI 앱 인스턴스 생성
# 이 순간 FastAPI 객체가 메모리에 생성되고, lifespan 함수가 연결됩니다.
# 서버 시작 시 lifespan의 yield 이전 코드가 실행됩니다.
app = FastAPI(
    title="AWS Bedrock 광고 문구 생성기",  # API 문서(/docs)에 표시될 제목
    description="Claude 3를 사용하여 SNS 광고 문구를 생성합니다.",
    version="2.0.0",  # 버전 관리
    lifespan=lifespan,  #  위에서 정의한 lifespan 함수 연결 → 서버 시작/종료 시 자동 호출됨
)


# ============================================================
# ZONE 7: 요청/응답 스키마 정의 (Pydantic Models)
# ============================================================
# [실무 패턴] 왜 Pydantic 모델을 쓰나요?
# 1. 클라이언트가 잘못된 데이터를 보내면 자동으로 422 에러를 반환합니다.
# 2. API 문서(Swagger)에 자동으로 스키마가 표시됩니다.
# 3. IDE가 자동완성을 해줘서 개발 속도가 빨라집니다.

#  Pydantic 요청 스키마 정의
# 클라이언트가 데이터를 보내면, FastAPI가 자동으로 이 클래스로 변환합니다.
# 타입이 맞지 않으면 422 에러가 자동 반환됩니다.
class AdRequest(BaseModel):
    """
    광고 문구 생성 요청 스키마
    
    클라이언트는 이 형식으로 데이터를 보내야 합니다.
    잘못된 형식이면 FastAPI가 자동으로 422 에러를 반환합니다.
    """
    product_name: str   #  문자열 타입임을 명시 (예: "다이슨 청소기")
    keywords: str       #  문자열 타입임을 명시 (예: "강력, 조용, 가벼움")
    
    class Config:
        # API 문서에 표시될 예시 데이터
        json_schema_extra = {
            "example": {
                "product_name": "초경량 무선 청소기",
                "keywords": "강력한 흡입력, 조용함, 가벼움"
            }
        }


#  Pydantic 응답 스키마 정의
# 함수가 리턴할 때 이 클래스로 감싸면, 자동으로 JSON으로 변환됩니다.
class AdResponse(BaseModel):
    """
    광고 문구 생성 응답 스키마
    
    성공 시 이 형식으로 데이터를 반환합니다.
    """
    result: str  #  생성된 광고 문구
    model_used: str  #  사용된 AI 모델 ID
    

class ErrorResponse(BaseModel):
    """
    에러 응답 스키마
    
    실패 시 이 형식으로 에러 정보를 반환합니다.
    """
    error: str  # 에러 메시지
    error_type: str  # 에러 종류 (AWS, Validation 등)


# ============================================================
# ZONE 8: API 엔드포인트 정의
# ============================================================

@app.post(
    "/generate-ad",
    response_model=AdResponse,  # 성공 시 반환 형식
    responses={
        500: {"model": ErrorResponse, "description": "서버 에러"},
    }
)
async def generate_ad_copy(request: AdRequest):
    """
    🎯 광고 문구 생성 API
    
    제품명과 키워드를 받아서 Claude 3가 매력적인 광고 문구를 생성합니다.
    
    [실무 패턴] 이 엔드포인트에서 배우는 것들:
    1. **세션 재사용**: Lifespan에서 만든 세션을 가져와서 사용합니다.
    2. **Async Context Manager**: `async with`로 클라이언트를 안전하게 관리합니다.
    3. **에러 핸들링**: AWS 에러와 기타 에러를 구분해서 처리합니다.
    4. **로깅**: 모든 중요한 이벤트를 로그로 기록합니다.
    """
    #  Step 1: 요청 로그 기록
    # logger.info() 호출 → 포맷에 맞춰 문자열 생성 → 콘솔 출력
    logger.info(f"📥 요청 수신: 제품명='{request.product_name}', 키워드='{request.keywords}'")
    
    #  Step 2: 전역 딕셔너리에서 AWS 세션 꺼내기
    # Q: "갑자기 이 세션은 어디서 튀어나왔나요?"
    # A: 맨 위 'lifespan' 함수(서버 켜질 때 실행됨)에서 미리 만들어서
    #    'app_state'라는 전역 사물함에 넣어둔 것입니다.
    #
    #    1. 서버 시작 -> lifespan 실행 -> AWS 세션 생성 -> app_state["session"]에 저장
    #    2. 요청 들어옴 -> 이 함수 실행 -> app_state.get("session")으로 꺼냄
    #
    # [핵심] 왜 이렇게 하나요? (성능 최적화)
    # -> 여기서 매번 세션을 새로 만들면(Connect) 요청마다 0.5초~1초씩 낭비됩니다.
    # -> 미리 만들어둔 걸 재사용하면 0.0001초면 가져옵니다.
    session = app_state.get("session")
    if session is None:
        #  세션이 없으면 에러 로그 후 500 에러 반환
        logger.error("❌ aioboto3 세션이 초기화되지 않았습니다!")
        raise HTTPException(status_code=500, detail="서버 초기화 오류")
    
    # [Step 3] 프롬프트 작성
    # AI에게 보낼 명령어를 구성합니다. f-string(f"...")을 사용하여 변수 값을 문자열에 삽입합니다.
    prompt = f"""
    당신은 전문 마케터입니다. 아래 제품에 대한 매력적인 SNS 광고 문구를 3줄 이내로 작성해주세요.
    
    제품명: {request.product_name}
    강조할 키워드: {request.keywords}
    
    광고 문구:
    """

    #  Step 4: Bedrock(Claude)에게 보낼 편지 봉투 만들기 (JSON 변환)
    
    # Q: "json.dumps()가 도대체 뭔가요? 이걸 왜 해야 하나요?"
    # A: "파이썬 딕셔너리(Dict)"를 "전송 가능한 문자열(String)"로 변환하는 과정입니다.
    #
    #    1. [Python Dict]: `{"a": 1}` -> 이건 파이썬 프로그램 안에서만 존재하고, 
    #       네트워크선을 타고 날아갈 수 없습니다. (메모리 덩어리)
    #    2. [JSON String]: `'{"a": 1}'` -> 이건 그냥 "글자(Text)"입니다.
    #       네트워크를 통해 AWS 서버로 쓩 날려보낼 수 있습니다.
    #
    #    ⚠️ `dumps` = Dump to String (문자열로 쏟아내다)의 약자입니다.
    #    AWS는 파이썬을 모릅니다. 만국 공용어인 JSON 문자열('글자')로 바꿔서 보내줘야 합니다.

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",  # Claude API 버전 (고정값)
        "max_tokens": settings.MAX_TOKENS,           # 설정에서 가져온 최대 토큰 수
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    try:
        #  Step 5: AWS Bedrock 부서와 '통화 연결' 하기 (Client 생성)
        
        # Q: "async with session.client(...) 이게 뭔가요?"
        # A: "Bedrock 부서(service_name)에 전화를 걸어서, 통화 중인 상태(client)"를 만드는 것입니다.
        #
        #    1. [session.client]: "Bedrock 담당자 바꿔주세요" 하고 연결을 시도합니다.
        #    2. [async with]: 이게 핵심입니다! "통화가 끝나면 자동으로 끊어주세요" 라는 뜻입니다.
        #       - 이 블록(들여쓰기)이 끝나면, 성공하든 에러가 나든 무조건 연결을 확실하게 닫습니다.
        #       - 이걸 안 하면, 전화기가 계속 '통화 중' 상태로 남아있어서 나중에 먹통이 됩니다. (Resource Leak)
        
        async with session.client(
            service_name='bedrock-runtime',          # "Bedrock 실행(Runtime) 부서 연결해주세요"
            region_name=settings.AWS_REGION,         # "미국 동부(Virginia) 지점으로요"
            aws_access_key_id=settings.AWS_ACCESS_KEY,     # "제 ID는 ... 입니다"
            aws_secret_access_key=settings.AWS_SECRET_KEY  # "비밀번호는 ... 입니다"
        ) as client:
            
            #  로그 기록 (연결 성공 확인용)
            logger.info(f"🔗 AWS Bedrock 연결 성공, 모델 호출 중: {settings.MODEL_ID}")

            # [실무 궁금증] "LangGraph나 LangChain은 안 쓰나요?"
            # A: 지금 같은 "단순 1회성 요청(Single Turn)"에는 과한 기술(Over-engineering)입니다.
            #    - LangGraph: "검색 -> 판단 -> 재질문"처럼 복잡한 '상태 관리'가 필요할 때 씁니다.
            #    - Direct SDK(현재 방식): 단순 요청은 이렇게 직접 짜는 게 훨씬 빠르고 가볍습니다.
            
            #  Step 6: AWS에 'JSON 편지' 보내고 답장 기다리기
            
            # Q: "지금 JSON을 던지고, JSON을 받는 건가요?"
            # A: 네, 정확합니다! (JSON 티키타카)
            #    1. [보낼 때] body(JSON 문자열)를 `contentType="application/json"`이라고 스티커 붙여서 보냅니다.
            #    2. [받을 때] AWS도 결과를 `accept="application/json"` 주문대로 JSON으로 포장해서 보내줍니다.
            #    -> 즉, 서로 'JSON 언어'로만 대화하는 것입니다.

            # await client.invoke_model() 상세 동작:
            #   1. AWS에 HTTPS POST 요청 전송 (JSON 발송)
            #   2. 응답 대기 중 다른 요청 처리 가능 (Non-blocking)
            #   3. 응답이 오면 response 변수에 할당 (아직은 포장된 상태)
            
            # Q: "ainvoke는 아닌가요? LangChain에서는 ainvoke 쓰던데?"
            # A: 좋은 지적입니다! 하지만 여기서는 **invoke_model**이 맞습니다.
            #    - `ainvoke`: **LangChain** 라이브러리에서 쓰는 함수 이름입니다.
            #    - `invoke_model`: **AWS 공식 SDK(boto3)**에서 정해놓은 함수 이름입니다.
            #    우리는 지금 LangChain 없이 AWS를 '직접' 찌르고 있으므로, AWS가 정한 이름을 써야 합니다.

            response = await client.invoke_model(
                body=body,                           # 위에서 만든 JSON body
                modelId=settings.MODEL_ID,           # Claude 3 Haiku 모델 ID
                accept="application/json",           # 응답받을 형식
                contentType="application/json"       # 보내는 형식
            )
            
            #  Step 7: "택배 상자" 뜯어서 내용물 꺼내기 (Stream Read)
            # response['body']는 스트림(Stream) 객체입니다. 데이터가 물처럼 흡러옵니다.
            # await .read() 호출 → 전체 데이터가 도착할 때까지 대기 → 바이트로 반환
            
            # Q: "왜 굳이 바이트(Bytes)로 받나요? 그냥 바로 텍스트로 주면 안 되나요?"
            # A: 네트워크의 기본 원리 때문입니다.
            #    1. 인터넷 선을 타고 오는 모든 데이터는 근본적으로 0과 1인 'Bytes'입니다.
            #    2. AWS(Boto3)는 응답이 엄청 클 수도 있어서(예: 1GB 파일), 
            #       한 번에 다 주지 않고 수도꼭지(Stream)처럼 연결만 해둡니다. (`StreamingBody`)
            #    3. 우리가 직접 `read()`를 호출해야 비로소 물(데이터)을 양동이(메모리)에 받아옵니다.
            #    -> 그래서 `read()`의 결과는 항상 원재료인 'Bytes' 상태입니다.
            response_body_bytes = await response['body'].read()
            
            #  Step 8: JSON 파싱
            # json.loads() 호출 → 바이트 데이터를 파이썬 dict로 변환
            
            # Q: "response_body_bytes.decode('utf-8')을 안 해도 되나요?"
            # A: 네, 필수는 아닙니다! 파이썬 `json.loads()`가 똑똑해서 Bytes 타입을 넣어도
            #    알아서 UTF-8로 디코딩해서 처리해줍니다.
            #    물론 `.decode('utf-8')`을 명시적으로 써주는 것도 아주 좋은 습관입니다. (명확하니까요)
            
            response_body = json.loads(response_body_bytes)
            
            #  Step 9: 복잡한 포장지 속에서 '알맹이'만 꺼내기
            # dict 인덱싱: response_body['content'] → 리스트 → [0] → dict → ['text'] → 문자열
            
            # Q: "['content'][0]['text']... 이게 무슨 암호인가요?"
            # A: Claude가 보내준 응답 데이터(JSON)의 구조를 보면 이해가 됩니다.
            #
            #    response_body = {
            #        "id": "msg_...",
            #        "role": "assistant",
            #        "content": [                 <-- 1. ['content'] : 리스트(List) 시작
            #            {                        <-- 2. [0] : 첫 번째 덩어리 선택
            #                "type": "text",
            #                "text": "광고 문구..."  <-- 3. ['text'] : 실제 글자 꺼내기
            #            }
            #        ]
            #    }
            #
            #    -> 양파 껍질 까듯이 하나씩 파고 들어가서 문자열을 가져오는 것입니다.
            result_text = response_body['content'][0]['text']
            
            #  Step 10: 성공 로그 기록
            logger.info(f"✅ 광고 문구 생성 성공! 길이: {len(result_text)}자")
            
            #  Step 11: 최종 결과물을 예쁘게 포장해서 고객에게 전달
            # AdResponse() 생성자 호출 → Pydantic이 자동으로 JSON으로 변환 → 클라이언트에게 반환
            #
            # Q: "그냥 딕셔너리로 리턴하면 안 되나요? 왜 AdResponse 객체를 쓰죠?"
            # A: 딕셔너리로 줘도 되지만, **AdResponse(Pydantic 모델)**를 쓰면 안전 장치가 생깁니다.
            #
            #    1. [자동 검증]: 실수로 숫자(int)가 들어갈 자리에 문자(str)를 넣으면 알아서 잡아줍니다.
            #    2. [자동 변환]: FastAPI가 이 객체를 받아서 자동으로 JSON으로 바꾼 뒤, 클라이언트에게 보냅니다.
            #    3. [문서화]: Swagger UI API 문서에 "이 API는 이런 모양의 데이터를 줍니다"라고 자동으로 표시됩니다.
            
            # AdResponse() 생성자 호출:
            # - result: Claude가 만든 실제 광고 문구
            # - model_used: 어떤 모델이 만들었는지 (증거 남기기)
            return AdResponse(
                result=result_text,
                model_used=settings.MODEL_ID
            )

    except client.exceptions.ValidationException as e:
        # [에러 처리 1] AWS 입력 검증 에러
        # 프롬프트가 너무 길거나 형식이 잘못된 경우
        logger.error(f"❌ AWS 입력 검증 에러: {str(e)}")
        raise HTTPException(status_code=400, detail=f"입력 검증 에러: {str(e)}")
    
    except client.exceptions.ThrottlingException as e:
        # [에러 처리 2] AWS 요청 제한 에러
        # 너무 많은 요청을 보내서 AWS가 잠시 막은 경우
        logger.warning(f"⚠️ AWS 요청 제한 (Rate Limit): {str(e)}")
        raise HTTPException(status_code=429, detail="요청이 너무 많습니다. 잠시 후 다시 시도하세요.")
    
    except Exception as e:
        # [에러 처리 3] 기타 모든 에러
        # AWS 연결 실패, 네트워크 문제 등
        logger.error(f"❌ 예상치 못한 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")


@app.get("/")
def read_root():
    """
    헬스 체크 엔드포인트
    
    서버가 정상적으로 실행 중인지 확인하는 용도입니다.
    실무에서는 로드 밸런서나 Kubernetes가 이 엔드포인트를 주기적으로 호출해서
    서버 상태를 확인합니다.
    """
    return {
        "message": "AWS Bedrock (Async) 서비스가 실행 중입니다!",
        "model": settings.MODEL_ID,
        "region": settings.AWS_REGION,
    }


@app.get("/health")
def health_check():
    """
    상세 헬스 체크 엔드포인트
    
    서버 상태뿐만 아니라 설정 정보도 함께 반환합니다.
    (보안상 프로덕션에서는 이 정보를 숨기거나 인증을 요구해야 합니다!)
    """
    return {
        "status": "healthy",
        "session_initialized": app_state.get("session") is not None,
        "config": {
            "region": settings.AWS_REGION,
            "model": settings.MODEL_ID,
            "max_tokens": settings.MAX_TOKENS,
        }
    }


# ============================================================
# ZONE 9: 서버 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    # 로그 출력: 서버 시작 전 안내
    logger.info("=" * 50)
    logger.info("🚀 AWS Bedrock 광고 생성기 서버를 시작합니다...")
    logger.info(f"📍 API 문서: http://localhost:8000/docs")
    logger.info("=" * 50)
    
    # Uvicorn으로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
