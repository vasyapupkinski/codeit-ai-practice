"""
🎯 FastAPI Lab 4 과제: RAG & History (혼자해보기)

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **Lab 4의 과제 1, 2를 구현하기 위한 템플릿**입니다.
기본 RAG 시스템을 바탕으로 다음 기능을 추가합니다:

**[기본 구현] 과제 0: 기본 RAG (완성됨)**
- 강의계획서 내용을 기반으로 답변하는 기본 RAG
- /ask-syllabus 엔드포인트

**[심화] 과제 1: 대화 히스토리 추가 (⭐⭐ 심화)**
- 이전 대화 내용을 기억하는 기능
- session_id를 통해 사용자별 대화 구분
- 프롬프트에 이전 대화 내용 포함

**[도전] 과제 2: 다중 문서 RAG (⭐⭐⭐ 도전)**
- API로 새로운 문서를 실시간으로 추가
- 추가된 문서들도 검색 대상에 포함
- /documents (문서 추가), /ask-all (전체 검색)

===============================================
💡 핵심 학습 포인트
===============================================
- **Memory**: 대화 맥락을 유지하는 방법 (Dictionary 활용)
- **VectorStore Update**: 실행 중에 새로운 지식 추가하기
- **Prompt Engineering**: History를 프롬프트에 주입하는 법

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install langchain-openai langchain-community chromadb tiktoken
   - .env 파일에 OPENAI_API_KEY 설정

2. 실행 방법:
   python ./lab4/rag_qa_practice.py
"""

# [파일 구조 및 순서 설명 (Map)]
# 이 파일은 위에서부터 아래로 읽히며 실행됩니다. 순서가 뒤바뀌면 에러가 납니다.
#
# 1. [재료 준비] 임포트 (Import)
#    - 요리에 필요한 도구(라이브러리)를 책상에 펼쳐놓는 단계입니다.
#    - 당연히 가장 먼저 해야 합니다. 도구 없이 요리 못하니까요.
#
# 2. [전역 설정] 환경변수 & 전역변수
#    - 요리할 때 계속 쓸 큰 그릇(변수)들을 미리 꺼내둡니다.
#
# 3. [개업 준비] Lifespan (수명주기)
#    - 가게 문 열기 전에 육수 끓이는 로직을 정의합니다.
#    - 아직 실행되는 건 아니고, "이렇게 준비할 거야"라고 계획표를 짜는 것입니다.
#
# 4. [주문서 양식] Pydantic 스키마 (Schema)
#    - "손님, 주문은 이 종이에 적어주세요"라고 양식을 만듭니다.
#    - **중요**: 이 양식(Class)이 있어야, 뒤에 나오는 API 함수(Handler)에서
#      "이 양식대로 주문 받았니?"라고 검사할 수 있습니다.
#    - 그래서 반드시 API 함수보다 **앞서서(위에)** 정의해야 합니다. (*가장 많이 묻는 질문)
#
# 5. [주문 처리] API 엔드포인트 (@app.post)
#    - 실제 손님을 응대하는 점원들입니다.
#    - 위에서 만든 '도구', '그릇', '양식'을 전부 사용합니다.
#    - 그래서 파일의 가장 **마지막**에 위치합니다.

#  필수 모듈 임포트
# (1) FastAPI 관련:
#    - FastAPI: 웹 서버를 만드는 도구. API 엔드포인트(/ask 등)를 생성합니다.
#    - HTTPException: 404(없음), 500(서버에러) 같은 HTTP 에러를 발생시킬 때 씁니다.
from fastapi import FastAPI, HTTPException # [초보] fastapi 모듈에서 FastAPI, HTTPException 클래스를 가져옵니다

# (2) Pydantic (데이터 검증):
#    - BaseModel: 입출력 데이터의 '설계도(Schema)'를 만드는 클래스입니다.
#    - 왜 쓰는가? "문자열 주세요" 했는데 "숫자"가 오면 에러를 내뿜어줍니다. (자동 검증)
from pydantic import BaseModel # [초보] pydantic 모듈에서 BaseModel 클래스를 가져옵니다

# (3) Lifespan (수명주기 관리):
#    - asynccontextmanager: 서버가 "켜질 때"와 "꺼질 때" 딱 한 번씩 실행할 코드를 정의합니다.
#    - 왜 쓰는가? DB 연결이나 AI 모델 로딩처럼 "오래 걸리는 작업"을 서버 켤 때 미리 해두기 위함입니다.
from contextlib import asynccontextmanager # [초보] contextlib 모듈에서 asynccontextmanager 함수를 가져옵니다

# (4) 환경 변수 관리:
#    - dotenv: .env 파일에 숨겨둔 비밀번호(API Key)를 코드로 불러옵니다.
#    - os: 운영체제의 환경 변수에 접근합니다. (os.getenv)
import os # [초보] os 모듈을 가져옵니다 (환경변수 사용)
from dotenv import load_dotenv # [초보] dotenv 모듈에서 load_dotenv 함수를 가져옵니다

#  LangChain 관련 임포트
# LangChain은 "AI 개발을 위한 레고 블록"입니다. 필요한 블록을 가져와서 조립합니다.

# (1) 모델 및 임베딩 (두뇌와 번역기):
#    - ChatOpenAI: GPT-4 같은 '언어 모델(두뇌)'입니다. 질문을 이해하고 답변을 합니다.
#    - OpenAIEmbeddings: 텍스트를 '숫자 리스트(벡터)'로 바꾸는 '번역기'입니다.
#      (예: "사과" -> [0.1, 0.5, ...]) 컴퓨터는 글자가 아니라 숫자로 의미를 이해하기 때문입니다.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # [초보] langchain_openai 모듈에서 ChatOpenAI와 OpenAIEmbeddings 클래스를 가져옵니다

# (2) 프롬프트 및 파싱 (지시서와 통역사):
#    - ChatPromptTemplate: GPT에게 보낼 '지시서 양식'을 만듭니다.
#      (예: "너는 친절한 봇이야. {질문}에 답해줘.")
#    - StrOutputParser: GPT의 복잡한 응답 객체에서 알맹이(텍스트)만 쏙 뽑아냅니다.
#    - RunnablePassthrough: 파이프라인에서 데이터를 "그냥 토스"해주는 역할을 합니다.
from langchain_core.prompts import ChatPromptTemplate # [초보] langchain_core.prompts 모듈에서 ChatPromptTemplate 클래스를 가져옵니다
from langchain_core.output_parsers import StrOutputParser # [초보] langchain_core.output_parsers 모듈에서 StrOutputParser 클래스를 가져옵니다
from langchain_core.runnables import RunnablePassthrough # [초보] langchain_core.runnables 모듈에서 RunnablePassthrough 클래스를 가져옵니다

# (3) 벡터 저장소 및 문서 (도서관과 책):
#    - Chroma: 변환된 숫자 벡터들을 저장하는 '전용 도서관'입니다. (빠른 검색 지원)
#    - Document: 텍스트와 메타데이터(페이지 번호, 제목 등)를 담는 '책' 표지입니다.
# [실무 패턴] operator.itemgetter
# - 딕셔너리에서 특정 키의 값을 뽑아내는 속도가 아주 빠른 도구입니다.
# - RAG 체인에서 {"question": ...} 처럼 데이터가 흐를 때, 값을 쏙쏙 뽑아내기 위해 씁니다.
from langchain_community.vectorstores import Chroma # [초보] langchain_community.vectorstores 모듈에서 Chroma 클래스를 가져옵니다
from langchain_core.documents import Document # [초보] langchain_core.documents 모듈에서 Document 클래스를 가져옵니다
from operator import itemgetter # [실무 패턴] 딕셔너리 값 추출용

#  환경변수 로드
# - .env 파일 내용을 읽어서 os가 알 수 있게 메모리에 올립니다.
load_dotenv() # [초보] load_dotenv() 함수를 호출하여 .env 파일을 읽고 환경변수로 등록합니다

# [전역 변수]
# [개념] 왜 전역 변수(Global Variable)를 쓰는가?
# - FastAPI는 여러 사용자의 요청을 동시에 처리(비동기)합니다.
# - 하지만 'AI 모델'이나 'DB 연결'은 아주 무겁습니다. 요청마다 새로 만들면 서버가 터집니다.
# - 그래서 전역 변수에 딱 하나만 만들어두고, 모든 요청이 이를 '공유'해서 씁니다.
# - 마치 식당에 정수기는 하나만 있고 손님들이 공유하는 것과 같습니다.

# 1. RAG 체인 (기본 과제용)
rag_chain = None # [초보] rag_chain 변수에 None 값을 할당합니다

# [개념] 왜 Vector Store를 전역 변수로 관리하는가? (과제 2)
# - 원칙적으로는 Lifespan 내부에서만 존재해도 됨 (retriever 만들고 끝)
# - 하지만 "실시간 문서 추가" 기능을 위해 외부에서 접근 필요
# - /documents 엔드포인트에서 이 변수에 접근하여 add_documents 호출

# 2. 벡터 저장소 (과제 2에서 문서 추가를 위해 전역 변수로 관리)
vectorstore = None # [초보] vectorstore 변수에 None 값을 할당합니다

# 3. 대화 히스토리 저장소 (과제 1용)
# 형식: {"session_id": ["Q: 안녕", "A: 반가워", ...]}
# [실무 주의] 현재는 파이썬 딕셔너리(RAM)에 저장하므로 서버를 껐다 켜면 다 날아갑니다. (휘발성)
# [실무 패턴] 실제 서비스에서는 Redis, DynamoDB, PostgreSQL 같은 '데이터베이스'에 저장해야 합니다.
#            그래야 서버가 재시작되어도 사용자 대화 기록이 유지됩니다.
chat_histories = {} # [초보] chat_histories 변수에 빈 딕셔너리 {}를 할당합니다

# [지식 데이터] 기본 강의계획서
syllabus_text = """
[FastAPI 및 AI 웹 개발 과정]
1주차: Python 기초 및 FastAPI 기본 구조
2주차: Pydantic 데이터 검증 및 비동기 처리
3주차: Hugging Face Transformers 활용
4주차: OpenAI API 및 LangChain 기초 (RAG)
5주차: LangGraph 에이전트 및 Streamlit 실습
평가: 출석 20%, 과제 30%, 프로젝트 50%
"""

# [Lifespan] 서버 시작 시 벡터 DB 구축
# [문법 설명] @asynccontextmanager가 뭔가요?
# - 파이썬의 '데코레이터(Decorator)' 문법입니다.
# - 역할: 아래 정의된 `lifespan` 함수를 "앞-뒤로 쪼개서 실행할 수 있는 특수한 함수"로 변신시킵니다.
#   1. `yield` 윗부분 -> (진입, Enter) -> 서버 켤 때 실행
#   2. `yield` 아랫부분 -> (종료, Exit) -> 서버 끌 때 실행
# - 만약 이 @가 없으면? 그냥 평범한 함수라서 한 번 실행하고 끝납니다. (서버 유지 불가)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
     서버 수명주기 관리 (Lifespan)
    
    이 함수는 서버가 켜지고 꺼지는 **전체 시간**을 관리합니다.
    순서가 헷갈리신다면 '식당 개업'을 상상해 보세요.
    
    [시간 순서 1: 개업 준비] (서버 시작 직후, yield 전)
    - 손님을 받기 전에 미리 재료를 손질하고 육수를 끓여놔야 합니다.
    - 여기서 무거운 작업(AI 모델 로딩, DB 구축)을 **딱 한 번만** 미리 해둡니다.
    - 만약 이걸 미리 안 해두면? 손님이 올 때마다 육수를 처음부터 다시 끓여야 합니다. (속도 느림)
       
    [시간 순서 2: 영업 시작] (yield)
    - 준비가 다 끝났으니 "영업 시작!" 팻말을 겁니다.
    - 이제부터 들어오는 요청(Request)을 처리할 수 있습니다.
    - 서버가 꺼질 때까지 여기서 대기합니다.
       
    [시간 순서 3: 영업 종료] (yield 이후)
    - 가게 문을 닫고 청소를 합니다. (리소스 정리)
    """
    print("🔄 [시작] 벡터 DB 구축 중...")
    
    #  전역 변수 선언
    # - rag_chain: 우리가 만든 AI 로봇 (모든 곳에서 씀)
    # - model, retriever: 추후 확장을 위해 전역으로 열어둠
    global rag_chain, model, retriever, vectorstore
    
    # [순서의 이유: 레고 조립과 같습니다]
    # 아래 코드의 순서는 내 마음대로 정한 것이 아니라, "앞에게 없으면 뒤에걸 못 만들기" 때문입니다.
    #
    # 1. '내용'이 있어야 책을 만들죠? -> (Docs 생성)
    # 2. '책'이 있어야 도서관에 꽂죠? -> (VectorStore 생성, Docs 필요)
    # 3. '도서관'이 있어야 사서를 고용하죠? -> (Retriever 생성, VectorStore 필요)
    # 4. '사서'와 '작가(AI)'가 있어야 일을 시키죠? -> (Chain 생성, Retriever & Model 필요)
    #
    # 그래서 반드시 이 순서대로 작성해야 합니다. 역순은 불가능합니다.
    
    # 1. 문서 생성
    # - "강의계획서"라는 텍스트를 포장지에 싸서 Document 객체로 만듭니다.
    docs = [Document(page_content=syllabus_text, metadata={"source": "강의계획서"})] # [초보] Document 객체를 생성하여 리스트에 넣고 docs 변수에 할당합니다
    
    # 2. 벡터 DB 생성 (핵심!)
    # - Chroma.from_documents()가 하는 일:
    #   a) docs 안의 텍스트를 OpenAI에 보냅니다.
    #   b) OpenAI가 [0.12, -0.98, ...] 같은 1536개 숫자(벡터)로 변환해줍니다. (임베딩)
    #   c) 이 벡터를 ChromaDB(내장 DB)에 저장합니다.
    # - 이제 컴퓨터는 이 텍스트의 "의미"를 숫자로 기억합니다.
    vectorstore = Chroma.from_documents(
        documents=docs,               # [초보] documents 매개변수에 docs 리스트를 전달합니다
        embedding=OpenAIEmbeddings(), # [초보] embedding 매개변수에 OpenAIEmbeddings 객체를 전달합니다
        collection_name="practice_collection" # [초보] collection_name 매개변수에 문자열을 전달합니다
    ) # [초보] Chroma.from_documents 메서드를 호출하고 반환값을 vectorstore 변수에 할당합니다
    
    # 3. Retriever 생성
    # - "검색기"입니다. 질문(Query)이 들어오면 벡터 DB에서 유사한 문서를 찾아줍니다.
    retriever = vectorstore.as_retriever() # [초보] vectorstore 객체의 as_retriever 메서드를 호출하고 반환값을 retriever 변수에 할당합니다
    
    # 4. 프롬프트 템플릿
    # - GPT에게 보낼 편지 양식입니다.
    # - {context}: 검색기가 찾은 문서 내용을 여기에 끼워넣습니다.
    # - {question}: 사용자의 질문을 여기에 끼워넣습니다.
    # [실무 패턴] 만능 단일 체인 (Universal Chain)
    # - 하나의 체인으로 "일반 대화"와 "히스토리 대화"를 모두 처리합니다.
    # - 핵심: history 필드가 들어오면 쓰고, 없으면 무시하는 유연한 구조 설계
    
    # 1. 유연한 프롬프트 정의
    # - {history}: 대화 기록 (있으면 들어가고, 없으면 빈칸)
    # - {context}: 검색된 문서
    # - {question}: 현재 질문
    template = """
    {history}

    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 2. 모델 생성
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # 3. 체인 조립 (itemgetter 활용)
    # - RunnablePassthrough.assign(): 들어온 입력(dict)에 새로운 키-값을 추가합니다.
    # - history 키가 없으면 기본값 ""(빈 문자열)을 넣어 에러를 방지합니다.
    rag_chain = (
        {
            # [여기가 핵심!] 전역 변수 'retriever' 사용
            # - 이 retriever는 vectorstore와 연결되어 있습니다.
            # - vectorstore에 문서가 추가되면, retriever는 자동으로 그 새 문서도 뒤적거립니다. (실시간 반영)
            "context": itemgetter("question") | retriever,    
            "question": itemgetter("question"),               
            "history": itemgetter("history")                  
        }

        | prompt
        | model
        | StrOutputParser()
    )
    
    print("✅ [완료] RAG 시스템 준비됨")
    yield # [초보] yield 문으로 제어를 호출자에게 반환하고 서버 실행 중 대기합니다
    print("🛑 [종료] 시스템 종료")

# [서버 앱 생성 및 수명주기 등록]
# [문법 설명] app = FastAPI(lifespan=lifespan) 이게 뭔가요?
# 1. FastAPI(): "서버 본체"를 만드는 생성자 함수입니다.
# 2. lifespan=lifespan:
#    - "이 서버의 수명(Life)은 저기 위에 정의한 `lifespan` 함수가 관리한다"라고 **등록**하는 것입니다.
#    - 왼쪽 `lifespan`: FastAPI가 정해놓은 설정 옵션 이름 (매개변수명)
#    - 오른쪽 `lifespan`: 위에서 우리가 만든 함수 이름 (변수명)
#    - 이걸 연결해 줘야 서버가 켜질 때 `lifespan` 함수를 자동으로 실행합니다. 안 적으면 실행 안 됩니다.
app = FastAPI(lifespan=lifespan)

# [요청 스키마 (Request Schema)]
# [개념] 도대체 스키마(Schema)가 뭔가요?
# - 스키마는 "주문서 양식"입니다.
# - 사용자가 API에 요청을 보낼 때, 아무렇게나 보내면 안 됩니다.
# - "반드시 `question`이라는 항목이 있어야 하고, 내용은 `글자(string)`여야 해!"라고 정하는 규칙입니다.
# - Pydantic의 BaseModel을 상속받으면, 이 규칙 검사를 100% 자동화해줍니다.
# - 만약 사용자가 숫자를 보내면? → FastAPI가 알아서 "너 형식 틀렸어"라고 에러를 뱉습니다.

# [문법 설명] 이것이 바로 'Pydantic(파이단틱)'입니다!
# - (BaseModel)을 상속받으면, 이 클래스는 "엄격한 검문소"가 됩니다.
# - 들어오는 데이터가 정해진 타입(str 등)이 아니면 에러를 내고 쫓아냅니다.
class QuestionRequest(BaseModel):
    # [초보] 사용자의 질문을 받는 틀
    question: str  # 필수 항목: 문자열

class HistoryQuestionRequest(BaseModel):
    # [초보] 대화 내용(History)을 위한 질문 틀
    question: str # [초보] 질문 내용 (문자열)
    session_id: str = "default"  # 선택 항목: 안 보내면 자동으로 "default"가 됨 (과제 1용)

class DocumentRequest(BaseModel):
    # [초보] 문서를 추가할 때 쓰는 틀 (과제 2용)
    title: str # [초보] 문서 제목
    content: str # [초보] 문서 내용

# ==========================================
# [기본] 과제 0: 기본 RAG (이미 구현됨)
# ==========================================
@app.post("/ask-syllabus")
async def ask_syllabus(request: QuestionRequest):
    """
    기본 강의계획서 질문 (프로덕션 패턴)
    
    [실무 패턴] 왜 async/await를 함께 쓰는가?
    - **async def**: "이 함수는 비동기로 동작할 수 있어요"라고 선언
    - **await**: "이 작업(AI 호출)이 끝날 때까지 다른 요청을 처리하세요"라고 지시
    
    [성능 차이]
    - ❌ invoke() (동기): 10명 동시 접속 시 → 마지막 사람은 50초 대기
    - ✅ await ainvoke() (비동기): 10명 동시 접속 시 → 모두 5초 안에 응답
    
    [실무 필수 사항]
    - OpenAI, DB, 외부 API 등 네트워크 I/O는 **반드시 await 사용**
    - 안 쓰면 서버가 한 명씩 순차 처리 → 트래픽 폭증 시 서버 다운
    """
    #  1. 예외 처리를 위한 try 블록 시작
    # - 에러가 발생할 수 있는 위험한 코드들을 감쌉니다.
    try:
        #  2. 입력 데이터 포장
        # - 체인이 좋아하는 딕셔너리 형태로 포장합니다.
        # [프로덕션 패턴] 비동기 체인 실행 (단일 체인 구조)
        # - 이전과 달리 입력값이 딕셔너리 형태여야 합니다. (itemgetter 때문)
        # - history는 없으므로 빈 문자열("") 또는 생략하면 itemgetter가 처리
        input_data = {
            "question": request.question,
            "history": "" # 히스토리 없는 일반 질문
        }
        #  3. RAG 체인 실행 (비동기)
        # - await: "답변 올 때까지 다른 손님 받고 있을게" (Non-blocking)
        # - ainvoke: "비동기로 실행해줘"
        response = await rag_chain.ainvoke(input_data)
    except Exception as e: # [초보] 예외가 발생하면 except 블록을 실행합니다
        #  4. 에러 잡기
        # - 위에서 무슨 일이든 터지면(Exception) 여기서 잡습니다.
        raise HTTPException(status_code=500, detail=str(e)) # [초보] HTTPException 객체를 생성하여 raise문으로 예외를 발생시킵니다

# ==========================================
# [심화] 과제 1: 대화 히스토리 추가
# ==========================================
@app.post("/ask-with-history")
async def ask_with_history(request: HistoryQuestionRequest):
    """
    [과제 1] 이전 대화를 기억하는 질문 엔드포인트
    
    [개념] 대화 히스토리 (Conversation History)
    - LLM은 기본적으로 'Stateless'(무상태)입니다. 이전 대화를 기억하지 못합니다.
    - 따라서 우리가 매번 이전 대화 내용을 프롬프트에 '문자열'로 넣어줘야 합니다.
    - 이를 'Memory' 기능이라고 합니다.
    
    TODO: 아래 단계를 구현하세요.
    1. chat_histories 딕셔너리에서 session_id에 해당하는 기록 가져오기
    2. 기록을 문자열로 포맷팅 (예: "User: 안녕\nAI: 반가워\n...")
    3. 프롬프트 템플릿 수정: {history} 변수 추가
    4. 체인 실행 시 {"history": history_str, ...} 전달
    5. 응답 받은 후, 현재 질문과 답변을 chat_histories에 추가
    """
    #  1. 예외 처리를 위한 try 블록 시작
    # - 에러가 발생할 수 있는 위험한 코드들을 감쌉니다.
    try:
        #  2. 히스토리 조회 (메모리 or DB)
        # - 딕셔너리에서 session_id에 해당하는 대화 기록을 가져옵니다.
        # [실무 패턴] 히스토리 조회
        # - session_id가 없으면 빈 리스트로 초기화
        # - setdefault(): 키가 있으면 값 반환, 없으면 빈 리스트 저장 후 반환
        history_list = chat_histories.setdefault(request.session_id, [])
        #  3. 히스토리 문자열 변환
        # - 리스트에 있는 대화들을 줄바꿈(\n)으로 합쳐서 하나의 긴 문자열로 만듭니다.
        history_str = "\n".join(history_list)
        
        #  4. 입력 데이터 포장
        # - 체인이 좋아하는 딕셔너리 형태로 포장합니다.
        # [실무 패턴] 단일 체인 재사용 (Single Chain Reuse)
        # - 이전처럼 체인을 새로 조립하지 않습니다. (오버헤드 방지)
        # - 만들어둔 rag_chain에 "history" 데이터만 추가해서 보냅니다.
        input_data = {
            "question": request.question,
            "history": history_str
        }
        #  5. RAG 체인 실행 (비동기)
        # - await: "답변 올 때까지 다른 손님 받고 있을게" (Non-blocking)
        response = await rag_chain.ainvoke(input_data)
        
        #  6. 대화 기록 업데이트 (RAM)
        # - 다음 대화를 위해 현재 질문과 답변을 저장해둡니다.
        # [실무 패턴] 대화 기록 저장
        # - AI 응답이 성공했을 때만 기록에 추가
        history_list.append(f"User: {request.question}")
        history_list.append(f"AI: {response}")
        
        # [실무 패턴] 여기서 DB/Redis에 저장
        # - redis_client.rpush(f"history:{session_id}", question, response)
        # - 그래야 서버가 재시작되어도 기억이 유지됨
        
        #  7. 결과 반환
        # - 사용자에게 JSON 형태로 답변을 줍니다.
        return {"answer": response}
        
    #  8. 에러 잡기
    # - 위에서 무슨 일이든 터지면(Exception) 여기서 잡습니다.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# [도전] 과제 2: 다중 문서 RAG
# ==========================================
@app.post("/documents")
async def add_document(doc: DocumentRequest):
    """
    [과제 2-1] 새로운 문서 추가
    
    TODO:
    1. Document 객체 생성 (content=doc.content, metadata={"title": doc.title})
    2. vectorstore.add_documents([new_doc]) 호출
    
    [개념] 벡터 DB 업데이트의 의미
    - 문서를 추가하면 즉시 임베딩 변환되어 저장됨
    - 별도로 Retriever를 재생성할 필요 없음 (연결되어 있음)
    """
    #  1. 예외 처리를 위한 try 블록 시작
    # - 에러가 발생할 수 있는 위험한 코드들을 감쌉니다.
    try:
        # TODO: 여기에 구현
        
        
        #  2. 결과 반환
        # - 사용자에게 성공 메시지를 JSON으로 줍니다.
        return {"status": "success", "message": f"Document '{doc.title}' added."} # [초보] 딕셔너리를 생성하여 return문으로 반환합니다
    except Exception as e: # [초보] 예외가 발생하면 except 블록을 실행합니다
        #  3. 에러 잡기
        # - 위에서 무슨 일이든 터지면(Exception) 여기서 잡습니다.
        raise HTTPException(status_code=500, detail=str(e)) # [초보] HTTPException 객체를 생성하여 raise문으로 예외를 발생시킵니다

@app.post("/ask-all")
async def ask_all_documents(request: QuestionRequest):
    """
    [과제 2-2] 전체 문서 대상 검색 및 답변
    
    [개념] 동적 지식 베이스
    - RAG의 강력함은 '데이터가 변해도 재학습 없이 반영된다'는 것입니다.
    - /documents로 문서를 추가하면 vectorstore에 실시간으로 반영됩니다.
    - rag_chain은 retriever를 참조하고 있고, retriever는 vectorstore를 참조하므로
      자동으로 최신 데이터를 검색하게 됩니다.
    
    [개념] 왜 코드가 ask-syllabus랑 똑같은가요?
    - 맞습니다. 코드(로직)는 100% 동일합니다.
    - 하지만 **"데이터"가 다릅니다.**
    - 1번(ask-syllabus) 때는 DB에 '강의계획서'만 들어있었습니다.
    - 2번(ask-all) 때는 사용자가 /documents로 '점심 메뉴' 등을 추가한 상태입니다.
    - 같은 기계(retriever)를 돌리지만, 창고(Vector Store)에 물건이 늘어났으니 더 많은 대답을 할 수 있는 것입니다.
    """
    #  1. 예외 처리를 위한 try 블록 시작
    # - 에러가 발생할 수 있는 위험한 코드들을 감쌉니다.
    try:
        #  2. 안전장치: RAG 체인이 준비되었는지 확인
        # - 서버가 켜질 때(lifespan) 만들어졌어야 하는데, 혹시나 없으면 에러를 냅니다.
        if rag_chain is None: # [초보] rag_chain이 None인지 비교합니다
            raise HTTPException(status_code=500, detail="RAG Chain not initialized") # [초보] 입구컷: 500 에러 발생
            
        #  3. 입력 데이터 포장
        # - 체인이 좋아하는 딕셔너리 형태로 포장합니다.
        # - {"question": "점심 뭐먹지?", "history": ""}
        input_data = {"question": request.question, "history": ""}
        
        #  4. AI에게 질문 던지기 (비동기)
        # - await: "답변 올 때까지 다른 손님 받고 있을게" (Non-blocking)
        # - ainvoke: "비동기로 실행해줘"
        response = await rag_chain.ainvoke(input_data)
        
        #  5. 결과 반환
        # - 사용자에게 JSON 형태로 답변을 줍니다.
        return {"answer": response} # [초보] 딕셔너리를 생성하여 return문으로 반환합니다

    #  6. 에러 잡기
    # - 위에서 무슨 일이든 터지면(Exception) 여기서 잡습니다.
    except Exception as e: # [초보] 예외가 발생하면 except 블록을 실행합니다
        # - 사용자에게 "서버 내부 에러(500)입니다"라고 정중히 알립니다.
        raise HTTPException(status_code=500, detail=str(e)) # [초보] HTTPException 객체를 생성하여 raise문으로 예외를 발생시킵니다

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # [초보] uvicorn.run() 함수를 호출하여 서버를 실행합니다
