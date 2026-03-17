"""
🎯 FastAPI 실습: LangChain RAG 시스템 구축

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **RAG(Retrieval-Augmented Generation) 시스템**을 구현합니다.
문서(강의계획서)에서 관련 정보를 검색한 후, 그 정보를 바탕으로 GPT가 답변을 생성합니다.

===============================================
🤔 왜 RAG인가? (설계 의도)
===============================================
1. **GPT의 한계**:
   - 학습 데이터 이후의 정보는 모름
   - 회사 내부 정보는 없음
   - 할루시네이션(거짓말) 발생

2. **RAG의 장점**:
   - 최신 정보 활용 가능
   - 회사 문서, DB 등 활용
   - 정확한 출처 기반 답변

3. **구조**:
   문서 → 임베딩 → 벡터 DB 저장
   질문 → 검색 → 관련 문서 + 질문 → GPT → 답변

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. 강의계획서 텍스트 준비
2. OpenAI로 임베딩 (텍스트 → 1536차원 벡터)
3. ChromaDB에 벡터 저장
4. LCEL 체인 구성

[요청 처리]
1. 질문 수신
2. Retriever가 질문과 유사한 문서 검색 (코사인 유사도)
3. 검색된 문서 + 질문을 프롬프트에 결합
4. GPT가 문서 기반으로 답변 생성
5. 결과 반환

===============================================
💡 핵심 학습 포인트
===============================================
- RAG 패턴: 검색 + 생성의 결합
- 벡터 DB: 의미 기반 검색
- LCEL: LangChain Expression Language
- 체인 조립: 파이프(|)로 연결

===============================================
📌 사전 준비 및 실행 가이드
===============================================
⚠️ Python 버전 주의:
- Python 3.14와 ChromaDB 호환 이슈 있음!
- Python 3.12 버전 또는 이전 버전 사용 권장
 
1. 사전 준비:
   pip install fastapi uvicorn langchain-openai langchain-community langchain-core langgraph chromadb tiktoken
   (.env 파일에 OPENAI_API_KEY=sk-xxx 추가)

2. 실행 방법:
   python ./lab4/rag_qa.py

💡 RAG 파이프라인 흐름:
┌─────────┐    ┌───────────┐    ┌────────┐    ┌─────┐    ┌────────┐
│ Question│ →  │ Retriever │ →  │ Prompt │ →  │ LLM │ →  │ Answer │
└─────────┘    └───────────┘    └────────┘    └─────┘    └────────┘
                  (검색)         (컨텍스트      (GPT)     (문자열)
                                  + 질문)
"""


# [파일 구조 및 순서 설명 (Map)]
# 이 파일은 위에서부터 아래로 읽히며 실행됩니다. 순서가 뒤바뀌면 에러가 납니다.
#
# 1. [재료 준비] 임포트 (Import)
#    - 요리에 필요한 도구(라이브러리)를 책상에 펼쳐놓는 단계입니다.
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
#    - 반드시 API 함수보다 **앞서서(위에)** 정의해야 합니다.
#
# 5. [주문 처리] API 엔드포인트 (@app.post)
#    - 실제 손님을 응대하는 점원들입니다.
#    - 위에서 만든 모든 것을 사용하여 요청을 처리합니다.

#  필수 모듈 임포트
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

#  LangChain 모듈 임포트
# LangChain v0.1 Core Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#  벡터 스토어 및 문서 클래스 임포트
# VectorStore & Document
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

#  환경변수 로드
# 환경변수 로드
load_dotenv()

#  지식 베이스 문자열
# 1. 지식 베이스 = 문서 (강의계획서라고 가정)
# [이게 뭐하는 코드?] 강의계획서 텍스트 = 우리의 "지식 데이터베이스"
# - 이 텍스트를 벡터 DB에 저장하고, 질문이 들어오면 여기서 검색함
# - 실무에서는: PDF 문서, 사내 위키, 고객센터 FAQ 등을 여기에 넣음
# - 왜 하드코딩?: 실습용. 실제론 파일 업로드나 DB에서 불러옴
syllabus_text = """
[FastAPI 및 AI 웹 개발 과정]
1주차: Python 기초 및 FastAPI 기본 구조 (Hello World, Path Param)
2주차: Pydantic 데이터 검증 및 비동기 처리 (Async/Await)
3주차: Hugging Face Transformers 활용 (감성분석, 이미지 분류)
4주차: OpenAI API 및 LangChain 기초 (RAG, Prompt Engineering)
5주차: LangGraph 에이전트 및 Streamlit 실습
평가 방법: 출석 20%, 중간 과제 30%, 최종 프로젝트 50%
"""

# [이게 뭐하는 코드?] RAG 체인을 저장할 전역 변수
# - 왜 전역 변수?: 모든 API 요청이 동일한 체인을 공유해야 하기 때문
# - None으로 초기화: 서버 시작 전에는 비어있고, lifespan에서 체워넣음
# - 추후 /ask-syllabus 엔드포인트에서 이 변수를 사용
rag_chain = None

# [개념] 왜 Lifespan 패턴을 쓰는가?
# - 문제: 요청마다 모델/DB를 로딩하면? → 매번 수 초씩 기다림 (성능 폭망)
# - 해결: 서버 시작 시 한 번만 로드 → 모든 요청이 공유 (0초)
# - 메모리 정리: 서버 종료 시 자동으로 리소스 해제 (메모리 누수 방지)
# - 실무 필수 패턴: DB 커넥션 풀, AI 모델, 캐시 등 모든 무거운 리소스에 사용
# - 없으면 어떻게 되나? → 동시 100명 접속 시 서버가 100번 모델 로딩 시도 → 서버 다운

#  Lifespan 함수 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
     서버 수명주기 관리 (Lifespan)
    
    이 함수는 서버가 켜지고 꺼지는 **전체 시간**을 관리합니다.
    순서가 헷갈리신다면 '식당 개업'을 상상해 보세요.
    
    [시간 순서 1: 개업 준비] (서버 시작 직후, yield 전)
    - 손님을 받기 전에 미리 재료를 손질하고 육수를 끓여놔야 합니다.
    - 여기서 무거운 작업(AI 모델 로딩, DB 구축)을 **딱 한 번만** 미리 해둡니다.
       
    [시간 순서 2: 영업 시작] (yield)
    - "영업 시작!" 팻말을 걸고 손님(Request)을 기다립니다.
       
    [시간 순서 3: 영업 종료] (yield 이후)
    - 가게 문을 닫고 청소를 합니다. (리소스 정리)
    """
    print("🔄 벡터 DB 구축 및 RAG 체인 생성 중...")
    
    #  전역 변수 선언
    global rag_chain

    # [순서의 이유: 레고 조립와 같습니다]
    # 1. '내용'이 있어야 책을 만들죠? -> (Docs 생성)
    # 2. '책'이 있어야 도서관에 꽂죠? -> (VectorStore 생성, Docs 필요)
    # 3. '도서관'이 있어야 사서를 고용하죠? -> (Retriever 생성, VectorStore 필요)
    # 4. '사서'와 '작가(AI)'가 있어야 일을 시키죠? -> (Chain 생성, Retriever & Model 필요)

    # [이게 뭐하는 코드?] 문서를 LangChain의 Document 객체로 변환
    # - Document 클래스: LangChain이 이해할 수 있는 표준 포맷
    # - page_content: 실제 텍스트 (위의 syllabus_text)
    # - metadata: 부가 정보 (출처, 저자, 날짜 등) - 추후 검색 결과에 표시 가능
    # - 왜 리스트로 감쌌?: 여러 문서를 한번에 처리할 수 있게 (현재는 1개지만 확장 용이)
    docs = [Document(page_content=syllabus_text, metadata={"source": "강의계획서"})]
    
    # [개념] 왜 Vector DB (임베딩)를 쓰는가?
    # - 일반 DB 키워드 검색의 한계:
    #   * "FastAPI 비동기" 검색 → 정확히 이 단어만 검색됨
    #   * "async", "asynchronous" 같은 동의어 검색 불가
    #   * 오타나 유사 표현 검색 불가
    # - Vector DB (의미 기반 검색):
    #   * "비동기 처리 방법" 검색 → "async/await 사용법" 문서도 찾아줌
    #   * 텍스트를 1536차원 숫자 벡터로 변환 → 의미상 유사한 벡터끼리 가까움
    #   * 코사인 유사도로 의미적으로 가까운 문서 검색
    # - 실무 활용: 추천 시스템, 문서 검색, 이미지 검색, 챗봇 등
    # - RAG의 핵심: GPT에게 "관련 있는" 문서만 정확하게 전달

    # [이게 뭐하는 코드?] 벡터 DB 생성 - OpenAI로 텍스트를 숫자로 변환해서 저장
    # - Chroma.from_documents(): ChromaDB라는 벡터 데이터베이스 생성
    # - 무슨 일이 일어나는가?
    #   1) OpenAIEmbeddings() 호출 → OpenAI API에 텍스트 전송
    #   2) OpenAI가 텍스트를 1536개 숫자 벡터로 변환 (임베딩)
    #   3) 이 벡터를 ChromaDB에 저장
    #   4) 나중에 질문이 들어오면, 질문도 벡터로 변환해서 비교
    # - 왜 벡터로?: "의미적으로 비슷한" 문장을 찾기 위해서
    # - collection_name: 데이터베이스 안에서 구분하는 이름 (여러 컨텍션 가능)

    #  벡터 DB 생성
    # 2. 임베딩 및 벡터 저장소 생성 
    # - Chroma.from_documents(docs, embedding):
    #   a) 문서 텍스트 추출 (page_content)
    #   b) OpenAIEmbeddings.embed_documents(texts) 호출:
    #      - OpenAI API 엔드포인트(/v1/embeddings)로 HTTP POST 요청
    #      - 텍스트 리스트를 전송하고 [1536] 차원 부동소수점 벡터 리스트 수신
    #   c) ChromaDB(DuckDB/SQLite 기반)에 저장:
    #      - 로컬 메모리/파일에 벡터 데이터 및 메타데이터 저장
    #      - HNSW (Hierarchical Navigable Small World) 알고리즘으로 ANN 인덱스 빌드
    #        (가장 가까운 이웃을 빠르게 찾기 위한 그래프 구조)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        collection_name="course_syllabus"
    )
    
    # [이게 뭐하는 코드?] Retriever = "검색 엔진"
    # - as_retriever(): 벡터 DB를 "검색기"로 변환
    # - Retriever가 하는 일:
    #   1) 질문을 받음 ("1주차에 뭐 배워?")
    #   2) 질문도 벡터로 변환
    #   3) 벡터 DB에서 유사한 문서 검색 (코사인 유사도)
    #   4) 가장 비슷한 문서 Top K개 반환 (기본 4개)
    # - 나중에 LCEL 체인에서 자동으로 사용됨

    #  Retriever 변환
    # - as_retriever(): 벡터 DB를 검색 인터페이스로 변환
    # - 질문이 들어오면 코사인 유사도로 유사한 문서를 찾아 반환
    retriever = vectorstore.as_retriever()

    # [이게 뭐하는 코드?] 프롬프트 템플릿 = GPT에게 주는 지시서 포맷
    # - ChatPromptTemplate: 변수를 넣을 수 있는 템플릿
    # - {context}: 여기에 Retriever가 찾은 문서가 자동 삽입됨
    # - {question}: 여기에 사용자 질문이 삽입됨
    # - 왜 필요한가?
    #   * GPT에게 "이 문서만 보고 답변해"라고 강제
    #   * 안 하면 GPT가 마음대로 지어낼 수 있음 (할루시네이션)

    #  프롬프트 템플릿 생성
    # 3. 프롬프트 템플릿 
    # - {context}: 검색된 문서가 자동으로 삽입될 자리
    # - {question}: 사용자 질문이 삽입될 자리
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # [이게 뭐하는 코드?] OpenAI GPT 모델 초기화
    # - ChatOpenAI(): OpenAI API와 통신할 클라이언트 객체
    # - model="gpt-4o-mini": 사용할 모델 종류 (빠르고 저렴)
    # - api_key: .env 파일에서 불러온 API 키

    #  LLM 모델 초기화
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    # [개념] 왜 LCEL (LangChain Expression Language) 파이프를 쓰는가?
    # - 기존 방식 (절차적 코드):
    #   context = retriever.get(question)
    #   filled_prompt = prompt.format(context=context, question=question)
    #   response = model.call(filled_prompt)
    #   result = parser.parse(response)
    # - LCEL 방식 (선언적 체인):
    #   chain = retriever | prompt | model | parser
    #   result = chain.invoke(question)
    # - 장점:
    #   * 간결함: 4줄 → 1줄로 표현
    #   * 병렬 처리 자동: 여러 retriever를 동시 실행 가능
    #   * 스트리밍 지원: 답변을 실시간으로 받을 수 있음
    #   * 디버깅 쉬움: 각 단계별 입출력 확인 간편
    # - 실무: 복잡한 AI 워크플로우를 선언적으로 구성 (6각형 → 통과)

    # [이게 뭐하는 코드?] LCEL 체인 조립 - 모든 단계를 | (파이프)로 연결
    # - rag_chain은 4단계 파이프라인:
    #   1단계: {"context": retriever, "question": RunnablePassthrough()}
    #           → 질문을 받아서, 검색하고, 딕셔너리로 만듦
    #   2단계: | prompt
    #           → 템플릿에 context와 question 삽입
    #   3단계: | model
    #           → GPT에게 전송, 답변 받음
    #   4단계: | StrOutputParser()
    #           → GPT 응답을 문자열로 변환
    # - 이 체인을 한 번 호출하면 위 4단계가 자동으로 순차 실행됨!
    # - RunnablePassthrough(): "입력값을 그대로 통과"시키는 특수 객체

    #  LCEL 체인 구성 (RunnableSequence)
    # - | 연산자(__or__)가 오버로딩되어 RunnableSequence 객체 생성
    # - 내부적으로 연결 리스트 형태의 호출 그래프 형성
    # - 실행 순서(invoke 호출 시):
    #   1. RunnableParallel(context, question): 병렬 실행
    #      - 사용자 입력이 2갈래로 복사됨
    #      - A트랙: retriever.invoke(input) → 코사인 유사도 검색 → 문서 리스트 반환
    #      - B트랙: RunnablePassthrough() → 입력값 그대로 통과
    #      - 병합: {"context": [Doc, ...], "question": "질문"} 딕셔너리 생성
    #   2. prompt.invoke(dict) → 프롬프트 문자열 생성
    #   3. model.invoke(string) → OpenAI API 호출 → AIMessage 반환
    #   4. parser.invoke(AIMessage) → content 속성 추출(문자열)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    print("✅ RAG 시스템 준비 완료!")
    
    #  yield - 서버 실행 대기
    yield
    
    # [Shutdown] 서버 종료 시 실행
    print("🛑 시스템 종료")

#  FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# [개념] 왜 Pydantic BaseModel을 쓰는가?
# - 수동 검증의 문제:
#   if not isinstance(data, dict): return error
#   if "question" not in data: return error
#   if not isinstance(data["question"], str): return error
#   → 코드 3줄, 휴먼에러 가능성 높음
# - Pydantic 자동 검증:
#   class QuestionRequest(BaseModel): question: str
#   → 타입 틀리면 자동으로 422 에러 (FastAPI가 처리)
# - 추가 장점:
#   * 자동 문서화: Swagger UI에 자동으로 스키마 표시
#   * IDE 자동완성: req.question 입력 시 자동완성 지원
#   * 타입 안전성: 실행 전에 타입 오류 발견 가능
# - 실무: dict로 받으면 버그 온상, Pydantic은 필수

#  요청 스키마 정의
# [문법 설명] 이것이 바로 'Pydantic(파이단틱)'입니다!
# - (BaseModel)을 상속받으면, 이 클래스는 "엄격한 검문소"가 됩니다.
class QuestionRequest(BaseModel):
    question: str

# [개념] 왜 async/await (비동기)를 쓰는가?
# - 동기 방식의 문제:
#   * 한 번에 한 요청만 처리 (커피 주문 → 완성까지 기다림 → 다음 손님)
#   * OpenAI API 호출은 3~5초 소요 → 이 시간 동안 서버가 멈춤
#   * 동시 10명 접속 → 10명 모두 대기 (마지막 사람은 30초 대기)
# - 비동기 방식의 장점:
#   * 여러 요청 동시 처리 (커피 주문받고 → 다른 손님 받고 → 완성되면 호출)
#   * await로 OpenAI 응답 대기하는 동안 다른 요청 처리
#   * 동시 10명 접속 → 모두 3~5초 안에 응답 (병렬 처리)
# - 핵심: I/O 대기 시간을 활용 → CPU 놀리지 않고 다른 일 처리
# - 실무: 데이터베이스, 외부 API 호출 등 모든 I/O 작업에 필수
# - 주의: CPU 집약적 작업(이미지 처리 등)은 비동기로 해결 안 됨 (별도 처리 필요)

# [이게 뭐하는 코드?] API 엔드포인트 - 사용자가 질문을 보내는 경로
# - @app.post("/ask-syllabus"): POST 요청을 받는 URL
# - async def: 비동기 함수 (여러 요청 동시 처리 가능)
# - req: QuestionRequest: Pydantic이 자동으로 JSON 유효성 검사
# - 흐름:
#   1) 사용자가 {"question": "1주차에 뭐 배워?"} 전송
#   2) rag_chain이 없으면 500 에러
#   3) rag_chain.ainvoke() 호출 → 위의 4단계 파이프라인 실행
#   4) {"answer": "Python 기초..."} 반환

#  API 엔드포인트
@app.post("/ask-syllabus")
async def ask_syllabus(req: QuestionRequest):
    #  체인 존재 여부 확인
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not init")
    
    # [개념] 왜 RAG (Retrieval-Augmented Generation) 패턴인가?
    # - GPT 단독 사용의 3가지 문제:
    #   1) 지식 컷오프: 2021년 이후 데이터 모름 (최신 정보 없음)
    #   2) 내부 데이터 접근 불가: 회사 문서, 사내 위키 등 못 봄
    #   3) 할루시네이션: 모르는 내용을 그럴싸하게 지어냄 (거짓 정보)
    # - RAG의 해결책:
    #   1) 최신 정보: 문서를 실시간으로 검색해서 GPT에게 제공
    #   2) 내부 데이터: 사내 문서를 벡터 DB에 저장 → 검색 가능
    #   3) 정확성: "이 문서 기반으로만 답변해"라고 강제 → 출처 명확
    # - 실무 활용 사례:
    #   * 사내 챗봇: 사내 규정, 매뉴얼 기반 답변
    #   * 고객 지원: 제품 문서 기반 자동 응답
    #   * 법률/의료: 판례, 논문 기반 정확한 정보 제공
    # - RAG = 검색(Retrieval) + 생성(Generation) 결합 패턴

    #  RAG 체인 실행
    # 체인 실행(ainvoke): # 비동기(async)로 RAG 체인을 실행하고 완료될 때까지 기다린 뒤 결과를 받음
    # - ainvoke(): 비동기로 체인 실행
    # - 내부적으로 검색 → LLM 호출 → 파싱이 순차 실행됨
    # - await: 네트워크 I/O 대기 (논블로킹)
    response = await rag_chain.ainvoke(req.question)
    
    #  결과 반환
    return {"answer": response}

#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)