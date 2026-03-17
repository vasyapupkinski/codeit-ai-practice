"""
🎯 FastAPI 실습: LangGraph 종합 실습 (4가지 패턴)

██╗      █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
██║     ██╔══██╗████╗  ██║██╔════╝ ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
██║     ███████║██╔██╗ ██║██║  ███╗██║  ███╗██████╔╝███████║██████╔╝███████║
██║     ██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝

핵심: AI 에이전트 서버 구축의 모든 것

================================================================================
  📚 [상세 가이드] 이 코드를 200% 활용하는 법
================================================================================
이 파일은 "AI 비서 4명이 살고 있는 아파트(서버)"를 짓는 설계도입니다.
복잡해 보이지만, 아래 설명을 천천히 읽으면 전체 그림이 보입니다.

--------------------------------------------------------------------------------
 [1] 🗺️ 전체 시스템 아키텍처 (System Architecture)
--------------------------------------------------------------------------------
1. 시각적 구조도 (Visual Diagram)
   * 폰트 깨짐 방지를 위해 영문 기호로 그렸습니다.

       (User)           (FastAPI App)               (Response)
    +--------+       +------------------+       +---------------+
    | CLIENT | ----> |   API ROUTER     | ----> |  JSON RESULT  |
    +--------+       +--------+---------+       +---------------+
                              |
          +-------------------+-------------------+
          |         [ Pattern 0: Master ]         |
          |       (Supervisor AI decides)         |
          +-------------------+-------------------+
                              |
          +-------------------+-------------------+-------------------+
          |                   |                   |                   |
    +-----v-----+       +-----v-----+       +-----v------+       +-----v-----+
    | Pattern 1 |       | Pattern 2 |       | Pattern 3  |       | Pattern 4 |
    | Classify  |       | Creative  |       | ReAct Loop |       | Chat Loop |
    +-----------+       +-----------+       +-----+------+       +-----+-----+
                                                  |                    |
                                            +-----v-----+        +-----v-----+
                                            |   TOOLS   |        |  HISTORY  |
                                            +-----------+        +-----------+

2. 데이터 흐름 시나리오 (Data Flow Scenario)

   상황: 사용자가 "내년 서울 인구가 몇 명이야?" (패턴 3) 라고 물어본 경우

    1. [입장] 사용자 -> FastAPI 서버 (@app.post)
       - 사용자의 질문이 서버 현관문(Endpoint)을 통과합니다.
       - 문지기(Pydantic)가 "질문 형식이 맞는지" 검사합니다.

    2. [배정] FastAPI -> AI 로봇 (Graph)
       - 서버가 미리 대기 중인 4명의 로봇 중 '패턴 3 로봇(해결사반)'을 깨웁니다.
       - "야, 이거 좀 알아봐" 하고 질문을 던져줍니다.

    3. [작업] AI 로봇 내부 (LangGraph Loop)
       - (생각) "음... 이건 검색을 해야겠어." (Reasoning Node)
       - (행동) 'Google 검색 도구' 실행 -> "2025 서울 인구 예측" 검색 (Tool Node)
       - (관찰) "약 930만 명이라네요." (Observation)
       - (판단) "정보 찾았다! 이제 답변하자." (Response)

    4. [퇴장] AI 로봇 -> 사용자 (JSON Response)
       - 로봇이 만든 답변을 예쁜 포장지(JSON)에 담아 사용자에게 줍니다.

 [핵심 개념] LangGraph의 3대 요소 (비유 설명)
 1. State (택배 상자):
    - 모든 노드가 공유하는 "데이터 저장소"입니다.
    - 처음엔 질문만 들어있다가, 노드를 거칠 때마다 분류표, 답변 등이 차곡차곡 쌓입니다.

 2. Node (작업자):
    - 택배 상자(State)를 받아서 일을 하는 "로봇"이나 "사람"입니다.
    - 예: `simple_classify`는 상자를 열어보고 "이건 기술 질문이네"라고 분류표를 써넣습니다.

 3. Route (신호등/표지판):
    - 작업이 끝난 상자를 "다음엔 어디로 보낼까?" 결정하는 로직입니다.
    - 예: "기술(Tech) 딱지가 붙어있으면 기술 전문가에게 보내라!"

 [Q&A] "순서가 중요해서 동기(Sync)를 쓰는 건가요?" (중요!)
 - 결론부터 말하면: **아닙니다! 순서는 비동기(Async)에서도 완벽하게 지켜집니다.**
 - 많은 분들이 "비동기는 동시에 제멋대로 실행되는 것"이라고 오해하시곤 합니다.
 - 하지만 `await node_a()`라고 쓰면, AI는 아무리 비동기라도 A가 끝날 때까지 절대로 B로 넘어가지 않습니다.
 - **[그럼 왜 동기를 쓰나요?]**: 오직 **'기다릴 일이 있느냐'** 때문입니다.
   - 노드: AI를 부르느라 3초를 **'기다려야'** 하므로 비동기(`async def`)를 씁니다.
   - 라우터: 기다릴 일 없이 0.001초 만에 끝나므로 굳이 복잡하게 비동기를 쓸 **'이유가 없어서'** 동기(`def`)를 쓰는 것입니다.
 - 즉, **'순서'는 그래프의 선(Edge)이 결정하는 것이지, 동기/비동기가 결정하는 것이 아닙니다!**

--------------------------------------------------------------------------------
 [2] 🧠 4가지 패턴 상세 분석 (Pattern Deep Dive)
--------------------------------------------------------------------------------
아키텍처에 등장하는 4가지 로봇의 정체와 작동 원리입니다.
  0. **Pattern 0: 마스터 라우터 (Master Router)**  🆕
     - [역할] "어디로 가야 하죠?" 결정 (Supervisor)
     - [구현] 사용자의 질문을 분석하여 패턴 1~4 중 적합한 곳으로 자동 연결
     - [코드 핵심] LLM을 활용한 의도 분류(Intent Classification)와 분기 처리

  1. **Pattern 1: 기본 라우팅 (Simple Routing)**
     - [역할] 사용자 질문이 "기술적인가?" vs "일상적인가?" 구분
     - [구현] 
        1. `simple_classify` 노드가 질문을 분석해 'category'를 결정
        2. `simple_route` 함수가 'category'를 보고 `tech_expert`나 `friendly_bot`으로 길을 안내 (Conditional Edge)
     - [코드 핵심] `add_conditional_edges` 사용법 익히기

  2. **Pattern 2: 3단계 라우팅 (Advanced Routing)**
     - [역할] 패턴 1에서 "창의적인 질문(시, 소설)"까지 처리하도록 확장
     - [구현]
        1. `advanced_classify` 노드가 질문을 Tech / Casual / **Creative** 3가지로 분류
        2. 창의적 질문이면 `creative_writer` 노드가 작동하여 문학적인 답변 생성
     - [코드 핵심] 복잡한 분기 처리 (`Literal['tech', 'casual', 'creative']`)

  3. **Pattern 3: ReAct 에이전트 (Reasoning + Acting)**
     - [역할] 혼자 답 못하는 질문(최신 데이터, 계산)을 도구 써서 해결
     - [구현]
        1. `react_think`: "내가 뭘 해야 하지?" 생각 (검색? 계산? 그냥 답변?)
        2. `react_act`: 도구가 필요하면 실제로 실행하고 결과를 가져옴
        3. **무한 루프**: 답이 완성될 때까지 1->2->1 과정을 계속 반복 (Loop)
     - [코드 핵심] 순환 그래프(Cyclic Graph)와 `tools_condition`

  4. **Pattern 4: 대화형 기억 에이전트 (Chat with Memory)**
     - [역할] "아까 내가 말한 거 기억해?" 구현 (멀티턴 대화)
     - [구현]
        1. 사용자의 `session_id`를 확인
        2. `chat_sessions` 딕셔너리에서 이전 대화(History)를 꺼내옴
        3. AI에게 "이전 대화 + 현재 질문"을 통째로 줘서 문맥을 파악하게 함
     - [코드 핵심] 외부 메모리(Dictionary) 연동과 State 업데이트

--------------------------------------------------------------------------------
 [3] 🛠️ 파일 해부도 (Code Anatomy)
--------------------------------------------------------------------------------
이 파일은 총 5개의 구역으로 철저하게 나뉘어 있습니다.

  📍 구역 1. [기초 공사] Imports & Globals (Line 90~120)
     - 필요한 장비(라이브러리)를 챙기고, 나중에 로봇을 담을 빈 상자(변수)를 준비합니다.
  
  📍 구역 2. [도구 제작] Tools Definition (Line 120~180)
     - 로봇들이 손에 쥘 도구(계산기, 검색기)를 미리 만들어 둡니다.
     - 예: `calculator_tool`, `knowledge_search_tool`
  
  📍 구역 3. [로봇 설계] Patterns 0~4 (Line 180~740) **(가장 중요!)**
     - 로봇 5대(반장 포함)의 뇌 구조(State)와 행동 방식(Node)을 하나하나 설계합니다.
     - 주의: 설계도일 뿐, 아직 로봇이 만들어진 건 아닙니다.
  
  📍 구역 4. [공장 가동] Lifespan (Line 740~810)
     - 서버가 켜지는 순간(Startup), 위에서 만든 설계도로 로봇을 실제로 '조립'합니다.
     - 여기서 `build_..._graph().compile()`이 실행됩니다.
  
  📍 구역 5. [민원 창구] Endpoints (Line 810~끝)
     - 사용자가 실제로 접속할 수 있는 URL 주소들입니다.
     - 여기서 로봇에게 일을 시킵니다 (`await graph.ainvoke(...)`).

--------------------------------------------------------------------------------
 [4] 💡 구체적인 학습 로드맵 (Study Roadmap)
--------------------------------------------------------------------------------
무작정 읽지 마시고, 다음 질문에 답을 찾는다는 기분으로 코드를 보세요.

  Step 0. "누가 이 모든 걸 지휘하지?"  🆕
     - `master_router` 함수를 가장 먼저 찾아보세요.
     - 사용자 질문이 들어오면 LLM이 어떻게 4가지 업무로 분류하는지(프롬프트) 확인하세요.

  Step 1. "로봇은 어떻게 조립되지?"
     - `build_pattern1_graph` 함수를 찾아보세요.
     - `add_node`(부품 추가)와 `add_edge`(전선 연결)가 보이면 성공입니다.
  
  Step 2. "로봇의 뇌는 어떻게 생겼지?"
     - `simple_classify` 함수를 찾아보세요.
     - AI에게 프롬프트를 보내고(`prompt | model`), 결과를 받는 과정을 확인하세요.
  
  Step 3. "이 모든 게 언제 시작되지?"
     - 맨 아래 `lifespan` 함수를 보세요.
     - `global`로 선언된 변수들에 진짜 로봇 객체가 들어가는 순간을 목격하세요.

  Step 4. "동기(Sync) vs 비동기(Async), 뭐가 다르지?" 🆕
     - 이 파일은 왜 `ainvoke`와 `await`로 도배되어 있을까요?
     - 하단의 [5]번 가이드를 통해 그 이유를 명확히 이해해 보세요.

--------------------------------------------------------------------------------
 [5] ⚡ 동기(Sync) vs 비동기(Async) 완벽 정리
--------------------------------------------------------------------------------
 1. 동기 (Synchronous): "나 끝날 때까지 기다려!"
    - 하나의 작업이 끝날 때까지 다음 작업은 멈춰 있습니다.
    - [언제 쓰나요?]: 계산이 아주 빠르거나, 반드시 순서대로 일어나야 하는 간단한 작업.
    - [단점]: AI가 답변하는 3초 동안 서버 전체가 '먹통'이 됩니다.

 2. 비동기 (Asynchronous): "나 일할 테니, 넌 볼일 봐!"
    - 무거운 작업(AI 호출)을 던져놓고, 결과가 나올 때까지 다른 일을 처리합니다.
    - [언제 쓰나요?]: 이 파일처럼 LLM 호출, 데이터베이스 조회 등 '기다림'이 필요한 작업.

 3. 왜 이 파일은 비동기인가요?
    - 우리 서버는 FastAPI(비동기 최적화) 기반입니다.
    - AI 답변은 보통 1~5초가 걸립니다. 동기 방식으로 짜면 한 명의 질문에 답변하는 동안
      다른 사용자들은 서버가 고장 난 것처럼 아무 응답도 받지 못하게 됩니다.
    - `await ainvoke()`를 사용함으로써, AI가 생각하는 동안에도 서버는 다른 요청을
      받을 수 있는 "실무급" 구조를 갖추게 된 것입니다.

--------------------------------------------------------------------------------
 [6] 👨‍🔧 Advanced Tip: Worker(워커)란 무엇인가?
--------------------------------------------------------------------------------
 1. 개념: 비동기(Async)가 '한 명의 점원이 여러 테이블을 보는 것'이라면,
    워커(Worker)는 '점원 자체를 여러 명 고용하는 것'입니다.

 2. 비동기(Async)의 한계:
    - 점원(CPU)이 아무리 빠릿빠릿해도, 요리(연산) 자체가 너무 많아지면 결국 지칩니다.
    - 파이썬은 기본적으로 하나의 몸(Process)으로 움직이기 때문에, CPU를 많이 쓰는 
      복잡한 계산(음식 조리)이 몰리면 비동기만으로는 한계가 옵니다.

 3. 워커(Worker)의 역할 (Parallelism):
    - 서버를 실행할 때 "워커를 4개 띄워줘!"라고 명령하면(예: `--workers 4`),
      똑같은 서버가 4개가 생겨서 각각 독립적으로 움직입니다.
    - 이제 우리 식당은 4명의 점원이 각자 여러 테이블을 동시에 관리하게 됩니다.

 4. [중요] 워커 사용 시 주의점 (Side Effects):
    - **메모리 파편화**: 각 워커는 자기만의 기억(메모리)을 가집니다. 
      워커A에 저장한 `chat_sessions` 기록을 워커B는 알 수 없습니다.
    - **해결책**: 그래서 실무에서는 메모리(변수)가 아닌 **Redis나 DB**에 대화 기록을 저장합니다.
    - **포트(Port) 조회**: 포트는 '대표 번호'와 같습니다. 포트 8000번은 하나만 뜨지만,
      그 뒤에서 움직이는 파이썬 프로세스(PID)는 워커 개수만큼 여러 개가 보이게 됩니다.
    - **Lifespan의 비밀**: 워커를 4개 띄우면 `lifespan` 함수도 **4번 실행**됩니다. 
      즉, AI 모델과 그래프 객체가 메모리에 4번 올라가므로 서버 사양(RAM)을 고려해야 합니다.

--------------------------------------------------------------------------------
 [7] 🧪 워커(Worker) 직접 테스트해보기 (실험실)
--------------------------------------------------------------------------------
 1. [테스트 방법]: 코드 맨 아래 `uvicorn.run`에 `workers=2`를 추가하고 실행해보세요.
 2. [실패 시나리오 (기억 상실)]:
    - 1번 호출: `/chat_start` (워커A가 받아서 세션 생성)
    - 2번 호출: `/chat_continue` (운 좋게 워커A가 받으면 성공!)
    - 3번 호출: `/chat_continue` (하필 워커B가 받으면 "세션 없음(404)" 에러 발생!)
    - [결론]: 이게 바로 워커 간의 '메모리 파편화' 현상입니다.

 3. [개념적 해결 코드 (Conceptual Fix)]:
    - **핵심**: 모든 워커가 '공통 장부'를 보게 만들어야 합니다.
    - **세션 저장용 DB (Session Store)**: 주로 **Redis**를 사용합니다. (가장 빠름)
    - **지식 저장용 DB (Vector DB)**: **Chroma, Qdrant** 같은 DB는 대화 기록이 아니라, 
      AI가 공부한 대량의 문서(PDF 등)를 저장하고 검색하는 데 쓰입니다. (RAG용)
    
    ```python
    # 실무 패러다임: 메모리에서 DB로의 전환
    # chat_sessions = {}  <- (X) 워커가 늘어나면 못 씀
    # redis_db.set(id, history) <- (O) 어떤 워커든 여기서 꺼내옴
    ```

 4. 요약:
    - [Async]: AI가 대답할 때까지 '기다리는 시간'을 효율적으로 쓰기 위해 (필수!)
    - [Worker]: 서버 자체의 '총 처리 능력(CPU 사용량)'을 키우기 위해 (운영 단계 필수!)

"""

# ===============================================
# 📍 [구역 1] 기초 공사 (Imports & Globals)
# ===============================================
#  재료 준비 (필수 라이브러리 임포트)
# ===============================================
# 1. 라이브러리 임포트 (Imports)
# ===============================================

# [핵심 개념] FastAPI vs LangGraph
# - FastAPI: 외부(사용자)와 소통하는 "창구" 역할 (HTTP 요청/응답 관리)
# - LangGraph: 내부적으로 복잡한 AI 로직을 처리하는 "공장" 역할 (상태 관리, 순서 제어)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# [핵심 개념] LangGraph의 3대 요소
# 1. StateGraph: 전체 흐름을 관장하는 "지도" (Workflow)
# 2. END: 작업이 끝났음을 알리는 "종착역" (Node)
# 3. TypedDict: 노드끼리 주고받는 "택배 상자"의 규격 (Type Hinting)
from langgraph.graph import StateGraph, END
from typing import TypedDict

# [핵심 개념] LangChain 컴포넌트
# - ChatPromptTemplate: AI에게 보낼 편지 양식 (Prompt Engineering)
# - ChatOpenAI: 실제 똑똑한 AI 모델 (LLM)
# - StrOutputParser: AI의 답변에서 핵심 텍스트만 뽑아내는 도구
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 및 기타 유틸리티
import os
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager # [실무 Tip] 서버 수명주기를 깔끔하게 관리하는 도구
import ast # [보안] eval() 사용 시 안전장치를 위해 필요

# 환경 변수 로드 (API 키 등 보안 정보)
load_dotenv()

# ================================================================================
# 🏢 [실무 권장 가이드] "API 키 관리"
# ================================================================================
# [현재 코드] os.getenv("OPENAI_API_KEY") - 환경 변수에서 바로 읽음 (.env 파일)
#
# [실무에서는?] 괜찮지만, 대규모 팀/데이터 보안이 중요한 환경에서는:
# 1. **AWS Secrets Manager**: 클라우드에 암호화되어 저장, 팀원 접근 권한 관리 가능
# 2. **HashiCorp Vault**: 카카오/배민 같은 대기업에서 쓰는 비밀 관리 시스템
# 3. **Azure Key Vault / GCP Secret Manager**: 각 클라우드의 관리형 서비스
#
# [왜?] .env 파일을 실수로 Git에 올리면 해커가 모든 API 키를 볼 수 있습니다.
#      위 서비스들은 "Git에 올리는 것 자체가 불가능"한 구조로 만들어주기 때문에 안전합니다.
# ================================================================================

# ===============================================
# 🔄 전역 변수 (Global Resources)
# ===============================================

# [실무 Tip] 왜 전역 변수를 `None`으로 초기화하나요? (Lazy Initialization)
# 1. "빈 상자" 전략 (Placeholder):
#    - 파이썬 파일이 실행(Import)되는 순간에는 아직 무거운 AI 모델을 만들지 않습니다.
#    - 그냥 "나중에 여기에 모델이 들어갈 거야"라고 이름표(변수)만 붙여두는 것입니다.
#    - [이유]: 서버가 켜지기도 전에 무거운 작업을 하면 시작 속도가 느려지고, 오류 진단이 어렵습니다.
#
# 2. 안전성 확보 (Safety):
#    - 만약 여기서 `model = ChatOpenAI(...)`를 바로 해버리면?
#    - API 키가 없거나 인터넷이 끊겨있을 때, 서버가 켜지지도 못하고 "퍽" 하고 죽어버립니다. (Startup Crash)
#    - 일단 `None`으로 두고, 나중에 `lifespan` 함수에서 안전장치를 갖추고 채워넣는 것이 정석입니다.

model = None            # [빈 상자 1] AI 모델이 들어갈 자리
simple_graph = None     # [빈 상자 2] 패턴 1 로봇이 들어갈 자리
advanced_graph = None   # 패턴 2 (3-way 라우팅) 로봇 담을 상자
react_graph = None      # 패턴 3 (ReAct 에이전트) 로봇 담을 상자
chat_graph = None       # 패턴 4 (기억 에이전트) 로봇 담을 상자

# [참고] 이 변수들은 맨 아래 'lifespan' 함수가 실행될 때 실제 객체들로 채워집니다.

# [실무 Tip] 메모리 세션 저장소의 한계
# - 현재는 파이썬 딕셔너리(RAM)에 저장하므로 서버 재시작 시 데이터가 날아갑니다.
# - 실무에서는 Redis, PostgreSQL 같은 외부 DB에 저장해야 안전합니다!
chat_sessions = {}      # 대화 기록 저장소 ({session_id: [messages...]})

# ===============================================
# 📍 [구역 2] 도구 제작 (Tools Definition)
# ===============================================
# 🔧 공용 도구 (Tools) 정의
# ===============================================

# [파이썬 문법 설명] 타입 힌트 (Type Hint)
# - expression: str -> "이 함수에는 문자열(str)을 넣어주세요"라는 명찰입니다.
# - -> str          -> "이 함수는 결과로 문자열(str)을 뱉어냅니다"라는 예고입니다.
# 즉, expression이라는 변수 안으로 "1+1" 같은 문자열 값이 들어오게 됩니다.
def calculator_tool(expression: str) -> str: # 매개변수 expression(문자열)을 받고 문자열을 반환하는 함수 정의
    """
    [도구 1] 안전한 계산기
    
    
    1. 입력된 수식의 양옆 공백 제거
    2. 위험한 단어(import, exec 등)가 있는지 검사 (해킹 방지)
    3. Python의 eval()로 계산하되, 기능 제한

    [🚨 보안 주의] 왜 eval()을 쓰나요?
    - 이 코드는 '도구가 어떻게 작동하는지' 원리를 배우기 위한 교육용 예제입니다.
    - 외부 라이브러리 설치 없이 파이썬 기본 기능만으로 구현하기 위해 eval을 사용했습니다.
    - [실무 권장] 실제로는 `import` 해서 검증된 도구를 써야 합니다.
      예: `from langchain_experimental.tools import PythonREPLTool` 또는 `numexpr` 라이브러리
    """
    try: # 예외 발생 가능성이 있는 코드를 감지합니다
        #  공백 제거
        # - 사용자가 " 1 + 1 " 처럼 입력할 수 있으므로 앞뒤 공백을 없앱니다
        expression = expression.strip() # .strip() 메서드를 호출하여 공백을 제거하고 expression 변수에 재할당합니다
        
        #  블랙리스트 필터링 (보안)
        # - 해킹에 사용될 수 있는 위험한 키워드들을 리스트로 정의합니다
        dangerous = ['import', '__', 'exec', 'eval', 'open'] # 위험 키워드 리스트 정의
        
        # - any() 함수를 사용하여 입력된 수식에 위험 키워드가 하나라도 포함되어 있는지 검사합니다
        if any(d in expression.lower() for d in dangerous): # 리스트 컴프리헨션으로 각 키워드 포함 여부를 확인하고 if문으로 분기합니다
            # - 위험 키워드가 발견되면 즉시 경고 메시지를 반환하고 종료합니다
            return "보안 경고: 허용되지 않는 표현식입니다." # 경고 문자열 반환
            
        #  제한된 환경에서 계산 실행
        # - eval() 함수는 문자열로 된 수식을 계산합니다 (예: "1 + 2" -> 3)
        # - {"__builtins__": {}}: 내장 함수 접근을 차단하여 보안을 강화합니다
        result = eval(expression, {"__builtins__": {}}, {}) # eval()을 호출하여 결과를 result 변수에 할당합니다
        
        #  결과 반환
        return f"계산 결과: {result}" # f-string으로 결과를 포맷팅하여 반환합니다
    except Exception as e: # 계산 중 에러(0으로 나누기, 문법 오류 등)가 발생하면 실행됩니다
        #  에러 처리
        return f"계산 오류: {str(e)}" # 에러 메시지를 문자열로 변환하여 반환합니다

def knowledge_search_tool(query: str) -> str: # 문자열 쿼리를 받아 문자열 결과를 반환하는 함수 정의
    """
    [도구 2] 지식 검색 (간단한 키워드 매칭)
    실제로는 구글 검색 API나 벡터 DB(Vector Store)를 연결하는 곳입니다.
    """
    #  지식 베이스 정의 (임시 데이터)
    # - 실제로는 DB에서 가져와야 하지만, 교육용으로 딕셔너리를 사용합니다
    knowledge_base = { # 딕셔너리 생성
        "fastapi": "FastAPI는 Python의 고성능 웹 프레임워크입니다.", # 키: 값 쌍 정의
        "langgraph": "LangGraph는 상태 기반 에이전트 워크플로우 도구입니다.",
        "python": "Python은 읽기 쉽고 강력한 프로그래밍 언어입니다."
    } # knowledge_base 변수에 할당
    
    #  키워드 매칭 검색 (Analogy: 숨은 그림 찾기)
    # - knowledge_base (딕셔너리)의 내용을 'key(키워드)'와 'value(정의)'로 하나씩 꺼내어 훑어봅니다.
    # - [비유]: key는 '찾을 사람 이름'이고, query(사용자 질문)는 '사람들이 북적이는 방'입니다.
    # - logic: "이 사람이 이 방 안에 있나?" (key in query)를 확인하는 과정입니다.
    for key, value in knowledge_base.items(): # 딕셔너리의 (항목명, 설명) 쌍을 하나씩 꺼냅니다.
        # - [중요] .lower()를 쓰는 이유: 
        #   사용자가 "FASTAPI"라고 대문자로 물어봐도 "fastapi"와 똑같은 것으로 인식하게 하기 위함입니다 (대소문자 무시).
        if key.lower() in query.lower(): # 질문(방) 안에 찾으려는 키워드(사람)가 포함되어 있는지 검사합니다.
            # - 포함되어 있다면(True), 더 찾을 필요 없이 바로 그에 대한 설명을 돌려주고 함수를 끝냅니다.
            return f"검색 결과: {value}" # 매칭 성공 시 설명을 반환합니다.
            
    #  검색 실패 시 처리
    return "검색 결과 없음" # 매칭되는 키워드가 없으면 이 문자열을 반환합니다

# [도구 모음] LLM이 선택할 수 있는 도구들의 명세서
# - 도구 이름(문자열)과 실제 함수 객체를 매핑합니다
TOOLS = { # 딕셔너리 생성
    "calculator": calculator_tool, # "calculator"라는 이름으로 calculator_tool 함수 등록
    "knowledge_search": knowledge_search_tool # "knowledge_search"라는 이름으로 knowledge_search_tool 함수 등록
} # TOOLS 상수에 할당

# ===============================================
# 📍 [구역 3] 로봇 설계 (Pattern 0 ~ 4 Implementation)
# ===============================================
# 📦 [패턴 0] Master Router (Supervisor)
# ===============================================

# [핵심 개념] Supervisor (반장 로봇) 패턴이란?
# - 거대 언어 모델(LLM)을 활용하여 "어떤 코드를 실행할지 결정"하는 아키텍처입니다.
# - 사용자의 모호한 질문을 4가지 명확한 실행 경로(Route) 중 하나로 매핑합니다.
# - [장점] 사용자는 메뉴를 고를 필요 없이 자연어로 질문하면 되고,
#          시스템은 새로운 기능(패턴)이 추가되어도 Supervisor만 교육하면 확장 가능합니다.
#
# - 왜 이렇게 순서가 엄격한가요? (계주 비유) 🏃‍♂️🏃‍♀️
#   1. LangGraph의 흐름은 마치 '이어달리기(Relay Race)'와 같습니다.
#   2. '상태(State)'는 주자들이 주고받는 '바톤(Baton)'입니다.
#   3. 앞 주자(Node A)가 바톤에 정보를 적어 넣지 않으면, 뒷 주자(Node B)는 무엇을 해야 할지 모릅니다.
#   4. [비동기의 역할]: 비동기(`await`)는 주자가 달리다 넘어지거나 멈추지 않고, 
#      자기 차례가 오면 가장 효율적으로 바톤을 넘겨받아 완주할 수 있게 돕는 장치입니다.
#      (순서를 무너뜨리는 게 아니라, 순서를 유지하되 대기 시간을 최소화합니다!)

async def master_router(question: str):
    """
    [반장 로봇] 질문 분류기 (Classifier)
    
    [역할]
    - 이 함수는 '답변'을 하는 게 아니라, '판단'만 합니다.
    - 복잡한 로직 없이 오직 "누구에게 일을 시킬까?"에만 집중합니다 (단일 책임 원칙).
    
    [입력] 사용자 질문 문자열
    [출력] 실행할 패턴 코드 ('pattern1' ~ 'pattern4')

    [비동기(Async) 설명]
    - 이 함수는 내부적으로 LLM을 호출하므로 I/O 바운드 작업입니다.
    - async def로 선언하고 await chain.ainvoke()를 써야 서버가 멈추지 않습니다.
    """
    
    #  디버깅 로그 (Traceability)
    # - 실제 서비스에서는 이 로그가 모니터링 시스템(Datadog, Sentry 등)으로 전송됩니다.
    # - "어떤 질문이 들어왔길래 AI가 저기로 보냈지?"를 추적할 때 필수적입니다.
    print(f"[Master] 질문 분석 중: {question}")
    
    #  슈퍼바이저 프롬프트 설계 (Prompt Engineering)
    # - AI에게 단순한 '텍스트 생성기'가 아닌 '시스템 관리자'라는 역할(Persona)을 부여합니다.
    # - [Few-Shot 기법] 예시를 주면 더 좋겠지만, 여기서는 각 부서의 업무 정의(Definition)를 명확히 하는 데 집중했습니다.
    # - 하단에 '부서 코드만 출력하라'는 제약사항(Constraint)을 주어 파싱을 쉽게 만듭니다.
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 이 AI 시스템의 총괄 관리자(Supervisor)입니다.
        사용자의 질문을 분석하여, 아래 4개 전문 부서 중 가장 적합한 곳으로 연결하세요.
        
        [🚨 부서 업무 정의서]
        
        1. [pattern2: 창의/감성 팀] (Creative)
           - 시, 소설, 에세이, 노래 가사 작성 등 '작문'이 필요한 경우
           - 위로가 필요하거나 감성적인 대화를 원하는 경우
           
        2. [pattern3: 해결사 팀] (ReAct Agent)
           - "지금", "오늘" 같은 실시간 정보가 필요한 경우 (예: 날씨, 주가)
           - 복잡한 수학 계산이나 논리적 추론이 필요한 경우
           - 단순 지식 검색이 필요한 경우
           
        3. [pattern4: 기억 팀] (Context Chat)
           - "아까 내가 말한 거", "이전 질문에 이어서" 같은 문맥이 필요한 경우
           - 특별한 목적 없이 길게 이어지는 대화
           
        4. [pattern1: 일반 팀] (Simple Routing)
           - 위 3가지에 해당하지 않는 단순 기술 질문
           - 가벼운 인사('안녕', '반가워')
           - 명확히 분류하기 어려운 경우의 기본값(Fallback)
        
        [⚠️ 출력 규칙]
        - 절대로 다른 말을 덧붙이지 마세요.
        - 오직 부서 코드 단어 하나만 출력하세요. (예: pattern2)
        
        질문: {question}
        담당 부서 코드:
        """
    )
    
    #  체인 실행 (LLM Inference)
    # - 프롬프트 주입 -> GPT 모델 생각 -> 문자열 추출
    # - 이 과정은 약 0.5~1.0초 정도 소요됩니다.
    # - [New] 비동기 처리: invoke 대신 ainvoke 사용
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({"question": question})
    
    #  데이터 클렌징 (Data Cleaning)
    # - AI는 가끔 "Answer: pattern1" 또는 "  pattern1. " 처럼 지저분한 답을 줄 때가 있습니다.
    # - .strip(): 앞뒤 공백 제거
    # - .lower(): 대소문자 통일 (실수 방지)
    # - 이런 방어적 코드(Defensive Coding)가 시스템의 안정성을 높입니다.
    chosen_pattern = result.strip().lower()
    
    #  유효성 검사 및 에러 핸들링 (Validation)
    # - 만약 AI가 프롬프트를 무시하고 "모르겠어요"라고 답한다면?
    # - 그대로 두면 시스템이 멈춥니다. 따라서 '허용된 목록'에 없으면 무조건 기본값(pattern1)으로 보냅니다.
    # - 이를 'Graceful Degradation'(우아한 성능 저하)라고 부릅니다.
    valid_patterns = ['pattern1', 'pattern2', 'pattern3', 'pattern4']
    if chosen_pattern not in valid_patterns:
        print(f"⚠️ [주의] AI가 알 수 없는 응답을 했습니다: '{chosen_pattern}' -> Pattern 1로 강제 이동")
        # [핵심] 여기서 'pattern1'을 return하는 것은 "출력"이 아니라, "결정권"을 넘기는 것입니다.
        return 'pattern1' # 이 문자열을 받은 호출자(Main API)가 실제로 패턴 1 로봇을 깨웁니다.
        
    print(f"✅ [Master] 라우팅 완료: {chosen_pattern}")
    # [핵심] 이 return 값은 '일감 배정 표'와 같습니다.
    return chosen_pattern # 이 값이 endpoint_master_bot의 if/elif 조건문으로 들어가 실제 실행을 트리거합니다.
    
# ===============================================
# 📦 [패턴 1] 기본 조건부 라우팅 (Simple Routing)
# ===============================================

# [1. 설계도] State 정의 (State)
# [핵심 개념] 상태(State)의 역할
# - LangGraph의 모든 노드는 이 State를 "공유"합니다.
# - 노드 A가 state['classification']을 채우면, 노드 B는 그걸 읽을 수 있습니다.
# - 마치 공장에서 컨베이어 벨트 위의 물건에 부품이 하나씩 조립되는 과정과 같습니다.
class SimpleRoutingState(TypedDict):
    question: str           # 사용자 질문 (입력)
    classification: str     # 분류 결과 ("TECHNICAL" vs "CASUAL")
    response: str           # 최종 답변 (출력)

# [2. 부품] 노드 함수 정의
# - 각 함수는 로봇의 "부품"입니다. State를 받아서 작업을 하고, 갱신할 데이터를 반환합니다.

async def simple_classify(state: SimpleRoutingState): # def 키워드로 함수를 정의하고 SimpleRoutingState 타입의 state 매개변수를 받습니다
    """
    [노드 1] 질문 분류기
    - 사용자의 질문을 분석하여 카테고리를 정하는 "두뇌" 역할을 합니다.
    - 리턴값 {"classification": ...}은 자동으로 State에 업데이트됩니다.
    """

    #  디버깅 출력
    # - print() 함수를 호출하여 콘솔에 현재 상태를 출력합니다
    print(f"[Pattern 1] 분류 중: {state['question']}") # f-string으로 문자열을 포맷팅하고 print()로 출력합니다
    
    #  LLM에게 지시할 프롬프트 생성
    # - ChatPromptTemplate.from_template() 메서드를 호출하여 프롬프트 객체를 생성합니다
    # - {question} 부분은 나중에 실제 값으로 치환되는 변수입니다
    prompt = ChatPromptTemplate.from_template( # from_template() 클래스 메서드를 호출하여 prompt 변수에 할당합니다
        "질문을 'TECHNICAL' 또는 'CASUAL'로 분류하세요.\n질문: {question}\n결과:" # 프롬프트 템플릿 문자열을 전달합니다
    ) # 호출 종료
    
    #  LCEL 문법 Prompt -> Model -> StringParser (포장 뜯기)
    # - 파이프(|) 연산자로 prompt, model, parser를 연결하여 체인을 만듭니다
    # - 순서: prompt(프롬프트 포매팅) -> model(AI 실행) -> StrOutputParser(문자열 추출)
    chain = prompt | model | StrOutputParser() # | 연산자로 세 객체를 연결하고 chain 변수에 할당합니다
    
    #  체인 실행
    # - await chain.ainvoke() 메서드를 호출하여 AI에게 질문을 보냅니다
    # - 딕셔너리를 전달하면 {question} 변수가 치환됩니다
    result = await chain.ainvoke({"question": state["question"]}) # {"question": ...} 딕셔너리를 생성하여 ainvoke()에 전달하고 반환값을 result에 할당합니다
    
    #  결과 정제 (혹시 모를 공백이나 대소문자 문제 처리)
    # - result.upper() 메서드로 문자열을 대문자로 변환합니다
    # - "TECHNICAL" in 연산자로 해당 단어가 포함되어 있는지 확인합니다
    # - 삼항 연산자 (if else)로 분류 결과를 결정합니다
    classification = "TECHNICAL" if "TECHNICAL" in result.upper() else "CASUAL" # 조건식을 평가하여 결과를 classification 변수에 할당합니다
    
    #  결과 반환 (State 업데이트)
    # - 딕셔너리를 반환하면 LangGraph가 자동으로 State에 병합합니다
    return {"classification": classification} # 딕셔너리를 생성하여 return문으로 반환합니다

async def simple_tech_expert(state: SimpleRoutingState): # SimpleRoutingState 타입의 state를 받는 함수 정의
    """[노드 2-A] 기술 전문가 페르소나 (딱딱하고 전문적인 말투)"""
    #  디버깅 메시지 출력
    print("[Pattern 1] 기술 전문가 실행") # print()로 콘솔에 현재 가동 중인 노드를 표시합니다
    
    #  페르소나를 담은 프롬프트 생성
    # - "당신은 시니어 개발자입니다" <- 이 문장이 AI의 말투와 태도를 결정합니다
    prompt = ChatPromptTemplate.from_template("당신은 시니어 개발자입니다. 질문: {question}") # 프롬프트 템플릿을 생성하여 prompt 변수에 할당합니다
    
    #  체인 구성 및 실행
    chain = prompt | model | StrOutputParser() # 파이프로 연결하여 체인을 만듭니다
    
    #  결과 반환
    # - await chain.ainvoke()로 AI를 실행하고 그 결과를 딕셔너리에 담아 바로 반환합니다
    
    # [핵심] 여기서 왜 {"question": ...} 딕셔너리를 만드나요?
    # - 위에서 만든 프롬프트에 `{question}`이라는 빈칸(변수)이 있기 때문입니다.
    # - "빈칸 이름: 채울 값" 형태로 짝을 맞춰서 던져줘야 체인이 제대로 돌아갑니다.
    return {"response": await chain.ainvoke({"question": state['question']})}

async def simple_friendly_bot(state: SimpleRoutingState): # 함수 정의
    """[노드 2-B] 친구 페르소나 (친근한 말투)"""
    #  패턴 2-A와 거의 동일하며 프롬프트만 다릅니다
    print("[Pattern 1] 친구 봇 실행") # 디버깅 메시지 출력
    prompt = ChatPromptTemplate.from_template("당신은 친절한 친구입니다. 질문: {question}") # 친근한 페르소나를 설정한 프롬프트 생성
    chain = prompt | model | StrOutputParser() # 체인 구성
    return {"response": await chain.ainvoke({"question": state['question']})} # AI 실행 후 결과를 딕셔너리로 반환

def simple_route(state: SimpleRoutingState): 
    """
    [라우터] 조건부 엣지(Conditional Edge)를 위한 함수
    - State의 'classification' 값을 보고 다음 갈 길(노드 이름)을 알려줍니다.
    
    [Q&A] 왜 여기는 async가 아니라 그냥 def 인가요?
    - 라우터는 이미 데이터(State)가 다 채워진 상자를 보고 "왼쪽? 오른쪽?" 결정만 하는 '교통정리자'입니다.
    - AI를 부르는 무거운 작업(I/O)이 아니기 때문에, 굳이 비동기(async)로 만들지 않아도 아주 빠릅니다.
    - [규칙]: CPU 연산만 하는 가벼운 로직은 `def`를 쓰는 것이 성능상 더 이득입니다.
    """
    #  조건부 분기
    # - state 딕셔너리에서 "classification" 키의 값을 가져와 "TECHNICAL"과 비교합니다
    if state["classification"] == "TECHNICAL": # state[...]로 딕셔너리 값에 접근하고 == 연산자로 비교합니다
        # - 조건이 참이면 "tech_expert" 문자열을 반환합니다
        return "tech_expert" # "tech_expert" 문자열을 return문으로 반환합니다
    # - 조건이 거짓이면 "friendly_bot" 문자열을 반환합니다
    return "friendly_bot" # else 없이 바로 return하므로 기본 경로로 사용됩니다

# [3. 조립] 그래프 빌더 함수 (Refactoring의 핵심!)
# - 이전에는 전역 공간에 흩어져 있던 조립 로직을 함수 하나로 깔끔하게 모았습니다.

def build_pattern1_graph():
    """
    패턴 1 그래프 조립 (Blueprint)
    
    [Q&A] 여기도 왜 그냥 def 인가요?
    - 이 함수는 건물을 짓기 전 '설계도'를 그리는 단계입니다.
    - 실제 AI가 작동하는 단계가 아니라, 로봇의 구조만 짜는 작업이라 비동기가 필요 없습니다.
    - 서버가 켜질 때(Lifespan) 딱 한 번 실행되는 일종의 '세팅' 함수입니다.
    
    
    1. 그래프 객체 생성
    2. 노드 3개 추가 (classifier, tech_expert, friendly_bot)
    3. 시작점(Entry Point) 설정
    4. 라우팅 로직(Conditional Edge) 연결
    5. 종료 엣지 연결
    6. 컴파일(Compile)
    
    [핵심 개념] 컴파일(Compile)이란?
    - 우리가 정의한 노드와 엣지는 그저 "설계도"일 뿐입니다.
    - .compile()을 호출해야 LangGraph 엔진이 이를 해석하여 실제 "실행 가능한 기계(Runnable)"로 만듭니다.
    - 이 과정에서 그래프의 유효성 검사(끊긴 길은 없는지 등)도 수행됩니다.
    """
    #  빈 그래프 생성 (설계도 전달)
    # StateGraph() 생성자를 호출하여 workflow 변수에 할당합니다
    workflow = StateGraph(SimpleRoutingState) # SimpleRoutingState 타입을 전달하여 그래프 객체를 생성합니다
    
    #  부품(노드) 장착 (이름표 붙이기)
    # add_node() 메서드를 호출하여 그래프에 노드를 추가합니다
    # 첫 번째 인자는 노드 이름(문자열), 두 번째 인자는 실행할 함수입니다
    workflow.add_node("classifier", simple_classify) # "classifier"라는 이름으로 simple_classify 함수를 노드로 등록합니다
    workflow.add_node("tech_expert", simple_tech_expert) # "tech_expert" 노드를 등록합니다
    workflow.add_node("friendly_bot", simple_friendly_bot) # "friendly_bot" 노드를 등록합니다
    
    #  전선(엣지) 연결
    # - 시작점(Entry Point) 설정: 전원이 켜지면 무조건 'classifier'부터 시작
    # set_entry_point() 메서드를 호출하여 첫 번째로 실행될 노드를 지정합니다
    workflow.set_entry_point("classifier") # "classifier" 문자열을 전달하여 시작 노드로 설정합니다
    
    #  조건부 분기점 연결
    #   classifier가 끝나면 -> simple_route 함수를 실행해서 -> 결과에 따라 다음 노드로 이동
    # add_conditional_edges() 메서드를 호출하여 조건부 분기를 설정합니다
    workflow.add_conditional_edges(
        "classifier",     # 출발 노드 (문자열)
        simple_route,     # 조건 판단 함수 (함수 객체)
        {
            "tech_expert": "tech_expert",     # simple_route()가 "tech_expert"를 반환하면 "tech_expert" 노드로 이동
            "friendly_bot": "friendly_bot"    # simple_route()가 "friendly_bot"을 반환하면 "friendly_bot" 노드로 이동
        } # 딕셔너리로 반환값과 목적지 노드를 매핑합니다
    )
    
    #  종료 지점 연결 (작업 끝나면 END)
    # add_edge() 메서드를 호출하여 특정 노드 실행 후 종료하도록 설정합니다
    workflow.add_edge("tech_expert", END) # "tech_expert" 노드 실행 후 그래프를 종료합니다
    workflow.add_edge("friendly_bot", END) # "friendly_bot" 노드 실행 후 그래프를 종료합니다
    
    #  컴파일 (실행 가능한 기계로 변환)
    # compile() 메서드를 호출하여 그래프를 실행 가능한 객체로 변환하고 반환합니다
    return workflow.compile() # workflow.compile()을 호출하여 컴파일된 그래프 객체를 반환합니다

# ===============================================
# 📦 [패턴 2] 3단계 라우팅 (Advanced Routing)
# ===============================================

# [1. 설계도] State (패턴 1과 비슷하지만 명시적으로 정의)
# - 패턴 1과 똑같아 보이지만, 나중에 확장될 가능성을 고려해 분리하는 것이 좋습니다.
# [Q&A] 왜 패턴마다 State를 따로 만드나요?
# - 지금은 SimpleRoutingState와 필드가 똑같지만, 나중에 패턴 2에만 새 필드가 추가될 수 있습니다.
# - 미리 분리해두면 패턴 1을 건드리지 않고 패턴 2만 수정할 수 있어요. (확장성)
# - 규칙: "지금 같더라도, 목적이 다르면 분리하라" (유지보수 편의)

class AdvancedRoutingState(TypedDict):
    question: str
    classification: str
    response: str

# [2. 부품] 추가 노드
# - 패턴 1의 함수들을 재사용할 수도 있지만, 교육 목적상 '질문 분류기'와 '창의적 작가'를 추가 구현합니다.

async def advanced_classify(state: AdvancedRoutingState): # async def로 비동기 함수 정의, AdvancedRoutingState 타입의 state를 매개변수로 받습니다
    """
    [노드 1] 3단계 질문 분류기
    - TECH / CASUAL / CREATIVE 3가지로 분류합니다.
    - 패턴 1의 2-way 분류기를 확장한 버전입니다.
    """
    #  디버깅 출력
    # - print() 함수로 현재 처리 중인 질문을 콘솔에 출력합니다
    # - f-string을 사용해 state['question'] 값을 문자열에 삽입합니다
    print(f"[Pattern 2] 3단계 분류 중: {state['question']}") # 디버깅용 로그 출력
    
    #  프롬프트 구성
    # - ChatPromptTemplate.from_template()로 프롬프트 객체를 생성합니다
    # - {question} 부분은 나중에 실제 질문으로 치환됩니다
    # - AI에게 3가지 중 하나만 골라 달라고 명확하게 지시합니다
    prompt = ChatPromptTemplate.from_template( # from_template() 메서드 호출
        "사용자의 질문을 'TECHNICAL', 'CASUAL', 'CREATIVE' 중 하나로 분류하세요.\n"
        "질문: {question}\n"
        "결과 (단어 하나만):"
    ) # 프롬프트 템플릿을 prompt 변수에 할당합니다
    
    #  LCEL 체인 구성 및 비동기 실행
    # - | 연산자로 prompt -> model -> StrOutputParser를 연결합니다
    # - await chain.ainvoke()로 비동기 실행하고, 결과가 올 때까지 기다립니다
    chain = prompt | model | StrOutputParser() # 파이프 연산자로 체인 구성
    result = await chain.ainvoke({"question": state["question"]}) # 딕셔너리를 전달하고 결과를 result에 할당
    
    #  결과 정제 (Sanitization)
    # - AI가 "TECHNICAL입니다" 같이 불필요한 말을 붙일 수 있으므로
    # - .upper()로 대문자로 통일하고, .strip()으로 앞뒤 공백을 제거합니다
    res = result.upper().strip() # 결과를 대문자로 변환하고 공백 제거
    
    #  조건부 분류
    # - in 연산자로 해당 키워드가 결과에 포함되어 있는지 확인합니다
    # - 우선순위: TECHNICAL > CREATIVE > CASUAL (기본값)
    if "TECHNICAL" in res: # res 문자열에 "TECHNICAL"이 포함되어 있으면
        classification = "TECHNICAL" # classification 변수에 "TECHNICAL" 할당
    elif "CREATIVE" in res: # 아니면 "CREATIVE"가 포함되어 있으면
        classification = "CREATIVE" # classification 변수에 "CREATIVE" 할당
    else: # 둘 다 아니면
        classification = "CASUAL" # 기본값으로 "CASUAL" 할당
    
    #  State 업데이트용 딕셔너리 반환
    # - 반환된 딕셔너리는 LangGraph 엔진이 자동으로 State에 병합합니다
    return {"classification": classification} # classification 키와 값을 가진 딕셔너리를 반환

async def creative_writer(state: AdvancedRoutingState): # 함수 정의
    """
    [노드 2-C] 창의적 작가 (신규 추가)
    - 시, 소설, 아이디어 같은 창의적 질문을 담당합니다.
    """
    #  디버깅
    print("[Pattern 2] 창의적 작가 실행") # 현재 노드 표시
    
    #  상상력과 창의성을 강조하는 프롬프트
    # - "당신은 시인입니다" <- AI의 창작 성향을 가이드합니다
    prompt = ChatPromptTemplate.from_template("당신은 시인입니다. 창의적으로 답변하세요.\n질문: {question}") # from_template()로 프롬프트 생성
    
    #  체인 구성
    chain = prompt | model | StrOutputParser() # | 연산자로 체인 구성
    
    #  실행 및 반환
    return {"response": await chain.ainvoke({"question": state['question']})} # AI 실행 결과를 딕셔너리로 반환

def advanced_route(state: AdvancedRoutingState): # 함수 정의
    """
    [라우터] 3가지 경로 결정
    - Python 딕셔너리의 .get() 메소드를 활용하여 깔끔하게 분기합니다.
    """
    #  딕셔너리 매핑 기법
    # - 딕셔너리를 만들어 classification 값을 키로 사용하고 .get()으로 값을 가져옵니다
    # - .get()의 두 번째 인자는 기본값입니다 (키가 없을 때 반환)
    return { # 딕셔너리를 생성합니다
        "TECHNICAL": "tech_expert", # "TECHNICAL" 키에 "tech_expert" 값 할당
        "CREATIVE": "creative_writer", # "CREATIVE" 키에 "creative_writer" 값 할당
        "CASUAL": "friendly_bot" # "CASUAL" 키에 "friendly_bot" 값 할당
    }.get(state["classification"], "friendly_bot") # .get() 메서드로 state["classification"]를 키로 사용하여 값을 가져오고 없으면 "friendly_bot"을 기본값으로 반환합니다

# [3. 조립] 패턴 2 빌더
def build_pattern2_graph():
    """
    패턴 2 그래프 조립 (3-way 분기)
    
    
    1. 그래프 객체 생성
    2. 노드 4개 추가 (재사용 포함)
    3. 시작점 설정
    4. 3갈래 조건부 분기 설정
    5. 종료 엣지 연결
    6. 컴파일
    """
    #  그래프 생성
    # - AdvancedRoutingState를 사용하는 StateGraph 객체를 생성합니다
    workflow = StateGraph(AdvancedRoutingState) # StateGraph 객체를 workflow 변수에 할당합니다
    
    #  노드 추가 (패턴 1의 함수 재사용 + 신규 함수)
    # - add_node() 메서드로 각 기능을 하는 함수들을 노드에 등록합니다
    workflow.add_node("classifier", advanced_classify) # "classifier"라는 이름으로 advanced_classify 함수 등록
    workflow.add_node("tech_expert", simple_tech_expert)   # "tech_expert" 이름으로 simple_tech_expert 함수 등록 (재사용)
    workflow.add_node("friendly_bot", simple_friendly_bot) # "friendly_bot" 이름으로 simple_friendly_bot 함수 등록 (재사용)
    workflow.add_node("creative_writer", creative_writer)  # "creative_writer" 이름으로 creative_writer 함수 등록 (신규)
    
    #  시작점 설정
    # - 그래프가 시작되면 가장 먼저 실행될 노드를 지정합니다
    workflow.set_entry_point("classifier") # "classifier"를 시작점으로 설정합니다
    
    #  3갈래 조건부 분기 연결
    # - classifier 실행 후 advanced_route 함수의 반환값에 따라 경로를 나눕니다
    workflow.add_conditional_edges(
        "classifier", # 출발 노드
        advanced_route, # 경로 결정 함수 (라우터)
        {
            "tech_expert": "tech_expert", # advanced_route가 "tech_expert"를 반환하면 이동
            "friendly_bot": "friendly_bot", # "friendly_bot" 반환 시 이동
            "creative_writer": "creative_writer" # "creative_writer" 반환 시 이동
        } # 딕셔너리로 경로 매핑
    ) # 조건부 엣지 추가 완료
    
    #  종료 엣지 연결
    # - 각 전문가 노드가 작업이 끝나면 그래프를 종료하도록 설정합니다
    workflow.add_edge("tech_expert", END) # tech_expert -> END 연결
    workflow.add_edge("friendly_bot", END) # friendly_bot -> END 연결
    workflow.add_edge("creative_writer", END) # creative_writer -> END 연결
    
    #  컴파일 및 반환
    # - 실행 가능한 그래프 객체로 변환합니다
    return workflow.compile() # workflow.compile() 호출 결과를 반환합니다

# ===============================================
# 📦 [패턴 3] ReAct Agent (Reasoning + Acting)
# ===============================================

# [핵심 개념] ReAct(Reasoning + Acting) 패턴이란?
# - 단순한 질문-답변(QA)을 넘어, AI가 스스로 "생각(Reasoning)"하고 "행동(Acting)"하며 문제를 해결하는 방식입니다.
# - Loop(반복): 생각 -> 도구 실행 -> 결과 관찰 -> 다시 생각 -> ... -> 최종 답변
# - 이 과정을 통해 복잡한 문제(예: 최신 정보 검색 후 요약)를 해결할 수 있습니다.

# [1. 설계도] 복잡한 State
# - ReAct 패턴은 "생각(Think) -> 행동(Act) -> 관찰(Observe)"을 계속 반복하므로
# - 중간 과정에서 생긴 데이터를 계속 누적하거나 전달해야 합니다.
class ReActState(TypedDict):
    question: str       # 사용자의 최초 질문
    thought: str        # "지금 상황이 이렇네, 다음엔 뭘 해야지" (Reasoning)
    action: str         # "계산기를 써야겠다" (도구 이름)
    action_input: str   # "15 * 23" (도구에 넣을 값)
    observation: str    # "345" (도구 실행 결과)
    iterations: int     # 무한 루프에 빠지지 않도록 안전장치 (반복 횟수)
    final_answer: str   # 사용자가 원하던 최종 답

# [2. 부품] 노드
async def react_think(state: ReActState): # 비동기 함수 정의
    """
    [노드 1] Reasoning (생각하기) 단계
    - 현재까지의 관찰 결과(Observation)를 보고 다음 행동(Action)을 결정합니다.
    """
    #  반복 횟수 증가
    # - state.get("iterations", 0)으로 iterations 키의 값을 가져오고 없으면 0을 기본값으로 사용합니다
    # - + 1을 하여 현재 반복 횟수를 계산합니다
    iter_count = state.get("iterations", 0) + 1 # state.get()으로 값을 가져와 1을 더하고 iter_count 변수에 할당합니다
    if iter_count >= 5:
        return {"final_answer": "I give up"}

    #  디버깅 출력
    print(f"[ReAct] 생각 중... (반복 {iter_count}회)") # f-string으로 반복 횟수를 포함한 문자열을 print()로 출력합니다
    
    #  이전 단계의 관찰 결과를 프롬프트에 포함 (Context)
    # - state에서 'observation' 키의 값을 가져오고 없으면 빈 문자열을 사용합니다
    obs = state.get('observation', '') # state.get()으로 observation 값을 가져와 obs 변수에 할당합니다
    
    #  프롬프트 템플릿 생성 (정석 방식)
    # - ChatPromptTemplate.from_template()으로 템플릿을 만들고
    # - {변수} 자리에 나중에 ainvoke()로 값을 전달합니다
    prompt = ChatPromptTemplate.from_template(
        "질문: {question}\n"
        "이전 관찰: {observation}\n\n"
        "사용 가능 도구: calculator, knowledge_search\n\n"
        "다음 형식으로 답변하세요:\n"
        "Thought: (생각 내용)\n"
        "Action: (도구 이름 또는 FINISH)\n"
        "Action Input: (입력값)"
    )
    
    #  LCEL 체인 구성 및 비동기 실행
    # - | 연산자(파이프)로 prompt -> model -> StrOutputParser를 연결합니다
    # - 파이프는 "A의 출력을 B의 입력으로 넘겨라"는 의미입니다
    chain = prompt | model | StrOutputParser() # 체인 객체를 생성하여 chain 변수에 할당합니다
    
    # - await chain.ainvoke()로 비동기 실행합니다
    # - 딕셔너리의 키 이름("question", "observation")이 템플릿의 {변수} 이름과 일치해야 합니다!
    result = await chain.ainvoke({
        "question": state["question"],  # 템플릿의 {question}에 들어갈 값
        "observation": obs              # 템플릿의 {observation}에 들어갈 값
    }) # AI가 답변을 생성하면 result에 문자열로 저장됩니다
    
    #  결과 파싱 (Thought / Action / Action Input 분리)
    # - AI가 여러 줄로 답변("Thought: ...\nAction: ...")을 주므로, 줄 단위로 쪼갭니다
    # - .strip()으로 앞뒤 공백을 제거하고, .split('\n')으로 줄바꿈 기준으로 리스트를 만듭니다
    lines = result.strip().split('\n') # ["Thought: ...", "Action: ...", "Action Input: ..."]
    
    #  각 필드에 대한 기본값 설정 (안전장치)
    # - AI가 이상한 답변을 하면 파싱이 실패할 수 있으므로, 기본값을 미리 정해둡니다
    # - action의 기본값을 "FINISH"로 두면, 파싱 실패 시 루프가 종료됩니다 (무한루프 방지)
    thought, action, action_input = "", "FINISH", "파싱 실패" # 튜플 언패킹으로 3개 변수에 한 번에 할당
    
    #  각 줄을 순회하며 필드 추출
    # - for문으로 lines 리스트의 각 요소(줄)를 line 변수에 담아 반복합니다
    # [Q&A] 도대체 이 for문이 왜 필요한가요?
    # 1. AI의 응답은 사실 하나의 **긴 문자열(Raw Text)** 덩어리입니다.
    #    예: "Thought: 계산해야지\nAction: calculator\nAction Input: 1+1"
    # 2. 컴퓨터는 이 문자열을 보고 바로 "아, 계산기 돌려야지"라고 알 수 없습니다.
    # 3. 그래서 이 for문이 문자열을 한 줄씩 읽으면서 "Action이 뭐지?", "Input이 뭐지?" 하고 변수에 담아주는 과정입니다.
    # 4. 마치 사람이 쓴 편지를 읽고 "수신인: 철수", "내용: 안녕" 이렇게 장부에 옮겨 적는 것과 같습니다.
    for line in lines: # lines 리스트를 순회합니다
        # - .startswith()로 해당 줄이 특정 접두사로 시작하는지 확인합니다
        if line.startswith("Thought:"): # "Thought:"로 시작하면
            # - .replace()로 접두사를 제거하고, .strip()으로 앞뒤 공백을 제거합니다
            thought = line.replace("Thought:", "").strip() # thought 변수에 할당
        elif line.startswith("Action:"): # "Action:"으로 시작하면
            # [핵심] "Action: calculator" -> "calculator" (껍질 까기)
            # - 앞에 붙은 "Action:"이라는 라벨(Label)을 제거해야, 컴퓨터가 변수명으로 인식합니다.
            action = line.replace("Action:", "").strip() # action 변수에 할당
        elif line.startswith("Action Input:"): # "Action Input:"으로 시작하면
            # "Action Input: 서울 날씨" -> "서울 날씨"
            action_input = line.replace("Action Input:", "").strip() # action_input 변수에 할당
    
    #  결과 반환 (State 업데이트)
    # - 반환된 딕셔너리는 LangGraph 엔진이 자동으로 기존 State에 병합(Merge)합니다
    # - 예: 기존 state = {"question": "..."} → 병합 후 state = {"question": "...", "thought": "...", ...}
    return {
        "thought": thought,         # AI의 생각 과정
        "action": action,           # 다음에 사용할 도구 이름 (또는 "FINISH")
        "action_input": action_input, # 도구에 전달할 입력값
        "iterations": iter_count    # 현재 반복 횟수
    } # 딕셔너리를 반환하여 State를 업데이트합니다

def react_act(state: ReActState): # ReActState 타입의 state를 받는 함수 정의
    """
    [노드 2] Acting (행동하기) 단계
    - 'think' 노드가 결정한 도구를 실제로 실행합니다.
    - 실행 결과(Observation)를 반환하여 다음 'think' 단계로 넘깁니다.
    """
    #  State에서 필요한 값 추출
    action = state.get("action") # state.get()로 "action" 키의 값을 가져와 action 변수에 할당합니다
    inp = state.get("action_input") # state.get()로 "action_input" 키의 값을 가져와 inp 변수에 할당합니다
    
    #  디버깅 출력
    print(f"[ReAct] 도구 실행: {action}({inp})") # f-string으로 도구 이름과 입력값을 포함한 문자열을 print()로 출력합니다
    
    #  종료 조건 (FINISH) 처리
    #
    # [Q&A] 갑자기 inp가 왜 '정답'이 되나요?
    # - 보통 `inp`는 계산기에 넣을 숫자("1+1") 같은 '부품'입니다.
    # - 하지만 `action`이 "FINISH"일 때, AI는 `inp` 자리에 '최종 답변'을 담아서 줍니다.
    # - 예: "Action: FINISH", "Action Input: 서울 날씨는 맑음입니다."
    # - 즉, 여기서 inp는 도구의 재료가 아니라, 사용자에게 줄 '선물(Result)'로 변신합니다.
    if action == "FINISH": # "일 다 끝났다"는 신호가 오면
        return {"final_answer": inp} # inp에 담긴 최종 답변을 그대로 반환하고 종료
    
    #  실제 도구 함수 호출
    #
    # [비유 설명] 작업 지시서(Work Order) 이행
    # - 앞 단계(Think)에서 "계산기 써라(action)"라는 '종이(Text)'를 받았습니다.
    # - 이제 진짜 '계산기 기계(Function)'를 꺼내서 버튼을 누르는(Call) 단계입니다.
    # - TOOLS[action]은 "이름표(String)를 보고 진짜 도구(Function)를 찾아내는 과정"입니다.
    if action in TOOLS: # action이 TOOLS 딕셔너리의 키에 포함되는지 in 연산자로 확인합니다
        # - TOOLS 딕셔너리에서 해당 도구 함수를 가져옵니다
        tool_func = TOOLS[action] # TOOLS[action]으로 함수 객체를 가져와 tool_func 변수에 할당합니다
        # - 도구 함수를 실행하고 결과를 observation에 저장합니다
        obs = tool_func(inp) # tool_func()를 inp 인자로 호출하고 반환값을 obs 변수에 할당합니다
    else: # action이 TOOLS에 없는 경우
        # - 에러 메시지를 observation으로 설정합니다
        obs = f"오류: {action} 알 수 없는 도구" # f-string으로 에러 메시지를 생성하여 obs 변수에 할당합니다
    
    #  관찰 결과 반환 (다음 사이클에서 사용)
    return {"observation": obs} # obs 값을 "observation" 키에 할당한 딕셔너리를 반환합니다

def react_condition(state: ReActState): # 함수 정의
    """
    [조건부 엣지] 계속할지(Loop) 끝낼지(End) 결정
    - 반복 횟수가 너무 많으면 강제 종료 (안전장치)
    - AI가 FINISH를 선언하면 종료
    - 아니면 다시 생각(think) 하러 이동
    """
    #  반복 횟수 제한 확인 (무한 루프 방지)
    # - state.get()으로 iterations 값을 가져와 5와 비교합니다
    if state.get("iterations", 0) >= 5: # state.get("iterations", 0)을 호출하고 >= 연산자로 5와 비교합니다
        # - 5회 이상 반복하면 "end"를 반환하여 루프 종료
        return "end" # "end" 문자열을 return문으로 반환합니다
    
    #  AI가 FINISH를 선언했는지 확인
    # - state.get()으로 action 값을 가져와 "FINISH"와 비교합니다
    if state.get("action", "") == "FINISH": # state.get("action", "")을 호출하고 == 연산자로 "FINISH"와 비교합니다
        # - FINISH면 "end"를 반환하여 종료
        return "end" # "end" 문자열을 반환합니다
    
    #  위 조건이 모두 거짓이면 계속 반복
    # - "continue"를 반환하면 다시 think 노드로 돌아갑니다
    return "continue" # "continue" 문자열을 반환하여 루프를 계속합니다

# [3. 조립] 패턴 3 빌더
def build_pattern3_graph():
    """
    패턴 3 (ReAct) 그래프 조립
    
    
    1. 그래프 객체 생성
    2. Think, Act 노드 추가
    3. 시작점은 Think
    4. Think -> Act 일반 엣지 연결
    5. Act -> (Think or End) 조건부 엣지 연결
    6. 컴파일
    """
    #  그래프 생성
    workflow = StateGraph(ReActState) # ReActState를 사용하는 StateGraph 객체 생성하여 할당
    
    #  노드 추가
    # - 생각하는 노드와 행동하는 노드 두 개를 추가합니다
    workflow.add_node("think", react_think) # "think"라는 이름으로 react_think 함수 등록
    workflow.add_node("act", react_act) # "act"라는 이름으로 react_act 함수 등록
    
    #  시작점 설정
    # - ReAct 패턴의 시작은 항상 '생각하기' 입니다
    workflow.set_entry_point("think") # "think"를 시작점으로 설정
    
    #  일반 엣지 연결
    # - 생각(Think)이 끝나면 무조건 행동(Act) 여부를 판단하거나 실행해야 합니다
    # [핵심 개념] 엣지(Edge)의 종류
    # 1. 일반 엣지: 조건 없이 무조건 다음 단계로 이동 (Think -> Act)
    # 2. 조건부 엣지: 상황에 따라 경로가 갈림 (Act -> Think 또는 End)
    workflow.add_edge("think", "act") # "think" 노드가 끝나면 "act" 노드로 가도록 정적 엣지 연결
    
    #  조건부 엣지 (Loop 결정)
    # - 행동(Act) 결과(관찰)를 보고 다시 생각할지(Loop), 아니면 끝낼지(End) 결정합니다
    # [실무 Tip] 순환 그래프(Cyclic Graph) 주의사항
    # - 잘못하면 무한 루프에 빠질 수 있으므로, 반드시 iterations 체크 같은 탈출 조건(Exit Condition)이 필요합니다.
    workflow.add_conditional_edges(
        "act", # 출발 노드: 행동(Act)
        react_condition, # 조건 판단 함수
        {"continue": "think", "end": END} # 매핑: continue면 think로, end면 종료
    ) # 조건부 엣지 추가
    
    #  컴파일
    return workflow.compile() # 컴파일된 그래프 객체 반환

# ===============================================
# 📦 [패턴 4] 대화 루프 Agent (Chat w/ History)
# ===============================================

# [핵심 개념] Stateful vs Stateless
# - Stateless (패턴 1, 2, 3): 매 요청마다 새로운 질문을 처리합니다. 과거 기억이 없습니다.
# - Stateful (패턴 4): "안녕" -> "반가워" -> "내 이름이 뭐게?" -> "기억 안나요" (문맥 유지)
# - 이를 위해 'session_id'로 사용자별 대화 기록(messages)을 따로 관리해야 합니다.

# [1. 설계도]
class ChatLoopState(TypedDict):
    session_id: str         # 사용자 구분용 ID (예: "user123")
    messages: List[str]     # 대화 히스토리 (["안녕", "반가워요"]) -> 누적됨
    current_input: str      # 방금 들어온 입력
    response: str           # AI 답변

# [2. 부품]
async def chat_respond(state: ChatLoopState): # ChatLoopState 타입의 state를 받는 함수 정의
    """대화 히스토리를 보고 답변"""
    #  디버깅
    print(f"[Chat] 답변 생성 중... (세션: {state['session_id']})") # f-string으로 세션 ID를 포함한 메시지를 print()로 출력합니다
    
    #  히스토리를 하나의 문자열로 합쳐서 문맥(Context) 제공
    # - state.get("messages", [])로 messages 리스트를 가져옵니다
    # - "\n".join()로 리스트의 각 요소를 줄바꿈으로 연결합니다
    history = "\n".join(state.get("messages", [])) # "\n".join()을 호출하여 리스트의 각 요소를 연결합니다
    
    #  프롬프트 템플릿 생성 (정석 방식)
    # - ChatPromptTemplate.from_template()으로 템플릿을 만듭니다
    # - {history}, {current_input} 변수는 ainvoke()에서 딕셔너리로 전달합니다
    prompt = ChatPromptTemplate.from_template(
        "이전 대화:\n{history}\n\n"
        "사용자: {current_input}\n\n"
        "당신은 친절한 AI 어시스턴트입니다. 자연스럽게 대화하세요."
    )
    
    #  LCEL 체인 구성 및 비동기 실행
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({
        "history": history,
        "current_input": state["current_input"]
    }) # 딕셔너리로 템플릿 변수에 값을 전달합니다
    
    #  결과 반환
    return {"response": response} # response 값을 딕셔너리에 담아 반환합니다

def chat_check_end(state: ChatLoopState): # 함수 정의
    """
    [조건부 엣지] 종료 키워드 감지
    - '종료', '끝' 같은 단어가 나오면 대화를 멈춥니다.
    """
    #  사용자 입력 추출 및 소문자 변환
    # - state.get()으로 current_input 값을 가져오고 .lower()로 소문자로 변환합니다
    user_input = state.get("current_input", "").lower() # state.get().lower()를 호출하여 소문자 문자열로 변환하고 user_input 변수에 할당합니다
    
    #  종료 키워드 확인
    # - 종료 키워드 리스트를 정의합니다
    end_keywords = ['종료', '끝', 'bye', 'quit', 'exit'] # 리스트를 생성하여 end_keywords 변수에 할당합니다
    
    #  각 키워드를 순회하며 포함 여부 확인
    for keyword in end_keywords: # end_keywords 리스트의 각 요소를 keyword 변수에 할당하며 반복합니다
        if keyword in user_input: # keyword가 user_input에 포함되는지 in 연산자로 확인합니다
            # - 키워드가 발견되면 "end"를 반환합니다
            return "end" # "end" 문자열을 return문으로 반환합니다
    
    #  종료 키워드가 없으면 계속
    #
    # [Q&A] 진짜 궁금한 건데, 그냥 창 끄면 끝 아닌가요? 이걸 왜 코드에서 따지죠?
    # - 아주 날카로운 질문입니다! 웹사이트라면 그냥 창 끄면 끝이 맞습니다.
    # - 하지만 **LangGraph 내부 로직** 입장에서는 이 그래프가 '무한 루프(Cycle)' 구조입니다.
    # - "대답했어? -> 다시 듣기 대기 -> 대답했어? -> ..." 이 굴레를 **논리적으로 끊어주는 장치(Break)**가 필요해서 그렇습니다.
    # - 안 그러면 그래프 입장에선 "아직 안 끝났는데 사용자가 도망갔네?" 하고 영원히 대기 상태로 남을 수도 있거든요.
    #
    # [실무에서는 어떻게 하나요?]
    # 1. **타임아웃 (Timeout)**: "30분 동안 말 없으면 자동 종료" (가장 흔함)
    # 2. **UI 버튼**: 사용자가 "새 대화 시작" 버튼을 누르면 강제로 종료 신호를 보냄
    # 3. **목표 달성**: 상담 챗봇이 "예약 완료되었습니다. 종료합니다." 하고 스스로 끊음
    # 이 3가지 중 하나를 쓰는데, 이 예제 코드는 UI도 없고 DB도 없어서 부득이하게 타이핑으로 흉내만 낸 것입니다
    return "continue" # for 루프가 종료되고 종료 키워드가 없었으므로 "continue" 문자열을 반환합니다

# [3. 조립] 패턴 4 빌더
def build_pattern4_graph():
    """
    패턴 4 (대화 루프) 그래프 조립
    
    
    1. 그래프 객체 생성
    2. Respond 노드 추가
    3. 시작점 설정
    4. Respond -> (End or End) 조건부 엣지 연결 (종료 조건 체크)
    5. 컴파일
    """
    #  그래프 생성
    # - ChatLoopState를 사용하는 StateGraph 객체를 생성합니다
    workflow = StateGraph(ChatLoopState) # StateGraph 객체를 workflow 변수에 할당합니다
    
    #  노드 추가
    # - 답변을 생성하는 단일 노드를 추가합니다
    workflow.add_node("respond", chat_respond) # "respond"라는 이름으로 chat_respond 함수 등록
    
    #  시작점 설정
    workflow.set_entry_point("respond") # "respond"를 시작점으로 설정
    
    #  답변 후 -> 계속 대화할지 종료할지 판단
    #
    # [Q&A] 진짜 이상하네요. "continue"인데 왜 END(종료)로 가나요?
    # - 이것은 이 코드가 **'웹 서버(API)'** 방식이기 때문입니다.
    # - **(일반 프로그램)**: `while True:`로 계속 돕니다. (continue -> 다시 처음으로)
    # - **(웹 API)**: 질문 1개 받고 -> 답변 1개 주고 -> **(연결을 끊어야 함!)** -> 다음 요청 대기
    # - 즉, 여기서 END는 "영원한 종료"가 아니라, "**이번 턴(Turn)을 마치고 응답을 보낸다**"는 뜻입니다.
    # - 만약 "continue"하면 클라이언트는 다음 답변을 **기다리지 않고** 서버 연결을 종료했다가, 나중에 다시 요청합니다.
    workflow.add_conditional_edges(
        "respond", # 출발 노드: 답변 생성
        chat_check_end, # 조건 판단 함수
        {"continue": END, "end": END} # 그래서 둘 다 END로 가지만 의미는 다릅니다. (다음 요청 대기 vs 아예 세션 파기)
    ) # 조건부 엣지 추가
    
    #  컴파일
    return workflow.compile() # 컴파일된 그래프 객체 반환

# ===============================================
# 📍 [구역 4] 공장 가동 (Lifespan: Startup/Shutdown)
# ===============================================

@asynccontextmanager
async def lifespan(app: FastAPI): # asynccontextmanager 데코레이터로 컨텍스트 매니저 정의
    """
    [서버 수명주기 관리자] (Context Manager)
    
    [핵심 개념] 왜 Lifespan을 쓰나요?
    1. 효율성: AI 모델이나 그래프 빌드는 무겁습니다. 매 요청마다 만들면 서버가 느려집니다.
    2. 안전성: DB 연결이나 파일 열기 같은 리소스는 서버 종료 시 안전하게 닫아야 합니다.
    
    [비유 설명] 식당 개업 준비 🍳
    - 손님이 올 때마다(API 요청) 가스레인지를 사러 가면 안 됩니다.
    - 가게 문 열기 전에 미리 가스레인지 설치하고(Startup), 재료 손질해둬야(Initialize),
    - 손님이 오면 바로 요리(Response)를 내어줄 수 있습니다.
    
    - 역할: 서버가 켜질 때(startup)와 꺼질 때(shutdown) 실행할 코드를 정의합니다.
    - 중요성: AI 모델이나 그래프 같은 무거운 객체는 여기서 '딱 한 번만' 만들어야 합니다.
    - 
      1. 서버 시작 -> 2. yield 전까지 코드 실행 (초기화) -> 3. 서버 작동 (yield) -> 4. 서버 종료 시 나머지 실행 (정리)
    """
    #  전역 변수 호출
    # - 함수 외부(Global)에 있는 변수들을 사용하겠다고 선언합니다
    global model, simple_graph, advanced_graph, react_graph, chat_graph # global 키워드 사용
    
    #  시작 로그 출력
    print("\n🚀 [시작] LangGraph 서버 초기화 중...") # 시작 메시지 출력
    print("   (잠시만 기다려주세요. AI 모델과 그래프를 조립합니다.)")
    
    #  AI 모델 생성 (빈 상자 채우기)
    # - 아까 맨 위에서 `model = None`으로 비워뒀던 상자에, 이제 진짜 AI 객체를 넣습니다.
    # - 이 작업은 서버가 켜질 때 **딱 한 번만** 실행됩니다. (Singleton)
    # - [장점] 이후에 들어오는 수천 명의 사용자는 이미 만들어진 이 `model`을 공유해서 씁니다. 엄청 빠르겠죠?
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    #  그래프 조립 (공장 가동)
    # - 설계도(함수)를 바탕으로 실제 작동하는 로봇(Graph)을 조립해서 전역 변수에 넣습니다.
    simple_graph = build_pattern1_graph() # 패턴 1 빌더 호출 및 할당
    print("✅ 패턴 1 (Simple Routing) 준비 완료") # 로딩 완료 로그
    
    advanced_graph = build_pattern2_graph() # 패턴 2 빌더 호출 및 할당
    print("✅ 패턴 2 (Advanced Routing) 준비 완료")
    
    react_graph = build_pattern3_graph() # 패턴 3 빌더 호출 및 할당
    print("✅ 패턴 3 (ReAct Agent) 준비 완료")
    
    chat_graph = build_pattern4_graph() # 패턴 4 빌더 호출 및 할당
    print("✅ 패턴 4 (Chat Loop) 준비 완료")
    
    print("✨ 모든 시스템 준비 완료! http://localhost:8000/docs 에 접속하세요.\n")
    
    #  서버 실행 대기 (Yield)
    # - 여기서 코드가 멈추고 서버가 클라이언트 요청을 받기 시작합니다
    # - 서버가 종료 신호(Ctrl+C)를 받을 때까지 여기서 대기합니다
    yield # [서버 작동 중] 여기가 서버가 켜져 있는 동안의 상태입니다.
    
    #  서버 종료 (정리 작업)
    # - yield 이후의 코드는 서버가 꺼질 때 실행됩니다
    # - DB 연결 해제, 파일 닫기 등을 여기서 수행합니다
    print("🛑 [종료] 서버가 종료되었습니다. 리소스를 정리합니다.") # 종료 로그 출력

# ===============================================
# 📍 [구역 5] 민원 창구 (API Endpoints)
# ===============================================
#
# [비유 설명] 은행 창구 / 민원실 🏦
# - 우리가 은행에 가면 "예금 창구", "대출 창구"가 따로 있듯이,
# - 서버에도 "패턴 1 창구(/pattern1)", "패턴 2 창구(/pattern2)"를 만들어 두는 곳입니다.
# - 사용자는 이 주소(URL)로 신청서(Request)를 내고, 결과(Response)를 받아갑니다.
# - 실제 사용자(User)는 오직 `/master_bot`라는 **"정문(Main Entrance)"** 하나만 이용합니다.
# - 정문에 있는 '안내 데스크(Master Router)'가 알아서 1번, 2번, 3번 창구로 안내해줍니다.

# [핵심] lifespan을 app에 등록해야 실제로 작동합니다.
# - lifespan 파라미터: 서버 시작/종료 시 실행할 함수를 지정합니다.
app = FastAPI(lifespan=lifespan) # FastAPI 객체 생성 시 lifespan 함수 등록

# ================================================================================
# 🏢 [실무 권장 가이드] "FastAPI 딱 한 줄에 다 끋내기"
# ================================================================================
# [현재 코드] `app = FastAPI(lifespan=lifespan)` <- 기본 설정만 있음
#
# [실무에서는?] 아래 설정들을 추가해야 합니다:
#
# 1. **CORS (Cross-Origin Resource Sharing)**:
#    - 프론트엔드(React 등)와 다른 도메인일 때 필수!
#    ```python
#    from fastapi.middleware.cors import CORSMiddleware
#    app.add_middleware(
#        CORSMiddleware,
#        allow_origins=["https://yourfrontend.com"],  # 허용할 도메인
#        allow_methods=["*"],
#        allow_headers=["*"],
#    )
#    ```
#
# 2. **Rate Limiting (요청 제한)**:
#    - DDoS 공격 방지, API 남용 막기
#    ```python
#    from slowapi import Limiter
#    limiter = Limiter(key_func=get_remote_address)
#    app.state.limiter = limiter
#    @app.get("/limited")
#    @limiter.limit("5/minute")  # 1분에 5번만 허용
#    async def limited_route(): ...
#    ```
#
# 3. **인증/인가 (Authentication/Authorization)**:
#    - 누가 요청했는지 확인, 권한 검사
#    ```python
#    from fastapi import Depends, Security
#    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#    security = HTTPBearer()
#    @app.get("/protected")
#    async def protected_route(creds: HTTPAuthorizationCredentials = Security(security)):
#        # JWT 토큰 검증 로직
#        ...
#    ```
#
# 4. **로깅 (Logging)**:
#    - print() 대신 구조화된 로그 사용 (ELK, CloudWatch 연동)
#    ```python
#    import logging
#    logging.basicConfig(level=logging.INFO)
#    logger = logging.getLogger(__name__)
#    logger.info(f"User {user_id} requested pattern {pattern}")
#    ```
# ================================================================================

# ===============================================
# 📝 데이터 모델 정의 (Pydantic Models)
# ===============================================
# [핵심 개념] Pydantic이 뭔가요?
# - 클라이언트가 보낸 데이터(Body)가 우리가 원하는 모양인지 "자동으로 검사"해주는 문지기입니다.
# - 예: "나이(age)는 숫자여야 해"라고 정의하면, 문자열 "열살"이 들어올 때 자동으로 에러(422)를 냅니다.
# - [장점] 더 이상 `if "question" not in body:` 같은 지루한 검사 코드를 짤 필요가 없습니다!
#
# [Q&A] TypedDict랑 뭐가 다른가요? (중요 ⭐)
#
# **1. Pydantic (`BaseModel`) = "입구 지키는 보안관(Sheriff)" 👮‍♂️**
#    - **언제 쓰나요?**: 외부 사용자(남)가 데이터를 보낼 때. (믿을 수 없음)
#    - **하는 일**: 데이터가 규칙에 맞는지 **현미경으로 검사**합니다.
#    - **[실제 상황 예시] 회원가입**
#      ```python
#      class SignupReq(BaseModel):
#          email: EmailStr  # "이메일 형식이 아니면 절대 안 돼!"
#          age: int         # "숫자만 가져와!"
#      
#      # 사용자가 {"email": "hello", "age": "열살"} 보냄
#      # -> 🚨 Pydantic: "이메일 형식이 틀렸고, 나이는 숫자여야 해!" (422 Error 방출)
#      # -> 서버 코드는 실행조차 안 됨 (안전 확보)
#      ```
#
# **2. TypedDict = "팀원끼리 쓰는 포스트잇(Post-it)" 📝**
#    - **언제 쓰나요?**: 내부 로직(우리)끼리 데이터를 옆으로 넘길 때. (믿을 수 있음)
#    - **하는 일**: "이 변수에는 이런 게 들어있을 거야"라고 **힌트만 줍니다.**
#    - **[실제 상황 예시] 내부 로직 처리**
#      ```python
#      class UserState(TypedDict):
#          status: str
#      
#      # 로봇1이 로봇2에게: {"status": "active", "bonus": 100}
#      # -> ✅ TypedDict: "어? 정의 안 된 'bonus'가 있네? 근데 뭐 우리끼리니까 OK."
#      # -> 에러 없이 유연하게 넘어갑니다. (속도 빠름, 오버헤드 없음)
#      ```
#
# **[결론]**
# - **API 요청 받을 때** -> 무조건 **Pydantic** (안전 제일)
# - **LangGraph 상태 관리** -> **TypedDict** (유연성 & 속도)
#
# **---------------------------------------------------**
# **[이 파일에서의 실제 적용 예시]**
# **1. Pydantic 사용 (1274라인 `QueryRequest`)**
#    - 사용자가 `{"question": 123}`이라고 숫자를 보내면?
#    - `QueryRequest`가 "어? `question`은 `str`이어야 하는데?" 하고 **422 에러로 막아줍니다.**
#
# **2. TypedDict 사용 (690라인 `AdvancedRoutingState`)**
#    - 그래프 내부에서 `{"classification": "TECH"}` 정보를 넘길 때?
#    - 복잡한 검사 없이 **빠르게 딕셔너리로 주고받습니다.**
#
# **[Q&A] 검사도 안 하는데 TypedDict는 왜 굳이 정의하나요? (귀찮게)**
# - **이유는 '오타 방지'와 '자동 완성' 때문입니다.** 🛠️
# 1. **자동 완성**: `state["q...`까지만 쳐도 `question`이 뜹니다. (개발 편의성)
# 2. **오타 검출**: 실수로 `state["queston"]`이라고 치면, 빨간 줄을 그어줍니다. (버그 예방)
# 3. **설명서**: "이 상태에는 question이랑 answer만 있어"라고 팀원에게 알려주는 역할입니다.
# -> 만약 이게 없으면, 코드를 다 뒤져봐야 "아, 여기에 answer라는 키가 들어가는구나" 하고 알 수 있게 됩니다.
# **---------------------------------------------------**

class QueryRequest(BaseModel):
    """
    [모델 1] 단순 질문 요청용 (Request Schema)
    - 가장 기본적인 형태의 입력 규격입니다.
    """
    # [설명] question 필드: 사용자가 물어볼 내용을 담습니다.
    # - 타입 힌트 (str): 반드시 문자열이어야 합니다.
    # - 필수 여부: 기본값(= "")이 없으므로 필수 항목입니다. 안 보내면 에러 납니다.
    question: str 

    # ================================================================================
    # 🏢 [실무 권장 가이드] "더 깐깐한 입력 검증"
    # ================================================================================
    # [현재 코드] `question: str` -> 빈 문자열("")이나 100만 자 텍스트도 통과됨 😱
    #
    # [실무에서는?] `pydantic.Field`를 써서 구체적으로 제한해야 합니다.
    # ```python
    # from pydantic import Field, field_validator
    # 
    # class StrictRequest(BaseModel):
    #     question: str = Field(..., min_length=2, max_length=1000) # 길이 제한
    #     
    #     @field_validator('question')
    #     def check_safe_text(cls, v):
    #         if "DROP TABLE" in v.upper():
    #             raise ValueError("SQL Injection 감지됨!")
    #         return v
    # ```
    # ================================================================================

class ChatRequest(BaseModel):
    """
    [모델 2] 대화형 질문 요청용 (Context Request)
    - "누구(session_id)"가 "뭐라고(message)" 했는지 식별하기 위한 규격입니다.
    """
    # [설명] session_id: 사용자 구분용 ID (예: "user_123", "guest_abc")
    # - 이 ID를 키(Key)로 사용하여 서버 메모리에서 대화 기록을 찾습니다.
    session_id: str 
    
    # [설명] message: 사용자의 실제 발화 내용
    # - default="": 만약 메시지 없이 ID만 보내도 에러가 나지 않도록 기본값을 빈 문자열로 둡니다.
    #   (예: 단순히 세션이 살아있는지 확인할 때 유용)
    message: str = ""


# ----------------------------------------------------
# [API 0] 통합 마스터 봇 (Master Bot)
# ----------------------------------------------------
@app.post("/master_bot")
async def endpoint_master_bot(req: ChatRequest):
    """
    [통합 창구] 반장 로봇(Master)에게 질문하기
    
    [역할]
    - 사용자가 "이거 누구한테 물어봐야 하지?" 고민할 필요가 없습니다.
    - 일단 이 창구로 질문을 던지면, Master AI가 내용을 보고 가장 적합한 로봇을 호출해줍니다.

    [🗺️ 전체 시스템 조감도 (The Big Picture)]
    이 함수 하나가 전체 시스템을 어떻게 끌어다 쓰는지 보여주는 지도입니다.

    Start (사용자 요청)
       ↓
    [Zone 5] Pydantic 검사 (`ChatRequest`)
       ↓ "통과! req 객체 생성"
    [Zone 5] API 엔드포인트 (`endpoint_master_bot`) 진입
       ↓ "반장님, 저기요(call)"
    [Zone 3] 패턴 0 (`master_router`) 호출 
       ↓ "내부적으로 AI(`gpt-4o`)에게 물어봄"
       ↓ "결과: pattern1으로 가라!"
    [Zone 3] 패턴 1 (`simple_graph`) 선택 및 실행 (`ainvoke`)
       ↓ "단순 답변 봇 가동"
       ↓ ([Zone 2] `ChatOpenAI` 모델 사용)
    [Zone 5] 결과 수신 및 JSON 포장
       ↓
    End (사용자 응답: `{"answer": "..."}`)

    -------------------------------------------------------
    [데이터 변신 과정 (Data Flow) 🕵️‍♂️]
       1. User JSON `{"message": "안녕"}`
    -> 2. Pydantic `req` 객체
    -> 3. Dict `{"question": "안녕"}` (Graph Input)
    -> 4. Dict `{"response": "안녕하세요"}` (Graph Output)
    -> 5. JSON `{"answer": "안녕하세요"}` (API Response)
    -------------------------------------------------------
    """

    # ==============================================================================
    # 🕵️‍♂️ [코드 해부학] 이 코드가 어떻게 도는지 1줄씩 뜯어보기 (Python Anatomy)
    # ==============================================================================
    
    # [STEP 1] 변수 출처 확인 & 라우팅
    # - req: 이 함수가 시작될 때 외부(사용자)에서 받아온 '입력 상자'입니다. (req: ChatRequest)
    # - req.message: 상자 안에서 'message'라는 내용물("파이썬이 뭐야?")을 꺼냅니다.
    # - master_router(...): 우리가 고용한 '안내원' 함수에게 이 내용물을 보여줍시다.
    # - await: 안내원이 "잠시만요, 생각 좀 할게요" 하니까 끝날 때까지 여기서 멈춰서 기다립니다.
    # - target_pattern =: 안내원이 "pattern1입니다"라고 뱉은 결과를 `target_pattern`이라는 스티커(변수)에 적어둡니다.
    target_pattern = await master_router(req.message) 
    print(f"👉 [Master] 결정된 담당자: {target_pattern}") # 로그 출력

    # [STEP 2] 분기 처리 (if-elif)
    # - 방금 적어둔 스티커(`target_pattern`)의 내용이 무엇인지 하나씩 확인합니다.
    
    # CASE 1: 안내원이 "pattern1"이라고 했다면?
    if target_pattern == "pattern1":
        # 1. {"question": req.message}: 그래프에게 줄 '작업지시서(Dict)'를 새로 만듭니다.
        #    -> 이유: pattern1 그래프는 "question"이라는 키워드를 원하니까 맞춰주는 겁니다.
        
        # [오해 금지 🚫] "저 밑에 있는 `endpoint_pattern1` 함수를 부르는 건가요?"
        # - 아닙니다! 그건 '다른 문'이고, 우리는 **'기계(simple_graph)'를 직접** 사용합니다.
        # - (매니저가 직접 커피머신 버튼을 누르는 거지, 다른 창구 직원한테 부탁하는 게 아닙니다.)
        
        # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
        try:
            result = await simple_graph.ainvoke({"question": req.message})
            return {
                "router_decision": "Pattern 1 (Simple)",
                "answer": result.get("response")
            }
        except Exception as e:
            # AI 호출 실패 시 사용자에게 친절한 에러 메시지 반환 (서버는 살아있음)
            return {"error": f"Pattern 1 실행 중 오류 발생: {str(e)}"}

    # CASE 2: 안내원이 "pattern2"라고 했다면?
    elif target_pattern == "pattern2":
        # [uc548전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
        try:
            result = await advanced_graph.ainvoke({"question": req.message})
            return {
                "router_decision": "Pattern 2 (Creative)",
                "answer": result.get("response")
            }
        except Exception as e:
            return {"error": f"Pattern 2 실행 중 오류 발생: {str(e)}"}

    # CASE 3: 안내원이 "pattern3"라고 했다면?
    elif target_pattern == "pattern3":
        # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
        try:
            result = await react_graph.ainvoke({
                "question": req.message,
                "iterations": 0
            })
            return {
                "router_decision": "Pattern 3 (ReAct)",
                "trace": result.get("thought"),
                "answer": result.get("final_answer")
            }
        except Exception as e:
            return {"error": f"Pattern 3 실행 중 오류 발생: {str(e)}"}

    # CASE 4: 안내원이 "pattern4"라고 했다면? (대화형)
    elif target_pattern == "pattern4":
        # [해부학] 여기는 '기억'이 필요한 곳이라 손님 명부(Session ID)를 확인합니다.
        
        # 1. sid 추출: 손님 가슴에 달린 이름표("user123")를 봅니다.
        sid = req.session_id 

        # 2. 명부 확인: "어? 처음 오신 분이네?"
        if sid not in chat_sessions:
            chat_sessions[sid] = [] # 빈 대화 노트(List)를 새로 만들어줍니다.
            
        # 3. 대화록 가져오기: 기존에 적어둔 대화 노트를 꺼냅니다.
        hist = chat_sessions[sid]
        
        # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
        try:
            result = await chat_graph.ainvoke({
                "session_id": sid,
                "messages": hist,
                "current_input": req.message
            })
            
            # 5. 후처리: AI 답변을 받아서 대화록에 기록합니다.
            response = result["response"]
            hist.append(f"User: {req.message}")
            hist.append(f"AI: {response}")
            
            # [Q&A] 로봇은 끝났는데(return), 왜 우리가 뒤처리를 하나요? 🧹
            if chat_check_end({"current_input": req.message}) == "end":
                del chat_sessions[sid]
                response += " (대화가 종료되었습니다. 메모리를 초기화합니다.)"

            return {
                "router_decision": "Pattern 4 (Chat)",
                "answer": response
            }
        except Exception as e:
            return {"error": f"Pattern 4 실행 중 오류 발생: {str(e)}"}

# ==============================================================================
# 🚧 [경계선] 여기서 `master_bot`의 역할은 끝났습니다! (THE END) 🚧
# ==============================================================================
# - 위에서 리턴한 값은 [API 1]로 내려가는 게 아니라, **바로 사용자에게 날아갑니다.**
# - 아래에 있는 `simple_routing` 함수는 그냥 **"또 다른 별개의 문"**일 뿐입니다.
# - (마치 옆집 사람이 내가 퇴근한다고 우리 집에 들어오지 않는 것과 같습니다.)
# ==============================================================================

# ----------------------------------------------------
# [API 1] Simple Routing 엔드포인트
# ----------------------------------------------------
# [Q&A] 위(`master_bot`)에도 있는데, 여기도 왜 또 리턴(return)이 있나요?
# - **"문이 다르면 나가는 곳(Exit)도 따로 있어야죠!"** 🚪
# - `master_bot`은 "정문"이고, 여기는 테스트용 "1번 창구 뒷문"입니다.
# - 1번 체크포인트: 정문으로 들어오든 뒷문으로 들어오든, **"나가는 건(Return) 사용자에게로"** 입니다.
# 
# [Q&A] 마스터 봇이 다 해주는데 엔드포인트 얘들은 왜 있나요? (존재 이유 🤷)
# - **"수동 모드(Manual Mode)"**가 필요하기 때문입니다!
# 1. **테스트용**: 마스터 봇(안내원)이 실수할 수도 있으니, 기계만 따로 점검할 때 씁니다.
# 2. **강제 실행**: "난 AI 판단 필요 없고, 무조건 1번 기계 쓸 거야!"라고 할 때 사용합니다.
#
# [Tip Q&A] 그럼 저~ 위에 있는 로봇(노드)들이 하는 리턴은요?
# - 그거랑 이거랑은 완전히 다른 **"내부 바통 터치"**입니다.
# 1. **노드 리턴**: "김 대리, 내 일 다 했어. 다음 받으세요!" (팀원끼리 주고받는 서류)
# 2. **API 리턴**: "손님, 여기 주문하신 결과입니다!" (최종 고객에게 주는 상품)
# 
# [Q&A] `@app.post("/simple_routing")` 저 문자열은 어디서 가져온 건가요?
# - **어디서 가져온 게 아니라, 방금 우리가 '작명(Naming)'한 겁니다!** 🏷️
# - "이 문패를 뭐라고 달까요?" -> "음, `simple_routing`이라고 합시다."
# - 그래서 주소가 `http://서버주소/simple_routing`이 되는 겁니다.
# - (만약 `("/pizza")`라고 적으면 주소가 `/pizza`가 됩니다.)
@app.post("/simple_routing")
async def endpoint_pattern1(req: QueryRequest):
    """
    [API 1] 패턴 1 실행 (Simple Routing)
    
    [핵심 개념] async/await의 마법
    - AI 모델 호출은 커피 주문 후 기다리는 것(I/O Bound)과 같습니다. (약 1~3초)
    - 동기(Sync)라면: 점원이 커피 나올 때까지 계산대 앞에 서서 아무것도 안 합니다. (서버 멈춤)
    - 비동기(Async)라면: 점원이 진동벨을 주고 다음 손님을 받습니다. (서버 원활)
    """
    # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
    # - try: 아래 코드를 "시도"하고, 만약 에러가 나면 except로 점프합니다.
    try:
        #  비동기 그래프 실행
        # - await: simple_graph.ainvoke()가 끝날 때까지 여기서 멈춰서 기다립니다.
        # - simple_graph: 맨 위에서 lifespan에서 만들어둔 패턴 1 그래프 객체입니다.
        # - .ainvoke(): 그래프를 비동기로 실행하는 메서드입니다. (a = async의 줄임말)
        # - {"question": req.question}: 딕셔너리를 생성합니다. "question"이라는 키에 req.question 값을 넣습니다.
        # - req.question: 사용자가 보낸 요청(req) 안에 들어있는 question 필드 값입니다.
        # - result =: ainvoke()가 반환하는 딕셔너리(결과물 상자)를 result라는 변수에 할당합니다.
        result = await simple_graph.ainvoke({"question": req.question}) # 딕셔너리를 생성하여 ainvoke()에 전달하고 반환값을 result 변수에 할당합니다
        
        #  결과 포장 후 반환
        # - return: 이 함수를 호출한 사람(FastAPI)에게 값을 돌려주고 함수를 종료합니다.
        # - { ... }: 새로운 딕셔너리를 생성합니다. 이것이 사용자가 받을 JSON 응답입니다.
        # - "pattern": 응답에 패턴 이름을 문자열로 포함합니다. (디버깅/확인용)
        # - result.get("classification"): result 딕셔너리에서 "classification" 키의 값을 꺼냅니다.
        #   만약 키가 없으면 None을 반환합니다. (KeyError를 방지하는 안전한 방법)
        # - result.get("response"): result 딕셔너리에서 "response" 키의 값을 꺼냅니다.
        return { # return문으로 딕셔너리를 반환합니다
            "pattern": "1. Simple Routing", # 패턴 이름 (문자열)
            "category": result.get("classification"), # result 딕셔너리에서 "classification" 키의 값을 가져옵니다
            "answer": result.get("response") # result 딕셔너리에서 "response" 키의 값을 가져옵니다
        } # 이 딕셔너리를 FastAPI가 자동으로 JSON으로 변환하여 사용자에게 전달합니다
    except Exception as e:
        # [에러 처리] 위의 try 블록에서 에러가 발생하면 여기로 옵니다.
        # - Exception as e: 발생한 에러 객체를 e라는 변수에 담습니다.
        # - str(e): 에러 객체를 문자열로 변환합니다. (예: "API key not found")
        # - f"...{str(e)}": f-string으로 동적 문자열을 생성합니다.
        return {"error": f"Pattern 1 실행 중 오류 발생: {str(e)}"} # 에러 메시지를 담은 딕셔너리를 반환합니다

# ----------------------------------------------------
# [API 2] 패턴 2 실행 (Advanced Routing)
# ----------------------------------------------------
@app.post("/advanced_routing") # POST /advanced_routing 엔드포인트 등록
async def endpoint_pattern2(req: QueryRequest):
    """
    [API 2] 패턴 2 실행 (Advanced 3-Way)
    - 기술(Tech), 일상(Casual), 창작(Creative) 3가지 중 하나로 분기합니다.
    """
    # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
    # - try: 아래 코드를 "시도"하고, 만약 에러가 나면 except로 점프합니다.
    try:
        #  비동기 그래프 실행
        # - await: advanced_graph.ainvoke()가 끝날 때까지 여기서 멈춰서 기다립니다.
        # - advanced_graph: lifespan에서 만들어둔 패턴 2 그래프 객체입니다.
        # - .ainvoke(): 그래프를 비동기로 실행하는 메서드입니다.
        # - {"question": req.question}: 딕셔너리를 생성합니다. "question"이라는 키에 req.question 값을 넣습니다.
        # - result =: ainvoke()가 반환하는 딕셔너리를 result라는 변수에 할당합니다.
        result = await advanced_graph.ainvoke({"question": req.question}) # 딕셔너리를 생성하여 그래프에 전달하고 반환값을 result 변수에 할당합니다
        
        #  결과 포장 후 반환
        # - return: 이 함수를 호출한 사람(FastAPI)에게 값을 돌려주고 함수를 종료합니다.
        # - { ... }: 새로운 딕셔너리를 생성합니다. 이것이 사용자가 받을 JSON 응답입니다.
        # - result.get("classification"): result 딕셔너리에서 "classification" 키의 값을 꺼냅니다.
        # - result.get("response"): result 딕셔너리에서 "response" 키의 값을 꺼냅니다.
        return { # return문으로 딕셔너리를 반환합니다
            "pattern": "2. Advanced Routing", # 패턴 이름 (문자열)
            "category": result.get("classification"), # result 딕셔너리에서 "classification" 키의 값을 가져옵니다
            "answer": result.get("response") # result 딕셔너리에서 "response" 키의 값을 가져옵니다
        } # 이 딕셔너리를 FastAPI가 자동으로 JSON으로 변환하여 사용자에게 전달합니다
    except Exception as e:
        # [에러 처리] 위의 try 블록에서 에러가 발생하면 여기로 옵니다.
        # - Exception as e: 발생한 에러 객체를 e라는 변수에 담습니다.
        # - str(e): 에러 객체를 문자열로 변환합니다.
        return {"error": f"Pattern 2 실행 중 오류 발생: {str(e)}"} # 에러 메시지를 담은 딕셔너리를 반환합니다

# ----------------------------------------------------
# [API 3] 패턴 3 실행 (ReAct Agent)
# ----------------------------------------------------
@app.post("/react_agent")
async def endpoint_pattern3(req: QueryRequest):
    """
    [API 3] 패턴 3 실행 (ReAct Agent)
    
    [주의사항] 루프 초기화
    - ReAct는 계속 생각(Think)하고 행동(Act)하는 루프 구조입니다.
    - 따라서 처음 시작할 때 "너 이제 막 시작했어(iterations=0)"라고 알려줘야 합니다.
    - 안 그러면 이전에 돌던 루프 횟수가 꼬일 수 있습니다.
    """
    # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
    # - try: 아래 코드를 "시도"하고, 만약 에러가 나면 except로 점프합니다.
    try:
        #  State 초기화 및 그래프 실행
        # - await: react_graph.ainvoke()가 끝날 때까지 여기서 멈춰서 기다립니다.
        # - react_graph: lifespan에서 만들어둔 패턴 3(ReAct) 그래프 객체입니다.
        # - .ainvoke(): 그래프를 비동기로 실행하는 메서드입니다.
        # - {"question": ..., "iterations": 0}: 다중 키를 가진 딕셔너리를 생성합니다.
        # - "iterations": 0: ReAct 루프 카운터를 0으로 초기화합니다. (5본 넘으면 강제 종료)
        # - result =: ainvoke()가 반환하는 딕셔너리를 result라는 변수에 할당합니다.
        result = await react_graph.ainvoke({ # await로 비동기 실행하고 반환값을 result에 할당합니다
            "question": req.question,  # 사용자 질문 (문자열)
            "iterations": 0  # 초기 반복 횟수는 0 (정수)
        }) # 딕셔너리를 생성하여 ainvoke()에 전달합니다
        
        #  결과 포장 후 반환
        # - return: 이 함수를 호출한 사람(FastAPI)에게 값을 돌려주고 함수를 종료합니다.
        # - result.get('iterations'): result 딕셔너리에서 루프 횟수를 꺼냅니다.
        # - result.get("final_answer", "실패"): final_answer를 꺼내되, 없으면 "실패"를 기본값으로 쓰ub2c8다.
        return { # return문으로 딕셔너리를 반환합니다
            "pattern": "3. ReAct Agent", # 패턴 이름 (문자열)
            "trace": f"총 {result.get('iterations')}회 사고 과정", # f-string으로 반복 횟수를 포함한 문자열을 생성합니다
            "final_answer": result.get("final_answer", "실패") # get() 메서드로 값을 가져오며 없으면 "실패"를 기본값으로 사용합니다
        } # 이 딕셔너리를 FastAPI가 자동으로 JSON으로 변환하여 사용자에게 전달합니다
    except Exception as e:
        # [에러 처리] 위의 try 블록에서 에러가 발생하면 여기로 옵니다.
        # - Exception as e: 발생한 에러 객체를 e라는 변수에 담습니다.
        # - str(e): 에러 객체를 문자열로 변환합니다.
        return {"error": f"Pattern 3 실행 중 오류 발생: {str(e)}"} # 에러 메시지를 담은 딕셔너리를 반환합니다

# ----------------------------------------------------
# [API 4-1] 대화 시작 (Session Start)
# ----------------------------------------------------
@app.post("/chat_start")
async def endpoint_chat_start(req: ChatRequest):
    """
    [API 4-1] 대화 세션 시작 (Session Initialization)
    
    [역할]
    - 사용자 "A"가 왔다는 것을 서버에 등록합니다.
    - A를 위한 빈 공책(Empty List)을 하나 마련해 둡니다.
    """
    # [안전장치 🛡️] 중복 세션 체크: 이미 존재하는 세션을 덮어쓰지 않도록 경고합니다.
    # - req.session_id: 사용자가 보낸 요청(req) 안에 들어있는 session_id 필드 값입니다.
    # - in chat_sessions: chat_sessions 딕셔너리에 해당 키가 존재하는지 확인합니다.
    # - if True: 이미 존재하면 경고 메시지를 반환하고 함수를 종료합니다.
    if req.session_id in chat_sessions: # req.session_id가 chat_sessions의 키에 존재하는지 in 연산자로 확인합니다
        return {"warning": f"세션 {req.session_id}이 이미 존재합니다. 기존 대화가 유지됩니다."} # f-string으로 경고 메시지를 생성하고 딕셔너리를 반환합니다
    
    #  세션 초기화
    # - chat_sessions: 서버 메모리에 존재하는 전역 딕셔너리입니다. {session_id: [messages...]}
    # - chat_sessions[req.session_id]: 새로운 키를 추가하고 빈 리스트를 값으로 할당합니다.
    # - []: 빈 리스트를 생성합니다. 나중에 대화 기록이 여기에 쌓입니다.
    chat_sessions[req.session_id] = [] # 빈 리스트 []를 생성하여 chat_sessions 딕셔너리의 req.session_id 키에 할당합니다
    
    #  결과 반환
    # - return: 이 함수를 호출한 사람(FastAPI)에게 값을 돌려주고 함수를 종료합니다.
    # - f"세션 {req.session_id} 시작됨": f-string으로 세션 ID를 포함한 성공 메시지를 생성합니다.
    return {"message": f"세션 {req.session_id} 시작됨"} # f-string으로 문자열을 생성하고 딕셔너리를 반환합니다

# ----------------------------------------------------
# [API 4-2] 대화 계속 (Chat Continue)
# ----------------------------------------------------
@app.post("/chat_continue")
async def endpoint_chat_continue(req: ChatRequest):
    """
    [API 4-2] 대화 이어가기 (Session Continue)
    
    
    1. [조회] session_id로 이전 대화 기록(History)을 꺼내옵니다.
    2. [실행] History + 현재 질문을 합쳐서 AI에게 던져줍니다.
    3. [갱신] AI 답변이 오면, 질문과 답변을 History에 추가(Append)합니다.
    4. [반환] 최종 답변을 사용자에게 줍니다.
    """
    #  세션 ID 추출
    # - req.session_id를 sid 변수에 할당합니다 (간결함을 위한 변수명 단축)
    sid = req.session_id # req 객체의 session_id 속성을 가져와 sid 변수에 할당합니다
    
    #  세션 존재 여부 확인 (안전장치)
    # - 세션이 없으면 404 에러를 발생시킵니다
    if sid not in chat_sessions: # sid가 chat_sessions 딕셔너리의 키에 없는지 확인합니다
        # HTTPException을 발생시켜 사용자에게 에러를 전달합니다
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없음 (먼저 /chat_start 하세요)") # HTTPException 객체를 생성하여 raise문으로 예외를 발생시킵니다
        
    #  히스토리 조회
    # - chat_sessions 딕셔너리에서 해당 세션의 대화 기록을 가져옵니다
    hist = chat_sessions[sid] # chat_sessions[sid]를 호출하여 리스트를 가져와 hist 변수에 할당합니다
    
    # [안전장치 🛡️] try-except: AI 호출 실패 시 서버가 죽지 않도록 보호합니다.
    # - try: 아래 코드를 "시도"하고, 만약 에러가 나면 except로 점프합니다.
    try:
        #  그래프 실행 (세션 ID와 히스토리 전달)
        # - await: chat_graph.ainvoke()가 끝날 때까지 여기서 멈춰서 기다립니다.
        # - chat_graph: lifespan에서 만들어둔 패턴 4(대화형) 그래프 객체입니다.
        # - .ainvoke(): 그래프를 비동기로 실행하는 메서드입니다.
        # - {"session_id": sid, "messages": hist, "current_input": ...}: 다중 키를 가진 딕셔너리를 생성합니다.
        # - result =: ainvoke()가 반환하는 딕셔너리를 result라는 변수에 할당합니다.
        result = await chat_graph.ainvoke({ # await 키워드로 ainvoke()를 호출하고 반환값을 result에 할당합니다
            "session_id": sid,      # 세션 ID를 전달합니다 (문자열)
            "messages": hist,       # 대화 히스토리 리스트를 전달합니다 (리스트)
            "current_input": req.message  # 현재 사용자 입력을 전달합니다 (문자열)
        }) # 딕셔너리를 생성하여 ainvoke()에 전달하고 그래프를 실행합니다
        
        #  응답 추출
        # - result["response"]: result 딕셔너리의 "response" 키로 접근하여 값을 꺼냅니다.
        # - response =: 꺼낸 값을 response라는 변수에 할당합니다.
        response = result["response"] # result 딕셔너리의 "response" 키로 접근하여 값을 response 변수에 할당합니다
        
        #  히스토리 업데이트 (대화 내용을 저장소에 추가)
        # - hist.append(): 리스트 맨 뒤에 새 항목을 추가합니다.
        # - f"User: {req.message}": f-string으로 사용자 메시지를 포함한 문자열을 생성합니다.
        # - f"AI: {response}": f-string으로 AI 응답을 포함한 문자열을 생성합니다.
        hist.append(f"User: {req.message}")  # f-string으로 문자열을 생성하고 hist.append()를 호출하여 리스트에 추가합니다
        hist.append(f"AI: {response}")       # AI 응답을 포함한 문자열을 히스토리에 추가합니다
        
        #  종료 체크 (그래프 내부가 아니라 여기서 최종적으로 상태를 보고 판단)
        # [Q&A] 로봇은 끝났는데(return), 왜 우리가 뒤처리를 하나요? 🧹
        # - 로봇(Graph)은 그냥 대화만 할 뿐, 메모리(Session)를 지우는 권한은 없습니다.
        # - "종료합니다"라는 말이 나오면, 카운터 직원(여기)이 방을 비워줘야(del) 합니다. (=호텔 체크아웃)
        # - chat_check_end(): 사용자 입력에 종료 키워드가 있는지 확인하는 함수입니다.
        if chat_check_end({"current_input": req.message}) == "end": # 딕셔너리를 생성하여 함수에 전달하고 반환값이 "end"인지 비교합니다
            # [종료 처리] 세션 삭제
            # - del: 파이썬 내장 키워드로 객체를 삭제합니다.
            # - chat_sessions[sid]: 삭제할 항목을 케론값으로 지정합니다.
            del chat_sessions[sid]  # del 문으로 chat_sessions 딕셔너리에서 sid 키를 삭제합니다 (방 빼기 = 메모리 삭제)
            return {"answer": response, "status": "ended"} # 종료 상태를 포함한 딕셔너리를 반환합니다
            
        #  진행 중 상태 반환
        # - return: 이 함수를 호출한 사람(FastAPI)에게 값을 돌려주고 함수를 종료합니다.
        # - "status": "continue": 대화가 계속됨을 나타냅니다.
        return {"answer": response, "status": "continue"} # 계속 상태를 포함한 딕셔너리를 반환합니다
    except Exception as e:
        # [에러 처리] 위의 try 블록에서 에러가 발생하면 여기로 옵니다.
        # - Exception as e: 발생한 에러 객체를 e라는 변수에 담습니다.
        # - str(e): 에러 객체를 문자열로 변환합니다. (예: "API key not found")
        return {"error": f"Pattern 4 실행 중 오류 발생: {str(e)}"} # 에러 메시지를 담은 딕셔너리를 반환합니다


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

