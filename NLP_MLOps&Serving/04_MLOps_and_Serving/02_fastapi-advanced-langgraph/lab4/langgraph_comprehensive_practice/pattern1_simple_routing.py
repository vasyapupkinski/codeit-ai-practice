"""
🎯 패턴 1: 기본 조건부 라우팅 (Simple Routing)

이 파일은 LangGraph의 가장 기초적인 패턴을 배우는 [Step 1] 단계입니다.
shared.py의 재료를 가져와서 "첫 번째 로봇"을 만들어봅니다.

================================================================================
📚 [상세 가이드] 이 파일이 하는 일
================================================================================
사용자의 질문을 분석해서 **2가지 중 하나**로 분류합니다:
- "기술적인 질문이네?" → 기술 전문가(Tech Expert)가 답변
- "일상 질문이네?" → 친구 봇(Friendly Bot)이 답변

┌─────────────────────────────────────────────────────────────────────────────┐
│ [실행 흐름 다이어그램]                                                        │
│                                                                             │
│   [시작] ──▶ [classifier 노드] ──▶ [분류 결과 확인]                         │
│                                        │                                    │
│                        ┌───────────────┴───────────────┐                   │
│                        │                               │                    │
│                   [TECHNICAL]                     [CASUAL]                  │
│                        │                               │                    │
│                        ▼                               ▼                    │
│               [tech_expert 노드]             [friendly_bot 노드]           │
│                        │                               │                    │
│                        ▼                               ▼                    │
│                     [종료]                          [종료]                  │
└─────────────────────────────────────────────────────────────────────────────┘
│

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **State(상태)란 무엇인가?**
   - 노드끼리 주고받는 "택배 상자"입니다.
   - 처음엔 question만 들어있다가, classifier가 classification을 채우고,
     전문가가 response를 채웁니다.

2. **Node(노드)는 어떻게 작동하는가?**
   - State를 받아서 일을 하고, 결과를 State에 업데이트하는 "작업자"입니다.
   - 파이썬 함수로 정의합니다.

3. **Conditional Edge(조건부 분기)는 어떻게 연결하는가?**
   - "분류 결과가 TECHNICAL이면 tech_expert로, 아니면 friendly_bot으로"
   - add_conditional_edges()로 설정합니다.

4. **그래프를 조립(build)하고 컴파일(compile)하는 방법**
   - StateGraph()로 빈 설계도를 만들고
   - add_node(), add_edge()로 부품을 배치하고
   - compile()로 실행 가능한 기계로 변환합니다.

--------------------------------------------------------------------------------
[2] 🔧 실행 방법
--------------------------------------------------------------------------------
1. 터미널에서 이 폴더로 이동:
   cd "C:\\AI Projects\\codeit\\코드잇 AI\\실습\\fastapi-실습-2\\lab4\\langgraph_comprehensive_practice"

2. 이 파일 실행:
   python pattern1_simple_routing.py

3. 브라우저에서 테스트:
   http://localhost:8000/docs

4. 테스트 예시:
   - 기술 질문: "파이썬에서 리스트와 튜플의 차이점은?"
   - 일상 질문: "오늘 기분이 어때?"

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================
#
#  이 파일이 실행되면 가장 먼저 임포트가 처리됩니다.
#
# [Q&A] from shared import ...가 뭔가요?
# - shared.py 파일에서 필요한 부품을 가져옵니다.
# - shared.py를 먼저 봤다면, 여기서 뭘 가져오는지 알 수 있습니다.
# - model: AI 모델 (ChatOpenAI 객체)
# - StateGraph, END: LangGraph의 핵심 클래스
# - ChatPromptTemplate, StrOutputParser: 프롬프트 관련 도구
# ===============================================

from fastapi import FastAPI          # 웹 서버 프레임워크
from pydantic import BaseModel       # 요청 데이터 검증용
from contextlib import asynccontextmanager  # 서버 수명 주기 관리

# [핵심] shared.py에서 공용 부품 가져오기
# 이 한 줄로 AI 모델과 LangGraph 핵심 클래스를 바로 사용할 수 있습니다.
from shared import model, StateGraph, END, ChatPromptTemplate, StrOutputParser


# ===============================================
# 📍 [구역 2] State 정의 (택배 상자의 규격)
# ===============================================
#
# [핵심 개념] State는 노드끼리 주고받는 "정보 저장소"입니다.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ [비유] 택배 상자 📦                                              │
# │                                                                 │
# │ 처음 상태:                                                       │
# │   { "question": "파이썬이 뭐야?", "classification": "", "response": "" }
# │                                                                 │
# │ classifier 노드 통과 후:                                         │
# │   { "question": "파이썬이 뭐야?", "classification": "TECHNICAL", "response": "" }
# │                                                                 │
# │ tech_expert 노드 통과 후:                                        │
# │   { "question": "파이썬이 뭐야?", "classification": "TECHNICAL", "response": "파이썬은..." }
# └─────────────────────────────────────────────────────────────────┘
#
# [Q&A] TypedDict가 뭔가요?
# - 딕셔너리인데, "이 키에는 이 타입만 들어갈 수 있어!"라고 명시하는 도구입니다.
# - 코드 작성 시 자동완성이 되고, 오타를 잡아줍니다.
# - 런타임에 실제로 검사하지는 않습니다. (타입 힌트 용도)
#
# [Q&A] 왜 BaseModel이 아니라 TypedDict를 쓰나요?
# - Pydantic의 BaseModel은 "외부 입력"을 검증할 때 씁니다. (API 요청)
# - TypedDict는 "내부 로직"에서 데이터를 주고받을 때 씁니다. (가볍고 빠름)
# - LangGraph는 내부에서 State를 빠르게 전달해야 하므로 TypedDict를 씁니다.
# ===============================================

from typing import TypedDict

class SimpleRoutingState(TypedDict):
    """
    패턴 1 전용 State 정의
    
    [필드 설명]
    - question (str): 사용자가 보낸 질문 원본
    - classification (str): 분류 결과 ("TECHNICAL" 또는 "CASUAL")
    - response (str): 최종 AI 답변
    """
    question: str           # [입력] 사용자 질문 (처음에 채워짐)
    classification: str     # [중간] 분류 결과 (classifier 노드가 채움)
    response: str           # [출력] 최종 답변 (전문가 노드가 채움)


# ===============================================
# 📍 [구역 3] 노드 함수 정의 (각 작업자의 역할)
# ===============================================
#
# [핵심 개념] 노드 = State를 받아서 일을 하는 "로봇"
#
# [Q&A] 왜 async def인가요? 그냥 def로 해도 되지 않나요?
# - AI 모델을 호출하면 1~5초를 기다려야 합니다.
# - 동기(def)로 하면: 기다리는 동안 서버 전체가 멈춥니다.
# - 비동기(async def)로 하면: 기다리는 동안 다른 요청을 처리할 수 있습니다.
# - FastAPI는 비동기 기반이므로, 노드도 비동기로 만드는 게 좋습니다.
#
# [Q&A] 노드 함수의 리턴값은 뭔가요?
# - 딕셔너리를 리턴합니다.
# - 이 딕셔너리의 내용이 State에 **자동으로 업데이트**됩니다.
# - 예: return {"classification": "TECHNICAL"}
#   → State의 classification 필드가 "TECHNICAL"로 바뀝니다.
# ===============================================

async def simple_classify(state: SimpleRoutingState) -> dict:
    """
    [노드 1] 질문 분류기 (Classifier)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할] 사용자의 질문을 분석하여 카테고리를 정합니다.            │
    │                                                                 │
    │ [입력] State에서 question을 읽습니다.                           │
    │ [출력] classification 필드를 업데이트합니다.                    │
    │                                                                 │
    │ [분류 기준]                                                      │
    │ - TECHNICAL: 프로그래밍, 기술, 과학 관련 질문                   │
    │ - CASUAL: 일상, 감정, 취미 관련 질문                            │
    └─────────────────────────────────────────────────────────────────┘
    """
    # [Step 1] 로그 출력 (디버깅용)
    # - 어떤 질문이 들어왔는지 터미널에서 확인할 수 있습니다.
    print(f"[Pattern 1] 🔍 분류 중: '{state['question']}'")
    
    # [Step 2] AI에게 분류를 요청하는 프롬프트 작성
    # - ChatPromptTemplate.from_template(): 변수({question})가 들어갈 자리를 만듭니다.
    # - 이 템플릿에 실제 질문을 넣으면 완성된 프롬프트가 됩니다.
    #
    # [Q&A] 프롬프트를 왜 이렇게 쓰나요?
    # - AI에게 **명확한 지시**를 내리기 위해서입니다.
    # - "TECHNICAL 또는 CASUAL로 분류하세요"라고 딱 정해주면,
    #   AI가 엉뚱한 답변("기술적인 것 같아요~")을 하지 않습니다.
    prompt = ChatPromptTemplate.from_template(
        "다음 질문을 'TECHNICAL' 또는 'CASUAL' 중 하나로 분류하세요.\n"
        "TECHNICAL: 프로그래밍, 기술, 과학, 수학 관련 질문\n"
        "CASUAL: 일상, 감정, 인사, 취미 관련 질문\n\n"
        "질문: {question}\n\n"
        "결과 (단어 하나만 출력):"
    )
    
    # [Step 3] 체인 구성 (프롬프트 → AI 모델 → 텍스트 추출)
    #
    # [핵심 개념] 파이프(|) 연산자
    # - LangChain에서는 | 기호로 처리 단계를 연결합니다.
    # - prompt: 질문을 프롬프트 형식으로 변환
    # - model: AI에게 보내서 답변을 받음
    # - StrOutputParser(): 답변에서 텍스트만 추출
    #
    # [비유] 조립 라인
    # - 원자재(질문) → 1차 가공(프롬프트) → 2차 가공(AI) → 포장(텍스트 추출)
    chain = prompt | model | StrOutputParser()
    
    # [Step 4] 비동기로 AI 호출
    # - await: AI가 답변할 때까지 기다립니다. (1~3초)
    # - ainvoke: async invoke의 줄임말. 비동기 호출입니다.
    # - {"question": state["question"]}: 템플릿의 {question}에 실제 값을 넣습니다.
    result = await chain.ainvoke({"question": state["question"]})
    
    # [Step 5] 결과 정제
    # - AI가 "TECHNICAL입니다" 같이 답변할 수도 있으므로
    # - upper()로 대문자로 바꾸고, "TECHNICAL"이 포함되어 있는지 확인합니다.
    classification = "TECHNICAL" if "TECHNICAL" in result.upper() else "CASUAL"
    
    # [Step 6] 로그 출력
    print(f"[Pattern 1] ✅ 분류 완료: {classification}")
    
    # [Step 7] State 업데이트용 딕셔너리 반환
    # - 이 딕셔너리가 State의 classification 필드를 업데이트합니다.
    return {"classification": classification}


async def simple_tech_expert(state: SimpleRoutingState) -> dict:
    """
    [노드 2-A] 기술 전문가 페르소나 (Tech Expert)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할] 딱딱하고 전문적인 말투로 기술 질문에 답변합니다.         │
    │                                                                 │
    │ [페르소나] 시니어 개발자                                        │
    │ - 정확하고 구체적인 설명을 제공합니다.                          │
    │ - 예시 코드를 포함할 수 있습니다.                               │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("[Pattern 1] 👨‍💻 기술 전문가 실행")
    
    # [프롬프트 설계] 페르소나 부여
    # - "당신은 시니어 개발자입니다"라고 역할을 명확히 해줍니다.
    # - 이렇게 하면 AI가 그 역할에 맞는 말투와 내용으로 답변합니다.
    prompt = ChatPromptTemplate.from_template(
        "당신은 10년 경력의 시니어 개발자입니다.\n"
        "전문적이고 정확하게 답변하세요.\n"
        "필요하면 예시 코드를 포함해도 좋습니다.\n\n"
        "질문: {question}\n\n"
        "답변:"
    )
    
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"question": state['question']})
    
    # State의 response 필드 업데이트
    return {"response": response}


async def simple_friendly_bot(state: SimpleRoutingState) -> dict:
    """
    [노드 2-B] 친구 페르소나 (Friendly Bot)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할] 친근한 말투로 일상 질문에 답변합니다.                    │
    │                                                                 │
    │ [페르소나] 친절한 친구                                          │
    │ - 편안하고 유머러스하게 대화합니다.                             │
    │ - 이모지를 적절히 사용합니다.                                   │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("[Pattern 1] 😊 친구 봇 실행")
    
    prompt = ChatPromptTemplate.from_template(
        "당신은 친절하고 유쾌한 친구입니다.\n"
        "편하게 대화하듯이 답변하세요.\n"
        "이모지를 적절히 사용해도 좋습니다.\n\n"
        "질문: {question}\n\n"
        "답변:"
    )
    
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"question": state['question']})
    
    return {"response": response}


# ===============================================
# 📍 [구역 4] 라우터 함수 (신호등 역할)
# ===============================================
#
# [핵심 개념] 라우터(Router)란?
# - 노드가 끝난 후 "다음에 어디로 갈지" 결정하는 함수입니다.
# - State의 값을 보고 다음 노드 이름을 문자열로 반환합니다.
#
# [Q&A] 왜 여기는 async가 아니라 그냥 def인가요?
# - 라우터는 이미 채워진 데이터를 보고 "왼쪽? 오른쪽?" 결정만 합니다.
# - AI를 호출하는 무거운 작업이 없습니다.
# - 0.001초면 끝나는 간단한 로직이라 굳이 비동기로 만들 이유가 없습니다.
#
# [중요!] 동기/비동기는 "순서"를 결정하지 않습니다!
# - 그래프의 Edge(선)가 순서를 결정합니다.
# - classifier → tech_expert 순서는 Edge가 정해놓은 것입니다.
# - 동기든 비동기든 이 순서는 절대 바뀌지 않습니다.
# ===============================================

def simple_route(state: SimpleRoutingState) -> str:
    """
    [라우터] 조건부 엣지를 위한 분기 함수
    
    
    1. State의 classification 필드를 읽습니다.
    2. "TECHNICAL"이면 "tech_expert" 문자열을 반환합니다.
    3. 그 외에는 "friendly_bot" 문자열을 반환합니다.
    
    [반환값의 의미]
    - 반환된 문자열은 **다음에 실행할 노드의 이름**입니다.
    - add_conditional_edges()에서 이 문자열을 보고 실제 노드로 연결합니다.
    """
    if state["classification"] == "TECHNICAL":
        return "tech_expert"
    return "friendly_bot"


# ===============================================
# 📍 [구역 5] 그래프 조립 함수 (설계도 → 기계)
# ===============================================
#
# [핵심 개념] 그래프 조립 과정
#
# 1. StateGraph(State): 빈 설계도를 만듭니다.
# 2. add_node("이름", 함수): 부품(노드)을 설계도에 붙입니다.
# 3. set_entry_point("이름"): 시작점을 정합니다.
# 4. add_edge("A", "B"): A 다음에 B가 실행되도록 연결합니다.
# 5. add_conditional_edges(): 조건에 따라 분기되도록 연결합니다.
# 6. compile(): 설계도를 실행 가능한 기계로 변환합니다.
#
# [비유] 레고 조립
# - StateGraph = 레고 베이스판
# - add_node = 블록 올려놓기
# - add_edge = 블록끼리 연결하기
# - compile() = 완성! 이제 가지고 놀 수 있음
# ===============================================

def build_pattern1_graph():
    """
    패턴 1 그래프 조립 함수
    
    [반환값]
    - CompiledGraph: ainvoke()로 실행할 수 있는 그래프 객체
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [조립 과정 시각화]                                              │
    │                                                                 │
    │ Step 1: 빈 설계도 생성                                          │
    │         workflow = StateGraph(SimpleRoutingState)               │
    │                                                                 │
    │ Step 2: 노드 3개 추가                                           │
    │         [classifier] [tech_expert] [friendly_bot]               │
    │                                                                 │
    │ Step 3: 시작점 설정                                             │
    │         entry_point ──▶ [classifier]                           │
    │                                                                 │
    │ Step 4: 조건부 분기 연결                                        │
    │         [classifier] ──TECHNICAL──▶ [tech_expert]              │
    │                      ──CASUAL────▶ [friendly_bot]              │
    │                                                                 │
    │ Step 5: 종료 연결                                               │
    │         [tech_expert] ──▶ END                                  │
    │         [friendly_bot] ──▶ END                                 │
    │                                                                 │
    │ Step 6: 컴파일                                                  │
    │         workflow.compile() ──▶ 실행 가능한 그래프!              │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    # [Step 1] 빈 그래프 생성
    # - SimpleRoutingState: 이 그래프에서 사용할 State의 "규격"을 알려줍니다.
    workflow = StateGraph(SimpleRoutingState)
    
    # [Step 2] 노드 추가
    # - add_node("이름", 함수): 노드를 등록합니다.
    # - "이름"은 나중에 edge 연결할 때 사용합니다.
    # - 함수는 비동기(async def)여도 됩니다.
    workflow.add_node("classifier", simple_classify)      # 분류 노드
    workflow.add_node("tech_expert", simple_tech_expert)  # 기술 전문가 노드
    workflow.add_node("friendly_bot", simple_friendly_bot)  # 친구 봇 노드
    
    # [Step 3] 시작점 설정
    # - set_entry_point(): 그래프 실행 시 가장 먼저 실행할 노드를 지정합니다.
    # - ainvoke()를 호출하면 이 노드부터 시작합니다.
    workflow.set_entry_point("classifier")
    
    # [Step 4] 조건부 분기 연결
    # - add_conditional_edges(): 조건에 따라 다른 노드로 가도록 설정합니다.
    #
    # [매개변수 설명]
    # 1. "classifier": 출발 노드 (이 노드가 끝난 후 분기)
    # 2. simple_route: 조건 판단 함수 (State를 보고 문자열 반환)
    # 3. 매핑 딕셔너리: {반환값: 도착 노드}
    #    - simple_route()가 "tech_expert" 반환 → tech_expert 노드로
    #    - simple_route()가 "friendly_bot" 반환 → friendly_bot 노드로
    workflow.add_conditional_edges(
        "classifier",       # 출발 노드
        simple_route,       # 조건 함수
        {
            "tech_expert": "tech_expert",      # 라우터가 "tech_expert" 반환 시
            "friendly_bot": "friendly_bot"     # 라우터가 "friendly_bot" 반환 시
        }
    )
    
    # [Step 5] 종료 연결
    # - add_edge("노드", END): 이 노드가 끝나면 그래프 종료
    # - END는 LangGraph의 특수 상수입니다. (더 이상 실행할 노드 없음)
    workflow.add_edge("tech_expert", END)
    workflow.add_edge("friendly_bot", END)
    
    # [Step 6] 컴파일
    # - compile(): 설계도를 실행 가능한 기계로 변환합니다.
    # - 이 시점에서 노드 연결이 올바른지 검증됩니다.
    # - 반환된 객체로 ainvoke()를 호출할 수 있습니다.
    return workflow.compile()


# ===============================================
# 📍 [구역 6] FastAPI 앱 설정
# ===============================================
#
# [핵심 개념] Lifespan이란?
# - 서버가 시작될 때(Startup)와 종료될 때(Shutdown) 실행되는 함수입니다.
# - 그래프 조립, DB 연결, 모델 로딩 등 "한 번만 할 일"을 여기서 합니다.
#
# [Q&A] 왜 lifespan에서 그래프를 만드나요?
# - 그래프 조립은 시간이 걸리는 작업입니다.
# - 매 요청마다 새로 만들면 느려집니다.
# - 서버 시작 시 한 번만 만들어서 재사용합니다.
# ===============================================

# 전역 변수: 그래프를 담을 빈 상자
# - None으로 초기화해두고, lifespan에서 실제 객체를 넣습니다.
graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [서버 수명 주기 관리] Lifespan 함수
    
    [실행 순서]
    1. 서버 시작 → yield 이전 코드 실행 (그래프 조립)
    2. 서버 운영 → 사용자 요청 처리
    3. 서버 종료 → yield 이후 코드 실행 (정리 작업)
    
    [Q&A] global이 뭔가요?
    - 함수 안에서 전역 변수를 수정하려면 global 선언이 필요합니다.
    - global graph라고 쓰면 "이 함수에서 graph를 바꾸면 전역 변수에 반영해줘"란 뜻입니다.
    """
    global graph
    
    # ===== 서버 시작 시 실행 =====
    print("🚀 [Pattern 1] 서버 시작 중...")
    
    # 그래프 조립
    graph = build_pattern1_graph()
    
    print("✅ [Pattern 1] 그래프 조립 완료!")
    print("📍 http://localhost:8000/docs 에서 테스트하세요")
    
    # yield: 여기서 서버가 요청을 받기 시작합니다.
    # Ctrl+C로 종료 신호가 올 때까지 대기합니다.
    yield
    
    # ===== 서버 종료 시 실행 =====
    print("🛑 [Pattern 1] 서버 종료")


# [FastAPI 앱 생성]
# - title, description: Swagger UI에 표시되는 정보
# - lifespan: 서버 수명 주기 관리 함수
app = FastAPI(
    title="패턴 1: 기본 조건부 라우팅",
    description="질문을 TECHNICAL/CASUAL로 분류하여 전문가가 답변합니다.",
    lifespan=lifespan
)


# ===============================================
# 📍 [구역 7] API 엔드포인트
# ===============================================
#
# [핵심 개념] 엔드포인트란?
# - 사용자가 접속할 수 있는 URL 주소입니다.
# - @app.post("/ask"): POST 방식으로 /ask 주소에 접속하면 이 함수가 실행됩니다.
#
# [Q&A] Pydantic BaseModel이 뭔가요?
# - API 요청 데이터를 검증하는 도구입니다.
# - QuestionRequest에 question 필드가 없으면 자동으로 422 에러가 납니다.
# - 보안과 안정성을 위해 필수입니다.
# ===============================================

class QuestionRequest(BaseModel):
    """
    API 요청 데이터 모델
    
    [필드]
    - question (str): 사용자의 질문 (필수)
    """
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """
    [엔드포인트] 질문을 받아서 분류 후 답변합니다.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                     │
    │                                                                 │
    │ 1. 사용자가 POST /ask 로 요청 전송                              │
    │    Body: {"question": "파이썬이 뭐야?"}                         │
    │                                                                 │
    │ 2. Pydantic이 데이터 검증 (question 필수 확인)                  │
    │                                                                 │
    │ 3. graph.ainvoke()로 그래프 실행                                │
    │    - classifier 노드 → 분류 (TECHNICAL)                        │
    │    - tech_expert 노드 → 답변 생성                               │
    │                                                                 │
    │ 4. 결과를 JSON으로 반환                                         │
    │    {"pattern": "1", "category": "TECHNICAL", "answer": "..."}   │
    └─────────────────────────────────────────────────────────────────┘
    
    [테스트 예시]
    - 기술: "파이썬에서 리스트와 튜플의 차이점은?"
    - 일상: "오늘 기분이 어때?"
    """
    # [Step 1] 그래프 실행
    # - ainvoke(): 비동기로 그래프를 실행합니다.
    # - {"question": req.question}: 초기 State를 전달합니다.
    # - 결과: 완성된 State 딕셔너리가 반환됩니다.
    result = await graph.ainvoke({"question": req.question})
    
    # [Step 2] 결과 반환
    # - 딕셔너리를 반환하면 FastAPI가 자동으로 JSON으로 변환합니다.
    # - result.get("키"): 키가 없으면 None을 반환 (에러 방지)
    return {
        "pattern": "1. Simple Routing",
        "category": result.get("classification"),
        "answer": result.get("response")
    }


# ===============================================
# 📍 [구역 8] 서버 실행
# ===============================================
#
# [Q&A] if __name__ == "__main__"이 뭔가요?
# - "이 파일을 직접 실행했을 때만" 아래 코드를 실행합니다.
# - 다른 파일에서 import할 때는 실행되지 않습니다.
#
# [Q&A] uvicorn.run()은 뭔가요?
# - FastAPI 앱을 실행하는 ASGI 서버입니다.
# - host="0.0.0.0": 모든 IP에서 접속 가능
# - port=8000: 8000번 포트 사용
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 패턴 1: 기본 조건부 라우팅 서버")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 테스트 예시:")
    print("   - 기술 질문: '파이썬에서 리스트와 튜플의 차이점은?'")
    print("   - 일상 질문: '오늘 기분이 어때?'")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
