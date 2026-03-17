"""
🎯 패턴 2: 3단계 라우팅 (Advanced Routing)

이 파일은 패턴 1을 확장한 [Step 2] 단계입니다.
2가지 분기(TECHNICAL/CASUAL)에서 **3가지 분기**로 늘려봅니다.

================================================================================
📚 [상세 가이드] 패턴 1과 무엇이 다른가요?
================================================================================

패턴 1:  질문 → [TECHNICAL] 또는 [CASUAL]  (2가지)
패턴 2:  질문 → [TECHNICAL] 또는 [CASUAL] 또는 [CREATIVE]  (3가지!)

┌─────────────────────────────────────────────────────────────────────────────┐
│ [실행 흐름 다이어그램]                                                        │
│                                                                             │
│   [시작] ──▶ [classifier 노드] ──▶ [분류 결과 확인]                         │
│                                        │                                    │
│            ┌───────────────────────────┼───────────────────────────┐        │
│            │                           │                           │        │
│       [TECHNICAL]                  [CASUAL]                  [CREATIVE]     │
│            │                           │                           │        │
│            ▼                           ▼                           ▼        │
│    [tech_expert]             [friendly_bot]             [creative_writer]   │
│            │                           │                           │        │
│            ▼                           ▼                           ▼        │
│         [종료]                      [종료]                      [종료]      │
└─────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **기존 코드 확장하기**
   - 패턴 1을 복사해서 CREATIVE 분기를 추가합니다.
   - 노드를 추가하고, 라우터를 수정하면 됩니다.

2. **3가지 이상의 분기 처리**
   - if-elif-else 대신 딕셔너리 .get()을 활용하면 깔끔합니다.
   - 예: {"TECHNICAL": "tech_expert", "CREATIVE": "creative_writer"}.get(분류)

3. **페르소나(Persona) 설계**
   - 각 노드에 다른 성격을 부여합니다.
   - 기술 전문가: 딱딱하고 정확하게
   - 친구 봇: 친근하게
   - 창의적 작가: 감성적이고 문학적으로

--------------------------------------------------------------------------------
[2] 🔧 실행 방법
--------------------------------------------------------------------------------
1. 터미널에서 이 폴더로 이동
2. python pattern2_advanced_routing.py 실행
3. http://localhost:8000/docs 에서 테스트

[테스트 예시]
- 기술: "FastAPI와 Flask의 차이점은?"
- 일상: "주말에 뭐 하면 좋을까?"
- 창작: "겨울 밤에 대한 짧은 시를 써줘" ← 이게 새로 추가된 것!

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================
#
# [패턴 1과 동일] shared.py에서 공용 부품을 가져옵니다.
# ===============================================

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import TypedDict

# shared.py에서 가져오기
from shared import model, StateGraph, END, ChatPromptTemplate, StrOutputParser


# ===============================================
# 📍 [구역 2] State 정의
# ===============================================
#
# [패턴 1과 동일] 구조는 같지만, classification의 가능한 값이 3개로 늘어납니다.
# ===============================================

class AdvancedRoutingState(TypedDict):
    """
    패턴 2 전용 State 정의
    
    [필드 설명]
    - question (str): 사용자 질문
    - classification (str): "TECHNICAL", "CASUAL", 또는 "CREATIVE" (3가지!)
    - response (str): 최종 AI 답변
    """
    question: str
    classification: str
    response: str


# ===============================================
# 📍 [구역 3] 노드 함수 정의
# ===============================================
#
# [패턴 1 대비 변경점]
# 1. classifier 노드: 3가지로 분류하도록 프롬프트 수정
# 2. creative_writer 노드: 신규 추가!
# ===============================================

async def advanced_classify(state: AdvancedRoutingState) -> dict:
    """
    [노드 1] 3단계 질문 분류기 (Advanced Classifier)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [패턴 1과의 차이점]                                              │
    │                                                                 │
    │ 패턴 1: TECHNICAL / CASUAL (2가지)                              │
    │ 패턴 2: TECHNICAL / CASUAL / CREATIVE (3가지!)                  │
    │                                                                 │
    │ [CREATIVE 판단 기준]                                            │
    │ - 시, 소설, 에세이 등 창작 요청                                  │
    │ - "~를 써줘", "~를 만들어줘" 같은 생성 요청                      │
    │ - 감성적이거나 예술적인 질문                                     │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(f"[Pattern 2] 🔍 3단계 분류 중: '{state['question']}'")
    
    # [프롬프트 수정] 3가지 분류 기준을 명시합니다.
    prompt = ChatPromptTemplate.from_template(
        "다음 질문을 'TECHNICAL', 'CASUAL', 'CREATIVE' 중 하나로 분류하세요.\n\n"
        "[분류 기준]\n"
        "- TECHNICAL: 프로그래밍, 기술, 과학, 수학 관련 질문\n"
        "- CASUAL: 일상, 감정, 인사, 취미, 가벼운 대화\n"
        "- CREATIVE: 시, 소설, 에세이, 노래 가사 등 창작 요청 ⭐ 새로 추가!\n\n"
        "질문: {question}\n\n"
        "결과 (단어 하나만 출력):"
    )
    
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({"question": state["question"]})
    
    # [결과 정제] 우선순위: TECHNICAL > CREATIVE > CASUAL
    # - 혹시 AI가 "CREATIVE한 기술 질문"처럼 모호하게 답변해도 대응합니다.
    res = result.upper().strip()
    
    if "TECHNICAL" in res:
        classification = "TECHNICAL"
    elif "CREATIVE" in res:
        classification = "CREATIVE"
    else:
        classification = "CASUAL"
    
    print(f"[Pattern 2] ✅ 분류 완료: {classification}")
    return {"classification": classification}


async def tech_expert(state: AdvancedRoutingState) -> dict:
    """
    [노드 2-A] 기술 전문가 (패턴 1과 동일)
    """
    print("[Pattern 2] 👨‍💻 기술 전문가 실행")
    
    prompt = ChatPromptTemplate.from_template(
        "당신은 10년 경력의 시니어 개발자입니다.\n"
        "전문적이고 정확하게 답변하세요.\n"
        "필요하면 예시 코드를 포함해도 좋습니다.\n\n"
        "질문: {question}\n\n"
        "답변:"
    )
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"question": state['question']})
    
    return {"response": response}


async def friendly_bot(state: AdvancedRoutingState) -> dict:
    """
    [노드 2-B] 친구 봇 (패턴 1과 동일)
    """
    print("[Pattern 2] 😊 친구 봇 실행")
    
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


async def creative_writer(state: AdvancedRoutingState) -> dict:
    """
    [노드 2-C] 창의적 작가 (⭐ 신규 추가!)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할] 시, 소설, 아이디어 같은 창의적 질문을 담당합니다.         │
    │                                                                 │
    │ [페르소나] 감성적인 시인/작가                                    │
    │ - 문학적이고 아름다운 표현을 사용합니다.                         │
    │ - 비유와 은유를 적절히 활용합니다.                               │
    │ - 감정을 섬세하게 표현합니다.                                    │
    └─────────────────────────────────────────────────────────────────┘
    
    [Q&A] 왜 creative_writer를 따로 만드나요?
    - 페르소나에 따라 AI의 답변 스타일이 완전히 달라집니다.
    - "시 써줘"라고 했을 때 기술 전문가가 답하면 딱딱해집니다.
    - 창의적 작가 페르소나를 주면 더 아름다운 결과가 나옵니다.
    """
    print("[Pattern 2] 🎨 창의적 작가 실행")
    
    prompt = ChatPromptTemplate.from_template(
        "당신은 감성적이고 창의적인 시인이자 작가입니다.\n\n"
        "[스타일 가이드]\n"
        "- 문학적이고 아름다운 표현을 사용하세요.\n"
        "- 비유와 은유를 적절히 활용하세요.\n"
        "- 감정을 섬세하게 표현하세요.\n"
        "- 형식보다는 감성을 우선시하세요.\n\n"
        "요청: {question}\n\n"
        "창작물:"
    )
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"question": state['question']})
    
    return {"response": response}


# ===============================================
# 📍 [구역 4] 라우터 함수 (3갈래 분기)
# ===============================================
#
# [패턴 1 대비 변경점]
# - 3가지 경로를 처리해야 합니다.
# - if-elif-else 대신 딕셔너리를 활용하면 더 깔끔합니다.
# ===============================================

def advanced_route(state: AdvancedRoutingState) -> str:
    """
    [라우터] 3가지 경로 결정
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [딕셔너리 활용 팁]                                               │
    │                                                                 │
    │ 나쁜 예 (if-elif가 늘어남):                                      │
    │   if classification == "TECHNICAL":                             │
    │       return "tech_expert"                                      │
    │   elif classification == "CREATIVE":                            │
    │       return "creative_writer"                                  │
    │   else:                                                         │
    │       return "friendly_bot"                                     │
    │                                                                 │
    │ 좋은 예 (딕셔너리로 한 줄):                                      │
    │   return {                                                      │
    │       "TECHNICAL": "tech_expert",                               │
    │       "CREATIVE": "creative_writer",                            │
    │       "CASUAL": "friendly_bot"                                  │
    │   }.get(classification, "friendly_bot")                         │
    │                                                                 │
    │ .get(키, 기본값): 키가 없으면 기본값을 반환합니다.               │
    └─────────────────────────────────────────────────────────────────┘
    """
    # 딕셔너리로 깔끔하게 분기
    # - 나중에 4번째, 5번째 분기가 추가되어도 한 줄만 추가하면 됩니다.
    return {
        "TECHNICAL": "tech_expert",
        "CREATIVE": "creative_writer",
        "CASUAL": "friendly_bot"
    }.get(state["classification"], "friendly_bot")  # 기본값: friendly_bot


# ===============================================
# 📍 [구역 5] 그래프 조립 함수
# ===============================================
#
# [패턴 1 대비 변경점]
# - creative_writer 노드 추가
# - add_conditional_edges에 creative_writer 경로 추가
# - add_edge에 creative_writer → END 추가
# ===============================================

def build_pattern2_graph():
    """
    패턴 2 (3-way 분기) 그래프 조립
    
    [패턴 1 대비 추가된 부분]
    1. add_node("creative_writer", creative_writer)
    2. 조건부 엣지에 "creative_writer" 경로 추가
    3. add_edge("creative_writer", END)
    """
    
    workflow = StateGraph(AdvancedRoutingState)
    
    # 노드 4개 추가 (패턴 1보다 1개 많음!)
    workflow.add_node("classifier", advanced_classify)
    workflow.add_node("tech_expert", tech_expert)
    workflow.add_node("friendly_bot", friendly_bot)
    workflow.add_node("creative_writer", creative_writer)  # ⭐ 신규!
    
    # 시작점
    workflow.set_entry_point("classifier")
    
    # 3갈래 조건부 분기
    workflow.add_conditional_edges(
        "classifier",
        advanced_route,
        {
            "tech_expert": "tech_expert",
            "friendly_bot": "friendly_bot",
            "creative_writer": "creative_writer"  # ⭐ 신규!
        }
    )
    
    # 종료 연결 (3개)
    workflow.add_edge("tech_expert", END)
    workflow.add_edge("friendly_bot", END)
    workflow.add_edge("creative_writer", END)  # ⭐ 신규!
    
    return workflow.compile()


# ===============================================
# 📍 [구역 6] FastAPI 앱 설정
# ===============================================

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    
    print("🚀 [Pattern 2] 서버 시작 중...")
    graph = build_pattern2_graph()
    print("✅ [Pattern 2] 그래프 조립 완료!")
    print("📍 http://localhost:8000/docs 에서 테스트하세요")
    
    yield
    
    print("🛑 [Pattern 2] 서버 종료")


app = FastAPI(
    title="패턴 2: 3단계 라우팅",
    description="질문을 TECHNICAL/CASUAL/CREATIVE 3가지로 분류하여 답변합니다.",
    lifespan=lifespan
)


# ===============================================
# 📍 [구역 7] API 엔드포인트
# ===============================================

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """
    [엔드포인트] 질문을 3가지로 분류 후 답변
    
    [테스트 예시]
    - 기술: "FastAPI와 Flask의 차이점은?"
    - 일상: "주말에 뭐 하면 좋을까?"
    - 창작: "겨울 밤에 대한 짧은 시를 써줘" ⭐
    """
    result = await graph.ainvoke({"question": req.question})
    
    return {
        "pattern": "2. Advanced Routing (3-Way)",
        "category": result.get("classification"),
        "answer": result.get("response")
    }


# ===============================================
# 📍 [구역 8] 서버 실행
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 패턴 2: 3단계 라우팅 서버")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 테스트 예시:")
    print("   - 기술: 'FastAPI와 Flask의 차이점은?'")
    print("   - 일상: '주말에 뭐 하면 좋을까?'")
    print("   - 창작: '겨울 밤에 대한 짧은 시를 써줘' ⭐")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
