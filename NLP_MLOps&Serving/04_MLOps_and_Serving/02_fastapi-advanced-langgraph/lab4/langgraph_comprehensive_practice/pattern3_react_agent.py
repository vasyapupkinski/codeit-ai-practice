"""
🎯 패턴 3: ReAct 에이전트 (Reasoning + Acting)

이 파일은 가장 고급 패턴인 [Step 3] 단계입니다.
AI가 스스로 "생각(Reasoning)"하고 "행동(Acting)"하며 문제를 해결합니다.

================================================================================
📚 [상세 가이드] ReAct 패턴이란?
================================================================================

[핵심 개념] AI가 단순히 답변만 하는 게 아니라:
1. **Think (생각)**: "이 문제를 풀려면 뭘 해야 하지?"
2. **Act (행동)**: "계산기를 써서 계산해보자" (도구 실행)
3. **Observe (관찰)**: "결과가 345네"
4. **반복**: "아직 답이 안 나왔으니 다시 생각해보자"
5. **Finish**: "이제 답을 알았으니 사용자에게 알려주자"

┌─────────────────────────────────────────────────────────────────────────────┐
│ [실행 흐름 다이어그램] - 루프 구조!                                          │
│                                                                             │
│   [시작] ──▶ [think 노드] ──▶ [act 노드] ──▶ [조건 확인]                    │
│                   ▲                               │                         │
│                   │                               ▼                         │
│                   │                    ┌────────────────────┐               │
│                   │                    │ FINISH? or 5회 초과?│               │
│                   │                    └────────────────────┘               │
│                   │                        │           │                    │
│                   │                       Yes         No                    │
│                   │                        │           │                    │
│                   │                        ▼           │                    │
│                   │                     [종료]         │                    │
│                   │                                    │                    │
│                   └────────────────────────────────────┘                    │
│                           (다시 think로!)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **순환 그래프(Cyclic Graph)**
   - 패턴 1, 2: 시작 → 끝 (일직선)
   - 패턴 3: 시작 → 반복 → 끝 (루프!)
   - act 노드가 끝난 후 조건에 따라 다시 think로 돌아갑니다.

2. **도구(Tools) 사용**
   - AI가 혼자서는 못하는 일(계산, 검색)을 외부 함수로 처리합니다.
   - shared.py의 calculator_tool, knowledge_search_tool을 사용합니다.

3. **무한 루프 방지**
   - AI가 끝없이 생각할 수 있으므로 iterations 카운터를 둡니다.
   - 5회 넘으면 강제 종료합니다.

4. **State가 복잡해짐**
   - thought, action, action_input, observation 등 중간 과정 데이터가 추가됩니다.

--------------------------------------------------------------------------------
[2] 🔧 실행 방법
--------------------------------------------------------------------------------
1. python pattern3_react_agent.py 실행
2. http://localhost:8000/docs 접속
3. /ask 엔드포인트에서 테스트:
   - 계산: "15 곱하기 23은 얼마야?"
   - 검색: "FastAPI가 뭐야?"
   - 복합: "파이썬이 뭔지 설명해줘"

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================
#
# [패턴 1, 2와의 차이점]
# - TOOLS를 shared.py에서 가져옵니다! (계산기, 검색 도구)
# ===============================================

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import TypedDict

# shared.py에서 가져오기 (TOOLS 추가!)
from shared import model, TOOLS, StateGraph, END, ChatPromptTemplate, StrOutputParser


# ===============================================
# 📍 [구역 2] State 정의 (ReAct 전용)
# ===============================================
#
# [핵심] ReAct 패턴은 State가 복잡합니다!
# - 중간 과정(생각, 행동, 관찰)을 모두 저장해야 합니다.
# - iterations로 루프 횟수를 추적합니다.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ [State 변화 예시] "15 * 23 계산해줘"                            │
# │                                                                 │
# │ 1회차 (think 노드 후):                                          │
# │   thought: "이건 계산이 필요하네"                               │
# │   action: "calculator"                                         │
# │   action_input: "15 * 23"                                       │
# │   iterations: 1                                                 │
# │                                                                 │
# │ 1회차 (act 노드 후):                                            │
# │   observation: "계산 결과: 345"                                 │
# │                                                                 │
# │ 2회차 (think 노드 후):                                          │
# │   thought: "계산 결과를 알았으니 답변하자"                      │
# │   action: "FINISH"                                              │
# │   action_input: "15 × 23 = 345입니다."                         │
# │   iterations: 2                                                 │
# │                                                                 │
# │ 2회차 (act 노드 후):                                            │
# │   final_answer: "15 × 23 = 345입니다."                         │
# └─────────────────────────────────────────────────────────────────┘
# ===============================================

class ReActState(TypedDict):
    """
    ReAct 패턴 전용 State
    
    [필드 설명]
    - question: 사용자의 최초 질문
    - thought: AI의 생각 과정 ("이건 계산이 필요하네")
    - action: 사용할 도구 이름 ("calculator" 또는 "FINISH")
    - action_input: 도구에 넣을 값 ("15 * 23") 또는 최종 답변
    - observation: 도구 실행 결과 ("345")
    - iterations: 반복 횟수 (무한 루프 방지용, 최대 5회)
    - final_answer: 최종 답변 (FINISH 시에만 채워짐)
    """
    question: str           # [입력] 사용자 질문
    thought: str            # [중간] AI의 생각
    action: str             # [중간] 선택한 도구 이름
    action_input: str       # [중간] 도구 입력값
    observation: str        # [중간] 도구 실행 결과
    iterations: int         # [제어] 반복 횟수
    final_answer: str       # [출력] 최종 답변


# ===============================================
# 📍 [구역 3] 노드 함수 정의
# ===============================================
#
# [ReAct의 2가지 노드]
# 1. react_think: "무엇을 할지" 생각하는 노드
# 2. react_act: 실제로 "도구를 실행"하는 노드
#
# [Q&A] 왜 2개로 나누나요?
# - 생각(Reasoning)과 행동(Acting)을 분리해야 디버깅이 쉽습니다.
# - 각 단계의 로그를 따로 볼 수 있습니다.
# - 나중에 한 단계만 교체하기도 쉽습니다.
# ===============================================

async def react_think(state: ReActState) -> dict:
    """
    [노드 1] Reasoning (생각하기) 단계
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할]                                                          │
    │ - 현재까지의 관찰 결과(Observation)를 보고                      │
    │ - 다음에 무슨 행동(Action)을 해야 할지 결정합니다.              │
    │                                                                 │
    │ [AI의 선택지]                                                   │
    │ - "calculator": 수학 계산이 필요할 때                           │
    │ - "knowledge_search": 지식 검색이 필요할 때                     │
    │ - "FINISH": 답변이 준비되었을 때                                │
    └─────────────────────────────────────────────────────────────────┘
    
    [무한 루프 방지]
    - iterations >= 5이면 강제로 FINISH합니다.
    - AI가 영원히 생각하는 것을 막습니다.
    """
    # [Step 1] 반복 횟수 확인 (무한 루프 방지)
    iter_count = state.get("iterations", 0) + 1
    
    if iter_count >= 5:
        print(f"[ReAct] ⚠️ 최대 반복 횟수(5) 초과! 강제 종료합니다.")
        return {
            "final_answer": "최대 반복 횟수를 초과했습니다. 답변을 생성할 수 없습니다.",
            "action": "FINISH",
            "iterations": iter_count
        }

    print(f"[ReAct] 🧠 생각 중... (반복 {iter_count}회)")
    
    # [Step 2] 이전 관찰 결과 가져오기
    # - 첫 번째 반복이면 observation이 비어있습니다.
    # - 두 번째 이후부터는 이전 도구 실행 결과가 들어있습니다.
    obs = state.get('observation', '')
    
    # [Step 3] AI에게 생각하도록 요청하는 프롬프트
    # - Thought, Action, Action Input 형식으로 답변하도록 지시합니다.
    # - 이 형식을 지켜야 파싱이 가능합니다.
    prompt = ChatPromptTemplate.from_template(
        "당신은 문제를 단계적으로 해결하는 AI 에이전트입니다.\n\n"
        "[사용 가능한 도구]\n"
        "- calculator: 수학 계산을 수행합니다. (예: '15 * 23')\n"
        "- knowledge_search: 지식을 검색합니다. (예: 'FastAPI')\n"
        "- FINISH: 최종 답변을 제출합니다.\n\n"
        "[현재 상황]\n"
        "질문: {question}\n"
        "이전 관찰 결과: {observation}\n\n"
        "[지시사항]\n"
        "아래 형식으로 정확히 답변하세요:\n\n"
        "Thought: (생각 내용)\n"
        "Action: (도구 이름: calculator, knowledge_search, 또는 FINISH)\n"
        "Action Input: (도구 입력값 또는 최종 답변)"
    )
    
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({
        "question": state["question"],
        "observation": obs if obs else "없음 (첫 번째 시도)"
    })
    
    # [Step 4] 결과 파싱 (줄 단위로 분해)
    # - AI의 응답에서 Thought, Action, Action Input을 추출합니다.
    #
    # [파싱 예시]
    # 입력: "Thought: 계산이 필요하네\nAction: calculator\nAction Input: 15 * 23"
    # 출력: thought="계산이 필요하네", action="calculator", action_input="15 * 23"
    lines = result.strip().split('\n')
    thought, action, action_input = "", "FINISH", "파싱 실패"
    
    for line in lines:
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()
    
    print(f"[ReAct] 💭 Thought: {thought}")
    print(f"[ReAct] 🎯 Action: {action}")
    print(f"[ReAct] 📥 Action Input: {action_input}")
    
    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "iterations": iter_count
    }


def react_act(state: ReActState) -> dict:
    """
    [노드 2] Acting (행동하기) 단계
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할]                                                          │
    │ - think 노드가 결정한 도구(Action)를 실제로 실행합니다.         │
    │ - 결과를 observation에 저장합니다.                              │
    │                                                                 │
    │ [분기 처리]                                                      │
    │ - action == "FINISH": 최종 답변을 final_answer에 저장           │
    │ - action in TOOLS: 도구 실행 후 결과를 observation에 저장       │
    │ - 그 외: 에러 메시지를 observation에 저장                       │
    └─────────────────────────────────────────────────────────────────┘
    
    [Q&A] 왜 여기는 async가 아니라 def인가요?
    - 현재 정의된 도구(calculator, knowledge_search)는 동기 함수입니다.
    - API 호출이나 DB 조회처럼 "기다릴 일"이 없습니다.
    - 나중에 외부 API를 호출하는 도구가 추가되면 async로 바꿔야 합니다.
    """
    action = state.get("action", "")
    action_input = state.get("action_input", "")
    
    print(f"[ReAct] 🔧 도구 실행: {action}('{action_input}')")
    
    # [분기 1] 종료 조건: AI가 FINISH를 선택함
    if action.upper() == "FINISH":
        print(f"[ReAct] ✅ 최종 답변 생성 완료!")
        return {"final_answer": action_input}
    
    # [분기 2] 도구 실행
    if action.lower() in TOOLS:
        # TOOLS 딕셔너리에서 함수를 가져와서 실행합니다.
        # TOOLS = {"calculator": calculator_tool, "knowledge_search": knowledge_search_tool}
        tool_func = TOOLS[action.lower()]
        obs = tool_func(action_input)
        print(f"[ReAct] 📋 관찰 결과: {obs}")
        return {"observation": obs}
    
    # [분기 3] 알 수 없는 도구
    error_msg = f"오류: '{action}'은(는) 알 수 없는 도구입니다. (사용 가능: calculator, knowledge_search)"
    print(f"[ReAct] ❌ {error_msg}")
    return {"observation": error_msg}


# ===============================================
# 📍 [구역 4] 조건부 엣지 함수 (루프 제어)
# ===============================================
#
# [핵심] 이 함수가 루프를 제어합니다!
# - "continue": think 노드로 다시 돌아감 (루프)
# - "end": 그래프 종료
# ===============================================

def react_condition(state: ReActState) -> str:
    """
    [조건부 엣지] 계속할지(Loop) 끝낼지(End) 결정
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [종료 조건]                                                      │
    │ 1. iterations >= 5: 무한 루프 방지                              │
    │ 2. action == "FINISH": AI가 답변 완료를 선언                    │
    │                                                                 │
    │ [계속 조건]                                                      │
    │ - 위 조건에 해당하지 않으면 think로 돌아가서 다시 생각           │
    └─────────────────────────────────────────────────────────────────┘
    
    [반환값]
    - "end": END로 이동 (그래프 종료)
    - "continue": think 노드로 이동 (루프)
    """
    # 조건 1: 최대 반복 횟수 초과
    if state.get("iterations", 0) >= 5:
        print("[ReAct] ⏹️ 조건: 최대 반복 초과 → 종료")
        return "end"
    
    # 조건 2: AI가 FINISH를 선택함
    if state.get("action", "").upper() == "FINISH":
        print("[ReAct] ⏹️ 조건: FINISH 선언 → 종료")
        return "end"
    
    # 그 외: 계속 진행
    print("[ReAct] 🔄 조건: 계속 진행 → think로 돌아감")
    return "continue"


# ===============================================
# 📍 [구역 5] 그래프 조립 함수
# ===============================================
#
# [핵심] 순환 그래프 (Cyclic Graph)
# - 패턴 1, 2: A → B → END (일직선)
# - 패턴 3: A → B → A → B → ... → END (루프)
# ===============================================

def build_pattern3_graph():
    """
    패턴 3 (ReAct) 그래프 조립
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [순환 그래프의 핵심]                                            │
    │                                                                 │
    │ 기존 (패턴 1, 2):                                               │
    │   classifier ──▶ expert ──▶ END                                │
    │                                                                 │
    │ 패턴 3:                                                         │
    │   think ──▶ act ──┬──▶ END (조건 충족 시)                       │
    │     ▲             │                                             │
    │     └─────────────┘ (조건 미충족 시: 다시 think로!)             │
    │                                                                 │
    │ [무한 루프 방지 필수!]                                          │
    │ - iterations 카운터 없이는 AI가 영원히 생각할 수 있습니다.       │
    └─────────────────────────────────────────────────────────────────┘
    """
    workflow = StateGraph(ReActState)
    
    # 노드 추가 (2개)
    workflow.add_node("think", react_think)
    workflow.add_node("act", react_act)
    
    # 시작점: 항상 생각(think)부터
    workflow.set_entry_point("think")
    
    # 일반 엣지: think → act (무조건 이동)
    # - 생각이 끝나면 반드시 행동으로 넘어갑니다.
    workflow.add_edge("think", "act")
    
    # 조건부 엣지: act → (think 또는 END)
    # - 여기가 루프의 핵심입니다!
    # - "continue" 반환 시: think로 다시 돌아감
    # - "end" 반환 시: END로 이동 (그래프 종료)
    workflow.add_conditional_edges(
        "act",              # 출발 노드
        react_condition,    # 조건 함수
        {
            "continue": "think",  # 루프! think로 다시
            "end": END            # 종료
        }
    )
    
    return workflow.compile()


# ===============================================
# 📍 [구역 6] FastAPI 앱 설정
# ===============================================

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    
    print("🚀 [Pattern 3] ReAct 에이전트 서버 시작 중...")
    graph = build_pattern3_graph()
    print("✅ [Pattern 3] 그래프 조립 완료!")
    print("📍 http://localhost:8000/docs 에서 테스트하세요")
    
    yield
    
    print("🛑 [Pattern 3] 서버 종료")


app = FastAPI(
    title="패턴 3: ReAct 에이전트",
    description="AI가 생각하고, 도구를 사용하고, 답변하는 에이전트입니다.",
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
    [엔드포인트] AI가 생각하고 도구를 사용하여 답변합니다.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [테스트 예시]                                                    │
    │                                                                 │
    │ 1. 계산: "15 * 23 계산해줘"                                     │
    │    → AI가 calculator 도구를 사용합니다.                         │
    │                                                                 │
    │ 2. 검색: "LangGraph가 뭐야?"                                    │
    │    → AI가 knowledge_search 도구를 사용합니다.                   │
    │                                                                 │
    │ 3. 복합: "파이썬이 뭔지 검색해서 설명해줘"                       │
    │    → 검색 후 결과를 정리해서 답변합니다.                        │
    └─────────────────────────────────────────────────────────────────┘
    
    [주의] iterations를 0으로 초기화해야 합니다!
    - 이전 요청의 카운터가 남아있으면 바로 종료될 수 있습니다.
    """
    # iterations: 0으로 초기화! (중요)
    result = await graph.ainvoke({
        "question": req.question,
        "iterations": 0
    })
    
    return {
        "pattern": "3. ReAct Agent",
        "trace": f"총 {result.get('iterations')}회 사고 과정",
        "thought": result.get("thought"),
        "final_answer": result.get("final_answer", "답변 생성 실패")
    }


# ===============================================
# 📍 [구역 8] 서버 실행
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 패턴 3: ReAct 에이전트 서버")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 테스트 예시:")
    print("   - 계산: '15 * 23 계산해줘'")
    print("   - 검색: 'LangGraph가 뭐야?'")
    print("   - 복합: '파이썬이 뭔지 검색해서 설명해줘'")
    print()
    print("⚠️ 주의: 도구 실행 결과가 터미널에 출력됩니다!")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
