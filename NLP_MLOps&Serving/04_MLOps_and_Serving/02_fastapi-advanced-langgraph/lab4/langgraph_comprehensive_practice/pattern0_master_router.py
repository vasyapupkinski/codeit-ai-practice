"""
🎯 패턴 0: 마스터 라우터 (Master Router / Supervisor)

이 파일은 모든 패턴을 통합하는 [Step 5] 단계입니다.
패턴 1~4를 다 배운 후에 보시면 됩니다!

================================================================================
📚 [상세 가이드] 마스터 라우터란?
================================================================================

[핵심 개념] Supervisor 패턴
- 사용자가 "/pattern1", "/pattern2" 중에 어디로 가야 할지 고민할 필요가 없습니다.
- 하나의 "반장 AI"가 모든 요청을 먼저 받습니다.
- 질문 내용을 분석하여 가장 적합한 패턴(1~4)으로 자동 연결합니다.

┌─────────────────────────────────────────────────────────────────────────────┐
│ [실행 흐름 다이어그램]                                                        │
│                                                                             │
│   [사용자 질문: "파이썬이 뭐야?"]                                            │
│              │                                                              │
│              ▼                                                              │
│   ┌───────────────────────────────┐                                        │
│   │     Master Router AI          │                                        │
│   │   "이건 기술 질문이네.        │                                        │
│   │    pattern1으로 보내야겠다."  │                                        │
│   └───────────────┬───────────────┘                                        │
│                   │ (라우팅 결정)                                           │
│       ┌───────────┼───────────┬───────────┐                                │
│       │           │           │           │                                 │
│       ▼           ▼           ▼           ▼                                 │
│   [Pattern 1] [Pattern 2] [Pattern 3] [Pattern 4]                          │
│   기술 질문    창작 요청    도구 필요   대화 맥락                           │
│       │           │           │           │                                 │
│       └───────────┴───────────┴───────────┘                                │
│                       │                                                     │
│                       ▼                                                     │
│               [최종 응답 반환]                                              │
└─────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **Supervisor(감독자) 패턴**
   - 모든 요청을 먼저 받아서 분류하는 AI
   - LangGraph에서는 이런 구조를 "Hierarchical Multi-Agent"라고 부릅니다.

2. **Intent Classification (의도 분류)**
   - AI에게 "이 질문은 어느 부서로 가야 해?"라고 물어봅니다.
   - 프롬프트 엔지니어링이 핵심입니다.

3. **여러 그래프 조합**
   - 4개의 그래프를 한 서버에서 동시에 운영합니다.
   - 상황에 따라 다른 그래프를 호출합니다.

4. **확장성(Scalability)**
   - 새로운 패턴 5, 6이 추가되어도 마스터 라우터만 수정하면 됩니다.
   - 각 패턴은 독립적으로 개발/테스트할 수 있습니다.

--------------------------------------------------------------------------------
[2] 🔧 실행 방법
--------------------------------------------------------------------------------
1. python pattern0_master_router.py 실행
2. http://localhost:8000/docs 접속
3. /ask 엔드포인트에서 아무 질문이나 해보세요!
   - AI가 질문을 분석해서 적절한 패턴을 선택합니다.

[테스트 예시]
- "파이썬이 뭐야?" → Pattern 1 (기술 질문)
- "가을에 대한 시를 써줘" → Pattern 2 (창작)
- "15 * 23 계산해줘" → Pattern 3 (도구 필요)
- "아까 내가 뭐라 했지?" → Pattern 4 (대화 맥락)

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================
#
# [핵심 포인트]
# - 각 패턴 파일에서 그래프 빌더를 가져옵니다!
# - 이렇게 하면 각 패턴을 독립적으로 개발하고 여기서 조합할 수 있습니다.
# ===============================================

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# shared.py에서 가져오기
from shared import model, ChatPromptTemplate, StrOutputParser

# [핵심] 각 패턴의 그래프 빌더를 가져옵니다!
# - 이렇게 하면 각 파일에서 정의한 그래프를 여기서 사용할 수 있습니다.
# - 모듈화(Modularization)의 힘!
from pattern1_simple_routing import build_pattern1_graph
from pattern2_advanced_routing import build_pattern2_graph
from pattern3_react_agent import build_pattern3_graph
from pattern4_chat_memory import build_pattern4_graph, chat_sessions, chat_check_end


# ===============================================
# 📍 [구역 2] 마스터 라우터 함수 (반장 AI)
# ===============================================
#
# [핵심 개념] 이 함수가 "어디로 보낼지" 결정합니다.
# - AI를 사용하여 질문의 의도(Intent)를 파악합니다.
# - 프롬프트 엔지니어링이 매우 중요합니다!
#
# [Q&A] 왜 AI를 써서 분류하나요? 그냥 키워드 매칭하면 안 되나요?
# - 키워드 매칭: "계산"이라는 단어가 있으면 Pattern 3
#   → "계산적으로 생각하면..." 같은 문장도 Pattern 3으로 잘못 분류
# - AI 분류: 문맥을 이해해서 진짜 의도를 파악
#   → "계산적으로 생각하면..." → Pattern 1 (일반 질문으로 인식)
# ===============================================

async def master_router(question: str) -> str:
    """
    [반장 로봇] 질문 의도 분류기
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ [역할]                                                          │
    │ - 이 함수는 '답변'을 하는 게 아니라, '판단'만 합니다!           │
    │ - "누구에게 일을 시킬까?"에만 집중합니다 (단일 책임 원칙).      │
    │                                                                 │
    │ [입력] 사용자 질문 문자열                                       │
    │ [출력] 실행할 패턴 코드 ('pattern1' ~ 'pattern4')               │
    └─────────────────────────────────────────────────────────────────┘
    
    [프롬프트 설계 핵심]
    - 각 패턴의 역할을 명확히 설명합니다.
    - "오직 부서 코드 하나만 출력하세요"로 출력 형식을 제한합니다.
    - 이렇게 하면 AI가 "pattern1입니다~" 대신 "pattern1"만 출력합니다.
    """
    print(f"[Master] 🧠 질문 분석 중: '{question}'")
    
    # [프롬프트 설계]
    # - 각 패턴의 역할을 AI에게 알려줍니다.
    # - AI는 이 정보를 바탕으로 가장 적합한 패턴을 선택합니다.
    prompt = ChatPromptTemplate.from_template(
        """당신은 이 AI 시스템의 총괄 관리자(Supervisor)입니다.
사용자의 질문을 분석하여, 아래 4개 전문 부서 중 가장 적합한 곳으로 연결하세요.

[🚨 부서 업무 정의서]

1. [pattern2: 창의/감성 팀] (Creative)
   - 시, 소설, 에세이, 노래 가사 작성 등 '작문'이 필요한 경우
   - 위로가 필요하거나 감성적인 대화를 원하는 경우
   - 예: "가을 시 써줘", "슬픈 이야기 만들어줘"

2. [pattern3: 해결사 팀] (ReAct Agent)
   - "지금", "오늘" 같은 실시간 정보가 필요한 경우 (예: 날씨, 주가)
   - 복잡한 수학 계산이나 논리적 추론이 필요한 경우
   - 지식 검색이 필요한 경우
   - 예: "15 * 23 계산해줘", "FastAPI 검색해줘"

3. [pattern4: 기억 팀] (Context Chat)
   - "아까 내가 말한 거", "이전 질문에 이어서" 같은 문맥이 필요한 경우
   - 특별한 목적 없이 길게 이어지는 대화
   - 예: "내 이름 뭐라고 했지?", "아까 그거 다시 설명해줘"

4. [pattern1: 일반 팀] (Simple Routing) ← 기본값(Fallback)
   - 위 3가지에 해당하지 않는 일반적인 질문
   - 간단한 인사 ('안녕', '반가워')
   - 기술적인 질문이나 일상 대화
   - 명확히 분류하기 어려운 경우
   - 예: "파이썬이 뭐야?", "오늘 기분 어때?"

[⚠️ 출력 규칙]
- 절대로 다른 말을 덧붙이지 마세요.
- 오직 부서 코드 단어 하나만 출력하세요.
- 가능한 답변: pattern1, pattern2, pattern3, pattern4

질문: {question}

담당 부서 코드:"""
    )
    
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({"question": question})
    
    # [결과 정제]
    # - AI가 "pattern1입니다" 같이 답변해도 대응합니다.
    chosen_pattern = result.strip().lower()
    
    # [유효성 검사]
    # - 허용된 패턴이 아니면 안전하게 pattern1로 폴백합니다.
    valid_patterns = ['pattern1', 'pattern2', 'pattern3', 'pattern4']
    
    if chosen_pattern not in valid_patterns:
        print(f"[Master] ⚠️ AI가 알 수 없는 응답: '{chosen_pattern}' → pattern1로 폴백")
        return 'pattern1'
    
    print(f"[Master] ✅ 라우팅 완료: {chosen_pattern}")
    return chosen_pattern


# ===============================================
# 📍 [구역 3] 전역 그래프 변수
# ===============================================
#
# [핵심] 4개의 그래프를 모두 메모리에 올립니다.
# - 각 패턴의 그래프를 미리 만들어 둡니다.
# - 요청이 올 때마다 적절한 그래프를 선택해서 실행합니다.
# ===============================================

pattern1_graph = None  # 패턴 1: 기본 라우팅
pattern2_graph = None  # 패턴 2: 창작 라우팅
pattern3_graph = None  # 패턴 3: ReAct 에이전트
pattern4_graph = None  # 패턴 4: 대화 기억


# ===============================================
# 📍 [구역 4] FastAPI 앱 설정
# ===============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [서버 수명 주기] 4개의 그래프를 모두 조립합니다.
    
    [Q&A] 왜 여기서 4개를 다 만드나요?
    - 요청이 올 때마다 만들면 느립니다.
    - 서버 시작 시 한 번에 만들어 두면 요청 처리가 빨라집니다.
    
    [Q&A] 메모리를 많이 쓰지 않나요?
    - 그래프 자체는 크지 않습니다. (KB 수준)
    - 진짜 무거운 건 AI 모델인데, 이건 shared.py에서 한 번만 로드됩니다.
    """
    global pattern1_graph, pattern2_graph, pattern3_graph, pattern4_graph
    
    print("🚀 [Master Router] 통합 서버 시작 중...")
    print("   4개의 그래프를 조립합니다...")
    
    pattern1_graph = build_pattern1_graph()
    print("   ✅ 패턴 1 (Simple Routing) 준비 완료")
    
    pattern2_graph = build_pattern2_graph()
    print("   ✅ 패턴 2 (Creative Routing) 준비 완료")
    
    pattern3_graph = build_pattern3_graph()
    print("   ✅ 패턴 3 (ReAct Agent) 준비 완료")
    
    pattern4_graph = build_pattern4_graph()
    print("   ✅ 패턴 4 (Chat Memory) 준비 완료")
    
    print()
    print("✨ 모든 시스템 준비 완료!")
    print("📍 http://localhost:8000/docs 에서 테스트하세요")
    print()
    
    yield
    
    print("🛑 [Master Router] 서버 종료")


app = FastAPI(
    title="패턴 0: 마스터 라우터",
    description="AI가 자동으로 최적의 패턴을 선택하여 답변합니다. 아무 질문이나 /ask에 보내보세요!",
    lifespan=lifespan
)


# ===============================================
# 📍 [구역 5] API 엔드포인트
# ===============================================

class MasterRequest(BaseModel):
    """
    마스터 라우터 요청 모델
    
    [필드]
    - session_id: 패턴 4(대화 기억)에서 사용. 기본값 "default"
    - message: 사용자 질문
    """
    session_id: str = "default"  # 패턴 4용
    message: str


@app.post("/ask")
async def master_ask(req: MasterRequest):
    """
    [엔드포인트] 통합 창구 - 아무 질문이나 하세요!
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                     │
    │                                                                 │
    │ 1. Master Router가 질문을 분석합니다.                           │
    │ 2. 적합한 패턴(1~4)을 선택합니다.                               │
    │ 3. 해당 그래프를 실행합니다.                                    │
    │ 4. 결과를 반환합니다.                                           │
    │                                                                 │
    │ [응답에 포함되는 정보]                                          │
    │ - router_decision: 어떤 패턴이 선택되었는지                     │
    │ - answer: 최종 답변                                             │
    │ - category (선택): 분류 결과 (패턴 1, 2)                        │
    │ - trace (선택): 사고 과정 (패턴 3)                              │
    └─────────────────────────────────────────────────────────────────┘
    
    [테스트 예시]
    - "파이썬이 뭐야?" → Pattern 1
    - "가을 시 써줘" → Pattern 2
    - "15 * 23 계산해줘" → Pattern 3
    - "아까 내 이름 뭐라 했지?" → Pattern 4
    """
    # [Step 1] 마스터 라우터가 어디로 보낼지 결정
    target_pattern = await master_router(req.message)
    print(f"👉 [Master] 라우팅 결정: {target_pattern}")
    
    try:
        # [Step 2] 결정된 패턴으로 분기
        
        # === Pattern 1: 기본 라우팅 ===
        if target_pattern == "pattern1":
            result = await pattern1_graph.ainvoke({"question": req.message})
            return {
                "router_decision": "Pattern 1 (Simple Routing)",
                "category": result.get("classification"),
                "answer": result.get("response")
            }
        
        # === Pattern 2: 창작 라우팅 ===
        elif target_pattern == "pattern2":
            result = await pattern2_graph.ainvoke({"question": req.message})
            return {
                "router_decision": "Pattern 2 (Creative Routing)",
                "category": result.get("classification"),
                "answer": result.get("response")
            }
        
        # === Pattern 3: ReAct 에이전트 ===
        elif target_pattern == "pattern3":
            # iterations: 0으로 초기화 필수!
            result = await pattern3_graph.ainvoke({
                "question": req.message,
                "iterations": 0
            })
            return {
                "router_decision": "Pattern 3 (ReAct Agent)",
                "trace": result.get("thought"),
                "answer": result.get("final_answer")
            }
        
        # === Pattern 4: 대화 기억 ===
        elif target_pattern == "pattern4":
            sid = req.session_id
            
            # 세션 없으면 자동 생성
            if sid not in chat_sessions:
                chat_sessions[sid] = []
            
            hist = chat_sessions[sid]
            
            result = await pattern4_graph.ainvoke({
                "session_id": sid,
                "messages": hist,
                "current_input": req.message
            })
            
            response = result["response"]
            
            # 히스토리 업데이트
            hist.append(f"User: {req.message}")
            hist.append(f"AI: {response}")
            
            # 종료 키워드 확인
            if chat_check_end({"current_input": req.message}) == "end":
                del chat_sessions[sid]
                response += " (대화가 종료되었습니다.)"
            
            return {
                "router_decision": "Pattern 4 (Chat Memory)",
                "answer": response
            }
    
    except Exception as e:
        # [에러 처리] 어떤 패턴에서 에러가 나도 서버는 죽지 않습니다.
        print(f"[Master] ❌ 에러 발생: {str(e)}")
        return {"error": f"실행 중 오류 발생: {str(e)}"}


# ===============================================
# 📍 [구역 6] 서버 실행
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 패턴 0: 마스터 라우터 서버 (통합 창구)")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 테스트 예시 (아무 질문이나 /ask에 보내보세요!):")
    print("   - '파이썬이 뭐야?' → Pattern 1 (기술)")
    print("   - '가을 시 써줘' → Pattern 2 (창작)")
    print("   - '15 * 23 계산해줘' → Pattern 3 (도구)")
    print("   - '아까 내 이름 뭐라 했지?' → Pattern 4 (기억)")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
