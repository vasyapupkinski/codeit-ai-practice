"""
🎯 통합 실행 파일 (All Patterns in One Server)

이 파일은 모든 패턴을 조합한 [Step 6] 마지막 단계입니다.
원본 langgraph_comprehensive_practice.py와 동일한 기능을 제공합니다.

================================================================================
📚 [상세 가이드] 이 파일의 역할
================================================================================

[목적] 패턴 0~4를 **모두 한 서버에서** 실행합니다.

[pattern0_master_router.py와의 차이점]
- pattern0: 마스터 라우터 하나만 노출 (/ask)
- 이 파일: 모든 엔드포인트를 노출 (/master_bot, /pattern1, /pattern2, ...)

┌─────────────────────────────────────────────────────────────────────────────┐
│ [엔드포인트 목록]                                                            │
│                                                                             │
│ 📍 통합 창구                                                                │
│    POST /master_bot     : AI가 자동으로 패턴 선택                           │
│                                                                             │
│ 📍 개별 패턴 직접 호출 (테스트/디버깅용)                                    │
│    POST /pattern1       : 패턴 1 직접 호출                                  │
│    POST /pattern2       : 패턴 2 직접 호출                                  │
│    POST /pattern3       : 패턴 3 직접 호출                                  │
│    POST /chat_start     : 패턴 4 세션 시작                                  │
│    POST /chat_continue  : 패턴 4 대화 계속                                  │
└─────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **모듈 조합**
   - 분리된 파일들(pattern1.py, pattern2.py, ...)을 하나로 모읍니다.
   - 각 파일의 build_graph 함수를 가져와서 사용합니다.

2. **여러 엔드포인트 노출**
   - 마스터 라우터 + 개별 패턴 엔드포인트를 모두 제공합니다.
   - 디버깅 시 특정 패턴만 테스트하고 싶을 때 유용합니다.

3. **실무 배포 구조**
   - 실제 배포할 때는 이런 통합 파일을 사용합니다.
   - 모든 기능을 하나의 서버에서 제공합니다.

--------------------------------------------------------------------------------
[2] 🔧 실행 방법
--------------------------------------------------------------------------------
1. python main_all_patterns.py 실행
2. http://localhost:8000/docs 접속
3. 모든 엔드포인트를 사용할 수 있습니다!

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# shared.py에서 가져오기
from shared import model, ChatPromptTemplate, StrOutputParser

# 각 패턴의 그래프 빌더 가져오기
from pattern1_simple_routing import build_pattern1_graph
from pattern2_advanced_routing import build_pattern2_graph
from pattern3_react_agent import build_pattern3_graph
from pattern4_chat_memory import build_pattern4_graph, chat_check_end


# ===============================================
# 📍 [구역 2] 전역 변수
# ===============================================

# 4개의 그래프를 담을 빈 상자
pattern1_graph = None
pattern2_graph = None
pattern3_graph = None
pattern4_graph = None

# 대화 세션 저장소 (패턴 4용)
chat_sessions = {}


# ===============================================
# 📍 [구역 3] 마스터 라우터 함수
# ===============================================

async def master_router(question: str) -> str:
    """
    [반장 로봇] 질문을 분석하여 적합한 패턴을 선택합니다.
    """
    prompt = ChatPromptTemplate.from_template(
        """당신은 AI 시스템의 총괄 관리자입니다.
사용자의 질문을 분석하여 적합한 부서로 연결하세요.

[부서 목록]
- pattern2: 시, 소설, 창작 요청
- pattern3: 계산, 검색, 실시간 정보
- pattern4: "아까", "이전에" 등 맥락 필요한 대화
- pattern1: 위에 해당 안 되는 일반 질문 (기본값)

오직 부서 코드 하나만 출력하세요.

질문: {question}
담당 부서:"""
    )
    
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({"question": question})
    
    chosen = result.strip().lower()
    valid = ['pattern1', 'pattern2', 'pattern3', 'pattern4']
    
    return chosen if chosen in valid else 'pattern1'


# ===============================================
# 📍 [구역 4] FastAPI 앱 설정
# ===============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 그래프를 조립합니다."""
    global pattern1_graph, pattern2_graph, pattern3_graph, pattern4_graph
    
    print("🚀 [통합 서버] 시작 중...")
    
    pattern1_graph = build_pattern1_graph()
    print("   ✅ 패턴 1 준비 완료")
    
    pattern2_graph = build_pattern2_graph()
    print("   ✅ 패턴 2 준비 완료")
    
    pattern3_graph = build_pattern3_graph()
    print("   ✅ 패턴 3 준비 완료")
    
    pattern4_graph = build_pattern4_graph()
    print("   ✅ 패턴 4 준비 완료")
    
    print()
    print("✨ 모든 시스템 준비 완료!")
    print("📍 http://localhost:8000/docs")
    print()
    
    yield
    
    print("🛑 [통합 서버] 종료")


app = FastAPI(
    title="LangGraph 통합 서버",
    description="패턴 0~4를 모두 포함한 통합 서버입니다. 원본 langgraph_comprehensive_practice.py와 동일한 기능을 제공합니다.",
    lifespan=lifespan
)


# ===============================================
# 📍 [구역 5] 데이터 모델
# ===============================================

class QueryRequest(BaseModel):
    """패턴 1, 2, 3용 단순 질문 요청"""
    question: str


class ChatRequest(BaseModel):
    """패턴 4 및 마스터 봇용 대화 요청"""
    session_id: str = "default"
    message: str = ""


# ===============================================
# 📍 [구역 6] API 엔드포인트
# ===============================================

# ─────────────────────────────────────────────
# [통합 창구] /master_bot
# ─────────────────────────────────────────────
@app.post("/master_bot")
async def endpoint_master_bot(req: ChatRequest):
    """
    [통합 창구] AI가 자동으로 패턴 선택
    
    아무 질문이나 보내면 AI가 분석하여 적절한 패턴으로 연결합니다.
    """
    target = await master_router(req.message)
    print(f"👉 [Master] 라우팅 결정: {target}")
    
    try:
        if target == "pattern1":
            result = await pattern1_graph.ainvoke({"question": req.message})
            return {"router": "Pattern 1", "answer": result.get("response")}
        
        elif target == "pattern2":
            result = await pattern2_graph.ainvoke({"question": req.message})
            return {"router": "Pattern 2", "answer": result.get("response")}
        
        elif target == "pattern3":
            result = await pattern3_graph.ainvoke({"question": req.message, "iterations": 0})
            return {"router": "Pattern 3", "answer": result.get("final_answer")}
        
        elif target == "pattern4":
            sid = req.session_id
            if sid not in chat_sessions:
                chat_sessions[sid] = []
            
            result = await pattern4_graph.ainvoke({
                "session_id": sid,
                "messages": chat_sessions[sid],
                "current_input": req.message
            })
            
            response = result["response"]
            chat_sessions[sid].append(f"User: {req.message}")
            chat_sessions[sid].append(f"AI: {response}")
            
            return {"router": "Pattern 4", "answer": response}
    
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# [개별 패턴] /pattern1, /pattern2, /pattern3
# ─────────────────────────────────────────────
@app.post("/pattern1")
async def endpoint_pattern1(req: QueryRequest):
    """패턴 1 직접 호출 (기본 라우팅)"""
    result = await pattern1_graph.ainvoke({"question": req.question})
    return {
        "pattern": "1. Simple Routing",
        "category": result.get("classification"),
        "answer": result.get("response")
    }


@app.post("/pattern2")
async def endpoint_pattern2(req: QueryRequest):
    """패턴 2 직접 호출 (3-way 라우팅)"""
    result = await pattern2_graph.ainvoke({"question": req.question})
    return {
        "pattern": "2. Advanced Routing",
        "category": result.get("classification"),
        "answer": result.get("response")
    }


@app.post("/pattern3")
async def endpoint_pattern3(req: QueryRequest):
    """패턴 3 직접 호출 (ReAct 에이전트)"""
    result = await pattern3_graph.ainvoke({"question": req.question, "iterations": 0})
    return {
        "pattern": "3. ReAct Agent",
        "answer": result.get("final_answer")
    }


# ─────────────────────────────────────────────
# [패턴 4] /chat_start, /chat_continue
# ─────────────────────────────────────────────
@app.post("/chat_start")
async def endpoint_chat_start(req: ChatRequest):
    """패턴 4: 세션 시작"""
    if req.session_id in chat_sessions:
        return {"warning": f"세션 '{req.session_id}'이 이미 존재합니다."}
    
    chat_sessions[req.session_id] = []
    return {"message": f"세션 '{req.session_id}' 시작됨"}


@app.post("/chat_continue")
async def endpoint_chat_continue(req: ChatRequest):
    """패턴 4: 대화 계속"""
    sid = req.session_id
    
    if sid not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션 없음. /chat_start 먼저 하세요.")
    
    hist = chat_sessions[sid]
    
    result = await pattern4_graph.ainvoke({
        "session_id": sid,
        "messages": hist,
        "current_input": req.message
    })
    
    response = result["response"]
    hist.append(f"User: {req.message}")
    hist.append(f"AI: {response}")
    
    if chat_check_end({"current_input": req.message}) == "end":
        del chat_sessions[sid]
        return {"answer": response, "status": "ended"}
    
    return {"answer": response, "status": "continue"}


# ===============================================
# 📍 [구역 7] 서버 실행
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 LangGraph 통합 서버 (All Patterns)")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 사용 가능한 엔드포인트:")
    print("   - POST /master_bot     : AI가 자동으로 패턴 선택")
    print("   - POST /pattern1       : 패턴 1 직접 호출")
    print("   - POST /pattern2       : 패턴 2 직접 호출")
    print("   - POST /pattern3       : 패턴 3 직접 호출")
    print("   - POST /chat_start     : 패턴 4 세션 시작")
    print("   - POST /chat_continue  : 패턴 4 대화 계속")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
