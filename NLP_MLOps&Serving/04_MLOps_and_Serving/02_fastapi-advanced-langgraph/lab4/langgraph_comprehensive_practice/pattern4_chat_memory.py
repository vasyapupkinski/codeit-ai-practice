"""
🎯 패턴 4: 대화형 기억 에이전트 (Chat with Memory)

이 파일은 세션(Session)과 기억(Memory)을 다루는 [Step 4] 단계입니다.
"아까 내가 말한 거 기억해?" 가 가능한 멀티턴 대화를 구현합니다.

================================================================================
📚 [상세 가이드] 왜 기억이 필요한가요?
================================================================================

[문제 상황] 패턴 1, 2, 3은 "단발성 질문"만 처리할 수 있습니다.
```
사용자: 나는 김철수야.
AI: 안녕하세요, 김철수님!

사용자: 내 이름이 뭐였지?
AI: 죄송합니다, 아직 이름을 알려주시지 않으셨네요. ← 이전 대화를 모름!
```

[해결책] 대화 기록(History)을 저장하고, AI에게 같이 보여줍니다.
```
[서버가 저장한 기록]
- "사용자: 나는 김철수야."
- "AI: 안녕하세요, 김철수님!"

사용자: 내 이름이 뭐였지?
AI: (기록을 보고) 김철수님이시죠! ← 기억함!
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ [실행 흐름 다이어그램]                                                        │
│                                                                             │
│   [/start 호출] ──▶ 세션 생성 (빈 대화 노트 준비)                           │
│         │                                                                   │
│         ▼                                                                   │
│   [/chat 호출] ──▶ 대화 노트 + 현재 질문 ──▶ AI 응답                        │
│         │                        │                                          │
│         │                        ▼                                          │
│         │              노트에 대화 기록 추가                                │
│         │                        │                                          │
│         ▼                        ▼                                          │
│   [종료 키워드 감지?] ─Yes─▶ 세션 삭제 (대화 노트 삭제)                     │
│         │                                                                   │
│        No                                                                   │
│         │                                                                   │
│         ▼                                                                   │
│   (다음 /chat 호출 대기)                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[1] 🎓 이 파일에서 배우는 핵심 개념
--------------------------------------------------------------------------------
1. **Session(세션) 관리**
   - 사용자마다 고유한 ID(session_id)를 부여합니다.
   - 서버는 이 ID를 key로, 대화 기록을 value로 딕셔너리에 저장합니다.
   - 예: chat_sessions = {"user1": ["안녕", "반가워"], "user2": [...]}

2. **대화 히스토리(History) 관리**
   - 리스트에 대화를 순서대로 쌓습니다.
   - AI에게 질문할 때 히스토리를 같이 전달합니다.
   - AI는 이전 맥락을 보고 답변합니다.

3. **종료 키워드 감지**
   - "종료", "끝", "bye" 같은 단어가 나오면 세션을 삭제합니다.
   - 메모리 누수를 방지합니다.

4. **Stateful vs Stateless**
   - Stateless (패턴 1, 2, 3): 요청 간에 연관성 없음. 매번 새로 시작.
   - Stateful (패턴 4): 요청 간에 상태(대화 기록)가 유지됨.

--------------------------------------------------------------------------------
[2] ⚠️ 실무 주의사항 (워커 문제)
--------------------------------------------------------------------------------
이 파일은 **chat_sessions 딕셔너리**에 대화를 저장합니다.
하지만 실무에서 서버를 여러 개(워커 4개) 띄우면 문제가 생깁니다!

[워커 문제 시나리오]
1. 요청 1: /chat_start (워커 A가 받음) → 워커 A 메모리에 세션 생성
2. 요청 2: /chat (워커 B가 받음) → 워커 B에는 세션이 없음 → 404 에러!

[해결책]
- Redis, PostgreSQL 같은 **외부 저장소**에 세션을 저장합니다.
- 모든 워커가 같은 저장소를 바라보면 문제 해결!

```python
# 메모리 기반 (이 파일): 워커 1개에서만 작동
chat_sessions = {}

# Redis 기반 (실무): 여러 워커에서 작동
import redis
r = redis.Redis()
r.set(session_id, json.dumps(history))
```

--------------------------------------------------------------------------------
[3] 🔧 실행 방법
--------------------------------------------------------------------------------
1. python pattern4_chat_memory.py 실행
2. http://localhost:8000/docs 접속
3. 테스트 순서:
   a. POST /start (session_id: "user1") ← 세션 시작
   b. POST /chat (session_id: "user1", message: "안녕! 나는 철수야.")
   c. POST /chat (session_id: "user1", message: "내 이름이 뭐라고?")
   d. POST /chat (session_id: "user1", message: "종료") ← 세션 삭제

================================================================================
"""

# ===============================================
# 📍 [구역 1] 필수 임포트
# ===============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import TypedDict, List

# shared.py에서 가져오기
from shared import model, StateGraph, END, ChatPromptTemplate, StrOutputParser


# ===============================================
# 📍 [구역 2] 전역 세션 저장소
# ===============================================
#
# [핵심 개념] chat_sessions 딕셔너리
# - 이 딕셔너리가 사용자별 대화 기록을 저장합니다.
# - 구조: {session_id: [대화1, 대화2, ...]}
#
# [예시]
# chat_sessions = {
#     "user1": ["User: 안녕", "AI: 안녕하세요!", "User: 파이썬이 뭐야?", ...],
#     "user2": ["User: 시 써줘", "AI: 가을 바람이...", ...],
# }
#
# [Q&A] 왜 전역 변수인가요?
# - 여러 엔드포인트(start, chat)에서 같은 저장소를 봐야 하기 때문입니다.
# - /start에서 생성한 세션을 /chat에서 사용해야 합니다.
#
# [Q&A] 서버를 껐다 키면 어떻게 되나요?
# - 다 날아갑니다! (메모리에만 저장되므로)
# - 실무에서는 Redis나 DB에 저장해야 합니다.
# ===============================================

chat_sessions = {}


# ===============================================
# 📍 [구역 3] State 정의
# ===============================================
#
# [패턴 4 State의 특징]
# - session_id: 사용자 구분용 (이전 패턴에는 없었음!)
# - messages: 대화 히스토리 리스트 (이전 패턴에는 없었음!)
# - current_input: 방금 들어온 사용자 메시지
# ===============================================

class ChatLoopState(TypedDict):
    """
    대화형 기억 에이전트 State
    
    [필드 설명]
    - session_id: 사용자 구분용 ID (예: "user1", "guest_abc")
    - messages: 대화 히스토리 리스트 (예: ["User: 안녕", "AI: 반가워요"])
    - current_input: 방금 들어온 사용자 메시지
    - response: AI 응답
    """
    session_id: str         # [식별] 사용자 고유 ID
    messages: List[str]     # [히스토리] 대화 기록 리스트
    current_input: str      # [입력] 현재 사용자 메시지
    response: str           # [출력] AI 응답


# ===============================================
# 📍 [구역 4] 노드 함수 정의
# ===============================================
#
# [패턴 4의 노드는 1개]
# - chat_respond: 히스토리를 보고 답변을 생성합니다.
#
# [Q&A] 왜 노드가 하나뿐인가요?
# - 이 패턴은 "분류"나 "도구 사용"이 목적이 아닙니다.
# - 오직 "히스토리를 보고 답변하기"만 하면 됩니다.
# - 필요하면 classifier 노드를 앞에 추가할 수도 있습니다.
# ===============================================

async def chat_respond(state: ChatLoopState) -> dict:
    """
    [노드] 대화 히스토리를 보고 답변 생성
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                     │
    │                                                                 │
    │ 1. state["messages"]에서 이전 대화 기록을 가져옵니다.           │
    │ 2. 리스트를 하나의 문자열로 합칩니다 (줄바꿈으로 연결).         │
    │ 3. AI에게 "이전 대화 + 현재 질문"을 보여줍니다.                 │
    │ 4. AI가 맥락을 이해하고 답변합니다.                             │
    └─────────────────────────────────────────────────────────────────┘
    
    [프롬프트 구조]
    ```
    이전 대화:
    User: 나는 철수야.
    AI: 안녕하세요, 철수님!
    
    사용자: 내 이름이 뭐라고?
    
    당신은 친절한 AI 어시스턴트입니다. 이전 대화 맥락을 고려하여 자연스럽게 대화하세요.
    ```
    """
    print(f"[Chat] 💬 답변 생성 중... (세션: {state['session_id']})")
    
    # [Step 1] 히스토리를 하나의 문자열로 합침
    # - ["User: 안녕", "AI: 반가워"] → "User: 안녕\nAI: 반가워"
    history = "\n".join(state.get("messages", []))
    
    # [Step 2] 프롬프트 구성
    # - 히스토리가 비어있으면 "(첫 대화입니다)"로 표시
    prompt = ChatPromptTemplate.from_template(
        "[이전 대화 기록]\n{history}\n\n"
        "[현재 질문]\n사용자: {current_input}\n\n"
        "[지시사항]\n"
        "당신은 친절한 AI 어시스턴트 '코디'입니다.\n"
        "위 대화 기록을 참고하여 자연스럽게 대화해주세요.\n"
        "사용자가 알려준 정보(이름, 취미 등)를 기억하고 활용하세요.\n\n"
        "AI 응답:"
    )
    
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({
        "history": history if history else "(첫 대화입니다)",
        "current_input": state["current_input"]
    })
    
    print(f"[Chat] ✅ 응답 생성 완료!")
    return {"response": response}


# ===============================================
# 📍 [구역 5] 종료 키워드 감지 함수
# ===============================================
#
# [역할] 사용자가 "종료", "끝" 같은 단어를 쓰면
#       세션을 삭제해야 함을 알려줍니다.
#
# [Q&A] 이 함수도 노드인가요?
# - 아닙니다! 이건 그래프 안에서 쓰이지 않습니다.
# - API 엔드포인트에서 "대화 종료 시 세션 삭제"를 위해 씁니다.
# - 필요하면 그래프 내 조건부 엣지로도 사용할 수 있습니다.
# ===============================================

def chat_check_end(state: dict) -> str:
    """
    [유틸 함수] 종료 키워드 감지
    
    [반환값]
    - "end": 종료 키워드가 있음 (세션 삭제 필요)
    - "continue": 종료 키워드 없음 (계속 대화)
    
    [종료 키워드 목록]
    - 한글: 종료, 끝
    - 영어: bye, quit, exit
    """
    user_input = state.get("current_input", "").lower()
    end_keywords = ['종료', '끝', 'bye', 'quit', 'exit']
    
    for keyword in end_keywords:
        if keyword in user_input:
            print(f"[Chat] 🔚 종료 키워드 감지: '{keyword}'")
            return "end"
    
    return "continue"


# ===============================================
# 📍 [구역 6] 그래프 조립 함수
# ===============================================
#
# [패턴 4 그래프의 특징]
# - 노드가 1개뿐입니다 (respond)
# - 루프가 없습니다 (웹 API는 요청-응답 후 연결 끊김)
# - 세션 관리는 그래프 바깥(엔드포인트)에서 합니다.
#
# [Q&A] 왜 그래프가 이렇게 단순한가요?
# - 이 그래프는 "한 번의 대화 턴"만 처리합니다.
# - 세션 생성, 히스토리 저장, 세션 삭제는 API 엔드포인트에서 합니다.
# - 그래프는 오직 "답변 생성"에만 집중합니다.
# ===============================================

def build_pattern4_graph():
    """
    패턴 4 (대화 루프) 그래프 조립
    
    [구조]
    entry → respond → (조건 확인) → END
    
    [참고] 조건 확인(chat_check_end)은 그래프가 아닌 엔드포인트에서 실행됩니다.
    """
    
    workflow = StateGraph(ChatLoopState)
    
    # 노드 추가 (1개)
    workflow.add_node("respond", chat_respond)
    
    # 시작점
    workflow.set_entry_point("respond")
    
    # 종료 연결 (조건 없이 바로 종료)
    # - 웹 API는 한 번 응답 후 연결이 끊기므로 루프가 없습니다.
    # - 다음 요청이 오면 그때 다시 그래프가 실행됩니다.
    workflow.add_edge("respond", END)
    
    return workflow.compile()


# ===============================================
# 📍 [구역 7] FastAPI 앱 설정
# ===============================================

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    
    print("🚀 [Pattern 4] 대화형 기억 에이전트 서버 시작 중...")
    graph = build_pattern4_graph()
    print("✅ [Pattern 4] 그래프 조립 완료!")
    print("📍 http://localhost:8000/docs 에서 테스트하세요")
    print("⚠️ 주의: 먼저 /start로 세션을 시작해야 합니다!")
    
    yield
    
    print("🛑 [Pattern 4] 서버 종료")
    print(f"   활성 세션 {len(chat_sessions)}개가 삭제됩니다.")


app = FastAPI(
    title="패턴 4: 대화형 기억 에이전트",
    description="대화 기록을 기억하는 멀티턴 챗봇입니다. /start로 세션을 먼저 시작하세요.",
    lifespan=lifespan
)


# ===============================================
# 📍 [구역 8] API 엔드포인트
# ===============================================
#
# [엔드포인트 3개]
# 1. POST /start: 세션 시작 (빈 히스토리 생성)
# 2. POST /chat: 대화 계속 (히스토리에 추가)
# 3. GET /sessions: 현재 활성 세션 조회 (디버깅용)
# ===============================================

class StartRequest(BaseModel):
    """세션 시작 요청"""
    session_id: str


class ChatRequest(BaseModel):
    """대화 요청"""
    session_id: str
    message: str


@app.post("/start")
async def start_chat(req: StartRequest):
    """
    [엔드포인트 1] 새로운 대화 세션을 시작합니다.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                     │
    │                                                                 │
    │ 1. session_id가 이미 존재하는지 확인                            │
    │ 2. 존재하면: 경고 메시지 반환 (기존 세션 유지)                  │
    │ 3. 존재하지 않으면: 빈 리스트로 세션 생성                       │
    └─────────────────────────────────────────────────────────────────┘
    
    [요청 예시]
    {"session_id": "user1"}
    
    [응답 예시]
    {"message": "세션 user1 시작됨"}
    """
    # [중복 체크] 이미 존재하는 세션인지 확인
    if req.session_id in chat_sessions:
        return {"warning": f"세션 '{req.session_id}'이 이미 존재합니다. 기존 대화가 유지됩니다."}
    
    # [세션 생성] 빈 리스트로 초기화
    chat_sessions[req.session_id] = []
    
    print(f"[Chat] 🆕 세션 생성: '{req.session_id}'")
    return {"message": f"세션 '{req.session_id}' 시작됨. 이제 /chat으로 대화하세요!"}


@app.post("/chat")
async def continue_chat(req: ChatRequest):
    """
    [엔드포인트 2] 대화를 이어갑니다.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                     │
    │                                                                 │
    │ 1. session_id로 히스토리 조회 (없으면 404)                       │
    │ 2. 그래프 실행 (히스토리 + 현재 메시지 → AI 응답)               │
    │ 3. 히스토리에 현재 대화 추가 (User: ..., AI: ...)               │
    │ 4. 종료 키워드 확인 → 맞으면 세션 삭제                          │
    │ 5. 응답 반환                                                     │
    └─────────────────────────────────────────────────────────────────┘
    
    [요청 예시]
    {"session_id": "user1", "message": "안녕! 나는 철수야."}
    
    [응답 예시]
    {"answer": "안녕하세요, 철수님! 만나서 반가워요.", "status": "continue"}
    """
    sid = req.session_id
    
    # [Step 1] 세션 존재 여부 확인
    if sid not in chat_sessions:
        # HTTPException: FastAPI의 에러 응답 도구
        # status_code=404: "찾을 수 없음" (HTTP 표준)
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{sid}'을 찾을 수 없습니다. 먼저 /start로 세션을 시작하세요."
        )
    
    # [Step 2] 히스토리 가져오기
    hist = chat_sessions[sid]
    
    # [Step 3] 그래프 실행
    result = await graph.ainvoke({
        "session_id": sid,
        "messages": hist,
        "current_input": req.message
    })
    
    response = result["response"]
    
    # [Step 4] 히스토리 업데이트
    # - 사용자 메시지와 AI 응답을 모두 기록합니다.
    hist.append(f"User: {req.message}")
    hist.append(f"AI: {response}")
    
    # [Step 5] 종료 키워드 확인
    if chat_check_end({"current_input": req.message}) == "end":
        # 세션 삭제 (메모리 해제)
        del chat_sessions[sid]
        print(f"[Chat] 🔚 세션 종료: '{sid}'")
        return {
            "answer": response,
            "status": "ended",
            "note": "대화가 종료되었습니다. 메모리가 초기화되었습니다."
        }
    
    # [Step 6] 응답 반환 (계속 진행)
    return {"answer": response, "status": "continue"}


@app.get("/sessions")
async def list_sessions():
    """
    [엔드포인트 3] 현재 활성화된 세션 목록을 조회합니다.
    
    [용도]
    - 디버깅: 어떤 세션이 살아있는지 확인
    - 모니터링: 동시 접속자 수 파악
    
    [응답 예시]
    {"active_sessions": ["user1", "user2"], "count": 2}
    """
    return {
        "active_sessions": list(chat_sessions.keys()),
        "count": len(chat_sessions)
    }


# ===============================================
# 📍 [구역 9] 서버 실행
# ===============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎯 패턴 4: 대화형 기억 에이전트 서버")
    print("=" * 60)
    print()
    print("📍 실행 후 아래 주소에서 테스트하세요:")
    print("   http://localhost:8000/docs")
    print()
    print("💡 테스트 순서:")
    print("   1. POST /start (session_id: 'user1')")
    print("   2. POST /chat (session_id: 'user1', message: '안녕! 나는 철수야.')")
    print("   3. POST /chat (session_id: 'user1', message: '내 이름이 뭐라고?')")
    print("   4. POST /chat (session_id: 'user1', message: '종료')")
    print()
    print("⚠️ 주의: 세션을 먼저 시작해야 대화할 수 있습니다!")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
