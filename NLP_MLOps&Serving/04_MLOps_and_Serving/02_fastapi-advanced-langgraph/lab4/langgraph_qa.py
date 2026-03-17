"""
🎯 FastAPI 실습: LangGraph 조건부 라우팅 에이전트

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **질문 유형을 분석하여 다른 AI 페르소나로 라우팅**하는 시스템입니다.
기술 질문은 전문가 모드로, 일상 대화는 친구 모드로 자동 분기하는 LangGraph 워크플로우를 구현합니다.

===============================================
🤔 왜 LangGraph인가? (설계 의도)
===============================================
1. **단순 체인의 한계**:
   - 모든 입력을 동일하게 처리
   - 조건부 로직 구현 어려움
   - 복잡한 워크플로우 불가

2. **LangGraph의 장점**:
   - 상태(State) 기반 워크플로우
   - 조건부 분기 가능
   - 노드 간 데이터 전달 명확

3. **사용 사례**:
   - 고객 문의 분류 시스템
   - 멀티 에이전트 협업
   - 게임 NPC AI 등

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[그래프 구조]
                    ┌─────────────┐
                    │  classifier │ (질문 분류)
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            │ TECHNICAL            CASUAL │
            ▼                             ▼
    ┌──────────────┐               ┌─────────────┐
    │ tech_expert  │               │ friendly_bot│
    │(시니어 개발자) │               │ (친절한 친구) |
    └──────┬───────┘               └──────┬──────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
                        [END]

[실행]
1. 질문 입력
2. Classifier 노드: GPT로 질문 분류
3. decide_route 함수: 분류 결과 확인
4. 조건부 분기:
   - TECHNICAL → tech_expert 노드
   - CASUAL → friendly_bot 노드
5. 해당 노드에서 답변 생성
6. END

===============================================
💡 핵심 학습 포인트
===============================================
- StateGraph: 상태 기반 워크플로우
- 조건부 엣지: 동적 라우팅
- 노드: 작업 단위
- TypedDict: 상태 스키마 정의

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   pip install fastapi uvicorn langgraph langchain-openai langchain-core
   (.env 파일에 OPENAI_API_KEY=sk-xxx 추가)

2. 실행 방법:
   python ./lab4/langgraph_qa.py

📌 테스트 (Swagger UI):
http://localhost:8000/docs → /smart_chat → Try it out
message 예제1: \"파이썬에서 데코레이터가 뭐야?\"
message 예제2: \"오늘 날씨 어때?\"
"""


# [파일 구조 및 순서 설명 (Map)]
# 이 파일도 위에서부터 아래로 순서대로 실행됩니다.
# LangGraph는 "로봇 조립"과 순서가 매우 비슷합니다.
#
# 1. [재료 준비] 임포트 & 환경설정
#    - 필요한 부품(모듈)과 배터리(API Key)를 준비합니다.
#
# 2. [설계도 작성] 상태(State) 정의 (`AgentState`)
#    - "이 로봇은 어떤 데이터(기억)를 가지고 다닐까?"를 정의합니다.
#    - 로봇의 뇌 구조(Schema)를 만드는 단계입니다. 가장 먼저 해야 합니다.
#
# 3. [부품 제작] 노드 함수 정의 (`classify`, `tech_expert`...)
#    - 로봇의 팔, 다리, 머리 역할을 하는 함수들을 만듭니다.
#    - 각 함수는 독립적으로 동작하지만, 2번에서 만든 '설계도(State)'를 공유합니다.
#
# 4. [조립] 그래프 구성 (`workflow.add_node`, `add_edge`)
#    - 만들어둔 부품들을 전선으로 연결합니다.
#    - "시작하면 머리(Classifier)로 가고, 그 다음엔 팔(Expert)로 가라"는 흐름을 만듭니다.
#
# 5. [가동 승인] 컴파일 (`workflow.compile`)
#    - 조립된 로봇이 실제로 움직일 수 있는지 검사하고 전원을 켭니다.
#    - 이때 `app_graph`라는 "움직이는 로봇 객체"가 탄생합니다.
#
# 6. [명령어 입력] API 엔드포인트 (`@app.post`)
#    - 사용자가 로봇에게 명령을 내릴 수 있는 리모컨 버튼을 만듭니다.

#  필수 모듈 임포트
from fastapi import FastAPI # [초보] fastapi 모듈에서 FastAPI 클래스를 가져옵니다
from pydantic import BaseModel
from typing import TypedDict, Literal

#  LangGraph 및 LangChain 임포트
# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

#  환경변수 로드
# 환경변수 로드
load_dotenv()

#  FastAPI 앱 생성
app = FastAPI()

# [이게 뭐하는 코드?] AgentState = 노드들 간에 공유하는 "데이터 저장소"
# - TypedDict: 타입을 지정할 수 있는 딕셔너리
# - question: 사용자가 보낸 질문 (처음에 저장)
# - classification: classifier 노드가 분류한 결과 ("TECHNICAL" or "CASUAL")
# - response: 최종 답변 (tech_expert 또는 friendly_bot이 생성)
# - 이 state가 모든 노드를 통과하면서 차기차기 채워짐!

# [기계적 흐릉] 상태(State) 정의
# 1. 상태(State) 정의: 그래프 내에서 공유되는 데이터
# - TypedDict: 타입 힌팅을 제공하는 딕셔너리 (Python 3.8+)
# - 노드 간 데이터를 주고받을 때 사용하는 스키마
class AgentState(TypedDict):
    question: str           # 사용자 질문
    classification: str     # 분류 결과 (TECHNICAL or CASUAL)
    response: str           # 최종 답변

#  LLM 모델 초기화
# 2. 모델 설정
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# [개념] 왜 노드 함수는 async def가 아닌 def인가? (중요!)
# - LangGraph 노드는 반드시 동기(sync) 함수여야 함:
#   def classify_input(state): ... ← 올바름
#   async def classify_input(state): ... ← 에러 발생!
# - 왜 그런가?
#   1) LangGraph가 내부적으로 state 관리를 동기적으로 처리
#   2) 각 노드가 순차적으로 실행되면서 state를 업데이트
#   3) 노드 간 의존성과 순서를 명확히 하기 위해
#   4) state 병합 시 경쟁 상태(race condition) 방지
# - 그럼 LangChain은 chain.invoke() 사용 (chain.ainvoke() 아님)
# - 하지만 FastAPI 엔드포인트는 async def + await app_graph.ainvoke()
#   → 그래프 전체 실행은 비동기, 노드 내부는 동기
#   → 이렇게 하면 그래프 실행 중에도 다른 요청 처리 가능!
# - 요약: 노드 = def, 체인 = invoke(), 엔드포인트 = async + ainvoke()

# [이게 뭐하는 코드?] classify_input = 질문을 분류하는 노드
# - 역할: "이건 기술 질문인가 vs 일상 대화인가?" 판단
# - 입력: state["question"] = 사용자 질문
# - 처리:
#   1) GPT에게 "이건 TECHNICAL인가 CASUAL인가?" 물어봄
#   2) GPT 응답에 "TECHNICAL" 문자 있으면 → TECHNICAL
#   3) 없으면 → CASUAL
# - 출력: {"classification": "TECHNICAL"} 또는 {"classification": "CASUAL"}
# - 이 결과가 state에 병합됨 → decide_route가 이걸 보고 다음 노드 결정

# [기계적 흐릉] 노드 함수 정의
# --- 노드(Nodes) 정의 ---
# 노드 1: 질문 분류기 (Classifier)
def classify_input(state: AgentState):
    #  디버그 출력
    print(f"--- 분류중: {state['question']}")
    
    #  프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_template(
        """다음 질문을 'TECHNICAL'(프로그래밍/코딩/기술 관련) 또는 'CASUAL'(일상 대화/인사) 중 하나로 분류하세요.
            질문: {question}
            결과:
        """
        )
    
    #  LCEL 체인 구성 및 실행
    # - prompt | model | StrOutputParser(): 파이프로 체인 연결
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"question": state["question"]})

    #  결과 정제
    # 결과를 깔끔하게 정제
    # - "TECHNICAL"이 결과에 포함되어 있으면 TECHNICAL, 아니면 CASUAL
    classification = "TECHNICAL" if "TECHNICAL" in result.upper() else "CASUAL"
    
    #  상태 업데이트 반환
    # - 반환된 딕셔너리가 state에 병합됨
    return {"classification": classification}

# [이게 뭐하는 코드?] handle_technical = 기술 전문가 모드로 답변하는 노드
# - 역할: 시니어 개발자 페르소나로 엄격하게 답변
# - 입력: state["question"] = 사용자 질문
# - 처리:
#   1) 프롬프트: "황은 시니어 개발자입니다. 명확하고 엄격하게 답변하세요."
#   2) GPT에게 질문 전송
#   3) GPT의 전문가 답변 받음
# - 출력: {"response": "기술적 답변..."}
# - 이게 최종 state에 저장됨

# [기계적 흐릉] 기술 전문가 노드
# 노드 2: 기술 전문가 답변
def handle_technical(state: AgentState):
    #  디버그 출력
    print("=========== 기술 전문가 모드")
    
    #  전문가 페르소나 프롬프트
    prompt = ChatPromptTemplate.from_template(
        "당신은 시니어 파이썬 개발자입니다. 질문에 대해 명확하고 엄격하게 답변하세요.\\n질문: {question}"
    )
    
    #  체인 실행
    chain = prompt | model | StrOutputParser()
    
    #  상태 업데이트 반환
    return {"response": chain.invoke({"question": state['question']})}
   
#  친절한 친구 노드
# 노드 3: 친절한 친구 답변
def handle_casual(state: AgentState):
    #  디버그 출력
    print("=========== 일상 대화 모드")
    
    #  친구 페르소나 프롬프트
    prompt = ChatPromptTemplate.from_template(
        "당신은 친절한 친구입니다. 이모지를 사용해서 따뜻하게 답변하세요.\\n질문: {question}"
    )
    
    #  체인 실행
    chain = prompt | model | StrOutputParser()
    
    #  상태 업데이트 반환
    return {"response": chain.invoke({"question": state['question']})}


# [개념] 왜 LangGraph를 쓰는가? (LCEL vs LangGraph)
# - LCEL(파이프)의 한계:
#   * 선형 흐름만 가능: retriever | prompt | model | parser
#   * 조건부 분기 어려움: "if A면 B로, else C로" 같은 로직 구현 불가
#   * 복잡한 워크플로우 표현 능력 부족
# - LangGraph의 장점:
#   * 상태 기반: AgentState로 노드 간 데이터 공유
#   * 조건부 분기: add_conditional_edges로 동적 라우팅
#   * 각 노드는 독립적인 AI 에이전트처럼 동작 가능
#   * 루프, 순환, 병렬 실행 모두 가능
# - 현재 파일의 예시:
#   * classifier 노드: 질문을 "TECHNICAL" vs "CASUAL" 분류
#   * decide_route: 분류 결과에 따라 다른 노드로 이동
#   * tech_expert / friendly_bot: 각각 다른 페르소나로 답변 생성
# - 실무 활용 사례:
#   * 고객 문의 분류 시스템 (기술지원/환불/판매 등)
#   * 멀티 에이전트 협업 (리서처 에이전트 + 라이팅 에이전트)
#   * 복잡한 의사결정 트리 (A면 B, B면 C, C면 D...)
# - LCEL은 간단한 파이프라인, LangGraph는 복잡한 워크플로우용

#  그래프 구성 (조립 시작)
# 여기서부터 본격적으로 "로봇 조립"을 시작합니다.
# 순서가 중요합니다: [본체(StateGraph)] -> [부품(Node)] -> [전선(Edge)]

# 1. 본체(Chassis) 준비
# - StateGraph: 상태(AgentState)를 담을 수 있는 빈 로봇 본체를 준비합니다.
workflow = StateGraph(AgentState) # [초보] StateGraph() 생성자를 호출하고 반환값을 workflow 변수에 할당합니다

#  노드 추가
# 노드 추가
# - add_node(이름, 함수): 그래프에 노드 추가
workflow.add_node("classifier", classify_input)
workflow.add_node("tech_expert", handle_technical)
workflow.add_node("friendly_bot", handle_casual)

#  진입점 설정
# 진입점 설정
# - set_entry_point(): 그래프 실행 시 가장 먼저 실행될 노드
workflow.set_entry_point("classifier")

#  라우팅 함수 정의
# 조건부 엣지 함수: 분류 결과에 따라 다음 노드 결정
def decide_route(state: AgentState):
    #  분류 결과에 따라 다음 노드 선택
    # - state["classification"]의 값에 따라 문자열 반환
    # - 반환된 문자열이 다음 노드의 이름이 됨
    if state["classification"] == "TECHNICAL":
        return "tech_expert"
    else:
        return "friendly_bot"
    
#  조건부 엣지 연결
# 조건부 엣지 연결 (classifier -> tech_expert OR friendly_bot)
# - add_conditional_edges(): 조건에 따라 다른 노드로 이동
# - 매핑 딕셔너리: decide_route의 반환값 → 실제 노드 이름
workflow.add_conditional_edges(
    "classifier",
    decide_route,
    {
        "tech_expert": "tech_expert",
        "friendly_bot": "friendly_bot"
    }
)

#  종료 엣지 연결
# 종료 엣지 연결
# - add_edge(노드, END): 해당 노드 실행 후 그래프 종료
workflow.add_edge("tech_expert", END)
workflow.add_edge("friendly_bot", END)

# [이게 뭐하는 코드?] app_graph = 실행 가능한 그래프로 변환
# - compile(): workflow를 "실제로 동작하는" 객체로 변환
# - 이 단계에서 내부적으로 검증:
#   * 모든 노드가 선언되었나?
#   * 엣지가 제대로 연결되었나?
#   * END로 가는 경로가 있나?
# - compile 후에는 app_graph.ainvoke()로 실행 가능
# - 이제 이걸 FastAPI 엔드포인트에서 사용함

#  그래프 컴파일 (조립 완료 & 전원 ON)
# 3. 그래프 컴파일 (실행 가능한 객체로 변환)
# - 지금까지는 그냥 '계획'이었습니다.
# - compile()을 호출하는 순간, 이 계획이 "실제로 돌아가는 기계(Runnable)"로 바뀝니다.
# - 내부적으로: 
#   1. 끊어진 전선(Edge)은 없나 확인
#   2. 시작점(Entry Point)과 끝점(END)이 잘 연결됐나 확인
#   3. 최적화된 실행 경로 생성 (Pregel)
app_graph = workflow.compile() # [초보] workflow.compile() 메서드를 호출하고 반환값을 app_graph 변수에 할당합니다


#  FastAPI 엔드포인트 정의
# --- FastAPI 엔드포인트 ---
# [문법 설명] 이것이 바로 'Pydantic(파이단틱)'입니다!
# - (BaseModel)을 상속받으면, 이 클래스는 "엄격한 검문소"가 됩니다.
class ChatRequest(BaseModel):
    message: str

#  API 엔드포인트
@app.post("/smart_chat")
async def smart_chat(req: ChatRequest):
    #  초기 상태 설정
    # - 그래프에 전달할 초기 state
    inputs = {"question": req.message}
    
    #  그래프 실행
    # - ainvoke(): 비동기로 그래프 실행 (Pregel 알고리즘 기반)
    # - 내부 동작 순서:
    #   1) 입력값(inputs)을 초기 상태(AgentState)에 병합
    #   2) Entry Point(classifier) 실행
    #   3) 노드 실행 결과 반환 -> 상태 업데이트 (기존 상태 덮어쓰기 또는 병합)
    #   4) Conditional Edge 평가 (decide_route) -> 다음 노드 결정
    #   5) 다음 노드 실행 -> 상태 업데이트
    #   6) END 도달 시 최종 상태 반환
    result = await app_graph.ainvoke(inputs)

    #  결과 반환
    # - result["classification"]: 분류 결과 (TECHNICAL or CASUAL)
    # - result["response"]: 최종 답변
    return {
        "type": result["classification"],
        "reply": result["response"]
    }

#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)