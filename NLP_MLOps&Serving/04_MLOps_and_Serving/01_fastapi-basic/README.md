# FastAPI 실습

FastAPI 실습 4개

## 실습 구성

| 실습 | 주제 | 핵심 학습 목표 | 난이도 |
|------|------|---------------|--------|
| 1 | FastAPI 기초 | 라우팅, 경로/쿼리 매개변수 | 
| 2 | Pydantic 데이터 검증 | Request/Response 모델링 |
| 3 | LLM API 래핑 | OpenAI API |
| 4 | RAG 챗봇 API + Streamlit | 실전 프로젝트 |

---

## 환경 설정

### 1. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
# 기본 패키지
pip install fastapi uvicorn

# 실습 3, 4용
pip install openai python-dotenv

# 실습 4용
pip install chromadb sentence-transformers streamlit requests
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 입력하세요:

```env
OPENAI_API_KEY=your-api-key-here
```

---

## 실습별 실행 방법

### 실습 1: FastAPI 기초

```bash
# 서버 실행
uvicorn lab1_fastapi_basics:app --reload

# 접속
# API 문서: http://localhost:8000/docs
# 메인: http://localhost:8000
```

**테스트해볼 것:**
- `GET /` - 기본 응답 확인
- `GET /users/1` - 경로 매개변수
- `GET /search?keyword=AI&limit=5` - 쿼리 매개변수

---

### 실습 2: Pydantic 데이터 검증

```bash
# 서버 실행
uvicorn lab2_pydantic_validation:app --reload

# 접속
# API 문서: http://localhost:8000/docs
```

**테스트해볼 것:**
- `POST /users` - 사용자 생성 (Swagger에서 테스트)
- `POST /products` - 상품 등록 (유효성 검사 테스트)
  - price에 음수 입력 → 에러!
  - category에 허용되지 않은 값 → 에러!

---

### 실습 3: LLM API 래핑

```bash
# .env 파일 설정 필수!
uvicorn lab3_llm_api_wrapper:app --reload

# 접속
# API 문서: http://localhost:8000/docs
```

**테스트해볼 것:**
- `POST /chat` - 기본 채팅
- `POST /summarize` - 텍스트 요약
- `POST /translate` - 번역
- `POST /analyze/sentiment` - 감정 분석
- `GET /templates` - 템플릿 목록
- `POST /generate` - 템플릿 기반 생성

---

### 실습 4: RAG 챗봇 (백엔드 + 프론트엔드)

**터미널 1: FastAPI 서버**
```bash
uvicorn lab4_rag_api:app --reload --port 8000
```

**터미널 2: Streamlit UI**
```bash
streamlit run lab4_streamlit_ui.py
```

**접속:**
- FastAPI 문서: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

**사용 방법:**
1. Streamlit 사이드바에서 "샘플 추가" 클릭
2. 채팅 탭에서 질문하기 (예: "FastAPI가 뭐야?")
3. 검색 탭에서 문서 검색 테스트

---

## 🔧 트러블슈팅

### 1. "ModuleNotFoundError" 발생

```bash
# 필요한 패키지 재설치
pip install -r requirements.txt
```

### 2. OpenAI API 키 오류

```bash
# .env 파일 확인
cat .env  # OPENAI_API_KEY가 올바르게 설정되어 있는지 확인
```

### 3. 포트 충돌

```bash
# 다른 포트로 실행
uvicorn lab1_fastapi_basics:app --reload --port 8001
```

### 4. ChromaDB 오류 (실습 4)

```bash
# 캐시 삭제 후 재설치
pip uninstall chromadb
pip install chromadb
```

### 5. Streamlit 연결 오류 (실습 4)
- FastAPI 서버가 먼저 실행되어 있어야 합니다
- 포트 8000이 사용 중인지 확인하세요
- 포트 관련 (linux/mac):
  - 실행시킬 떄 마다 포트가 누적됨
  - 포트확인 `lsof -i:{port}`
  - 포트종료 'kill -9 {pid}`
- 포트 관련 (window)
  - 포트 확인 `netstat -aon | findstr :{port}`
  - 마지막 숫자 pid `Stop-Process -Id {pid} -Force`
  - `$pid = (netstat -aon | findstr :{port} | Select-String "LISTENING" | ForEach-Object { $_ -split "\s+" } | Select-Object -Last 1); if ($pid) { Stop-Process -Id $pid -Force }`
---

## 📖 추가 학습 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/ko/)
- [Pydantic 공식 문서](https://docs.pydantic.dev/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [ChromaDB 공식 문서](https://docs.trychroma.com/)

---

