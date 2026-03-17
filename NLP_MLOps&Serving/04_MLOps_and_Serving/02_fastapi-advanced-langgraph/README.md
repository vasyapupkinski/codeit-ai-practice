# FastAPI 실습 2

FastAPI를 활용한 AI 모델 서빙 실습 자료입니다.

## 📚 Lab 구성

| Lab | 주제 | 핵심 내용 |
|-----|------|----------|
| lab0 | Lifespan | 서버 시작/종료 시 리소스 관리, ML 모델 1회 로딩 패턴 |
| lab1 | 텍스트 요약 | HuggingFace 모델(한/영) + OpenAI GPT API 서빙 |
| lab2 | 음성→텍스트 | Whisper 모델을 활용한 STT(Speech-to-Text) |
| lab3 | MNIST 숫자 인식 | PyTorch CNN 모델 학습 및 API 서빙 |
| lab4 | RAG & LangGraph | ChromaDB 기반 RAG + 조건부 라우팅 에이전트 |
| lab5 | AWS Bedrock | boto3로 Claude 3 호출, 광고 문구 생성기 |

## 🚀 시작하기

```bash
pip install -r requirements.txt
```

각 lab 폴더의 Python 파일 상단 주석에서 상세 실행 방법을 확인하세요.