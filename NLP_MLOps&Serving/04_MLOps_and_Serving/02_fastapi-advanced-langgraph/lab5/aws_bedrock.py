"""
🎯 FastAPI 실습: AWS Bedrock + Claude 3 광고 문구 생성기

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **AWS Bedrock을 통해 Claude 3 모델을 사용**하여 광고 문구를 생성하는 API입니다.
OpenAI 대신 AWS의 AI 서비스를 활용하여, 클라우드 벤더 종속성을 줄이는 멀티 클라우드 전략을 배웁니다.

===============================================
🤔 왜 AWS Bedrock인가? (설계 의도)
===============================================
1. **멀티 클라우드 전략**:
   - OpenAI에만 의존하지 않음
   - 가격/성능 비교 가능
   - 리스크 분산

2. **AWS 생태계 활용**:
   - 이미 AWS 쓰는 회사에 유리
   - IAM으로 보안 관리
   - Lambda, ECS 등과 통합 쉬움

3. **Claude의 특징**:
   - Anthropic이 개발
   - 긴 컨텍스트 처리 강점
   - 윤리적 AI 강조

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. .env에서 AWS 키 로드
2. boto3 클라이언트 생성
3. 서버 대기

[요청 처리]
1. 제품명, 키워드 수신
2. Claude API 형식으로 JSON 구성
3. Bedrock invoke_model() 호출
4. StreamingBody 읽기
5. JSON 파싱 → 텍스트 추출

===============================================
💡 핵심 학습 포인트
===============================================
- Boto3: AWS Python SDK
- StreamingBody: AWS 응답 형식
- IAM 인증: API 키 대신 역할 기반
- 멀티 클라우드: 종속성 탈피

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - pip install fastapi uvicorn boto3 python-dotenv
   - AWS 계정에서 Bedrock 모델 액세스 활성화 (Claude 3 Haiku)
   - .env 파일 설정:
     AWS_ACCESS_KEY=AKIA...
     AWS_SECRET_KEY=...
     (AWS IAM에서 AmazonBedrockFullAccess 권한 필요)

2. 실행 방법:
   python ./lab5/aws_bedrock.py

💡 OpenAI vs AWS Bedrock 비교:
┌─────────────┬──────────────────┬──────────────────┐
│             │ OpenAI API       │ AWS Bedrock      │
├─────────────┼──────────────────┼──────────────────┤
│ 인증        │ API Key          │ IAM (Access/Secret)│
│ 클라이언트    │ openai 라이브러리 │ boto3            │
│ 모델         │ gpt-4o-mini 등   │ Claude, Titan 등 │
│ 과금        │ OpenAI 직접 결제  │ AWS 청구서 통합    │
│ 보안        │ API Key 관리     │ IAM 정책 관리      │
└─────────────┴──────────────────┴──────────────────┘

💡 Bedrock 모델 ID 예시:
- anthropic.claude-3-haiku-20240307-v1:0 (빠름, 저렴)
- anthropic.claude-3-sonnet-20240229-v1:0 (균형)
- anthropic.claude-3-opus-20240229-v1:0 (고성능)

⚠️ 주의사항:
- AWS 리전별로 사용 가능한 모델이 다름 (us-east-1 권장)
- Bedrock 콘솔에서 모델 액세스 요청 필요 (승인까지 몇 분 소요)
- IAM 사용자에게 AmazonBedrockFullAccess 정책 연결 필요
"""

#  필수 모듈 임포트
import json
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

#  환경변수 로드
# 환경변수 로드
load_dotenv()

#  FastAPI 앱 생성
app = FastAPI()

#  AWS 자격 증명 로드
# --------------------------------------------------------
# 1. AWS Bedrock 클라이언트 설정
# --------------------------------------------------------
# 실습용: 본인의 AWS Access Key와 Secret Key를 여기에 체크합니다.
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"                # Bedrock 모델이 활성화된 리전 (예: us-east-1, us-west-2)

#  OpenAI API 키 (주석 처리 가능)
api_key=os.getenv("OPENAI_API_KEY")

#  Boto3 클라이언트 생성
# boto3 클라이언트 생성 (boto3가 AWS와 통신합니다)
# 1. boto3.client(): AWS 서비스 클라이언트 생성
# 2. service_name='bedrock-runtime': Bedrock 런타임 서비스 지정
# 3. region_name: AWS 리전 (Bedrock 모델이 있는 곳)
# 4. aws_access_key_id, aws_secret_access_key: IAM 인증 정보
# [개념] 왜 멀티 클라우드 전략인가? (왜 AWS Bedrock도 다루나?)
# - OpenAI 단독 사용의 리스크:
#   * API 다운: OpenAI 서버 장애 시 서비스 전체 마비 (Single Point of Failure)
#   * 가격 변동: 갑작스러운 가격 인상 시 대응 불가
#   * 벤더 종속: OpenAI 정책 변경에 끼이기만 함
#   * 지역 제한: 특정 국가에서 OpenAI 차단 시 대응 불가
# - 멀티 클라우드의 장점:
#   * 안정성: 한 곳이 다운되면 다른 곳으로 즉시 전환
#   * 경제성: 가격 비교하여 저렴한 곳 사용
#   * 기능 선택: Claude는 긴 문서, GPT는 코드 생성 등 각 장점 활용
# - AWS Bedrock의 특장:
#   * IAM 통합: AWS 기존 보안 체계 그대로 사용
#   * 비용 통합: AWS 청구서에 통합 (회계 처리 간편)
#   * 다양한 모델: Claude, Titan, Llama 등 한 곳에서 접근
#   * Lambda 통합: 서버리스로 쉽게 배포 가능
# - 실무 패턴:
#   * 폴백 시스템: OpenAI 에러 시 자동으로 Bedrock으로 전환
#   * 비용 최적화: 요청 종류별로 가장 저렴한 API 선택
#   * 지역 분산: 미국은 OpenAI, 아시아는 AWS 서울 리전 등
# - 이 파일의 목적: OpenAI에만 의존하지 않고 AWS도 다룰 줄 아는 개발자 되기

#  Boto3 클라이언트 생성
# 1. boto3.client(...):
#    a) botocore 세션 생성 (AWS 설정 로드)
#    b) credential provider chain 동작:
#       - 환경변수 (AWS_ACCESS_KEY_ID) -> ~/.aws/credentials -> IAM Role 순으로 자격증명 탐색
#    c) Service description(JSON) 로드:
#       - Bedrock Runtime 서비스의 API 정의(메서드, 파라미터) 로드
#    d) 동적 메서드 생성:
#       - invoke_model 등의 메서드가 런타임에 동적으로 객체에 바인딩됨
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

#  요청 스키마 정의
# --------------------------------------------------------
# 2. 데이터 모델 정의 (Pydantic)
# --------------------------------------------------------
class AdRequest(BaseModel):
    product_name: str   # 예: "초경량 무선 청소기"
    keywords: str       # 예: "강력한 흡입력, 조용함, 가벼움"

#  API 엔드포인트
# --------------------------------------------------------
# 3. API 엔드포인트 정의
# --------------------------------------------------------
@app.post("/generate-ad")
async def generate_ad_copy(request: AdRequest):
    #  try-except 블록
    try:
        #  프롬프트 작성
        # (1) 프롬프트 작성: AI에게 지시할 내용을 만듭니다.
        prompt = f"""
        당신은 전문 마케터입니다. 아래 제품에 대한 매력적인 SNS 광고 문구를 3줄 이내로 작성해주세요.
        
        제품명: {request.product_name}
        강조할 키워드: {request.keywords}
        
        광고 문구:
        """

        #  Bedrock 요청 body 구성
        # (2) Bedrock (Claude 3) 요청 바디 구성
        # Claude 3는 'messages' 형식을 사용합니다.
        # - anthropic_version: Claude API 버전
        # - max_tokens: 생성할 최대 토큰 수
        # - messages: 대화 형식의 메시지 배열
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        #  Bedrock API 호출
        # (3) 모델 호출 (invoke_model)
        # ========== modelId는 AWS 콘솔에서 확인 가능 (여기선 Claude 3 Haiku 사용) =========
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        # - invoke_model(): 동기식 API 호출 (비동기 버전도 있음)
        # - body: JSON 문자열로 인코딩된 요청
        # - modelId: Bedrock 모델 식별자
        # - accept, contentType: HTTP 헤더
        #  Bedrock API 호출
        # 3. bedrock_client.invoke_model():
        #    a) 요청 서명 (SigV4):
        #       - Access Key, Secret Key, 타임스탬프 등을 조합하여 HMAC-SHA256 서명 생성
        #       - 'Authorization' 헤더에 서명 추가
        #    b) HTTPS POST 요청 전송:
        #       - 엔드포인트: https://bedrock-runtime.us-east-1.amazonaws.com/model/.../invoke
        #       - TLS 암호화 터널을 통해 전송
        #    c) 동기 대기:
        #       - AWS 서버의 처리가 끝날 때까지 스레드 블로킹
        #       - 응답이 오면 HTTP 상태 코드(200 OK 등) 확인
        response = bedrock_client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )

        #  응답 파싱
        # (4) 응답 파싱
        # 1. response.get("body"): StreamingBody 객체
        # 2. .read(): 바이트 스트림 읽기
        # 3. json.loads(): JSON 문자열 → Python dict 변환
        response_body = json.loads(response.get("body").read())
        
        #  텍스트 추출
        # - response_body['content'][0]['text']: Claude가 생성한 텍스트
        result_text = response_body['content'][0]['text']

        #  결과 반환
        return {"result": result_text}

    except Exception as e:
        #  에러 처리
        # 에러 발생 시 상세 내용 반환
        raise HTTPException(status_code=500, detail=str(e))

#  기본 루트 엔드포인트
# 서버 실행 테스트를 위한 기본 루트
@app.get("/")
def read_root():
    return {"message": "AWS Bedrock FastAPI 서비스가 실행 중입니다!"}


#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)