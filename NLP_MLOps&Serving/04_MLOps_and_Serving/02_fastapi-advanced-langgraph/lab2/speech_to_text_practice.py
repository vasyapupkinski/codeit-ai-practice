"""
🎯 FastAPI Lab 2 과제: Whisper 음성-텍스트 변환 (혼자해보기)

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **Lab 2의 과제 1, 2, 3을 모두 구현한 완성본**입니다.
OpenAI Whisper 모델을 사용하여 음성 파일을 텍스트로 변환하며, 다음 3가지 개선사항을 포함합니다:

**과제 1: 지원 형식 검증**
- 허용된 오디오 형식(.mp3, .wav, .m4a, .flac)만 처리
- 잘못된 형식 업로드 시 400 에러 반환

**과제 2: 타임스탬프 포함 출력**
- Whisper의 세그먼트 정보를 활용
- 각 문장의 시작/종료 시간을 응답에 포함

**과제 3: 언어 지정 옵션**
- 사용자가 언어를 미리 지정할 수 있음 (ko, en, ja, zh 등)
- 지정하지 않으면 Whisper가 자동으로 언어 감지

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **파일 형식 검증의 필요성 (과제 1)**:
   - Whisper는 ffmpeg를 통해 오디오 디코딩
   - 지원하지 않는 형식(예: .exe, .txt)을 받으면 서버 에러 발생
   - 사전 검증으로 명확한 에러 메시지 제공 가능

2. **타임스탬프 정보의 활용 (과제 2)**:
   - 자막(Subtitle) 생성에 필수
   - 영상 편집, 회의록 작성 등에 활용
   - Whisper는 기본적으로 세그먼트 정보를 제공하지만 불필요한 데이터가 많음
   - start, end, text만 추출하여 깔끔한 응답 제공

3. **언어 지정 옵션의 장점 (과제 3)**:
   - 자동 감지는 편리하지만 때로 부정확
   - 한국어 발음이 영어로 인식되는 경우 방지
   - 언어를 미리 알고 있으면 인식 정확도 향상

4. **임시 파일 패턴 사용**:
   - Whisper는 파일 경로를 입력으로 요구 (메모리 스트림 불가)
   - 업로드 데이터를 임시 파일로 저장 후 처리
   - finally 블록으로 임시 파일 삭제 보장 (디스크 공간 관리)

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. FFmpeg 경로 자동 설정 (Windows 환경)
2. Whisper "base" 모델 로딩 (~140MB)
3. 전역 딕셔너리에 모델 저장

[요청 처리]
1. 사용자가 오디오 파일 + 언어(선택) 업로드
2. 파일 확장자 검증 (과제 1) -> 실패 시 400 에러
3. 임시 파일 생성 및 데이터 저장
4. Whisper 추론 실행 (language 옵션 적용, 과제 3)
5. 세그먼트 정보 가공 (start/end/text만 추출, 과제 2)
6. JSON 응답 반환
7. **finally: 임시 파일 삭제** (필수!)

===============================================
💡 핵심 학습 포인트
===============================================
- **파일 검증**: Set을 이용한 O(1) 확장자 체크
- **리스트 컴프리헨션**: 데이터 필터링의 Python다운 방법
- **쿼리 파라미터**: FastAPI가 자동으로 파싱하는 선택적 파라미터
- **try-finally**: 리소스 정리가 반드시 실행되도록 보장
- **Whisper API**: transcribe() 메서드의 language 옵션 활용

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - 시스템에 ffmpeg 설치 (필수!)
   - pip install openai-whisper python-multipart

2. 실행 방법:
   python ./lab2/speech_to_text_practice.py

3. 테스트 예시 (curl):
   # 기본 요청 (자동 언어 감지)
   curl -X POST "http://localhost:8000/transcribe" \\
     -F "file=@audio.mp3"
   
   # 한국어 지정
   curl -X POST "http://localhost:8000/transcribe?language=ko" \\
     -F "file=@audio.wav"
"""

#  필수 모듈 임포트
# 1. import 문법:
#    - Python이 외부 라이브러리의 코드를 메모리에 로드하는 명령어
#    - sys.path에서 모듈을 찾아 바이트코드로 컴파일 후 네임스페이스에 바인딩
# 2. from A import B:
#    - A 모듈에서 B만 선택적으로 가져옴 (메모리 절약)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import traceback
from contextlib import asynccontextmanager
import whisper
import shutil,os, tempfile

#  FFmpeg 경로 설정 함수 정의
# 1. def 키워드:
#    - 함수 객체를 생성하여 현재 네임스페이스에 "setup_ffmpeg_path"라는 이름으로 바인딩
#    - 이 시점에는 함수 내부 코드가 실행되지 않음 (정의만 함)
def setup_ffmpeg_path():
    confirmed_path = r"C:\ffmpeg\bin"
    conda_path = r"C:\Users\daboi\anaconda3\envs\yolo_env_py311\Library\bin"

    print(f"🔧 FFmpeg 경로 설정 시도: {confirmed_path}")
    
    if os.path.exists(confirmed_path):
        os.environ["PATH"] = confirmed_path + ";" + os.environ["PATH"]
        print(f"✅ PATH에 추가됨: {confirmed_path}")
    elif os.path.exists(conda_path):
        os.environ["PATH"] = conda_path + ";" + os.environ["PATH"]
        print(f"✅ (Conda) PATH에 추가됨: {conda_path}")
    else:
        print("❌ 경고: 확인된 FFmpeg 경로를 찾을 수 없습니다.")

    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🎉 FFmpeg 실행 테스트 성공! (준비 완료)")
    except Exception as e:
        print(f"🔥 FFmpeg 실행 테스트 실패: {e}")
        print("👉 브라우저에서 500 에러가 발생하면 이 로그를 확인하세요.")

#  함수 호출
# 1. 함수명() 문법:
#    - 위에서 정의한 함수 객체를 실제로 실행
#    - 새로운 스택 프레임 생성 -> 함수 내부 코드 실행 -> 스택 프레임 제거
setup_ffmpeg_path()

#  전역 딕셔너리 변수 초기화
# 1. {} 문법:
#    - 빈 딕셔너리(해시 테이블) 객체를 힙 메모리에 생성
#    - ml_models라는 이름을 이 객체의 메모리 주소에 바인딩
# 2. 전역 변수를 쓰는 이유:
#    - 모든 요청에서 같은 모델 객체를 공유하여 메모리 절약
ml_models = {}

#  데코레이터 문법
# 1. @asynccontextmanager:
#    - 데코레이터는 함수를 "감싸는" 함수
#    - lifespan 함수를 asynccontextmanager로 전달 -> 컨텍스트 매니저 객체로 변환
#    - 이렇게 하면 yield 전후로 startup/shutdown 로직 실행 가능
@asynccontextmanager
#  비동기 함수 정의
# 1. async def 키워드:
#    - 일반 함수가 아닌 "코루틴 객체"를 반환하는 함수 생성
#    - 내부에서 await 키워드 사용 가능 (이벤트 루프에 제어권 양보)
async def lifespan(app: FastAPI):
    print("Model Loading...")

    #  딕셔너리 항목 추가 (과제에서 Whisper 모델 로딩)
    # 1. 딕셔너리["키"] = 값 문법:
    #    - 해시 함수로 "whisper" 문자열의 해시값 계산
    #    - 해시 테이블에서 해당 버킷을 찾아 값 저장
    # 2. whisper.load_model("base"):
    #    - Whisper 모델 파일을 디스크에서 읽어 메모리(RAM)에 로드
    #    - 약 140MB 크기의 PyTorch 모델 객체 생성
    ml_models["whisper"] = whisper.load_model("base")
    print("Model Loading Complete")

    #  yield 키워드
    # 1. yield:
    #    - 제너레이터/컨텍스트 매니저에서 사용하는 특수 키워드
    #    - 여기서 함수 실행을 일시 정지하고, 서버가 돌아가는 동안 대기
    #    - 서버 종료 신호가 오면 yield 다음 줄부터 재개
    yield

    #  딕셔너리 초기화 (메모리 해제 유도)
    # 1. .clear() 메서드:
    #    - 딕셔너리의 모든 키-값 쌍 제거
    #    - 모델 객체에 대한 참조가 사라지면 가비지 컬렉터가 메모리 회수
    ml_models.clear()

#  FastAPI 애플리케이션 객체 생성
# 1. FastAPI() 생성자:
#    - FastAPI 클래스의 인스턴스를 힙 메모리에 생성
#    - lifespan 파라미터: 서버 시작/종료 시 실행할 함수 지정
# 2. 키워드 인자 (lifespan=...)
#    - 함수 호출 시 "파라미터명=값" 형식으로 전달
#    - 순서 상관없이 명확하게 인자를 전달 가능
app = FastAPI(lifespan=lifespan)

# [개념] 왜 async 함수를 쓰는가? (파일 업로드는 I/O 작업)
# - 파일 업로드의 특성:
#   * 클라이언트가 수 MB 파일을 천천히 전송 (1~10초 소요)
#   * 이 시간 동안 서버가 블로킹되면? → 다른 요청 처리 불가
# - async/await의 장점:
#   * 파일 수신 대기 중 다른 사용자 요청 처리 가능
#   * 동시에 10명이 파일 업로드 → 모두 병렬 처리
# - 실무: 파일 업로드, 다운로드 등 모든 I/O 작업은 비동기 필수

#  FastAPI 엔드포인트 정의
# 1. @app.post("/transcribe"):
#    - 데코레이터: transcribe_audio 함수를 HTTP POST 엔드포인트로 등록
#    - FastAPI가 내부 라우팅 테이블에 "/transcribe" 경로 추가
# 2. async def:
#    - 파일 업로드는 I/O 작업 (네트워크 대기 시간이 김)
#    - async를 쓰면 파일 수신 대기 중 다른 요청 처리 가능
@app.post("/transcribe")
#  함수 시그니처 (매개변수 정의)
# 1. file: UploadFile = File(...):
#    - file이라는 변수에 업로드된 파일 객체가 바인딩됨
#    - UploadFile: FastAPI의 특수 타입 (파일 스트림을 다루는 객체)
#    - File(...): FastAPI에게 "이건 필수 파일 업로드 파라미터"라고 알림
# 2. language: str = None:
#    - 선택적 쿼리 파라미터 (URL의 ?language=ko 형식으로 받음)
#    - None이 기본값: 사용자가 안 보내면 None으로 설정
async def transcribe_audio(file: UploadFile = File(...), language: str = None):
    #  try 블록 시작
    # 1. try:
    #    - 예외가 발생할 수 있는 코드를 감싸는 블록
    #    - 에러 발생 시 except 블록으로 이동 (프로그램 종료 방지)
    try:
        # [개념] 왜 파일 형식을 미리 검증하는가? (과제 1의 의도)
        # - 문제점:
        #   * Whisper에 지원하지 않는 파일(.txt, .exe 등)을 넘기면?
        #   * ffmpeg가 디코딩 실패 → 복잡한 에러 메시지 발생
        #   * 사용자가 원인 파악 어려움
        # - 해결책:
        #   * 사전에 확장자 검증 → 명확한 에러 메시지 제공
        #   * "지원하지 않는 파일 형식입니다" (사용자 친화적)
        # - Set을 쓰는 이유:
        #   * List는 검색이 O(n), Set은 O(1)로 훨씬 빠름
        #   * 확장자 추가/제거가 쉬움
        
        #  집합(Set) 생성 (과제 1: 파일 형식 검증)
        # 1. {} 안에 값들:
        #    - 중괄호 + 값 = Set (집합) 객체 생성
        #    - Set은 중복 제거 + 빠른 조회(O(1)) 가능
        # 2. 대문자 변수명 (ALLOWED_EXTENSIONS):
        #    - Python 관례상 "변하지 않는 상수"를 의미
        ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
        
        #  파일 확장자 추출
        # 1. os.path.splitext(파일명):
        #    - 파일명을 (이름, 확장자) 튜플로 분리
        #    - 예: "audio.mp3" -> ("audio", ".mp3")
        # 2. [1]:
        #    - 튜플의 두 번째 요소(확장자) 선택 (인덱스는 0부터 시작)
        # 3. .lower():
        #    - 문자열을 소문자로 변환 (대소문자 구분 없이 비교하기 위함)
        ext = os.path.splitext(file.filename)[1].lower()

        #  조건문 + 예외 발생 (과제 1: 검증 로직)
        # 1. if ... not in ...:
        #    - "만약 ext가 ALLOWED_EXTENSIONS 안에 없으면" 조건
        #    - Set의 in 연산은 해시 테이블 조회로 매우 빠름 (O(1))
        # 2. raise HTTPException:
        #    - 예외 객체를 생성하여 "던지기" (throw)
        #    - FastAPI가 이를 받아 클라이언트에게 400 에러 응답 전송
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="the file format is not allowed")

        #  with 문 (컨텍스트 매니저)
        # 1. with ... as 변수:
        #    - 리소스를 안전하게 관리하는 Python 문법
        #    - __enter__ 메서드 호출 -> 블록 실행 -> __exit__ 메서드 호출 (자동 정리)
        # 2. tempfile.NamedTemporaryFile:
        #    - 임시 파일을 OS 임시 폴더에 생성 (Windows: C:\Users\...\Temp)
        #    - delete=False: 파일 닫혀도 삭제 안 함 (수동 삭제 필요)
        #    - suffix=ext: 파일 끝에 확장자 붙임 (Whisper가 확장자 보고 형식 인식)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            #  파일 스트림 복사
            # 1. shutil.copyfileobj(원본, 대상):
            #    - 원본 스트림에서 데이터를 청크 단위로 읽어 대상에 쓰기 반복
            #    - User Space -> Kernel Space -> Disk에 순차적으로 저장
            # 2. file.file:
            #    - UploadFile 객체의 .file 속성 = 실제 파일 스트림 객체
            shutil.copyfileobj(file.file, temp_file)
            #  임시 파일 경로 저장
            # 1. temp_file.name:
            #    - 생성된 임시 파일의 절대 경로 (문자열)
            #    - 예: "C:\\Users\\...\\Temp\\tmp1a2b3c4d.mp3"
            temp_file_path = temp_file.name

        #  중첩된 try 블록 (모델 실행 + 파일 정리 보장)
        try:
            #  Whisper 모델 추론 실행 (과제 3: language 옵션)
            # 1. ml_models["whisper"]:
            #    - 딕셔너리에서 Whisper 모델 객체 조회 (해시 테이블 O(1))
            # 2. .transcribe(경로, language=...):
            #    - Whisper 모델의 메서드 호출
            #    - 내부: 오디오 로드 -> 스펙트로그램 변환 -> 신경망 추론 -> 텍스트 생성
            #    - language=None이면 자동 언어 감지, "ko"면 한국어로 강제 인식
            # 3. result:
            #    - 딕셔너리 객체 반환 {"text": ..., "language": ..., "segments": [...]}
            result = ml_models["whisper"].transcribe(temp_file_path, language=language)
            
            # [개념] 왜 리스트 컴프리헨션을 쓰는가? (과제 2 구현 패턴)
            # - Whisper 원본 segments 구조:
            #   {"id": 0, "seek": 0, "start": 0.0, "end": 2.5, "text": "안녕",
            #    "tokens": [1, 2, 3], "temperature": 0.0, "avg_logprob": -0.5, ...}
            # - 문제점:
            #   * 불필요한 정보가 너무 많음 (tokens, temperature 등)
            #   * 응답 크기가 커짐 (네트워크 낭비)
            # - 해결책:
            #   * start, end, text만 추출하여 깔끔한 응답 제공
            #   * 리스트 컴프리헨션: 반복문을 한 줄로 간결하게 표현
            # - 대안 (for 루프):
            #   segments = []
            #   for s in result["segments"]:
            #       segments.append({"start": s["start"], ...})
            #   위 방식도 동일하지만, 컴프리헨션이 더 "Python다움" (Pythonic)
            
            #  리스트 컴프리헨션 (과제 2: 타임스탬프 정보 가공)
            # 1. [표현식 for 변수 in 리스트]:
            #    - 반복문을 한 줄로 압축한 문법 (메모리 효율적)
            #    - result["segments"]의 각 요소를 s에 할당하며 반복
            # 2. {...} 딕셔너리 리터럴:
            #    - 각 반복마다 새로운 딕셔너리 객체 생성
            #    - s["start"], s["end"], s["text"]만 추출 (불필요한 정보 제거)
            # 3. 왜 이렇게?
            #    - Whisper 원본 segments에는 id, seek, tokens 등 쓸데없는 정보 많음
            #    - 과제 요구사항: start, end, text만 포함하도록 필터링
            segments = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"]
                } for s in result["segments"]
            ]
            
            #  응답 딕셔너리 생성 및 반환
            # 1. return {...}:
            #    - 딕셔너리 객체를 생성하여 함수 호출자에게 반환
            #    - FastAPI가 이를 받아 자동으로 JSON으로 직렬화하여 HTTP 응답 전송
            # 2. "키": 값 쌍:
            #    - 각 줄마다 키-값 쌍을 딕셔너리에 추가
            #    - "segments": segments -> 위에서 가공한 리스트를 segments 키에 할당
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": segments
            }
        # [개념] 왜 Try-Finally로 리소스를 정리하는가?
        # - 문제점:
        #   * 임시 파일을 삭제하지 않으면?
        #   * 하루 100개 에러 → 100개 쓰레기 파일 (수십 GB)
        #   * 일주일이면 디스크 가득 참 → 서버 다운
        # - 해결책: Finally 블록
        #   * try 성공하든 except 에러나든 "무조건" 실행
        #   * 임시 파일 삭제 보장
        # - 실무 필수 패턴:
        #   * 파일 처리 (업로드, 다운로드, 변환)
        #   * DB 커넥션 (반드시 close 호출)
        #   * 외부 API 호출 (타임아웃 처리)
        # - Python의 with문도 내부적으로 try-finally 사용
        
        #  finally 블록 (리소스 정리 보장)
        # 1. finally:
        #    - try 블록이 성공하든, 에러가 나든 "무조건" 실행되는 블록
        #    - 파일 삭제, DB 연결 종료 등 "정리 작업"에 필수
        # 2. 왜 필요?
        #    - 임시 파일을 안 지우면 디스크가 가득 차서 서버 다운
        #    - return으로 함수가 끝나도 finally는 실행됨 (보장)
        finally:
            #  파일 존재 확인 + 삭제
            # 1. os.path.exists(경로):
            #    - 파일 시스템에 해당 경로가 존재하는지 확인 (True/False)
            # 2. os.remove(경로):
            #    - 파일 시스템에서 파일 삭제 (inode 제거)
            # 3. if로 감싸는 이유:
            #    - 파일이 없는데 remove 호출하면 에러 발생 방지
            if os.path.exists(temp_path):
                os.remove(temp_path)

    #  except 블록 (예외 처리)
    # 1. except Exception as e:
    #    - try 블록에서 발생한 모든 예외를 잡아냄
    #    - 예외 객체를 e라는 변수에 바인딩
    # 2. 왜 필요?
    #    - 에러 발생 시 서버가 멈추지 않고, 사용자에게 유용한 에러 메시지 전달
    except Exception as e:
        #  f-string (포맷 문자열)
        # 1. f"...{변수}...":
        #    - 중괄호 안의 Python 표현식을 문자열로 변환하여 삽입
        #    - str(e): 예외 객체를 문자열로 변환
        #    - traceback.format_exc(): 에러 발생 위치 전체 스택 트레이스 문자열로 반환
        # 2. \n:
        #    - 줄바꿈 문자 (escape sequence)
        error_msg = f"서버 에러 발생: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        #  HTTP 에러 응답 반환
        # 1. JSONResponse(...):
        #    - FastAPI의 특수 응답 객체 생성
        #    - status_code=500: "Internal Server Error" 상태 코드
        #    - content={...}: 응답 본문을 딕셔너리로 지정 (자동으로 JSON 변환)
        # 2. return:
        #    - 이 객체를 반환하면 FastAPI가 클라이언트에게 에러 응답 전송
        return JSONResponse(status_code=500, content={"detail": error_msg, "hint": "ffmpeg might not be installed please check the PATH"})
