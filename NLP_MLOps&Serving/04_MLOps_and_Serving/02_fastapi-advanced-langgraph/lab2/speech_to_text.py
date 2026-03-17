"""
🎯 FastAPI 실습: OpenAI Whisper를 활용한 음성-텍스트 변환: Speech-to-Text (STT)

===============================================
📋 이 파일이 하는 일 (전체 목적)
===============================================
이 파일은 **OpenAI Whisper 모델을 사용하여 음성 파일을 텍스트로 변환**하는 실습입니다.
오디오 파일을 업로드하면 AI가 음성을 인식하여 텍스트로 변환해주는 Speech-to-Text API를 만듭니다.

===============================================
🤔 왜 이렇게 하는가? (설계 의도)
===============================================
1. **파일 업로드 처리의 특수성**:
   - 텍스트와 달리 오디오는 바이너리 데이터
   - 크기가 큼 (수 MB)
   - Whisper는 "파일 경로"를 입력으로 요구

2. **임시 파일을 쓰는 이유**:
   - Whisper API: transcribe(file_path) 형식
   - 메모리 스트림을 직접 못 받음
   - 따라서 잠깐 디스크에 저장 필요

3. **cleanup이 필수인 이유**:
   - 임시 파일을 안 지우면 디스크 폭발
   - 하루 1000개 요청 = 수십 GB 쓰레기
   - finally 블록으로 "반드시" 삭제 보장

===============================================
🔄 전체 실행 흐름 (워크플로우)
===============================================
[서버 시작]
1. Whisper "base" 모델 다운로드 (처음 한 번, 140MB)
2. 메모리에 모델 로드

[요청 처리]
1. 클라이언트가 오디오 파일 업로드 (mp3, wav 등)
2. /tmp에 임시 파일 생성
3. 업로드 스트림 → 임시 파일로 복사
4. Whisper로 추론:
   오디오 → 스펙트로그램 → 신경망 → 텍스트
5. 결과 반환 (텍스트 + 감지된 언어)
6. **finally: 임시 파일 삭제** (필수!)

===============================================
💡 핵심 학습 포인트
===============================================
- UploadFile: FastAPI의 파일 업로드 처리
- TemporaryFile: 임시 파일 생성/관리
- try-finally: 리소스 정리 보장
- shutil.copyfileobj: 스트림 복사
- Whisper: 다국어 음성 인식 (한국어 포함)

===============================================
📌 사전 준비 및 실행 가이드
===============================================
1. 사전 준비:
   - 시스템에 ffmpeg 설치
     * Mac: brew install ffmpeg
     * Ubuntu: apt install ffmpeg
     * Windows: ffmpeg 다운로드 후 Path 설정
   - pip install openai-whisper python-multipart

2. 실행 방법:
   python ./lab2/speech_to_text.py
"""

#  필수 모듈 임포트
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import traceback
from contextlib import asynccontextmanager
#  Whisper 모듈 임포트
import whisper                  # https://pypi.org/project/openai-whisper/
#  파일 관리 유틸리티 임포트
import shutil,os, tempfile

#  FFmpeg 자동 탐지 및 경로 추가 (Windows)
# 사용자가 다운로드한 폴더 내부에 중첩된 폴더가 있을 수 있으므로(예: bin 폴더),
# 자동으로 ffmpeg.exe 위치를 찾아서 PATH에 등록합니다.
#  FFmpeg 찾기 및 경로 설정
# find_by_name 도구로 확인된 확실한 경로를 추가합니다.
def setup_ffmpeg_path():
    # 1. 2026-01-12 확인된 경로: C:\ffmpeg\bin
    confirmed_path = r"C:\ffmpeg\bin"
    
    # 2. 콘다 환경 경로 (백업)
    conda_path = r"C:\Users\daboi\anaconda3\envs\yolo_env_py311\Library\bin"

    print(f"🔧 FFmpeg 경로 설정 시도: {confirmed_path}")
    
    # 경로가 존재하면 PATH에 가장 앞에 추가 (우선순위 높임)
    if os.path.exists(confirmed_path):
        os.environ["PATH"] = confirmed_path + ";" + os.environ["PATH"]
        print(f"✅ PATH에 추가됨: {confirmed_path}")
    elif os.path.exists(conda_path):
        # 백업 경로 사용
        os.environ["PATH"] = conda_path + ";" + os.environ["PATH"]
        print(f"✅ (Conda) PATH에 추가됨: {conda_path}")
    else:
        print("❌ 경고: 확인된 FFmpeg 경로를 찾을 수 없습니다.")

    # [진단] 실제 호출 가능한지 테스트
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🎉 FFmpeg 실행 테스트 성공! (준비 완료)")
    except Exception as e:
        print(f"🔥 FFmpeg 실행 테스트 실패: {e}")
        print("👉 브라우저에서 500 에러가 발생하면 이 로그를 확인하세요.")

setup_ffmpeg_path()

#  전역 모델 변수
ml_models = {}

#  Lifespan 컨텍스트 매니저
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시 실행
    print("====== 모델 로딩중 ....")
    
    #  Whisper 모델 로딩
    # 'base' 모델은 가볍고 빠름 (정확도를 높이려면 'small', 'medium' 사용)
    # 1. whisper.load_model("base"):
    #    a) ~/.cache/whisper에서 모델 파일 확인
    #    b) 없으면 인터넷에서 다운로드 (약 140MB)
    #    c) 메모리에 PyTorch 모델 로드
    # 2. Whisper 모델: OpenAI가 만든 다국어 음성 인식 모델
    #    - 99개 언어 지원 (한국어 포함)
    #    - 자동 언어 감지
    ml_models["whisper"] = whisper.load_model("base")
    
    print("✅ 모델 로딩 완료!")
    
    #  yield - 서버 실행 대기
    yield
    
    # [Shutdown] 서버 종료 시 메모리 정리
    ml_models.clear()

#  FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# [개념] 왜 async 함수를 쓰는가? (파일 업로드는 I/O 작업)
# - 파일 업로드의 특성:
#   * 클라이언트가 수 MB 파일을 천천히 전송 (1~10초 소요)
#   * 이 시간 동안 서버가 블로킹되면? → 다른 요청 처리 불가
# - async/await의 장점:
#   * 파일 수신 대기 중 다른 사용자 요청 처리 가능
#   * 동시에 10명이 파일 업로드 → 모두 병렬 처리
# - 실무: 파일 업로드, 다운로드 등 모든 I/O 작업은 비동기 필수

#  API 엔드포인트
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    #  임시 파일 생성 및 업로드 데이터 복사
    # 1. 업로드된 파일을 임시 파일로 저장 (Whisper는 파일 경로를 요구함)
    # 2. with 문: 컨텍스트 매니저로 파일 핸들 자동 관리 (__enter__, __exit__)
    # 3. tempfile.NamedTemporaryFile():
    #    - OS 임시 폴더(예: /tmp 또는 C:\Users\...\AppData\Local\Temp)에 파일 생성
    #    - 고유한 랜덤 파일명 생성 (충돌 방지)
    #    - 커널로부터 파일 디스크립터(fd) 할당받음
    #    - delete=False: close() 후에도 파일 삭제 안 함 (직접 삭제 필요)
    #    - suffix=".mp3": 파일 끝에 확장자 붙임
    #  에러 캡처 래퍼 (브라우저에 에러 내용 표시)
    try:
        #  임시 파일 생성 및 업로드 데이터 복사
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            #  스트림 복사 (Memory -> Disk)
            # 1. file.file: SpooledTemporaryFile 객체 (메모리 또는 디스크상의 임시 파일)
            # 2. shutil.copyfileobj(src, dst):
            #    - 청크 단위(기본 16KB ~ 1MB)로 읽고 쓰기 반복
            #    - User Space 버퍼 -> Kernel Space -> Disk Write
            #    - 메모리 효율적 (큰 파일도 한꺼번에 메모리에 올리지 않음)
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        #  모델 조회 및 추론
        try:
            model = ml_models["whisper"]
            # 💡 여기서 에러가 나면 아래 except 블록으로 이동합니다.
            result = model.transcribe(temp_path)

            return {
                "filename": file.filename,
                "text": result["text"],
                "language": result["language"]
            }
        finally:
            # [Cleanup] 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
        # [개념] 왜 Try-Finally로 에러를 처리하는가?
        # - 문제: 에러 발생 시 임시 파일이 삭제되지 않으면?
        #   * 하루 100개 에러 → 100개 쓰레기 파일 (수십 GB)
        #   * 일주일이면 디스크 가득 참 → 서버 다운
        # - 해결: Finally 블록
        #   * try 성공하든 except 에러나든 "무조건" 실행
        #   * 임시 파일 삭제 보장
        # - 실무 필수 패턴:
        #   * 파일 처리 (업로드, 다운로드, 변환)
        #   * DB 커넥션 (반드시 close 호출)
        #   * 외부 API 호출 (타임아웃 처리)
        # - Python의 with문도 내부적으로 try-finally 사용

    except Exception as e:
        # 🔥 에러 발생 시 브라우저에 자세한 내용을 출력합니다 (디버깅용)
        error_msg = f"서버 에러 발생: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg) # 서버 로그에도 출력
        return JSONResponse(status_code=500, content={"detail": error_msg, "hint": "ffmpeg가 설치되었나요? PATH를 확인하세요."})



#  메인 블록
if __name__ == "__main__":
    import uvicorn
    #  서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)