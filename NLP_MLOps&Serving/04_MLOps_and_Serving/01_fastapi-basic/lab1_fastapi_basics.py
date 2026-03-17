"""
🎯 FastAPI 실습1: 기본기 다지기
============================================
이 파일은 "웹 서버"라는 것을 내 컴퓨터에 띄우는 가장 기초적인 실습입니다.

[전체 흐름 요약]
1. FastAPI라는 도구를 가져와서 (import)
2. '앱(app)'이라는 이름의 서버 객체를 만들고 (인스턴스 생성)
3. "@app.get(...)" 같은 스티커(데코레이터)를 붙여서
    - "이 주소로 요청이 오면, 저 함수를 실행해줘!" 라고 연결합니다.
4. 마지막으로 uvicorn이 이 'app'을 실행시키면 서버가 켜집니다.

📌 실행 방법:
터미널에 아래 명령어를 입력하세요.
uvicorn lab1_fastapi_basics:app --reload
(해석: lab1_fastapi_basics 파일 안에 있는 app 객체를 실행해라. --reload는 코드 고치면 알아서 껐다 켜라는 뜻)
"""

# ============================================
# 0단계: 필수 라이브러리 가져오기 (Import)
# ============================================
# [기계적 동작]
# 1. 파이썬 인터프리터가 이 줄을 읽으면, 설치된 패키지 폴더에서 'fastapi'를 찾습니다.
# 2. fastapi 패키지 안의 FastAPI 클래스(설계도)를 메모리에 로드합니다.
# 3. 이제 이 파일 어디서든 'FastAPI'라는 이름으로 그 클래스를 사용할 수 있게 됩니다.
from fastapi import FastAPI

# ============================================
# 1단계: FastAPI 앱 인스턴스 생성
# ============================================
# [기계적 동작 - 아주 중요!]
# 1. FastAPI() 생성자를 호출하면, 메모리에 FastAPI 객체가 만들어집니다.
# 2. title, description, version은 자동 생성 문서(Swagger UI)에 표시될 정보입니다.
# 3. 이 객체(app)는 앞으로 모든 API 엔드포인트의 "중앙 관제탑" 역할을 합니다.
# 4. 'app'이라는 변수명은 관례입니다. (다른 이름도 가능하지만 보통 app으로 씁니다)
app = FastAPI(
    title="첫번째 FastAPI",           # 문서 제목
    description="AI 엔지니어 FastAPI 실습 1",  # 문서 설명
    version="1.0.0"                    # API 버전
)

# ============================================
# 1단계: 가장 기본적인 GET 요청
# ============================================
# @app.get("/") 의 의미:
# - @: 데코레이터. "아래 있는 함수를 꾸며준다/등록한다"는 뜻.
# - app.get: "누군가 정보를 달라고(GET) 요청하면"
# - ("/"): "인터넷 주소 뒤에 아무것도 안 붙은 메인 화면(/)으로 들어오면"
@app.get("/")
def home():
    """루트 경로 - 서버가 잘 동작하는지 확인"""
    # 동작 순서:
    # 1. 사용자가 브라우저 주소창에 http://localhost:8000/ 입력 (GET 요청)
    # 2. FastAPI가 "/" 주소를 보고 이 home() 함수를 찾아냄 (라우팅)
    # 3. 함수 실행 -> {"message": ...} 파이썬 딕셔너리 리턴
    # 4. [중요] FastAPI가 이 딕셔너리를 'JSON'이라는 텍스트 형식으로 자동 변환 (Serialization)
    # 5. 브라우저에게 최종적으로 텍스트 데이터를 전송 (Response)
    return {"message": "FastAPI 서버입니다!"}

@app.get("/health")
def health_check():
    """헬스체크 - 서버 상태 확인용 (실무에서 필수!)"""
    # 비유: 친구가 "야 너 자니?"(GET /health) 라고 물어보면 "아니 깨어있어"(healthy)라고 답장하는 것과 같음.
    # 클라우드(AWS 등)에서는 이 주소를 주기적으로 찔러보며 서버가 죽었나 살았나 확인함.
    return {"status": "healthy"}


# ============================================
# 2단계: 경로 매개변수 (Path Parameter)
# URL 경로에 변수를 포함시키는 방식
# ============================================
# 상황: 사용자가 한두 명이 아니라 100명, 1000명인데 주소를 다 따로 만들 수는 없음.
# 해결: "/users/{변수}" 처럼 구멍을 뚫어놓고, 그 자리에 들어온 값을 함수로 넘겨받음.

@app.get("/users/{user_id}")  # {user_id} 자리에 뭐가 들어오든 아래 함수의 user_id 변수로 쏙 들어감!
def get_user(user_id: int):   # 타입 힌트(int)를 적으면, FastAPI가 알아서 문자를 숫자로 바꿔줌 (자동 변환)
    """
    [기계적 작동 원리]
    1. 요청: GET /users/123
    2. 매칭: "/users/{user_id}" 패턴과 일치함. user_id 문자열 "123"을 추출.
    3. 검증 및 변환: 
       - 함수 인자에 `user_id: int`라고 적혀있음.
       - FastAPI가 "123"(문자)을 123(숫자)으로 `int("123")` 실행해서 변환.
       - 실패하면(예: "abc") 에러, 성공하면 변수에 대입.
    4. 실행: `get_user(user_id=123)` 함수 실행
    """
    return {
        "user_id": user_id,
        "message": f"{user_id}번 사용자 정보입니다"
    }

@app.get("/items/{item_name}")
def get_item(item_name: str):
    # [경로 파라미터 - 문자열 버전]
    # 이번에는 숫자(int)가 아닌 문자열(str) 그대로 받는 버전입니다.
    # 예: /items/apple -> item_name = "apple"
    # FastAPI는 타입에 맞게 자동 변환하지만, 문자열은 그냥 그대로 전달합니다.
    
    # [딕셔너리 생성 및 리턴]
    # 1. 메모리에 딕셔너리 객체를 생성합니다.
    # 2. item_name 변수 값을 딕셔너리에 담습니다.
    # 3. f-string으로 메시지를 만들어 담습니다.
    # 4. FastAPI가 이 딕셔너리를 JSON으로 변환해서 응답합니다.
    return {
        "item_name": item_name,
        "message": f"'{item_name}' 아이템을 조회합니다."
    }

# ============================================
# 3단계: 쿼리 매개변수 (Query Parameter)
# URL 뒤에 ?key=value 형태로 전달
# ============================================
# 경로 파라미터({path})가 아닌데 함수 인자로 적혀있는 녀석들은
# 자동으로 "쿼리 파라미터(물음표 뒤에 붙는 옵션)"로 인식됩니다.

@app.get("/search")
def search_items(
    keyword: str,           # 필수! (기본값이 없으므로 안 보내면 에러 남)
    limit: int = 10,        # 선택 (안 보내면 자동으로 10이 들어감)
    skip: int = 0           # 선택 (안 보내면 자동으로 0이 들어감)
):
    """
    [기계적 작동 원리]
    1. 요청: GET /search?keyword=파이썬&limit=5
    2. 해석:
       - keyword="파이썬" (찾아서 넣음)
       - limit=5 (찾아서 넣음)
       - skip? (요청에 없네? -> 기본값 0 사용)
    3. 실행: search_items(keyword="파이썬", limit=5, skip=0)
    """
    return {
        "keyword": keyword,
        "limit": limit,
        "skip": skip,
        "message": f"{keyword}로 검색, {skip}번째부터 {limit}개 조회"
    }

# ============================================
# 4단계: 경로 + 쿼리 매개변수 조합
# ============================================
@app.get("/categories/{category}/products")
def get_products_by_category(
    category: str,          # 경로 매개변수 (주소에 {category} 있음)
    min_price: int = 0,     # 쿼리 매개변수 (주소에 없음 -> ?min_price=... 로 받음)
    max_price: int = 100000,
    sort_by: str = "name"
):
    """
    실전 예시: 카테고리별 상품 조회
    
    [상황 예시]
    요청: GET /categories/electronics/products?min_price=1000&sort_by=price
    
    [매핑 과정]
    1. URL의 'electronics' -> category 변수로
    2. ?뒤의 'min_price=1000' -> min_price 변수로 (문자 "1000"을 숫자 1000으로 자동 변환)
    3. ?뒤의 'sort_by=price' -> sort_by 변수로
    4. max_price는 없으니까 -> 기본값 100000 사용
    """
    return {
        "category": category,
        "filters": {
            "min_price": min_price,
            "max_price": max_price,
            "sort_by": sort_by
        },
        "message": f"{category} 카테고리 상품 조회"
    }



# ============================================
# 체크포인트 : API 문서 자동 생성 확인하기
# ============================================
"""
FastAPI는 자동으로 API 문서를 생성합니다!

🔹 Swagger UI: http://localhost:8000/docs
   - 개발자가 테스트하기 딱 좋은 화면
   - "Try it out" 버튼 누르면 실제 요청을 날려볼 수 있음

🔹 ReDoc: http://localhost:8000/redoc
   - 깔끔하게 문서만 보고 싶을 때 사용
"""


# ============================================
# 혼자해보기 1
# ============================================
"""
아래 엔드포인트를 직접 만들어보기:

1. GET /greeting/{name}
   - 경로로 이름을 받아서 "안녕하세요, {name}님!" 반환
   
2. GET /calculate
   - 쿼리 파라미터: a (int), b (int), operation (str, 기본값="add")
   - operation이 "add"면 a+b, "multiply"면 a*b 반환

3. GET /movies/{genre}/list
   - 경로: genre (str)
   - 쿼리: year (int, 선택), rating (float, 기본값=0.0)
   - 필터 조건과 함께 메시지 반환
"""

@app.get("/greeting/{name}")
def greeting(name: str):
    """
    [문제 1: 경로 파라미터 실습]
    
    [구동 과정]
    1. 요청 도착: /greeting/철수
    2. URL 파싱: {name} 자리에 "철수"가 캡처됨.
    3. 함수 실행: greeting(name="철수")
    """
    # [응답 생성]
    # 1. f-string으로 "Hello 철수" 문자열 생성
    # 2. {"message": "Hello 철수"} 딕셔너리 생성
    # 3. JSON으로 변환되어 반환
    return {"message": f"Hello {name}"}

@app.get("/calculate")
def calculate(
    a: int,              # [쿼리 파라미터] ?a=10 -> 숫자 10으로 변환
    b: int,              # [쿼리 파라미터] ?b=5 -> 숫자 5로 변환
    operation: str = "add"  # [선택 파라미터] 안 보내면 "add"가 기본값
):
    """
    [문제 2: 계산기 API 실습]
    
    
    1. 쿼리 스트링 파싱: ?a=10&b=20&operation=multiply
    2. 타입 변환: 문자열 "10", "20"을 정수(int)로 변환
    3. 로직 분기: operation 값에 따라 다른 연산 수행
    """
    
    # [조건 분기 (Branching)]
    # operation 변수의 값이 "add"와 같은지 메모리상에서 문자열 비교
    if operation == "add":
        # [덧셈 연산]
        # 1. a와 b의 값을 CPU 레지스터로 로드
        # 2. 덧셈 수행 (ALU)
        # 3. 결과값을 사용하여 딕셔너리 생성
        return {"result": a + b}
        
    elif operation == "multiply":
        # [곱셈 연산]
        # a와 b를 곱한 결과를 리턴
        return {"result": a * b}
        
    else:
        # [예외 처리]
        # 약속되지 않은 operation이 오면 에러 메시지 반환
        return {"result": "Invalid operation"}

@app.get("/movies/{genre}/list")
def get_movies_by_genre(
    genre: str,             # [경로 파라미터] URL의 {genre} 부분
    year: int = None,       # [쿼리/선택] ?year=2023 (없으면 None)
    rating: float = 0.0     # [쿼리/선택] ?rating=8.5 (없으면 0.0)
):
    """
    [문제 3: 복합 파라미터 실습]
    경로 파라미터(장르)와 쿼리 파라미터(필터)를 섞어서 사용
    """
    # [응답 구성]
    # 받은 모든 파라미터를 그대로 딕셔너리에 담아 리턴 (Echo Server)
    # 메모리에서 일어나는 일:
    # 1. 딕셔너리 객체 힙(Heap) 메모리에 할당
    # 2. genre, year, rating 변수의 값을 키(key)에 매핑
    # 3. f-string 포맷팅 실행
    return {
        "genre": genre,
        "year": year,
        "rating": rating,
        "message": f"Movies in {genre} genre"
    }