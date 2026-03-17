"""
🎯 FastAPI 실습2: 데이터 검증의 파수꾼, Pydantic
============================================
이 파일은 "데이터 입국 심사"를 하는 방법을 배웁니다.
API 서버를 만들 때 가장 귀찮고 힘든 게 "사용자가 이상한 값을 보내면 어쩌지?" 하는 걱정입니다.
FastAPI는 'Pydantic'이라는 도구를 써서 이 걱정을 완벽하게 해결해 줍니다.

[핵심 개념: Pydantic]
- 클럽 입구의 바운서(Bouncer)라고 생각하면 됩니다.
- "신분증(필수 데이터) 보여주세요", "미성년자(잘못된 값)는 안됩니다" 라고 알아서 막아줍니다.
- 우리가 할 일은 "입장 조건(모델)"만 적어두면 됩니다.

📌 실행 방법:
uvicorn lab2_pydantic_validation:app --reload
"""

# ============================================
# 0단계: 필수 라이브러리 가져오기 (Import)
# ============================================
# [각 라인의 역할]

# FastAPI: 웹 서버 프레임워크
# HTTPException: 에러 발생 시 클라이언트에게 정확한 에러 코드(404, 500 등)를 보내기 위한 도구
from fastapi import FastAPI, HTTPException

# BaseModel: Pydantic의 핵심! 이걸 상속받으면 자동 검증 기능이 활성화됨
# Field: 필드에 상세 규칙(min_length, max_length 등)을 적용할 때 사용
# field_validator: 파이썬 함수로 직접 커스텀 검증 로직을 작성할 때 사용
from pydantic import BaseModel, Field, field_validator

# Optional: "이 값은 있을 수도 있고 None일 수도 있다"를 표현할 때 사용
# List: "이 값은 리스트(배열)다"를 타입 힌트로 표현할 때 사용
from typing import Optional, List

# datetime: 현재 시간을 기록할 때 사용 (created_at 같은 타임스탬프)
from datetime import datetime

# ============================================
# 1단계: FastAPI 앱 생성
# ============================================
# [기계적 동작]
# 메모리에 FastAPI 객체를 만들고, 'app'이라는 변수에 저장합니다.
# 모든 API 엔드포인트는 이 app 객체에 등록됩니다.
app = FastAPI(title="Pydantic 데이터 검증 실습")

# ============================================
# 1단계: 기본 Pydantic 모델
# ============================================
# 기계적 동작: BaseModel을 상속받은 클래스는 "데이터 검증 설계도"가 됩니다.

class UserCreate(BaseModel):
    """사용자 생성 요청 모델"""
    # [입장 조건 명세서]
    username : str  # "username은 무조건 문자열이어야 해!"
    email: str      # "email도 문자열!"
    age: int        # "age는 무조건 숫자야! 문자열 '스물다섯' 보내면 에러 낼 거야"

    # 예시 데이터 (Swagger UI에서 자동으로 표시됨)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "홍길동",
                    "email": "hong@example.com",
                    "age": 25
                }
            ]
        }   
    }


class UserResponse(BaseModel):
    """사용자 응답 모델"""
    # 응답 보낼 때도 "이 규칙대로 포장해서 내보내라"고 정할 수 있습니다.
    id: int
    username: str
    email: str
    age: int
    created_at: str

# 간단한 인메모리 저장소 (DB 대용)
fake_db = []
user_id_counter = 1

# [중요] 함수 인자에 타입을 UserCreate로 지정했습니다.
# 동작 원리:
# 1. POST 요청의 Body(본문)에 JSON 데이터가 들어옴
# 2. FastAPI가 UserCreate 설계도를 보고 하나씩 대조함
# 3. 만약 age에 "백살" 같은 문자가 들어있다? -> 함수 시작도 안 하고 바로 에러 뱉고 돌려보냄 (422 에러)
# 4. 검사가 통과되면? -> user 변수에 깔끔하게 정리된 객체가 담겨서 함수가 실행됨
@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    """
    사용자 생성 API
    
    [상세 기계적 구동 과정]
    1. **요청 도착 (Request Arrival)**
       - 클라이언트(브라우저)가 POST /users 주소로 JSON 데이터를 보냅니다.
       - 예시 데이터: `{"username": "철수", "email": "a@a.com", "age": 20}`
    
    2. **Pydantic 검문소 (Validation)**
       - FastAPI는 함수를 실행하기 전에 `user: UserCreate` 타입을 확인합니다.
       - UserCreate 설계도를 펼쳐놓고 들어온 JSON 데이터를 하나씩 대조합니다.
         - "username이 문자열인가?" -> OK
         - "email이 문자열인가?" -> OK
         - "age가 정수인가?" -> OK
       - 만약 하나라도 틀리면(예: age="스물")? -> 함수 실행 전 즉시 **422 에러**를 뱉고 종료합니다.
    
    3. **데이터 변환 (Parsing)**
       - 검사가 통과되면, JSON(딕셔너리)을 파이썬 객체(`user`)로 변환합니다.
       - 이제부터 코드 안에서는 `user.username` 처럼 점(.)을 찍어서 편하게 쓸 수 있습니다.
    
    4. **함수 실행 (Execution)**
       - 이제 비로소 `create_user` 함수가 실행됩니다.
    """
    global user_id_counter

    # [메모리 동작]
    # 새로운 딕셔너리(new_user)를 메모리에 생성합니다.
    # user 객체에서 .username, .email 값을 꺼내와서 옮겨 담습니다.
    # datetime.now()를 호출해 현재 시간을 문자열로 만들어 넣습니다.
    new_user = {
        "id": user_id_counter,
        "username": user.username,
        "email": user.email,
        "age": user.age,
        "created_at": datetime.now().isoformat()
    }
    
    # [DB 저장 시뮬레이션]
    # 전역 변수 리스트(fake_db)에 방금 만든 딕셔너리를 추가합니다.
    fake_db.append(new_user)
    
    # 다음 유저를 위해 번호표를 1 증가시킵니다. (1 -> 2)
    user_id_counter += 1

    # [응답 반환]
    # new_user 딕셔너리를 리턴하면, FastAPI가 다시 UserResponse 모델에 맞춰서 검사하고
    # 최종적으로 JSON으로 바꿔서 클라이언트에게 응답합니다.
    return new_user

@app.get("/users", response_model=List[UserResponse])
def get_users():
    """
    저장된 사용자 목록 조회
    
    [주의: 리스트 반환]
    - response_model=List[UserResponse] (UserResponse가 여러 개 들어있는 박스)
    - 이유: '목록' 조회를 하면, 당연히 1명이 아니라 여러 명이 나올 수 있으니까.
    - 리턴값: 딕셔너리들의 리스트 ([{...}, {...}])
    
    [핵심 차이점]
    - create_user: 사과 1개를 줌 (Model)
    - get_users: 사과 박스(여러 개)를 줌 (List[Model])
    이거 헷갈려서 List 안 붙이면 "왜 1개 준다해놓고 여러 개 줘?" 하고 에러 납니다!
    """
    return fake_db

# ============================================
# 2단계: 필드 검증 (Field Validation)
# ============================================
# 단순히 "문자열이다/숫자다" 만으로는 부족할 때가 있습니다.
# "0보다 커야 함", "10자 이내여야 함" 같은 디테일한 규칙을 정할 때 Field()를 씁니다.

class ProductCreate(BaseModel):
    """상품 등록 모델 - 필드 검증 포함"""
    
    name: str = Field(
        min_length=2,           # "최소 2글자는 써라"
        max_length=100,         # "너무 길게 100자 넘기지 마라"
        description="상품명"
    )
    
    price: int = Field(
        gt=0,                   # greater than 0: "0원은 안 돼" (양수만)
        le=10000000,            # less than or equal: "천만원 넘으면 안 돼"
        description="가격 (원)"
    )
    
    quantity: int = Field(
        ge=0,                   # greater than or equal: "0개 포함해서 그 이상 (음수 불가)"
        default=0,              # "값 안 보내면 0개로 칠게" (기본값)
        description="재고 수량"
    )
    
    category: str = Field(
        pattern=r"^(전자제품|의류|식품|기타)$",  # "이 4개 단어 중 하나만 허용한다" (정규표현식)
        description="카테고리"
    )
    
    description: Optional[str] = Field(
        default=None,           # "없으면 None(빈 값)으로 둬라" (선택 입력)
        max_length=500,
        description="상품 설명 (선택)"
    )

# 상품 저장소
products_db = []
product_id_counter = 1


@app.post("/products")
def create_product(product: ProductCreate):
    """
    상품 등록 API
    
    테스트해보기:
    - price에 -100 입력 → 에러! (gt=0 조건 위반)
    - name에 "A" 한 글자 입력 → 에러! (min_length=2 위반)
    """
    global product_id_counter

    new_product = {
        "id": product_id_counter,
        
        # [모델 덤프와 언패킹 설명 - 아주 중요!]
        # 1. product.model_dump()의 역할:
        #    - 현재 product 객체는 Pydantic 객체입니다. (예: ProductCreate(name="TV", price=100...))
        #    - 이걸 순수한 파이썬 딕셔너리로 바꿉니다. 
        #    -> {"name": "TV", "price": 100, "quantity": 5, ...}
        
        # 2. ** (별표 두 개)의 역할: "가방 털기"
        #    - 방금 만든 딕셔너리 안의 내용물을 와르르 쏟아서 현재 위치(new_product 딕셔너리)에 뿌립니다.
        #    - 결과적으로 아래와 똑같은 코드가 됩니다:
        #      "name": product.name,
        #      "price": product.price,
        #      "quantity": product.quantity,
        #      ... (일일이 다 안 써도 돼서 엄청 편함!)
        **product.model_dump(),
        
        "created_at": datetime.now().isoformat()
    }
    products_db.append(new_product)
    product_id_counter += 1

    return {
        "message": "상품 등록 성공",
        "product": new_product
    }

@app.get("/products")
def get_products():
    """상품 목록 조회"""
    return {"products": products_db, "total": len(products_db)}


# ============================================
# 3단계: 커스텀 Validator
# ============================================
# "길이 제한" 같은 단순한 거 말고, 복잡한 로직으로 검사해야 할 때
# 직접 파이썬 함수를 짜서 검사관으로 채용합니다 (field_validator)

class ChatMessage(BaseModel):
    """채팅 메시지 모델  - 커스텀 검증"""

    role: str = Field(description="메시지 역할")
    content: str = Field(min_length=1, description="메시지 내용")

    # @field_validator('role') : "role 필드는 내가 직접 검사하겠다" 선언
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """role은 무조건 user, assistant, system 중 하나여야 함"""
        # v: 사용자가 입력한 값
        allowed_roles = ['user','assistant','system']
        if v not in allowed_roles:
            # 기준에 안 맞으면 판사처럼 유죄 선고(ValueError 발생) -> FastAPI가 422 에러로 변환해서 알려줌
            raise ValueError(f"role은 {allowed_roles} 중 하나여야 합니다.")
        return v # 통과하면 값 그대로 리턴

class ChatRequest(BaseModel):
    """채팅 요청 모델 - 여러 필드가 결합된 복잡한 요청 예시"""
    
    # [중첩 모델 - 리스트 안에 객체]
    # messages 필드는 ChatMessage 객체들의 리스트입니다.
    # 예시: [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "반가워요"}]
    # Field의 min_length=1은 "최소 1개의 메시지는 있어야 한다"는 규칙입니다.
    messages: List[ChatMessage] = Field(
        min_length=1,           # 빈 리스트는 안 됨! 최소 1개 필요
        description="대화 메시지 목록"
    )
    
    # [온도 설정]
    # temperature는 AI의 창의성을 조절하는 0.0~2.0 사이의 실수입니다.
    # default=0.7: 사용자가 안 보내면 자동으로 0.7이 들어갑니다.
    # ge=0.0: greater or equal (0.0 이상)
    # le=2.0: less or equal (2.0 이하)
    temperature: float = Field(
        default=0.7,            # 기본값
        ge=0.0,                 # 최솟값 (0.0보다 작으면 에러)
        le=2.0,                 # 최댓값 (2.0보다 크면 에러)
        description="창의성 조절(0~2)"
    )
    
    # [토큰 제한]
    # max_tokens는 AI가 생성할 최대 단어(토큰) 개수입니다.
    # 1~4096 사이의 정수만 허용됩니다.
    max_tokens: int = Field(
        default=1000,           # 기본값: 1000토큰
        ge=1,                   # 최소 1토큰은 생성해야 함
        le=4096,                # 최대 4096토큰까지만
        description="최대 생성 토큰 수"
    )

    # 예시 데이터 (Swagger UI에서 자동으로 표시됨)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "system", "content":"당신은 친절한 AI 어시스턴트입니다."},
                        {"role": "user", "content":"안녕하세요!"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ]
        }
    }

@app.post("/chat")
def chat(request: ChatRequest):
    """
    채팅 API (아직 실제 AI 연동 전, 구조만 확인하는 버전)
    
    [구동 과정]
    1. POST 요청이 들어오면 Pydantic이 request 본문을 ChatRequest 모델로 검증합니다.
    2. 검증 통과 시 이 함수가 실행됩니다.
    3. request 객체에서 필요한 정보를 꺼내 응답 딕셔너리를 만듭니다.
    """
    # [응답 딕셔너리 생성]
    # 1. len(request.messages): messages 리스트의 길이(개수)를 세서 넣습니다.
    # 2. request.messages[-1]: 리스트의 마지막 요소(인덱스 -1)를 가져옵니다.
    # 3. .content: 그 메시지 객체의 content 필드 값을 가져옵니다.
    return {
        "message": "채팅 요청 접수",
        "received": {
            "message_count": len(request.messages),       # 메시지 개수
            "temperature": request.temperature,           # 온도 값
            "max_tokens": request.max_tokens,            # 최대 토큰
            "last_message": request.messages[-1].content  # 마지막 메시지 내용
        }
    }

# ============================================
# 4단계: 응답 모델로 필터링 (보안의 핵심!)
# ============================================
# [왜 필요한가?]
# 실제 서비스에서는 DB에 비밀번호, 권한 정보 등 민감한 데이터가 저장되어 있습니다.
# 하지만 API 응답에는 이런 민감한 정보를 포함하면 안 됩니다.
# Pydantic의 response_model을 사용하면 "이 필드만 보내라"고 필터링할 수 있습니다.

class UserInternal(BaseModel):
    """내부용 사용자 모델 - DB에 저장된 전체 정보"""
    # [모든 필드 포함]
    # 서버 내부에서는 이 모든 정보가 필요합니다.
    id: int                # 사용자 ID
    username: str          # 사용자 이름
    email: str             # 이메일 (민감정보)
    password_hash: str     # 암호화된 비밀번호 (절대 노출 금지!)
    is_admin: bool         # 관리자 여부 (보안 정보)

class UserPublic(BaseModel):
    """외부 노출용 사용자 모델 - 공개해도 안전한 정보만"""
    # [선택적 필드만 포함]
    # email, password_hash, is_admin은 여기에 없으므로
    # response_model=UserPublic일 때 자동으로 제외됩니다!
    id: int                # 사용자 ID만 공개
    username: str          # 사용자 이름만 공개

# ============================================
# 테스트용 가짜 사용자 데이터 생성
# ============================================
# [메모리에 UserInternal 객체 생성]
# 실제 DB 대신 리스트에 Pydantic 객체를 직접 넣어둡니다.
# UserInternal() 생성자를 호출하면 메모리에 객체가 만들어지고,
# 리스트에 그 객체의 참조(주소)가 저장됩니다.
internal_users = [
    # [첫 번째 사용자 객체 생성]
    # 1. UserInternal 클래스의 생성자를 호출합니다.
    # 2. 각 필드에 값을 대입하여 객체를 초기화합니다.
    # 3. 생성된 객체를 리스트에 추가합니다.
    UserInternal(
        id=1, 
        username="admin", 
        email="admin@test.com",
        password_hash="hashed_secret_123",  # 실제로는 bcrypt 등으로 암호화된 값
        is_admin=True
    ),
    # [두 번째 사용자 객체 생성]
    UserInternal(
        id=2, 
        username="user1", 
        email="user1@test.com",
        password_hash="hashed_password_456",
        is_admin=False
    )
]

# response_model=UserPublic 라고 지정하면?
# 함수가 리턴할 때는 password가 들어있더라도, FastAPI가 응답을 만들 때 필터링해서 싹 지워버림.
@app.get("/users/{user_id}/public", response_model=UserPublic)
def get_user_public(user_id: int):
    """
    사용자 공개 정보 조회
    
    !!! response_model을 지정하면 해당 필드만 응답에 포함! - 민감정보가 실수로 노출되는 것을 방지합니다.
    """
    for user in internal_users:
        if user.id == user_id:
            return user  # 여기선 user 객체(비번 포함)를 통째로 리턴하지만, 받는 사람은 비번 못 봄!
        
    raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")


@app.get("/users/{user_id}/internal", response_model=UserInternal)
def get_user_internal(user_id: int):
    """
    사용자 내부 정보 조회 (관리자용 - 실제로는 인증 필요!!!)
    """
    for user in internal_users:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")




# ============================================
# 혼자해보기 2
# ============================================
"""
[혼자해보기 2-1] BookCreate 모델 만들기:
- title: str (2~200자)
- author: str (필수)
- isbn: str (정확히 13자리 숫자) - 힌트: pattern=r"^[0-9]{13}$"
- price: int (1000원 이상, 100만원 이하)
- published_year: int (1900~2025)
- genre: str ("소설", "비문학", "자기계발", "기술" 중 하나)

[혼자해보기 2-2] POST /books 엔드포인트 만들기:
- BookCreate를 받아서 저장
- BookResponse 모델로 응답 (id, created_at 추가)
"""

# [메모리 초기화]
# 서버가 실행될 때 빈 리스트(book_db)와 카운터 변수(book_id_counter)를 메모리에 할당합니다.
book_db = []
book_id_counter = 1

class BookCreate(BaseModel):
    """도서 생성 요청 모델"""
    # [입력 데이터 검증 규칙]
    # 요청 들어온 JSON 데이터를 이 규칙대로 검사합니다.
    title: str = Field(min_length=2, max_length=200)      # 제목은 2~200자 사이
    author: str                                           # 저자는 필수 문자열
    isbn: str = Field(pattern=r"^[0-9]{13}$")             # ISBN은 정확히 숫자 13자리 (정규식 검사)
    price: int = Field(gt=1000, le=1000000)               # 가격은 1000원 초과 ~ 100만원 이하
    published_year: int = Field(ge=1900, le=2025)         # 출판년도는 1900년 ~ 2025년 사이
    genre: str = Field(pattern=r"^(소설|비문학|자기계발|기술)$") # 정해진 장르 중 하나만 허용

class BookResponse(BaseModel):
    """도서 응답 모델"""
    # [응답 데이터 직렬화]
    # 서버가 리턴하는 딕셔너리를 이 모양의 JSON으로 변환합니다.
    id: int
    title: str
    author: str
    isbn: str
    price: int
    published_year: int
    genre: str
    created_at: str

@app.post("/books", response_model=BookResponse)
def create_book(book: BookCreate):
    """
    [구동 과정 설명]
    1. 요청 수신: POST /books 요청이 오면 Pydantic이 Body를 BookCreate 모델로 검사합니다.
    2. 검증 통과: 유효한 데이터라면 이 함수가 실행됩니다.
    3. 메모리 조작: 딕셔너리를 만들고 리스트에 추가합니다.
    4. 응답 반환: BookResponse 모델에 맞춰 JSON으로 변환해 응답합니다.
    """
    # [전역 변수 접근]
    # 함수 밖의 book_id_counter 변수를 수정하기 위해 global 키워드 사용
    global book_id_counter
    
    # [새 도서 딕셔너리 생성]
    new_book = {
        "id": book_id_counter,                # 현재 ID 카운터 값 사용
        **book.model_dump(),                  # 입력받은 book 객체를 딕셔너리로 풀어서 병합 (** 언패킹)
        "created_at": datetime.now().isoformat()  # 현재 시간을 문자열로 변환해 추가
    }
    
    # [메모리 저장]
    # book_db 리스트에 새로 만든 딕셔너리를 추가합니다.
    book_db.append(new_book)
    
    # [카운터 증가]
    # 다음 책을 위해 ID 카운터를 1 증가시킵니다.
    book_id_counter += 1
    
    # [리턴]
    # 딕셔너리를 리턴하면, FastAPI가 response_model(BookResponse)을 보고
    # 자동으로 JSON으로 변환하여 클라이언트에게 보냅니다.
    return new_book