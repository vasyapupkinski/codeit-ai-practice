from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="Pydantic Data Validation Practice")

class UserCreate(BaseModel):
    username: str
    email: str
    age: int
    
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
    id: int
    username: str
    email: str
    age: int
    created_at: str

fake_db = []
user_id_counter = 1

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    global user_id_counter
    new_user = {
        "id": user_id_counter,
        "username": user.username,
        "email": user.email,
        "age": user.age,
        "created_at": datetime.now().isoformat()
    }
    fake_db.append(new_user)
    user_id_counter += 1

    return new_user

@app.get("/users", response_model=list[UserResponse])
def get_users():
    return fake_db

class ProductCreate(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100,
        description="상품명"
    )
    price: int = Field(
        gt=0,
        le=10000000,
        description="가격"
    )
    quantity: int = Field(
        ge=0,
        description="재고"
    )
    category: str = Field(
        pattern=r"^(전자제품|의류|식품|기타)$",
        description="카테고리"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="상품 설명 (선택)"
    )

products_db = []
product_id_counter = 1

@app.post("/products")
def create_product(product: ProductCreate):
    global product_id_counter
    new_product = {
        "id": product_id_counter,
        **product.model_dump(),
        "created_at": datetime.now().isoformat()
    }
    products_db.append(new_product)
    product_id_counter += 1

    return {
        "message": "Product created successfully",
        "product": new_product
    }

@app.get("/products")
def get_products():
    return {"products": products_db, "total": len(products_db)}

class ChatMessage(BaseModel):
    role: str = Field(description="Message role")
    content: str = Field(min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"Invalid role: {value}, role should be one of {allowed_roles}")
        return v

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        min_length=1,
        description="List of chat messages"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for the model"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="Maximum number of tokens"
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
    return {
        "message": "Chatting request received",
        "recieved": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "last_message": request.messages[-1].content
        }
    }

class UserInternal(BaseModel):
    id: int
    username: str
    email: str
    password_hash: str
    is_admin: bool

class UserPublic(BaseModel):
    id: int
    username: str

internal_users = [
    UserInternal(
        id=1,
        username="admin",
        email="admin@example.com",
        password_hash="hashed_secret_123",
        is_admin=True
    ),
    UserInternal(
        id=2,
        username="user1",
        email="user1@test.com",
        password_hash="hashed_password_456",
        is_admin=False
    )
]

@app.get("/users/{user_id}/public", response_model=UserPublic)
def get_user_public(user_id: int):
    for user in internal_users:
        if user.id == user_id:
            return user
    
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/users/{user_id}/internal", response_model=UserInternal)
def get_user_internal(user_id: int):
    for user in internal_users:
        if user.id == user_id:
            return user
    
    raise HTTPException(status_code=404, detail="User not found")


book_db = []
book_id_counter = 1

class BookCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    author: str
    isbn: str = Field(pattern=r"^[0-9]{13}$", description="13 ISBN digits")
    price: int = Field(gt=999, le=1000000, description="Price")
    published_year: int = Field(ge=1900, le=2025, description="Published year")
    genre: str = Field(pattern=r"^(소설|비문학|자기계발|기술)$", description="Genre")

class BookResponse(BaseModel):
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
    global book_id_counter
    new_book = {
        "id": book_id_counter,
        **book.model_dump(),
        "created_at": datetime.now().isoformat()
    }
    book_db.append(new_book)
    book_id_counter += 1
    return new_book