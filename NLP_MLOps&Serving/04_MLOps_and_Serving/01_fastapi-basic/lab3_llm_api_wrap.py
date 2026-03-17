import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="My LLM API Server")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ]
            }]
        }
    }

class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict

@app.get("/")
def home():
    return {"message": "LLM API Server is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return ChatResponse(
            response=response.choices[0].message.content,
            model=request.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": request.question}
            ]
        )

        return AskResponse(answer=response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))