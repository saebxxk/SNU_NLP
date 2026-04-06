import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 기존 모듈을 불러오기 위해 경로 추가
QA_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samsung_audit_qa_v2"))
if QA_ROOT_PATH not in sys.path:
    sys.path.insert(0, QA_ROOT_PATH)

from chatbot import AuditReportChatbot
from db_schema import DB_PATH

app = FastAPI(title="Samsung Audit QA Web API")

# CORS 설정 (프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 챗봇 인스턴스 (메모리에 유지)
# 실제 서비스라면 세션별로 관리해야 하지만, 테스트 환경이므로 전역 변수로 관리
chatbot = AuditReportChatbot(db_path=DB_PATH)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    status: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.query.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
            
        answer = chatbot.chat(user_input, verbose=True)
        return ChatResponse(answer=answer, status="success")
    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(answer=f"오류가 발생했습니다: {str(e)}", status="error")

@app.get("/history")
async def get_history():
    return {"history": chatbot.conversation.history}

@app.post("/clear")
async def clear_history():
    chatbot.conversation.clear()
    return {"message": "대화 기록이 초기화되었습니다."}

if __name__ == "__main__":
    # 서버 실행: python backend.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
