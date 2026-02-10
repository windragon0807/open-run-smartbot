"""
Smart Chatbot - FastAPI 서버
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from schemas import (
    ChatRequest, ChatResponse, TokenUsage, StatusResponse, ModelsResponse,
    DocumentListResponse, DocumentInfo, SyncResponse, ResetResponse,
    RAGRequest, RAGResponse, SourceDocument,
)
from rag.document_loader import list_knowledge_files
from rag.vector_store import reset as reset_db
from rag.watcher import watch_knowledge_folder, sync_all
from rag.chain import query as rag_query

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 라이프사이클 관리"""
    # 서버 시작: knowledge/ 폴더 감시 백그라운드 태스크 실행
    watcher_task = asyncio.create_task(watch_knowledge_folder())
    yield
    # 서버 종료: 감시 태스크 취소
    watcher_task.cancel()


# FastAPI 앱 생성
app = FastAPI(
    title="Smart Chatbot API",
    description="OpenAI GPT 모델을 활용한 챗봇 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (프론트엔드 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API 엔드포인트 ===

@app.get("/", response_model=StatusResponse)
def root():
    """서버 상태 확인"""
    return StatusResponse(status="running", message="Smart Chatbot API 서버가 실행 중입니다.")


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """사용 가능한 GPT 모델 목록 조회"""
    try:
        models = client.models.list()
        gpt_models = sorted([m.id for m in models if "gpt" in m.id.lower()])
        return ModelsResponse(models=gpt_models, count=len(gpt_models))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """챗봇에게 메시지를 보내고 응답을 받습니다."""
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 친절하고 도움이 되는 한국어 AI 어시스턴트입니다.",
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        return ChatResponse(
            reply=response.choices[0].message.content,
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === RAG 엔드포인트 ===

@app.get("/documents", response_model=DocumentListResponse)
def get_documents():
    """knowledge/ 폴더의 문서 목록을 조회합니다."""
    files = list_knowledge_files()
    return DocumentListResponse(
        documents=[DocumentInfo(**f) for f in files],
        count=len(files),
    )


@app.post("/documents/sync", response_model=SyncResponse)
def sync_documents():
    """knowledge/ 폴더의 모든 문서를 수동으로 벡터 DB에 동기화합니다."""
    try:
        result = sync_all()
        return SyncResponse(
            synced_files=result["synced_files"],
            total_chunks=result["total_chunks"],
            message=f"{result['synced_files']}개 파일, {result['total_chunks']}개 청크가 동기화되었습니다.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/reset", response_model=ResetResponse)
def reset_documents():
    """벡터 DB를 초기화합니다. knowledge/ 폴더의 파일은 유지됩니다."""
    try:
        reset_db()
        return ResetResponse(message="벡터 DB가 초기화되었습니다. /documents/sync로 다시 동기화할 수 있습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGResponse)
def rag_question(request: RAGRequest):
    """knowledge/ 폴더의 문서를 기반으로 질문에 답변합니다."""
    try:
        result = rag_query(request.question)
        return RAGResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
