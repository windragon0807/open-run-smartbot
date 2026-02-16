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
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from schemas import (
    ChatRequest, ChatResponse, TokenUsage, StatusResponse, ModelsResponse,
    DocumentListResponse, DocumentInfo, SyncResponse, ResetResponse,
    RAGRequest, RAGResponse, SourceDocument,
    DocumentContentResponse, DocumentUpdateRequest, DocumentUpdateResponse,
    LocateRequest, LocateResponse, DocumentLocation,
    EditRequest, EditResponse,
    SyncStatusResponse, FileSyncStatus, SyncEvent, DbHealth,
)
from rag.document_loader import list_knowledge_files, KNOWLEDGE_DIR
from rag.vector_store import reset as reset_db
from rag.watcher import watch_knowledge_folder, sync_all, is_sync_ready, get_sync_status
from rag.chain import query as rag_query, locate as rag_locate, edit as rag_edit

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 라이프사이클 관리"""
    logging.info("서버 시작 완료 - 포트 8000에서 요청 대기 중")
    yield
    logging.info("서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="Smart Chatbot API",
    description="OpenAI GPT 모델을 활용한 챗봇 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 서버 시작 후 백그라운드에서 knowledge/ 폴더 감시 시작
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(watch_knowledge_folder())

# CORS 설정 (프론트엔드 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (관리 UI)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


@app.get("/documents/status", response_model=SyncStatusResponse)
def get_documents_status():
    """문서 동기화 상태를 조회합니다."""
    try:
        status = get_sync_status()
        return SyncStatusResponse(
            is_ready=status["is_ready"],
            last_sync_at=status["last_sync_at"],
            last_sync_result=status["last_sync_result"],
            synced_files=status["synced_files"],
            total_chunks=status["total_chunks"],
            db_chunk_count=status["db_chunk_count"],
            errors=status["errors"],
            file_statuses=[FileSyncStatus(**f) for f in status["file_statuses"]],
            recent_events=[SyncEvent(**e) for e in status["recent_events"]],
            db_health=DbHealth(**status["db_health"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/sync", response_model=SyncResponse)
def sync_documents():
    """knowledge/ 폴더의 모든 문서를 수동으로 벡터 DB에 동기화합니다.
    기존 컬렉션을 초기화하고 전체를 다시 저장합니다."""
    try:
        result = sync_all(force_reset=True)
        errors = result.get("errors", [])
        msg = f"{result['synced_files']}개 파일, {result['total_chunks']}개 청크가 동기화되었습니다."
        if errors:
            msg += f" (실패: {', '.join(errors)})"
        return SyncResponse(
            synced_files=result["synced_files"],
            total_chunks=result["total_chunks"],
            message=msg,
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
    if not is_sync_ready():
        raise HTTPException(status_code=503, detail="문서 동기화가 아직 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
    try:
        result = rag_query(request.question)
        return RAGResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/locate", response_model=LocateResponse)
def rag_locate_endpoint(request: LocateRequest):
    """질문과 관련된 내용이 어느 문서의 어느 위치에 있는지 찾아줍니다."""
    if not is_sync_ready():
        raise HTTPException(status_code=503, detail="문서 동기화가 아직 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
    try:
        result = rag_locate(request.question)
        return LocateResponse(
            answer=result.get("answer", ""),
            locations=[DocumentLocation(**loc) for loc in result.get("locations", [])],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/edit", response_model=EditResponse)
def rag_edit_endpoint(request: EditRequest):
    """사용자의 수정 요청을 받아 AI가 수정안을 생성합니다."""
    if not is_sync_ready():
        raise HTTPException(status_code=503, detail="문서 동기화가 아직 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
    try:
        result = rag_edit(request.question)
        return EditResponse(
            filename=result.get("filename", ""),
            original=result.get("original", ""),
            revised=result.get("revised", ""),
            summary=result.get("summary", ""),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 관리 UI ===

@app.get("/manage", response_class=HTMLResponse)
def manage_page():
    """관리 UI 페이지를 서빙합니다."""
    html_path = STATIC_DIR / "manage.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="관리 UI 파일이 없습니다.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# === 문서 CRUD 엔드포인트 ===

@app.get("/documents/{filename}", response_model=DocumentContentResponse)
def get_document(filename: str):
    """특정 문서의 내용을 조회합니다."""
    file_path = KNOWLEDGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {filename}")
    if file_path.suffix not in (".txt", ".md"):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    content = file_path.read_text(encoding="utf-8")
    return DocumentContentResponse(
        filename=filename,
        content=content,
        size_bytes=file_path.stat().st_size,
    )


@app.put("/documents/{filename}", response_model=DocumentUpdateResponse)
def update_document(filename: str, request: DocumentUpdateRequest):
    """문서 내용을 수정합니다. 저장 후 watcher가 자동으로 벡터 DB에 반영합니다."""
    file_path = KNOWLEDGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {filename}")
    if file_path.suffix not in (".txt", ".md"):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 파일 저장 (watcher.py가 변경을 감지하여 ChromaDB에 자동 반영)
    file_path.write_text(request.content, encoding="utf-8")

    return DocumentUpdateResponse(
        filename=filename,
        size_bytes=file_path.stat().st_size,
        message=f"문서가 저장되었습니다. 벡터 DB에 자동 반영됩니다.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
