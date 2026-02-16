"""
요청/응답 스키마 모델 정의
"""

from pydantic import BaseModel


# === 요청 모델 ===

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    model: str = "gpt-4o-mini"

    class Config:
        json_schema_extra = {
            "example": {
                "message": "안녕하세요! 오늘 날씨가 어때요?",
                "model": "gpt-4o-mini",
            }
        }


# === 응답 모델 ===

class TokenUsage(BaseModel):
    """토큰 사용량 모델"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    reply: str
    model: str
    usage: TokenUsage


class StatusResponse(BaseModel):
    """서버 상태 응답 모델"""
    status: str
    message: str


class ModelsResponse(BaseModel):
    """모델 목록 응답 모델"""
    models: list[str]
    count: int


# === RAG 관련 모델 ===

class DocumentInfo(BaseModel):
    """문서 정보 모델"""
    filename: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    """문서 목록 응답 모델"""
    documents: list[DocumentInfo]
    count: int


class SyncResponse(BaseModel):
    """문서 동기화 응답 모델"""
    synced_files: int
    total_chunks: int
    message: str


class ResetResponse(BaseModel):
    """DB 초기화 응답 모델"""
    message: str


class SourceDocument(BaseModel):
    """참고 문서 모델"""
    content: str
    source: str


class RAGRequest(BaseModel):
    """RAG 질문 요청 모델"""
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "이 문서의 주요 내용은 무엇인가요?",
            }
        }


class RAGResponse(BaseModel):
    """RAG 질문 응답 모델"""
    answer: str
    sources: list[SourceDocument]


# === 문서 관리 모델 ===

class DocumentContentResponse(BaseModel):
    """문서 내용 응답 모델"""
    filename: str
    content: str
    size_bytes: int


class DocumentUpdateRequest(BaseModel):
    """문서 수정 요청 모델"""
    content: str

    class Config:
        json_schema_extra = {
            "example": {
                "content": "# 문서 제목\n\n수정된 내용입니다.",
            }
        }


class DocumentUpdateResponse(BaseModel):
    """문서 수정 응답 모델"""
    filename: str
    size_bytes: int
    message: str


# === 문서 위치 찾기 모델 ===

class LocateRequest(BaseModel):
    """문서 위치 찾기 요청 모델"""
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "벙 만드는 방법은 어디에 있어?",
            }
        }


class DocumentLocation(BaseModel):
    """문서 위치 정보"""
    filename: str
    section: str
    snippet: str


class LocateResponse(BaseModel):
    """문서 위치 찾기 응답 모델"""
    answer: str
    locations: list[DocumentLocation]


# === 문서 수정 모델 ===

class EditRequest(BaseModel):
    """문서 수정 요청 모델"""
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "벙 만들기 문서에서 최소 인원을 3명으로 바꿔줘",
            }
        }


class EditResponse(BaseModel):
    """문서 수정 제안 응답 모델"""
    filename: str
    original: str
    revised: str
    summary: str


# === 동기화 상태 모델 ===

class SyncEvent(BaseModel):
    """동기화 이벤트 로그"""
    timestamp: str
    type: str
    filename: str
    success: bool
    detail: str = ""


class FileSyncStatus(BaseModel):
    """파일별 동기화 상태"""
    filename: str
    size_bytes: int
    synced_at: str | None = None
    has_error: bool = False


class DbHealth(BaseModel):
    """벡터 DB 건강 상태"""
    chroma_connected: bool = False
    chroma_error: str | None = None
    collection_exists: bool = False
    chunk_count: int = -1
    chroma_dir: str = ""
    chroma_dir_exists: bool = False
    chroma_dir_writable: bool = False
    embedding_model_ok: bool = False
    embedding_error: str | None = None


class SyncStatusResponse(BaseModel):
    """동기화 상태 전체 응답 모델"""
    is_ready: bool
    last_sync_at: str | None = None
    last_sync_result: str | None = None
    synced_files: int = 0
    total_chunks: int = 0
    db_chunk_count: int = 0
    errors: list[str] = []
    file_statuses: list[FileSyncStatus] = []
    recent_events: list[SyncEvent] = []
    db_health: DbHealth = DbHealth()
