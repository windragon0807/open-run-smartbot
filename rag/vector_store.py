"""
ChromaDB 벡터 저장소 관리 모듈
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger("vector_store")

# ChromaDB 데이터 저장 경로
# Cloud Run은 컨테이너 파일시스템이 읽기 전용이므로 /tmp 사용
_default_chroma_dir = Path(__file__).resolve().parent.parent / "chroma_db"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_default_chroma_dir)))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# 컬렉션 이름
COLLECTION_NAME = "smart_chatbot_docs"

# ChromaDB 클라이언트 싱글턴 (매번 새로 생성하지 않도록)
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    """ChromaDB PersistentClient 싱글턴을 반환합니다."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logger.info(f"ChromaDB 클라이언트 생성: {CHROMA_DIR}")
    return _chroma_client


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델을 반환합니다."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vector_store() -> Chroma:
    """ChromaDB 벡터 저장소 인스턴스를 반환합니다."""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def add_documents(documents: list) -> int:
    """
    문서 청크들을 벡터 저장소에 추가합니다.

    Args:
        documents: LangChain Document 리스트

    Returns:
        추가된 문서 청크 수
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return len(documents)


def get_chunk_count() -> int:
    """
    벡터 저장소에 저장된 총 청크 수를 반환합니다.

    Returns:
        저장된 문서 청크 수 (오류 시 -1)
    """
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return -1


def check_db_health() -> dict:
    """
    벡터 DB의 건강 상태를 종합 진단합니다.

    Returns:
        {
            "chroma_connected": bool,       # ChromaDB 클라이언트 연결 가능 여부
            "chroma_error": str | None,     # 연결 오류 메시지
            "collection_exists": bool,      # 컬렉션 존재 여부
            "chunk_count": int,             # 저장된 청크 수 (-1이면 조회 실패)
            "chroma_dir": str,              # ChromaDB 저장 경로
            "chroma_dir_exists": bool,      # 저장 경로 존재 여부
            "chroma_dir_writable": bool,    # 저장 경로 쓰기 가능 여부
            "embedding_model_ok": bool,     # 임베딩 모델(OpenAI API) 사용 가능 여부
            "embedding_error": str | None,  # 임베딩 오류 메시지
        }
    """
    result = {
        "chroma_connected": False,
        "chroma_error": None,
        "collection_exists": False,
        "chunk_count": -1,
        "chroma_dir": str(CHROMA_DIR),
        "chroma_dir_exists": CHROMA_DIR.exists(),
        "chroma_dir_writable": os.access(str(CHROMA_DIR), os.W_OK) if CHROMA_DIR.exists() else False,
        "embedding_model_ok": False,
        "embedding_error": None,
    }

    # 1. ChromaDB 연결 확인
    try:
        client = _get_chroma_client()
        client.heartbeat()
        result["chroma_connected"] = True
    except Exception as e:
        result["chroma_error"] = str(e)
        return result

    # 2. 컬렉션 존재 여부 확인
    try:
        existing = [c.name for c in client.list_collections()]
        result["collection_exists"] = COLLECTION_NAME in existing
    except Exception as e:
        result["chroma_error"] = f"컬렉션 목록 조회 실패: {e}"

    # 3. 청크 수 조회
    try:
        if result["collection_exists"]:
            collection = client.get_collection(name=COLLECTION_NAME)
            result["chunk_count"] = collection.count()
        else:
            result["chunk_count"] = 0
    except Exception as e:
        result["chroma_error"] = f"청크 수 조회 실패: {e}"

    # 4. 임베딩 모델 확인 (OpenAI API 키 유효성)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            result["embedding_error"] = "OPENAI_API_KEY 환경변수가 설정되지 않음"
        else:
            embeddings = get_embeddings()
            embeddings.embed_query("health check")
            result["embedding_model_ok"] = True
    except Exception as e:
        result["embedding_error"] = str(e)

    return result


def search(query: str, k: int = 3) -> list:
    """
    질문과 가장 관련 있는 문서 청크를 검색합니다.

    Args:
        query: 검색할 질문
        k: 반환할 문서 수

    Returns:
        관련 Document 리스트
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def delete_by_source(filename: str) -> None:
    """
    특정 파일의 벡터 데이터를 DB에서 삭제합니다.

    Args:
        filename: 삭제할 파일명 (예: "문서.txt")
    """
    vector_store = get_vector_store()
    results = vector_store.get(where={"source": filename})
    if results and results["ids"]:
        vector_store.delete(ids=results["ids"])


def reset() -> None:
    """벡터 DB를 완전히 초기화합니다. (컬렉션 삭제 방식, 파일시스템 의존 없음)"""
    global _chroma_client
    try:
        client = _get_chroma_client()
        # 컬렉션이 존재하면 삭제
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"컬렉션 삭제 완료: {COLLECTION_NAME}")
        else:
            logger.info(f"삭제할 컬렉션 없음: {COLLECTION_NAME}")
    except Exception as e:
        logger.warning(f"컬렉션 삭제 중 오류 (무시하고 계속): {e}")
        # 클라이언트가 손상된 경우 재생성
        _chroma_client = None
