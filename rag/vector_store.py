"""
ChromaDB 벡터 저장소 관리 모듈
"""

import os
import shutil
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ChromaDB 데이터 저장 경로
# Cloud Run은 컨테이너 파일시스템이 읽기 전용이므로 /tmp 사용
_default_chroma_dir = Path(__file__).resolve().parent.parent / "chroma_db"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_default_chroma_dir)))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# 컬렉션 이름
COLLECTION_NAME = "smart_chatbot_docs"


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델을 반환합니다."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vector_store() -> Chroma:
    """ChromaDB 벡터 저장소 인스턴스를 반환합니다."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
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
    # source 메타데이터가 일치하는 문서를 찾아서 삭제
    results = vector_store.get(where={"source": filename})
    if results and results["ids"]:
        vector_store.delete(ids=results["ids"])


def reset() -> None:
    """벡터 DB를 완전히 초기화합니다."""
    # chroma_db 폴더 삭제 후 재생성
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)
