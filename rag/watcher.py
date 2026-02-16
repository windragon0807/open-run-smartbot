"""
knowledge/ 폴더 실시간 감시 모듈
파일 추가/수정 시 벡터 DB에 자동 반영, 삭제 시 DB에서 제거
"""

import asyncio
import logging
from pathlib import Path
from watchfiles import awatch, Change

from rag.document_loader import KNOWLEDGE_DIR, load_and_split, list_knowledge_files
from rag.vector_store import add_documents, delete_by_source, reset as reset_db

logger = logging.getLogger("watcher")

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# 동기화 상태 플래그
_sync_ready = False


def is_sync_ready() -> bool:
    """초기 동기화가 완료되었는지 반환합니다."""
    return _sync_ready


def is_supported_file(path: Path) -> bool:
    """지원하는 파일 형식인지 확인합니다."""
    return path.suffix in SUPPORTED_EXTENSIONS and path.is_file()


def sync_file(file_path: Path) -> int:
    """
    단일 파일을 벡터 DB에 동기화합니다.
    기존 데이터를 삭제하고 새로 추가합니다.

    Returns:
        추가된 청크 수
    """
    delete_by_source(file_path.name)
    documents = load_and_split(file_path)
    return add_documents(documents)


def sync_all() -> dict:
    """
    knowledge/ 폴더의 모든 문서를 벡터 DB에 동기화합니다.
    기존 컬렉션을 초기화하고 전체를 다시 저장합니다.

    Returns:
        동기화 결과 (파일 수, 총 청크 수)
    """
    global _sync_ready
    _sync_ready = False

    # 컬렉션 초기화 (파일시스템 삭제가 아닌 컬렉션 삭제 방식)
    reset_db()

    files = list_knowledge_files()
    total_chunks = 0
    errors = []

    for file_info in files:
        file_path = KNOWLEDGE_DIR / file_info["filename"]
        try:
            documents = load_and_split(file_path)
            add_documents(documents)
            total_chunks += len(documents)
            logger.info(f"동기화 완료: {file_info['filename']} ({len(documents)}개 청크)")
        except Exception as e:
            errors.append(file_info["filename"])
            logger.error(f"동기화 실패: {file_info['filename']} - {e}")

    if errors:
        logger.warning(f"동기화 실패 파일 {len(errors)}개: {errors}")

    _sync_ready = True
    return {
        "synced_files": len(files) - len(errors),
        "total_chunks": total_chunks,
        "errors": errors,
    }


async def watch_knowledge_folder():
    """knowledge/ 폴더를 실시간으로 감시합니다."""
    logger.info(f"knowledge/ 폴더 감시 시작: {KNOWLEDGE_DIR}")

    # 서버 시작 시 기존 문서 전체 동기화 (재시도 포함)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, sync_all)
            logger.info(
                f"초기 동기화 완료 (시도 {attempt}/{max_retries}): "
                f"{result['synced_files']}개 파일, {result['total_chunks']}개 청크"
            )
            if result["total_chunks"] > 0:
                break
            logger.warning("동기화된 청크가 0개입니다. 재시도합니다...")
        except Exception as e:
            logger.error(f"초기 동기화 실패 (시도 {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            await asyncio.sleep(2 * attempt)

    # 폴더 변경 감시
    async for changes in awatch(KNOWLEDGE_DIR):
        for change_type, change_path in changes:
            path = Path(change_path)

            if change_type in (Change.added, Change.modified):
                if is_supported_file(path):
                    try:
                        chunks = sync_file(path)
                        logger.info(f"파일 반영: {path.name} ({chunks}개 청크)")
                    except Exception as e:
                        logger.error(f"파일 반영 실패: {path.name} - {e}")

            elif change_type == Change.deleted:
                if path.suffix in SUPPORTED_EXTENSIONS:
                    try:
                        delete_by_source(path.name)
                        logger.info(f"파일 삭제 반영: {path.name}")
                    except Exception as e:
                        logger.error(f"파일 삭제 반영 실패: {path.name} - {e}")
