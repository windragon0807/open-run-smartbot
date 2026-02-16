"""
knowledge/ 폴더 실시간 감시 모듈
파일 추가/수정 시 벡터 DB에 자동 반영, 삭제 시 DB에서 제거
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from watchfiles import awatch, Change

from rag.document_loader import KNOWLEDGE_DIR, load_and_split, list_knowledge_files
from rag.vector_store import add_documents, delete_by_source, reset as reset_db, get_chunk_count, check_db_health

logger = logging.getLogger("watcher")

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# 동기화 상태 플래그
_sync_ready = False

# === 동기화 상태 추적 ===
_sync_status = {
    "last_sync_at": None,           # 마지막 전체 동기화 시각 (ISO 문자열)
    "last_sync_result": None,       # 마지막 동기화 결과 ("success" | "partial" | "failed")
    "synced_files": 0,              # 동기화된 파일 수
    "total_chunks": 0,              # 총 청크 수
    "errors": [],                   # 동기화 실패 파일 목록
    "file_sync_log": {},            # 파일별 마지막 동기화 시각 {filename: ISO 문자열}
    "recent_events": [],            # 최근 동기화 이벤트 로그 (최대 50개)
}


def _now_iso() -> str:
    """현재 시각을 ISO 형식 문자열로 반환합니다."""
    return datetime.now(timezone.utc).isoformat()


def _add_event(event_type: str, filename: str, success: bool, detail: str = ""):
    """동기화 이벤트를 로그에 추가합니다."""
    _sync_status["recent_events"].append({
        "timestamp": _now_iso(),
        "type": event_type,
        "filename": filename,
        "success": success,
        "detail": detail,
    })
    # 최대 50개 유지
    if len(_sync_status["recent_events"]) > 50:
        _sync_status["recent_events"] = _sync_status["recent_events"][-50:]


def get_sync_status() -> dict:
    """현재 동기화 상태 정보를 반환합니다."""
    # 벡터 DB 건강 상태 진단
    db_health = check_db_health()

    # knowledge/ 파일 목록과 벡터 DB 동기화 여부 비교
    files = list_knowledge_files()
    file_statuses = []
    for f in files:
        fname = f["filename"]
        synced_at = _sync_status["file_sync_log"].get(fname)
        has_error = fname in _sync_status["errors"]
        file_statuses.append({
            "filename": fname,
            "size_bytes": f["size_bytes"],
            "synced_at": synced_at,
            "has_error": has_error,
        })

    return {
        "is_ready": _sync_ready,
        "last_sync_at": _sync_status["last_sync_at"],
        "last_sync_result": _sync_status["last_sync_result"],
        "synced_files": _sync_status["synced_files"],
        "total_chunks": _sync_status["total_chunks"],
        "db_chunk_count": db_health["chunk_count"],
        "errors": _sync_status["errors"],
        "file_statuses": file_statuses,
        "recent_events": _sync_status["recent_events"][-20:],
        "db_health": db_health,
    }


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
    now = _now_iso()

    for file_info in files:
        file_path = KNOWLEDGE_DIR / file_info["filename"]
        try:
            documents = load_and_split(file_path)
            add_documents(documents)
            total_chunks += len(documents)
            _sync_status["file_sync_log"][file_info["filename"]] = now
            _add_event("full_sync", file_info["filename"], True, f"{len(documents)}개 청크")
            logger.info(f"동기화 완료: {file_info['filename']} ({len(documents)}개 청크)")
        except Exception as e:
            errors.append(file_info["filename"])
            _add_event("full_sync", file_info["filename"], False, str(e))
            logger.error(f"동기화 실패: {file_info['filename']} - {e}")

    if errors:
        logger.warning(f"동기화 실패 파일 {len(errors)}개: {errors}")

    _sync_ready = True

    # 상태 업데이트
    _sync_status["last_sync_at"] = now
    _sync_status["synced_files"] = len(files) - len(errors)
    _sync_status["total_chunks"] = total_chunks
    _sync_status["errors"] = errors
    if len(errors) == 0:
        _sync_status["last_sync_result"] = "success"
    elif len(errors) < len(files):
        _sync_status["last_sync_result"] = "partial"
    else:
        _sync_status["last_sync_result"] = "failed"

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
                        _sync_status["file_sync_log"][path.name] = _now_iso()
                        event_type = "file_added" if change_type == Change.added else "file_modified"
                        _add_event(event_type, path.name, True, f"{chunks}개 청크")
                        logger.info(f"파일 반영: {path.name} ({chunks}개 청크)")
                    except Exception as e:
                        _add_event("file_sync_error", path.name, False, str(e))
                        logger.error(f"파일 반영 실패: {path.name} - {e}")

            elif change_type == Change.deleted:
                if path.suffix in SUPPORTED_EXTENSIONS:
                    try:
                        delete_by_source(path.name)
                        _sync_status["file_sync_log"].pop(path.name, None)
                        _add_event("file_deleted", path.name, True)
                        logger.info(f"파일 삭제 반영: {path.name}")
                    except Exception as e:
                        _add_event("file_delete_error", path.name, False, str(e))
                        logger.error(f"파일 삭제 반영 실패: {path.name} - {e}")
