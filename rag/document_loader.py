"""
문서 로드 및 청크 분할 모듈
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# knowledge/ 폴더 경로
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
KNOWLEDGE_DIR.mkdir(exist_ok=True)


def load_and_split(file_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    텍스트 파일을 읽어서 청크로 분할합니다.

    Args:
        file_path: 텍스트 파일 경로
        chunk_size: 청크 하나의 최대 글자 수
        chunk_overlap: 청크 간 겹치는 글자 수 (문맥 유지를 위해)

    Returns:
        분할된 Document 리스트
    """
    text = file_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": file_path.name}],
    )

    return documents


def list_knowledge_files() -> list[dict]:
    """knowledge/ 폴더의 문서 목록을 반환합니다."""
    files = []
    for f in KNOWLEDGE_DIR.iterdir():
        if f.is_file() and f.suffix in (".txt", ".md"):
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            })
    return files
