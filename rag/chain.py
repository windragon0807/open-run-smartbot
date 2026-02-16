"""
LangChain RAG 체인 구성 모듈
"""

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.vector_store import get_vector_store
from rag.document_loader import KNOWLEDGE_DIR

# 시스템 프롬프트: 검색된 문서를 기반으로만 답변하도록 설정
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 제공된 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n"
        "아래 문서 내용을 참고하여 정확하게 답변해 주세요.\n"
        "문서에 없는 내용은 '제공된 문서에서 해당 정보를 찾을 수 없습니다.'라고 답변하세요.\n\n"
        "=== 참고 문서 ===\n"
        "{context}\n"
        "================",
    ),
    ("human", "{question}"),
])

# 시스템 프롬프트: 문서 위치 찾기 전용
LOCATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 knowledge 문서 관리를 돕는 어시스턴트입니다.\n"
        "사용자가 특정 내용이 어느 문서에 있는지 물어보면, 검색된 문서를 기반으로 정확한 위치를 안내합니다.\n\n"
        "반드시 아래 JSON 형식으로만 응답하세요:\n"
        '{{\n'
        '  "answer": "사용자에게 보여줄 안내 메시지 (한국어, 친절하게)",\n'
        '  "locations": [\n'
        '    {{\n'
        '      "filename": "정확한 파일명 (예: 01_서비스_개요.md)",\n'
        '      "section": "해당 섹션/제목 (예: ## 서비스 소개)",\n'
        '      "snippet": "관련 내용 일부 발췌 (50자 이내)"\n'
        '    }}\n'
        '  ]\n'
        '}}\n\n'
        "규칙:\n"
        "- filename은 반드시 검색된 문서의 source 메타데이터에 있는 실제 파일명을 사용하세요.\n"
        "- locations는 관련도가 높은 순서로 최대 3개까지 반환하세요.\n"
        "- 찾을 수 없으면 locations를 빈 배열로, answer에 안내 메시지를 넣으세요.\n\n"
        "=== 검색된 문서 ===\n"
        "{context}\n"
        "==================",
    ),
    ("human", "{question}"),
])


def format_docs(docs: list) -> str:
    """검색된 문서들을 하나의 문자열로 합칩니다."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """RAG 체인을 생성하여 반환합니다."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def query(question: str) -> dict:
    """
    질문을 받아 RAG 체인으로 답변을 생성합니다.

    Args:
        question: 사용자 질문

    Returns:
        답변과 참고 문서 정보를 포함한 딕셔너리
    """
    # 관련 문서 검색
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=3)

    # RAG 체인으로 답변 생성
    chain = get_rag_chain()
    answer = chain.invoke(question)

    # 참고 문서 정보
    sources = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in relevant_docs
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


def locate(question: str) -> dict:
    """
    질문을 받아 관련 내용이 어느 문서의 어느 위치에 있는지 찾아줍니다.

    Args:
        question: 사용자 질문 (예: "벙 만드는 방법은 어디에 있어?")

    Returns:
        안내 메시지와 문서 위치 정보를 포함한 딕셔너리
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 관련 문서 검색 (위치 찾기는 더 넓게 검색)
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=5)

    # 검색된 문서를 source 정보와 함께 포맷
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[파일: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # LLM으로 위치 분석
    chain = LOCATE_PROMPT | llm | StrOutputParser()
    raw_answer = chain.invoke({"context": context, "question": question})

    # JSON 파싱
    try:
        result = json.loads(raw_answer)
    except json.JSONDecodeError:
        result = {
            "answer": raw_answer,
            "locations": [],
        }

    return result


# 시스템 프롬프트: 문서 수정 전용
EDIT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 knowledge 문서를 수정하는 전문 편집 어시스턴트입니다.\n"
        "사용자가 문서의 특정 부분을 수정해달라고 요청하면, 원본 문서를 기반으로 수정된 전체 문서를 반환합니다.\n\n"
        "반드시 아래 JSON 형식으로만 응답하세요:\n"
        '{{\n'
        '  "revised": "수정된 문서의 전체 내용 (마크다운 형식 유지)",\n'
        '  "summary": "변경 내용을 한국어로 간결하게 요약 (1~2문장)"\n'
        '}}\n\n'
        "규칙:\n"
        "- 사용자가 요청한 부분만 정확하게 수정하고, 나머지 내용은 원본 그대로 유지하세요.\n"
        "- 마크다운 구조(제목, 리스트, 표 등)를 반드시 보존하세요.\n"
        "- 원본에 없는 내용을 임의로 추가하지 마세요.\n"
        "- revised에는 수정된 문서 전체 내용을 빠짐없이 포함하세요.\n"
        "- JSON 외의 텍스트를 절대 포함하지 마세요.\n\n"
        "=== 원본 문서 ({filename}) ===\n"
        "{document}\n"
        "============================",
    ),
    ("human", "{question}"),
])


def edit(question: str) -> dict:
    """
    사용자의 수정 요청을 받아 관련 문서를 찾고, LLM으로 수정안을 생성합니다.

    Args:
        question: 사용자의 수정 요청 (예: "벙 만들기 문서에서 최소 인원을 3명으로 바꿔줘")

    Returns:
        filename, original, revised, summary를 포함한 딕셔너리
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 1. RAG 검색으로 관련 문서 파일명 특정
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=3)

    if not relevant_docs:
        return {
            "filename": "",
            "original": "",
            "revised": "",
            "summary": "관련 문서를 찾을 수 없습니다.",
        }

    # 가장 관련도 높은 문서의 파일명 추출
    target_filename = relevant_docs[0].metadata.get("source", "")
    if not target_filename:
        return {
            "filename": "",
            "original": "",
            "revised": "",
            "summary": "문서 파일명을 확인할 수 없습니다.",
        }

    # 2. 해당 파일의 전체 원본을 직접 읽기
    file_path = KNOWLEDGE_DIR / target_filename
    if not file_path.exists():
        return {
            "filename": target_filename,
            "original": "",
            "revised": "",
            "summary": f"파일을 찾을 수 없습니다: {target_filename}",
        }

    original_content = file_path.read_text(encoding="utf-8")

    # 3. LLM에 원본 + 수정 요청 전달
    chain = EDIT_PROMPT | llm | StrOutputParser()
    raw_answer = chain.invoke({
        "filename": target_filename,
        "document": original_content,
        "question": question,
    })

    # 4. JSON 파싱
    try:
        result = json.loads(raw_answer)
    except json.JSONDecodeError:
        return {
            "filename": target_filename,
            "original": original_content,
            "revised": "",
            "summary": "수정안 생성에 실패했습니다. 다시 시도해주세요.",
        }

    return {
        "filename": target_filename,
        "original": original_content,
        "revised": result.get("revised", ""),
        "summary": result.get("summary", ""),
    }
