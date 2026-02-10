"""
LangChain RAG 체인 구성 모듈
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.vector_store import get_vector_store

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
