"""
실행형 Assistant 플래너

질문/요청 문장을 다음 중 하나로 분기합니다.
- qa: 일반 질의응답
- action_collect: 실행 의도는 있으나 필수 파라미터 누락
- action_ready: 실행 가능
- action_navigate: 원샷 실행이 아니라 화면 이동이 필요한 요청
- action_unavailable: 자동 실행 미지원
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.chain import query as rag_query
from rag.vector_store import get_vector_store


CONFIDENCE_THRESHOLD = 0.65

ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "bung.complete": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "execute",
    },
    "bung.join": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "execute",
    },
    "bung.leave": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "execute",
    },
    "bung.delete": {
        "required": ["bungId"],
        "danger": "high",
        "mode": "execute",
    },
    "bung.certify": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "execute",
    },
    "bung.delegate_owner": {
        "required": ["bungId", "targetUserId"],
        "danger": "high",
        "mode": "execute",
    },
    "bung.kick_member": {
        "required": ["bungId", "targetUserId"],
        "danger": "high",
        "mode": "execute",
    },
    "bung.modify": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.create": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.invite_members": {
        "required": [],
        "danger": "low",
        "mode": "unavailable",
    },
    "user.delete_account": {
        "required": [],
        "danger": "high",
        "mode": "execute",
    },
    "challenge.mint_nft": {
        "required": ["challengeId"],
        "danger": "low",
        "mode": "execute",
    },
    "bung.send_feedback": {
        "required": ["bungId", "targetUserIds"],
        "danger": "low",
        "mode": "execute",
    },
}

FIELD_LABELS = {
    "bungId": "대상 벙",
    "targetUserId": "대상 멤버",
    "targetUserIds": "피드백 대상 멤버",
    "challengeId": "대상 도전과제",
}

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 OpenRun 실행형 어시스턴트 플래너입니다.\n"
            "반드시 JSON 객체만 출력하세요.\n\n"
            "목표:\n"
            "1) 사용자의 메시지가 일반 질문인지(action이 아닌지) 판별\n"
            "2) action이면 actionKey와 params를 추출\n"
            "3) confidence를 0~1로 반환\n\n"
            "응답 JSON 키 요구사항:\n"
            '- kind: "qa | action_collect | action_ready | action_navigate | action_unavailable"\n'
            '- reply: 사용자에게 보여줄 한국어 메시지\n'
            "- confidence: 0~1 숫자\n"
            "- proposal: 객체 (qa면 null 가능)\n"
            "- proposal.actionKey, proposal.summary, proposal.params, proposal.missingFields,\n"
            "  proposal.dangerLevel, proposal.navigation(type/href/modalKey/prefill)\n\n"
            "규칙:\n"
            "- actionKey는 catalog에 있는 값만 사용\n"
            "- 불확실하면 kind=qa, confidence를 낮게 설정\n"
            "- 원샷 불가 액션은 action_navigate 또는 action_unavailable로 반환\n"
            "- params에는 찾은 값만 넣고, 없는 값은 누락\n"
            "- missingFields에는 필수 파라미터 중 비어있는 값을 넣음\n"
            "- 텍스트에 특정 벙/멤버/도전과제가 명시되지 않으면 ID를 추측하지 말 것\n"
            "- 단순 정보 질문(예: 진도, 과제 설명)은 qa\n"
            "- JSON 이외 텍스트 금지",
        ),
        (
            "human",
            "[ACTION CATALOG]\n{catalog}\n\n"
            "[RECENT HISTORY]\n{history}\n\n"
            "[PENDING ACTION]\n{pending_action}\n\n"
            "[RAG CONTEXT]\n{context}\n\n"
            "[USER MESSAGE]\n{message}",
        ),
    ]
)


def _extract_json_block(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(_extract_json_block(raw))
    except Exception:
        return None


def _compact_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "[]"
    compact = history[-10:]
    return json.dumps(compact, ensure_ascii=False)


def _rag_context(question: str) -> str:
    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(question, k=3)
        parts: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[파일: {source}]\n{doc.page_content[:600]}")
        return "\n\n".join(parts)
    except Exception:
        return ""


def _normalize_params(
    parsed_params: dict[str, Any] | None,
    pending_action: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(pending_action, dict):
        proposal = pending_action.get("proposal")
        if isinstance(proposal, dict):
            existing = proposal.get("params")
            if isinstance(existing, dict):
                merged.update(existing)
    if isinstance(parsed_params, dict):
        for key, value in parsed_params.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            merged[key] = value
    return merged


def _default_navigation(action_key: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if action_key == "bung.create":
        draft = {
            "name": params.get("name"),
            "title": params.get("title"),
            "description": params.get("description"),
            "location": params.get("location"),
            "detailedAddress": params.get("detailedAddress"),
            "distance": params.get("distance"),
            "memberNumber": params.get("memberNumber"),
            "hasAfterRun": params.get("hasAfterRun"),
            "afterRunDescription": params.get("afterRunDescription"),
            "hashtags": params.get("hashtags"),
            "runningTime": params.get("runningTime"),
            "paceMinute": params.get("paceMinute"),
            "paceSecond": params.get("paceSecond"),
        }
        cleaned_draft = {k: v for k, v in draft.items() if v is not None}
        return {
            "type": "modal",
            "modalKey": "create-bung",
            "prefill": {
                "initialStep": "create",
                "draft": cleaned_draft,
            },
        }

    if action_key == "bung.modify":
        bung_id = params.get("bungId")
        href = f"/bung/{bung_id}?chatAction=modify" if bung_id else "/"
        return {
            "type": "route",
            "href": href,
        }

    if action_key == "bung.invite_members":
        return {
            "type": "modal",
            "modalKey": "create-bung",
            "prefill": {
                "initialStep": "invitation",
            },
        }
    return None


def _build_missing_fields(action_key: str, params: dict[str, Any]) -> list[str]:
    required = ACTION_CATALOG.get(action_key, {}).get("required", [])
    missing: list[str] = []
    for field in required:
        value = params.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and value.strip() == "":
            missing.append(field)
            continue
        if isinstance(value, list) and len(value) == 0:
            missing.append(field)
            continue
    return missing


def _fallback_collect_reply(action_key: str, missing_fields: list[str]) -> str:
    labels = [FIELD_LABELS.get(field, field) for field in missing_fields]
    if not labels:
        return "실행을 진행할게요. 아래 부탁하기를 눌러주세요."
    joined = ", ".join(labels)
    return f"{action_key} 실행을 위해 {joined} 정보가 필요합니다. 아래 버튼에서 선택해 주세요."


def _fallback_ready_reply(action_key: str) -> str:
    return f"{action_key} 요청을 실행할 준비가 됐습니다. 아래 `부탁하기`를 눌러 진행하세요."


def _fallback_navigate_reply(action_key: str) -> str:
    if action_key == "bung.invite_members":
        return "이 요청은 자동 실행이 아직 지원되지 않습니다. 관련 화면으로 이동해 안내를 이어가겠습니다."
    return "이 요청은 채팅 원샷 실행 대신 관련 화면에서 마무리하는 방식이 안전합니다. 아래 `부탁하기`를 눌러 이동하세요."


def _proposal_response(
    action_key: str,
    parsed_params: dict[str, Any] | None,
    pending_action: dict[str, Any] | None,
    confidence: float,
    summary: str | None = None,
) -> dict[str, Any]:
    params = _normalize_params(parsed_params, pending_action)
    missing_fields = _build_missing_fields(action_key, params)

    action_mode = ACTION_CATALOG[action_key]["mode"]
    if action_mode == "execute":
        normalized_kind = "action_collect" if missing_fields else "action_ready"
    elif action_mode == "navigate":
        normalized_kind = "action_navigate"
    else:
        normalized_kind = "action_unavailable"

    if not summary:
        summary = f"{action_key} 실행"

    if normalized_kind == "action_collect":
        reply = _fallback_collect_reply(action_key, missing_fields)
    elif normalized_kind == "action_ready":
        reply = _fallback_ready_reply(action_key)
    else:
        reply = _fallback_navigate_reply(action_key)

    return {
        "kind": normalized_kind,
        "reply": reply,
        "sources": [],
        "proposal": {
            "actionKey": action_key,
            "summary": summary,
            "params": params,
            "missingFields": missing_fields,
            "dangerLevel": ACTION_CATALOG[action_key]["danger"],
            "navigation": _default_navigation(action_key, params),
            "confidence": confidence,
        },
    }


def _looks_like_command(message: str) -> bool:
    normalized = message.lower().replace(" ", "")
    command_tokens = (
        "해줘",
        "해주세요",
        "도와줘",
        "도와주세요",
        "부탁",
        "실행",
        "만들어줘",
        "생성해줘",
        "완료해줘",
        "취소해줘",
        "참여해줘",
        "삭제해줘",
        "넘겨줘",
        "내보내줘",
        "탈퇴해줘",
        "받아줘",
        "남겨줘",
    )
    return any(token in normalized for token in command_tokens)


def _rule_based_plan(message: str, pending_action: dict[str, Any] | None) -> dict[str, Any] | None:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    is_command = _looks_like_command(lowered)

    if (is_command or "도와" in lowered) and "벙" in lowered and ("만들" in lowered or "생성" in lowered):
        params: dict[str, Any] = {}

        title_from_quote = re.search(r"[\"'“”](.+?)[\"'“”]", message)
        if title_from_quote:
            title = title_from_quote.group(1).strip()
            if title:
                params["name"] = title
        else:
            title_from_phrase = re.search(r"(.+?)라는\s*타이틀", message)
            if title_from_phrase:
                title = title_from_phrase.group(1).strip()
                if title:
                    params["name"] = title

        location_match = re.search(r"([가-힣A-Za-z0-9\s]+)에서", message)
        if location_match:
            location = location_match.group(1).strip()
            if location and len(location) <= 40:
                params["location"] = location

        return _proposal_response(
            action_key="bung.create",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.95,
            summary="벙 생성 화면 이동",
        )

    if is_command and "벙" in lowered and "완료" in lowered:
        return _proposal_response(
            action_key="bung.complete",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.9,
            summary="벙 완료",
        )

    if is_command and ("참여인증" in normalized or ("참여" in lowered and "인증" in lowered)):
        return _proposal_response(
            action_key="bung.certify",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.9,
            summary="참여 인증",
        )

    return None


def _qa_response(question: str) -> dict[str, Any]:
    result = rag_query(question)
    return {
        "kind": "qa",
        "reply": result["answer"],
        "sources": result["sources"],
        "proposal": None,
    }


def plan_assistant(
    message: str,
    history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or []

    ruled = _rule_based_plan(message, pending_action)
    if ruled is not None:
        return ruled

    context = _rag_context(message)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    chain = PLANNER_PROMPT | llm | StrOutputParser()

    raw = chain.invoke(
        {
            "catalog": json.dumps(ACTION_CATALOG, ensure_ascii=False, indent=2),
            "history": _compact_history(history),
            "pending_action": json.dumps(pending_action or {}, ensure_ascii=False),
            "context": context[:3000],
            "message": message,
        }
    )
    parsed = _safe_json_loads(raw)

    if not isinstance(parsed, dict):
        return _qa_response(message)

    kind = parsed.get("kind", "qa")
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    reply = parsed.get("reply", "").strip()

    if kind == "qa" or confidence < CONFIDENCE_THRESHOLD:
        return _qa_response(message)

    proposal = parsed.get("proposal")
    if not isinstance(proposal, dict):
        return _qa_response(message)

    action_key = proposal.get("actionKey")
    if action_key not in ACTION_CATALOG:
        return _qa_response(message)

    summary = proposal.get("summary")
    summary_value = summary if isinstance(summary, str) and summary.strip() else None
    result = _proposal_response(
        action_key=action_key,
        parsed_params=proposal.get("params"),
        pending_action=pending_action,
        confidence=confidence,
        summary=summary_value,
    )

    # LLM이 작성한 안내 문구가 있으면 우선 사용
    if reply:
        result["reply"] = reply

    navigation = proposal.get("navigation")
    if isinstance(navigation, dict) and isinstance(result.get("proposal"), dict):
        result["proposal"]["navigation"] = navigation

    return result
