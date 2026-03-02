from __future__ import annotations

import re
from typing import Any


_ORDINAL_TOKEN_TO_INDEX = {
    "첫번째": 1,
    "첫째": 1,
    "두번째": 2,
    "둘째": 2,
    "세번째": 3,
    "셋째": 3,
    "네번째": 4,
    "넷째": 4,
    "다섯번째": 5,
    "여섯번째": 6,
    "일곱번째": 7,
    "여덟번째": 8,
    "아홉번째": 9,
    "열번째": 10,
}

_DEICTIC_TOKENS = (
    "그벙",
    "방금벙",
    "방금말한벙",
    "이벙",
    "해당벙",
    "저벙",
    "그거",
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _compact_entity(entity: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "bungId": entity.get("bungId"),
        "name": entity.get("name"),
        "currentMemberCount": entity.get("currentMemberCount"),
        "status": entity.get("status"),
        "order": _coerce_int(entity.get("order")) or index + 1,
    }


def get_bung_entities(conversation_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(conversation_state, dict):
        return []
    raw_entities = conversation_state.get("entities")
    if not isinstance(raw_entities, dict):
        return []
    bungs = raw_entities.get("bungs")
    if not isinstance(bungs, list):
        return []

    compacted = [_compact_entity(item, index) for index, item in enumerate(bungs) if isinstance(item, dict)]
    return sorted(compacted, key=lambda item: (_coerce_int(item.get("order")) or 999999))


def get_focus_bung_id(conversation_state: dict[str, Any] | None) -> str | None:
    if not isinstance(conversation_state, dict):
        return None
    focus = conversation_state.get("focus")
    if not isinstance(focus, dict):
        return None
    bung_id = focus.get("lastBungId")
    return bung_id if isinstance(bung_id, str) and bung_id.strip() else None


def parse_bung_ref_from_message(message: str, conversation_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    normalized = _normalize_text(message)

    index_match = re.search(r"(\d+)\s*번", message)
    if index_match:
        return {"type": "index", "value": int(index_match.group(1))}

    for token, index in _ORDINAL_TOKEN_TO_INDEX.items():
        if token in normalized:
            return {"type": "index", "value": index}

    if any(token in normalized for token in _DEICTIC_TOKENS):
        return {"type": "deictic", "value": "last"}

    entities = get_bung_entities(conversation_state)
    if not entities:
        return None

    message_norm = _normalize_text(message)
    matched_names: list[str] = []
    for entity in entities:
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name_norm = _normalize_text(name)
        if not name_norm:
            continue
        if name_norm in message_norm:
            matched_names.append(name)

    if not matched_names:
        return None

    matched_names.sort(key=len, reverse=True)
    return {"type": "name", "value": matched_names[0]}


def resolve_bung_ref(
    bung_ref: dict[str, Any] | None,
    conversation_state: dict[str, Any] | None,
) -> dict[str, Any]:
    entities = get_bung_entities(conversation_state)

    if not bung_ref:
        return {"status": "not_found", "matches": [], "reason": "missing_ref"}

    ref_type = bung_ref.get("type")
    ref_value = bung_ref.get("value")

    if ref_type == "id":
        if not isinstance(ref_value, str):
            return {"status": "not_found", "matches": [], "reason": "invalid_id"}
        matches = [entity for entity in entities if entity.get("bungId") == ref_value]
        if len(matches) == 1:
            return {"status": "resolved", "match": matches[0], "matches": matches}
        if len(matches) > 1:
            return {"status": "ambiguous", "matches": matches, "reason": "duplicated_id"}
        return {"status": "not_found", "matches": [], "reason": "id_not_found"}

    if ref_type == "index":
        index = _coerce_int(ref_value)
        if index is None or index < 1:
            return {"status": "not_found", "matches": [], "reason": "invalid_index"}
        for entity in entities:
            if _coerce_int(entity.get("order")) == index:
                return {"status": "resolved", "match": entity, "matches": [entity]}
        if index <= len(entities):
            match = entities[index - 1]
            return {"status": "resolved", "match": match, "matches": [match]}
        return {"status": "not_found", "matches": [], "reason": "index_out_of_range"}

    if ref_type == "deictic":
        focus_bung_id = get_focus_bung_id(conversation_state)
        if focus_bung_id:
            matches = [entity for entity in entities if entity.get("bungId") == focus_bung_id]
            if len(matches) == 1:
                return {"status": "resolved", "match": matches[0], "matches": matches}
        if entities:
            return {"status": "resolved", "match": entities[0], "matches": [entities[0]]}
        return {"status": "not_found", "matches": [], "reason": "missing_focus"}

    if ref_type == "name":
        if not isinstance(ref_value, str) or not ref_value.strip():
            return {"status": "not_found", "matches": [], "reason": "invalid_name"}
        target = _normalize_text(ref_value)
        if not target:
            return {"status": "not_found", "matches": [], "reason": "invalid_name"}

        exact = [entity for entity in entities if _normalize_text(str(entity.get("name", ""))) == target]
        if len(exact) == 1:
            return {"status": "resolved", "match": exact[0], "matches": exact}
        if len(exact) > 1:
            return {"status": "ambiguous", "matches": exact, "reason": "duplicated_name"}

        partial = [entity for entity in entities if target in _normalize_text(str(entity.get("name", "")))]
        if len(partial) == 1:
            return {"status": "resolved", "match": partial[0], "matches": partial}
        if len(partial) > 1:
            return {"status": "ambiguous", "matches": partial, "reason": "duplicated_partial_name"}
        return {"status": "not_found", "matches": [], "reason": "name_not_found"}

    return {"status": "not_found", "matches": [], "reason": "unsupported_ref_type"}
