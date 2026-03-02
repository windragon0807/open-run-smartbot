"""
실행형 Assistant 플래너

질문/요청 문장을 다음 중 하나로 분기합니다.
- qa: 일반 질의응답
- read: 현재 로그인 사용자의 개인 데이터 조회 제안
- action_collect: 실행 의도는 있으나 필수 파라미터 누락
- action_ready: 실행 가능
- action_navigate: 원샷 실행이 아니라 화면 이동이 필요한 요청
- action_unavailable: 자동 실행 미지원
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.chain import query as rag_query
from rag.entity_resolver import parse_bung_ref_from_message, resolve_bung_ref


CONFIDENCE_THRESHOLD = 0.65
ROUTER_MODE = os.getenv("ASSISTANT_ROUTER_MODE", "active").strip().lower()
ROUTER_LOGGER = logging.getLogger("assistant_router")
ROUTER_CACHE_TTL_SECONDS = int(os.getenv("ASSISTANT_ROUTER_CACHE_TTL_SECONDS", "20"))
_LLM_ROUTER_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}
GENERAL_CHAT_CACHE_TTL_SECONDS = int(os.getenv("ASSISTANT_CHAT_CACHE_TTL_SECONDS", "30"))
_GENERAL_CHAT_CACHE: dict[str, tuple[float, str]] = {}
NLG_CACHE_TTL_SECONDS = int(os.getenv("ASSISTANT_NLG_CACHE_TTL_SECONDS", "120"))
_NLG_CACHE: dict[str, tuple[float, str]] = {}

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
    "home.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "challenge.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "challenge.progress.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "challenge.general.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "challenge.repetitive.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "challenge.completed.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "avatar.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "profile.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "profile.modify.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "profile.notification.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "explore.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.search.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.detail.open_page": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.manage_members.open_page": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.delegate_owner.open_page": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "navigate",
    },
    "auth.signin.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "auth.register.open_page": {
        "required": [],
        "danger": "low",
        "mode": "navigate",
    },
    "bung.send_feedback": {
        "required": ["bungId", "targetUserIds"],
        "danger": "low",
        "mode": "execute",
    },
}

READ_CATALOG: dict[str, dict[str, Any]] = {
    "read.my_bungs.count": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.my_bungs.names": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.my_bung.member_count": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "read",
    },
    "read.my_bung.members": {
        "required": ["bungId"],
        "danger": "low",
        "mode": "read",
    },
    "read.weather.current": {
        "required": ["location"],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.claimable.count": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.claimable.list": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.completed.count": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.completed.list": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.progress.summary": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.challenge.by_name.status": {
        "required": ["challengeName"],
        "danger": "low",
        "mode": "read",
    },
    "read.user.profile.summary": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.profile.nickname": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.profile.email": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.profile.wallet": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.running.preferences": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.suggestions.count": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.user.suggestions.list": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.bungs.explore.count": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
    "read.bungs.explore.names": {
        "required": [],
        "danger": "low",
        "mode": "read",
    },
}

CAPABILITY_CATALOG: dict[str, dict[str, Any]] = {
    **ACTION_CATALOG,
    **READ_CATALOG,
}

FIELD_LABELS = {
    "bungId": "대상 벙",
    "targetUserId": "대상 멤버",
    "targetUserIds": "피드백 대상 멤버",
    "challengeId": "대상 도전과제",
    "challengeName": "도전과제 이름",
    "location": "지역",
}

OPENRUN_DOMAIN_TOKENS = (
    "오픈런",
    "벙",
    "도전과제",
    "챌린지",
    "아바타",
    "탐색",
    "프로필",
    "러닝",
    "nft",
    "민팅",
    "피드백",
    "참여인증",
    "벙주",
    "과제",
    "달리기모임",
    "보상받기",
    "보상버튼",
)

def _catalog_for_prompt() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in CAPABILITY_CATALOG.items():
        catalog[key] = {
            "mode": value.get("mode"),
            "required": value.get("required", []),
            "danger": value.get("danger", "low"),
        }
    return catalog


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 OpenRun 챗봇의 의도 라우터입니다. 반드시 JSON만 출력하세요.\n\n"
            "출력 스키마:\n"
            "{{\n"
            '  "intent": "qa|action|read|clarify|refuse",\n'
            '  "confidence": 0.0,\n'
            '  "key": "catalog key or null",\n'
            '  "params": {{}},\n'
            '  "summary": "짧은 설명",\n'
            '  "reply": "필요 시 사용자 메시지"\n'
            "}}\n\n"
            "판단 규칙:\n"
            "- qa: 문서 기반 설명 질문\n"
            "- action: 무언가를 실행/이동/변경해달라는 요청\n"
            "- read: 현재 로그인 사용자 본인 데이터 조회 요청\n"
            "- refuse: 타인 개인정보 조회 요청\n"
            "- clarify: 의도가 모호함\n"
            "- key는 반드시 catalog 안의 값만 사용\n"
            "- ID(bungId, targetUserId, challengeId)는 텍스트에 없으면 추측하지 말 것\n"
            "- read.my_bungs.*에서는 params.scope를 가능하면 채워라: active|all\n"
            "- 특정 벙 조회(read.my_bung.*)는 params.bungRef를 활용할 수 있다.\n"
            '- bungRef 형식: {{"type":"id|name|index|deictic","value":"..."}}\n'
            "- '지금/현재/참여 중'이면 scope=active 우선\n"
            "- '그 이름', '각각 이름', '목록'처럼 직전 맥락을 이어서 묻는 경우 history/conversation_state를 반영\n"
            "- 예: 직전 대화가 '내가 참여 중인 벙 몇 개야?'이고 현재 질문이 '그 이름이 각각 뭐야?'면 intent=read, key=read.my_bungs.names\n"
            "- 예: 직전 목록 이후 '1번 벙 참여자가 몇 명이야?'면 intent=read, key=read.my_bung.member_count\n"
            "- 타인 정보 요청(OO님, 다른 사람, 친구 등)은 intent=refuse\n"
            "- JSON 외 텍스트 금지",
        ),
        (
            "human",
            "[CAPABILITY CATALOG]\n{catalog}\n\n"
            "[RECENT HISTORY]\n{history}\n\n"
            "[CONVERSATION STATE]\n{conversation_state}\n\n"
            "[PENDING ACTION]\n{pending_action}\n\n"
            "[USER MESSAGE]\n{message}",
        ),
    ]
)


GENERAL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 OpenRun 앱 안에서 동작하는 한국어 AI 어시스턴트다.\n"
            "- 일반적인 대화 질문에 자연스럽고 짧게 답한다.\n"
            "- 친근하되 과장하지 말고 명확하게 답한다.\n"
            "- 모르면 모른다고 말하고 가능한 다음 질문을 제안한다.\n"
            "- 답변은 1~3문장으로 유지한다.\n"
            "- 오픈런 서비스의 고유 사실(운영 정보/인물/정책)을 추측해서 단정하지 않는다.\n"
            "- 본 답변에서는 문서 출처나 버튼 안내를 하지 않는다.",
        ),
        (
            "human",
            "[RECENT HISTORY]\n{history}\n\n"
            "[USER MESSAGE]\n{message}",
        ),
    ]
)

ACTION_NLG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 OpenRun 어시스턴트의 응답 문구 생성기다.\n"
            "- 입력 JSON 상황을 읽고 사용자에게 보여줄 한국어 한두 문장을 만든다.\n"
            "- 친근하지만 간결하게 쓴다.\n"
            "- 실행형(action)일 때는 버튼 누르기 전에는 실제 실행이 되지 않음을 자연스럽게 암시한다.\n"
            "- read 모드에서는 버튼/클릭/실행 문구를 쓰지 않는다.\n"
            "- mode가 domain_not_found면 사실/수치/고유정보를 추측하지 말고 '현재 확인된 데이터가 없다'는 취지로만 답한다.\n"
            "- 과장, 추측, 장황한 설명 금지.\n"
            "- 출력은 문장 텍스트만, JSON 금지.",
        ),
        ("human", "{payload}"),
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
    compact = history[-6:]
    return json.dumps(compact, ensure_ascii=False)


def _compact_conversation_state(conversation_state: dict[str, Any] | None) -> str:
    if not isinstance(conversation_state, dict):
        return "{}"

    entities = conversation_state.get("entities")
    bungs: list[dict[str, Any]] = []
    if isinstance(entities, dict) and isinstance(entities.get("bungs"), list):
        for item in entities.get("bungs", [])[:6]:
            if not isinstance(item, dict):
                continue
            bungs.append(
                {
                    "bungId": item.get("bungId"),
                    "name": item.get("name"),
                    "order": item.get("order"),
                    "status": item.get("status"),
                    "currentMemberCount": item.get("currentMemberCount"),
                }
            )

    summary = {
        "focus": conversation_state.get("focus", {}),
        "lastReadResult": conversation_state.get("lastReadResult", {}),
        "pendingClarification": conversation_state.get("pendingClarification", {}),
        "entities": {"bungs": bungs},
    }
    return json.dumps(summary, ensure_ascii=False)


def _normalize_text(value: str) -> str:
    return value.lower().replace(" ", "")


def _is_ack_only_message(message: str) -> bool:
    normalized = _normalize_text(message)
    return normalized in {"응", "네", "예", "ㅇㅇ", "맞아", "아니", "아니오", "취소"}


def _looks_like_smalltalk(message: str) -> bool:
    normalized = _normalize_text(message)
    tokens = (
        "안녕",
        "하이",
        "hello",
        "내말이들려",
        "들려?",
        "고마워",
        "감사",
        "반가워",
        "기분어때",
        "뭐해",
    )
    return any(token in normalized for token in tokens)


def _looks_like_weather_request(message: str) -> bool:
    normalized = _normalize_text(message)
    weather_tokens = ("날씨", "기온", "온도", "비와", "눈와", "강수", "추워", "더워")
    return any(token in normalized for token in weather_tokens)


def _looks_like_challenge_claimable_query(message: str) -> bool:
    normalized = _normalize_text(message)
    challenge_tokens = ("도전과제", "챌린지", "과제")
    claimable_tokens = (
        "보상받기",
        "보상받을",
        "보상받을수",
        "보상을받을",
        "보상을받을수",
        "보상수령",
        "수령가능",
        "완료가능",
        "완료할수",
        "완료할수있는",
        "민팅가능",
        "보상가능",
    )
    return any(token in normalized for token in challenge_tokens) and any(token in normalized for token in claimable_tokens)


def _looks_like_reward_button_explanatory_qa(message: str) -> bool:
    normalized = _normalize_text(message)
    if "보상받기" not in normalized:
        return False

    trigger_tokens = ("누르면", "누를", "클릭", "터치", "탭", "하면", "버튼")
    question_tokens = ("뭐", "뭘", "무얼", "무엇", "무슨", "어떤", "왜", "어떻게", "설명", "의미")
    reward_tokens = ("받", "얻", "보상", "리워드", "nft")
    return (
        any(token in normalized for token in trigger_tokens)
        and any(token in normalized for token in question_tokens)
        and any(token in normalized for token in reward_tokens)
    )


def _looks_like_openrun_feature_explanatory_qa(
    message: str,
    history: list[dict[str, Any]],
    conversation_state: dict[str, Any] | None,
) -> bool:
    normalized = _normalize_text(message)
    if _looks_like_reward_button_explanatory_qa(message):
        return True

    ui_trigger_tokens = (
        "누르면",
        "누를",
        "클릭",
        "탭",
        "버튼",
        "기능",
        "의미",
        "역할",
        "차이",
        "어떻게써",
        "어떻게사용",
    )
    question_tokens = ("뭐", "뭘", "무엇", "무슨", "어떤", "왜", "어떻게", "설명")
    data_query_tokens = (
        "몇개",
        "개수",
        "갯수",
        "목록",
        "리스트",
        "이름",
        "몇명",
        "인원",
        "참여자",
        "멤버",
    )

    if not any(token in normalized for token in ui_trigger_tokens):
        return False
    if not ("?" in message or any(token in normalized for token in question_tokens)):
        return False
    if any(token in normalized for token in data_query_tokens):
        return False

    if _looks_like_openrun_domain_text(message):
        return True
    if isinstance(conversation_state, dict):
        if isinstance(conversation_state.get("lastReadResult"), dict):
            return True
        if isinstance(conversation_state.get("focus"), dict):
            return True
    return _history_is_openrun_context(history)


def _looks_like_challenge_claimable_count_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_claimable_query(message):
        return False
    if _looks_like_challenge_claimable_list_request(message):
        return False
    if _looks_like_reward_button_explanatory_qa(message):
        return False
    count_tokens = ("몇개", "개수", "갯수", "얼마나", "몇개야", "몇개지", "총몇", "있니", "있어", "있나")
    return any(token in normalized for token in count_tokens)


def _looks_like_challenge_claimable_list_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_claimable_query(message):
        return False
    list_tokens = ("이름", "목록", "리스트", "뭐야", "뭐가", "어떤", "알려", "보여")
    return any(token in normalized for token in list_tokens)


def _is_personal_subject(message: str) -> bool:
    normalized = _normalize_text(message)
    explicit_tokens = (
        "내",
        "내가",
        "내정보",
        "내프로필",
        "내계정",
        "내닉네임",
        "내이메일",
        "내지갑",
        "회원님",
    )
    if any(token in normalized for token in explicit_tokens):
        return True
    if re.search(r"(^|[\s,])내(?:[\s,]|$)", message):
        return True
    return bool(re.search(r"(^|[\s,])(?:나|저|제가)(?:[\s,]|$)", message))


def _looks_like_user_profile_summary_request(message: str) -> bool:
    normalized = _normalize_text(message)
    profile_tokens = ("내정보", "프로필", "회원정보", "계정정보", "내계정")
    if not (_is_personal_subject(message) and any(token in normalized for token in profile_tokens)):
        return False

    if _has_navigation_intent(normalized):
        return False
    return True


def _has_navigation_intent(normalized: str) -> bool:
    navigation_tokens = (
        "이동",
        "페이지",
        "화면",
        "탭",
        "메뉴",
        "데려가",
        "가줘",
        "들어가",
        "열어",
        "보내",
        "띄워",
        "가고싶",
        "열고싶",
        "보여줘",
    )
    return any(token in normalized for token in navigation_tokens)


def _has_page_move_intent(normalized: str) -> bool:
    page_move_tokens = (
        "이동",
        "페이지",
        "화면",
        "탭",
        "메뉴",
        "데려가",
        "가줘",
        "들어가",
        "열어",
        "보내",
        "띄워",
        "가고싶",
        "열고싶",
    )
    return any(token in normalized for token in page_move_tokens)


def _looks_like_profile_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    profile_tokens = ("프로필", "마이페이지", "mypage", "profile")
    if not any(token in normalized for token in profile_tokens):
        return False

    scoped_tokens = ("수정", "알림설정", "notification", "modify-user", "set-notification")
    if any(token in normalized for token in scoped_tokens):
        return False

    return _has_navigation_intent(normalized)


def _looks_like_home_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    home_tokens = ("홈", "홈화면", "메인", "메인화면", "첫화면")
    if not any(token in normalized for token in home_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_profile_modify_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    modify_tokens = (
        "정보수정",
        "프로필수정",
        "사용자정보수정",
        "닉네임수정",
        "페이스수정",
        "빈도수정",
        "modify-user",
    )
    if not any(token in normalized for token in modify_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_profile_notification_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    notification_tokens = (
        "알림설정",
        "알림페이지",
        "알림화면",
        "푸시설정",
        "notification",
    )
    if not any(token in normalized for token in notification_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_signin_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    signin_tokens = ("로그인", "signin", "로그인페이지", "로그인화면", "로그인창")
    if not any(token in normalized for token in signin_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_register_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    register_tokens = ("회원가입", "가입", "register", "회원가입페이지", "가입페이지", "회원가입화면")
    if not any(token in normalized for token in register_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_bung_create_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    create_tokens = ("벙만들", "벙생성", "새벙", "모임만들", "모임생성", "번개만들")
    if not any(token in normalized for token in create_tokens):
        return False
    if any(token in normalized for token in ("방법", "어떻게", "설명", "왜")):
        return False
    return _has_navigation_intent(normalized) or _looks_like_desire_request(message)


def _looks_like_challenge_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    challenge_tokens = ("도전과제", "챌린지", "challenge")
    if not any(token in normalized for token in challenge_tokens):
        return False
    if not _has_page_move_intent(normalized):
        return False

    qa_tokens = ("뭐", "무엇", "설명", "어떻게", "왜", "의미", "차이")
    if any(token in normalized for token in qa_tokens):
        return False

    read_tokens = ("목록", "리스트", "몇개", "개수", "갯수", "상태", "현황", "가능", "보상받")
    if any(token in normalized for token in read_tokens):
        return False
    return True


def _looks_like_challenge_progress_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_page_navigation_request(message):
        return False
    progress_tokens = ("진행중", "진행", "progress")
    return any(token in normalized for token in progress_tokens)


def _looks_like_challenge_general_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_page_navigation_request(message):
        return False
    general_tokens = ("일반", "general")
    return any(token in normalized for token in general_tokens)


def _looks_like_challenge_repetitive_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_page_navigation_request(message):
        return False
    repetitive_tokens = ("반복", "repetitive")
    return any(token in normalized for token in repetitive_tokens)


def _looks_like_challenge_completed_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_page_navigation_request(message):
        return False
    completed_tokens = ("완료탭", "완료", "completed")
    return any(token in normalized for token in completed_tokens)


def _looks_like_explore_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _has_page_move_intent(normalized):
        return False
    explore_tokens = ("탐색", "벙탐색", "찾기페이지")
    if not any(token in normalized for token in explore_tokens):
        return False
    qa_tokens = ("뭐", "무엇", "설명", "어떻게", "왜", "의미", "차이")
    return not any(token in normalized for token in qa_tokens)


def _looks_like_bung_search_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _has_navigation_intent(normalized):
        return False
    search_tokens = ("벙검색", "검색페이지", "search")
    return any(token in normalized for token in search_tokens)


def _looks_like_avatar_page_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    avatar_tokens = ("아바타", "avatar", "꾸미러", "캐릭터", "옷아이콘")
    if not any(token in normalized for token in avatar_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_bung_detail_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    if not any(token in normalized for token in ("상세", "디테일", "자세히", "정보")):
        return False
    if any(token in normalized for token in ("수정", "변경", "바꿔", "고쳐", "삭제", "완료", "참여", "취소")):
        return False
    return _has_navigation_intent(normalized) or _looks_like_desire_request(message)


def _looks_like_bung_manage_members_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    manage_tokens = ("멤버관리", "멤버목록", "manage-members", "내보내기화면")
    if not any(token in normalized for token in manage_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_bung_delegate_owner_navigation_request(message: str) -> bool:
    normalized = _normalize_text(message)
    owner_tokens = ("벙주", "권한")
    delegate_tokens = ("넘기", "위임", "양도", "delegate-owner")
    if not any(token in normalized for token in owner_tokens):
        return False
    if not any(token in normalized for token in delegate_tokens):
        return False
    return _has_navigation_intent(normalized)


def _looks_like_user_nickname_request(message: str) -> bool:
    normalized = _normalize_text(message)
    return _is_personal_subject(message) and "닉네임" in normalized


def _looks_like_user_email_request(message: str) -> bool:
    normalized = _normalize_text(message)
    return _is_personal_subject(message) and "이메일" in normalized


def _looks_like_user_wallet_request(message: str) -> bool:
    normalized = _normalize_text(message)
    wallet_tokens = ("지갑주소", "지갑", "월렛주소", "월렛", "블록체인주소")
    return _is_personal_subject(message) and any(token in normalized for token in wallet_tokens)


def _looks_like_running_preferences_request(message: str) -> bool:
    normalized = _normalize_text(message)
    running_tokens = ("러닝페이스", "페이스", "러닝빈도", "빈도", "주몇회", "뛰는빈도", "러닝패턴")
    return _is_personal_subject(message) and any(token in normalized for token in running_tokens)


def _looks_like_suggestions_query(message: str) -> bool:
    normalized = _normalize_text(message)
    suggestion_tokens = ("추천러너", "추천유저", "함께뛴", "자주함께", "추천친구")
    return any(token in normalized for token in suggestion_tokens)


def _looks_like_suggestions_count_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_suggestions_query(message):
        return False
    return any(token in normalized for token in ("몇명", "몇명이나", "개수", "몇개", "몇명이야", "얼마나"))


def _looks_like_suggestions_list_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_suggestions_query(message):
        return False
    return any(token in normalized for token in ("누구", "목록", "리스트", "이름", "알려", "보여"))


def _looks_like_explore_bungs_query(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    explore_tokens = ("탐색", "찾", "전체", "오픈된", "열린", "available")
    return any(token in normalized for token in explore_tokens) and not _is_personal_subject(message)


def _looks_like_explore_bungs_count_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_explore_bungs_query(message):
        return False
    return any(token in normalized for token in ("몇개", "개수", "갯수", "얼마나"))


def _looks_like_explore_bungs_list_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_explore_bungs_query(message):
        return False
    return any(token in normalized for token in ("목록", "리스트", "이름", "뭐가", "어떤", "보여", "알려", "뭐있", "있어"))


def _looks_like_challenge_completed_query(message: str) -> bool:
    normalized = _normalize_text(message)
    challenge_tokens = ("도전과제", "챌린지", "과제")
    completed_tokens = ("완료한", "완료된", "완료", "끝낸", "달성한")
    return any(token in normalized for token in challenge_tokens) and any(token in normalized for token in completed_tokens)


def _looks_like_challenge_completed_count_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_completed_query(message):
        return False
    return any(token in normalized for token in ("몇개", "개수", "갯수", "얼마나"))


def _looks_like_challenge_completed_list_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not _looks_like_challenge_completed_query(message):
        return False
    return any(token in normalized for token in ("목록", "리스트", "이름", "뭐가", "어떤", "보여", "알려"))


def _looks_like_challenge_progress_summary_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if _extract_named_challenge_from_message(message):
        return False
    if any(token in normalized for token in ("완료할수", "완료가능", "보상받", "보상수령", "민팅")):
        return False
    challenge_tokens = ("도전과제", "챌린지", "과제")
    summary_tokens = ("현황", "요약", "진행상황", "정리", "전체")
    return any(token in normalized for token in challenge_tokens) and any(token in normalized for token in summary_tokens)


def _strip_topic_particles(value: str) -> str:
    stripped = value.strip()
    stripped = re.sub(r"^(?:(?:아니\s*그러니까|그러니까|그중에서|중에서|중에|그중)\s*)+", "", stripped).strip()
    for suffix in ("도전과제", "챌린지"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
    for suffix in ("을", "를", "이", "가", "은", "는", "의", "에", "에서"):
        if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
            stripped = stripped[: -len(suffix)].strip()
    return stripped


def _extract_named_challenge_from_message(message: str) -> str | None:
    quote_pattern = re.compile(r"[\"'“”‘’「」『』]([^\"'“”‘’「」『』]{2,40})[\"'“”‘’「」『』]")
    for matched in quote_pattern.findall(message):
        candidate = _strip_topic_particles(matched)
        if len(candidate) >= 2:
            return candidate

    topic_pattern = re.compile(r"([가-힣a-zA-Z0-9][가-힣a-zA-Z0-9\s]{1,40}?)\s*(?:도전과제|챌린지)")
    candidates: list[str] = []
    for topic_match in topic_pattern.finditer(message):
        candidate = _strip_topic_particles(topic_match.group(1))
        if len(candidate) >= 2 and candidate not in {"완료", "진행", "보상", "가능", "일반", "반복"}:
            candidates.append(candidate)
    if candidates:
        return candidates[-1]

    return None


def _looks_like_named_challenge_status_request(message: str) -> bool:
    normalized = _normalize_text(message)
    name = _extract_named_challenge_from_message(message)
    if not name:
        return False
    status_tokens = ("완료", "가능", "상태", "보상", "수령", "달성", "가능해", "할수있")
    challenge_tokens = ("도전과제", "챌린지", "과제")
    return any(token in normalized for token in status_tokens) and any(token in normalized for token in challenge_tokens)


def _extract_weather_location(message: str) -> str | None:
    text = re.sub(r"[?!,.]", " ", message).strip()
    if not text:
        return None

    # "서울시 날씨 어때?" / "서대문구 날씨 알려줘"
    pattern = re.compile(r"([가-힣0-9\s]{1,20}?(?:시|도|군|구|동|읍|면))(?:의|\s)?\s*(?:현재|지금)?\s*(?:날씨|기온|온도)")
    match = pattern.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    # "서울 날씨" 같이 행정구역 접미사가 없는 경우
    loose = re.compile(r"([가-힣0-9\s]{1,20})\s*(?:현재|지금)?\s*(?:날씨|기온|온도)")
    loose_match = loose.search(text)
    if loose_match:
        candidate = loose_match.group(1).strip()
        noise = {"지금", "현재", "오늘", "내일"}
        if candidate and candidate not in noise:
            return candidate
    return None


def _chat_response(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return _general_chat_response(message, history or [])


def _build_pending_clarification(question: str, candidate_lanes: list[str]) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "id": f"clarify-{uuid.uuid4().hex[:12]}",
        "question": question,
        "candidateLanes": candidate_lanes,
        "createdAt": now,
        "resolved": False,
    }


def _pending_clarification(conversation_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(conversation_state, dict):
        return None
    pending = conversation_state.get("pendingClarification")
    if not isinstance(pending, dict):
        return None
    return pending


def _resolve_pending_patch(pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(pending, dict):
        return None
    resolved = dict(pending)
    resolved["resolved"] = True
    return {"pendingClarification": resolved}


def _merge_state_patch(base: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(base, dict):
        return patch if isinstance(patch, dict) else None
    if not isinstance(patch, dict):
        return base
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _looks_like_ambiguous_action_request(message: str) -> bool:
    normalized = _normalize_text(message)
    vague_tokens = ("그거", "그걸", "그것", "이거", "저거")
    action_tokens = ("해줘", "해주세요", "실행", "처리", "해봐")
    return any(token in normalized for token in vague_tokens) and any(token in normalized for token in action_tokens)


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

    if action_key == "home.open_page":
        return {
            "type": "route",
            "href": "/",
        }

    if action_key == "challenge.open_page":
        return {
            "type": "route",
            "href": "/challenges?list=progress&category=general",
        }

    if action_key == "challenge.progress.open_page":
        return {
            "type": "route",
            "href": "/challenges?list=progress&category=general",
        }

    if action_key == "challenge.general.open_page":
        return {
            "type": "route",
            "href": "/challenges?list=progress&category=general",
        }

    if action_key == "challenge.repetitive.open_page":
        return {
            "type": "route",
            "href": "/challenges?list=progress&category=repetitive",
        }

    if action_key == "challenge.completed.open_page":
        return {
            "type": "route",
            "href": "/challenges?list=completed",
        }

    if action_key == "avatar.open_page":
        return {
            "type": "route",
            "href": "/avatar",
        }

    if action_key == "profile.open_page":
        return {
            "type": "route",
            "href": "/profile",
        }

    if action_key == "profile.modify.open_page":
        return {
            "type": "route",
            "href": "/profile/modify-user",
        }

    if action_key == "profile.notification.open_page":
        return {
            "type": "route",
            "href": "/profile/set-notification",
        }

    if action_key == "explore.open_page":
        return {
            "type": "route",
            "href": "/explore",
        }

    if action_key == "bung.search.open_page":
        return {
            "type": "route",
            "href": "/explore",
        }

    if action_key == "bung.detail.open_page":
        bung_id = params.get("bungId")
        href = f"/bung/{bung_id}" if bung_id else "/explore"
        return {
            "type": "route",
            "href": href,
        }

    if action_key == "bung.manage_members.open_page":
        bung_id = params.get("bungId")
        href = f"/bung/{bung_id}/manage-members" if bung_id else "/profile"
        return {
            "type": "route",
            "href": href,
        }

    if action_key == "bung.delegate_owner.open_page":
        bung_id = params.get("bungId")
        href = f"/bung/{bung_id}/delegate-owner" if bung_id else "/profile"
        return {
            "type": "route",
            "href": href,
        }

    if action_key == "auth.signin.open_page":
        return {
            "type": "route",
            "href": "/signin",
        }

    if action_key == "auth.register.open_page":
        return {
            "type": "route",
            "href": "/register",
        }
    return None


def _build_missing_fields(action_key: str, params: dict[str, Any]) -> list[str]:
    required = CAPABILITY_CATALOG.get(action_key, {}).get("required", [])
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


def _nlg_reply(payload: dict[str, Any], default: str) -> str:
    cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    cached = _NLG_CACHE.get(cache_key)
    if cached:
        cached_at, cached_reply = cached
        if (time.time() - cached_at) <= NLG_CACHE_TTL_SECONDS:
            return cached_reply
        _NLG_CACHE.pop(cache_key, None)

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.25,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        chain = ACTION_NLG_PROMPT | llm | StrOutputParser()
        reply = str(chain.invoke({"payload": json.dumps(payload, ensure_ascii=False)})).strip()
        if not reply:
            reply = default
    except Exception:
        reply = default

    _NLG_CACHE[cache_key] = (time.time(), reply)
    return reply


def _fallback_collect_reply(action_key: str, missing_fields: list[str]) -> str:
    labels = [FIELD_LABELS.get(field, field) for field in missing_fields]
    default = "진행에 필요한 정보를 한 번만 더 확인할게요."
    return _nlg_reply(
        {
            "mode": "collect",
            "actionKey": action_key,
            "missingFields": labels,
            "isReadAction": action_key.startswith("read."),
        },
        default,
    )


def _fallback_ready_reply(action_key: str) -> str:
    return _nlg_reply(
        {
            "mode": "ready",
            "actionKey": action_key,
        },
        "요청을 실행할 준비가 되었어요. 계속 진행할까요?",
    )


def _fallback_navigate_reply(action_key: str) -> str:
    return _nlg_reply(
        {
            "mode": "navigate",
            "actionKey": action_key,
            "isUnsupportedAutoRun": action_key == "bung.invite_members",
        },
        "해당 작업을 진행할 수 있는 화면으로 안내해드릴게요.",
    )


def _looks_like_desire_request(message: str) -> bool:
    normalized = message.lower().replace(" ", "")
    desire_tokens = (
        "하고싶",
        "원해",
        "가고싶",
        "열고싶",
        "찾고싶",
        "보고싶",
        "보고싶어",
    )
    return any(token in normalized for token in desire_tokens)


def _looks_like_navigation_request(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    nav_tokens = (
        "페이지",
        "화면",
        "탭",
        "메뉴",
        "이동",
        "데려가",
        "들어가",
        "가줘",
        "열어",
        "보여",
    )
    return any(token in lowered or token in normalized for token in nav_tokens)


def _needs_action_clarification(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    if any(token in normalized for token in ("뭐야", "무엇", "어떻게", "왜", "언제", "어디", "설명", "알려")):
        return False
    if "?" in message and not _looks_like_navigation_request(message):
        return False
    action_tokens = (
        "해줘",
        "해주세요",
        "부탁",
        "실행",
        "만들어",
        "생성",
        "완료",
        "취소",
        "참여",
        "삭제",
        "넘겨",
        "내보내",
        "탈퇴",
        "받아",
        "남겨",
    )
    return (
        _looks_like_navigation_request(message)
        or _looks_like_desire_request(message)
        or any(token in normalized for token in action_tokens)
    )


def _looks_like_plain_qa(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    question_tokens = (
        "뭐야",
        "무엇",
        "설명",
        "알려",
        "어떻게",
        "왜",
        "언제",
        "어디",
        "의미",
        "차이",
    )
    action_tokens = (
        "해줘",
        "해주세요",
        "실행",
        "이동",
        "데려가",
        "가줘",
        "열어줘",
        "만들어줘",
        "삭제해줘",
        "완료해줘",
        "참여해줘",
        "취소해줘",
        "부탁",
    )
    if any(token in normalized for token in action_tokens):
        return False
    return ("?" in message) or any(token in normalized for token in question_tokens)


def _looks_like_other_person_data_request(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    if "벙" not in lowered:
        return False

    if not any(token in normalized for token in ("이름", "목록", "리스트", "몇개", "개수", "갯수", "참여", "참가")):
        return False

    if any(token in normalized for token in ("다른사람", "타인", "남의", "친구", "회원", "유저", "사용자", "누구", "누가")):
        return True

    if re.search(r"[가-힣a-z0-9_]{2,}\s*님", message):
        return True

    if re.search(r"[가-힣a-z0-9_]{2,}\s*의\s*벙", message) and not re.search(r"(내|나|저|제)\s*의?\s*벙", message):
        return True

    return False


def _looks_like_other_person_user_data_request(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    personal_data_tokens = ("이메일", "닉네임", "지갑", "월렛", "프로필", "계정", "러닝페이스", "빈도")
    if not any(token in normalized for token in personal_data_tokens):
        return False

    if re.search(r"[가-힣a-z0-9_]{2,}\s*님", message):
        return True

    if any(token in normalized for token in ("다른사람", "타인", "남의", "친구", "회원", "유저", "사용자", "누구", "누가")):
        return True

    if re.search(r"[가-힣a-z0-9_]{2,}\s*의\s*(이메일|닉네임|프로필|계정|지갑)", message):
        return True

    return False


def _looks_like_my_bung_count_request(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    if "벙" not in lowered:
        return False

    count_tokens = ("몇개", "개수", "갯수", "수량", "총몇", "얼마나")
    has_count_intent = any(token in normalized for token in count_tokens) or bool(
        re.search(r"(참여중인|참여하고있는|참가중인|참가하고있는)?벙수", normalized)
    )
    if not has_count_intent:
        return False

    personal_tokens = (
        "내가",
        "내벙",
        "나",
        "저",
        "제가",
        "참여중",
        "참여하고있는",
        "참가중",
        "참가하고있는",
    )
    return any(token in normalized for token in personal_tokens)


def _looks_like_my_bung_names_request(message: str) -> bool:
    lowered = message.lower()
    normalized = lowered.replace(" ", "")
    if "벙" not in lowered:
        return False

    name_tokens = ("이름", "목록", "리스트", "나열")
    if not any(token in normalized for token in name_tokens):
        return False

    personal_tokens = (
        "내가",
        "내벙",
        "나",
        "저",
        "제가",
        "참여중",
        "참여하고있는",
        "참가중",
        "참가하고있는",
    )
    return any(token in normalized for token in personal_tokens)


def _looks_like_my_bung_member_count_request(message: str) -> bool:
    normalized = _normalize_text(message)
    list_tokens = ("누구", "목록", "리스트", "명단")
    if any(token in normalized for token in list_tokens):
        return False
    count_tokens = ("몇명", "인원", "참여자수", "참가자수", "멤버수", "몇명이", "몇명인지")
    if not any(token in normalized for token in count_tokens):
        return False
    return "벙" in normalized or any(token in normalized for token in ("1번", "첫번째", "두번째", "그벙", "방금벙"))


def _looks_like_my_bung_members_request(message: str) -> bool:
    normalized = _normalize_text(message)
    member_tokens = ("참여자", "멤버", "누가", "목록", "리스트", "명단")
    if not any(token in normalized for token in member_tokens):
        return False
    if any(token in normalized for token in ("몇명", "인원", "수")):
        return False
    return "벙" in normalized or any(token in normalized for token in ("1번", "첫번째", "두번째", "그벙", "방금벙"))


def _looks_like_bung_join_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    if not any(token in normalized for token in ("참여", "참가", "들어가")):
        return False
    if any(token in normalized for token in ("취소", "나가", "탈퇴", "나갈")):
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "원해", "하고싶", "싶어"))


def _looks_like_bung_leave_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    leave_tokens = ("참여취소", "참여를취소", "참가취소", "나가", "나갈래", "떠나", "빠져")
    if any(token in normalized for token in leave_tokens):
        return True
    return (
        any(token in normalized for token in ("참여", "참가"))
        and "취소" in normalized
        and any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "싶어"))
    )


def _looks_like_bung_complete_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    if "완료" not in normalized:
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "처리", "싶어"))


def _looks_like_bung_certify_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not any(token in normalized for token in ("인증", "참여인증")):
        return False
    if "벙" not in normalized and "참여" not in normalized:
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "처리", "싶어"))


def _looks_like_bung_modify_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    if not any(token in normalized for token in ("수정", "변경", "바꿔", "고쳐")):
        return False
    if any(token in normalized for token in ("어디서", "방법", "어떻게", "설명")):
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "원해", "하고싶", "싶어"))


def _looks_like_bung_delete_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "벙" not in normalized:
        return False
    if not any(token in normalized for token in ("삭제", "지워", "없애", "폭파")):
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "원해", "싶어", "처리"))


def _looks_like_delegate_owner_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not any(token in normalized for token in ("벙주", "권한")):
        return False
    if not any(token in normalized for token in ("넘겨", "위임", "양도")):
        return False
    return any(token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "원해", "싶어", "줘"))


def _looks_like_kick_member_request(message: str) -> bool:
    normalized = _normalize_text(message)
    kick_tokens = ("내보내", "강퇴", "추방", "빼줘", "빼내", "제외")
    target_tokens = ("사람", "멤버", "참여자", "유저", "회원")
    if not any(token in normalized for token in kick_tokens):
        return False
    if "벙" in normalized:
        return True
    return any(token in normalized for token in target_tokens)


def _looks_like_invite_members_request(message: str) -> bool:
    normalized = _normalize_text(message)
    invite_tokens = ("초대", "invite")
    return any(token in normalized for token in invite_tokens) and any(
        token in normalized for token in ("해줘", "해주세요", "부탁", "해주", "해줄", "원해", "싶어")
    )


def _looks_like_delete_account_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "탈퇴" not in normalized:
        return False
    if re.search(r"[가-힣a-z0-9_]{2,}\s*님", message):
        return False
    if any(token in normalized for token in ("다른사람", "타인", "남의", "친구", "타계정")):
        return False
    return any(token in normalized for token in ("계정", "회원", "서비스", "오픈런", "회원탈퇴", "계정탈퇴"))


def _build_bung_ref_params(
    message: str,
    conversation_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    params: dict[str, Any] = {}
    state_patch: dict[str, Any] | None = None

    bung_ref = parse_bung_ref_from_message(message, conversation_state)
    if bung_ref:
        params["bungRef"] = bung_ref
        resolved = resolve_bung_ref(bung_ref, conversation_state)
        if resolved.get("status") == "resolved" and isinstance(resolved.get("match"), dict):
            match = resolved["match"]
            bung_id = match.get("bungId")
            if isinstance(bung_id, str) and bung_id.strip():
                params["bungId"] = bung_id
                state_patch = _build_focus_state_patch(bung_id)

    return params, state_patch


def _infer_scope_from_message(message: str) -> str:
    normalized = message.lower().replace(" ", "")
    if any(token in normalized for token in ("지금", "현재", "참여중", "참여하고있는", "참가중", "진행중")):
        return "active"
    return "all"


def _build_focus_state_patch(bung_id: str) -> dict[str, Any]:
    return {
        "focus": {"lastBungId": bung_id},
        "lastReadResult": {
            "type": "my_bung_focus",
            "bungId": bung_id,
        },
    }


def _deterministic_route(
    message: str,
    history: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
    conversation_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _looks_like_smalltalk(message):
        return _chat_response(message, history)

    if _looks_like_openrun_feature_explanatory_qa(message, history, conversation_state):
        return _qa_response(message, history)

    if _looks_like_delete_account_request(message):
        return _proposal_response(
            action_key="user.delete_account",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="계정 탈퇴 실행",
        )

    if _looks_like_ambiguous_action_request(message):
        return _clarify_action_response(candidate_lanes=["action", "read", "qa"])

    if _looks_like_other_person_data_request(message):
        return {
            "kind": "qa",
            "reply": "개인정보 보호 정책상 다른 사용자의 벙 정보는 조회할 수 없습니다.",
            "sources": [],
            "proposal": None,
            "statePatch": None,
        }

    if _looks_like_other_person_user_data_request(message):
        return {
            "kind": "qa",
            "reply": "개인정보 보호 정책상 다른 사용자의 정보는 조회할 수 없습니다.",
            "sources": [],
            "proposal": None,
            "statePatch": None,
        }

    if _looks_like_home_page_navigation_request(message):
        return _proposal_response(
            action_key="home.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="홈 페이지 이동",
        )

    if _looks_like_profile_modify_navigation_request(message):
        return _proposal_response(
            action_key="profile.modify.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.96,
            summary="프로필 정보 수정 페이지 이동",
        )

    if _looks_like_profile_notification_navigation_request(message):
        return _proposal_response(
            action_key="profile.notification.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.96,
            summary="알림 설정 페이지 이동",
        )

    if _looks_like_profile_page_navigation_request(message):
        return _proposal_response(
            action_key="profile.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.97,
            summary="프로필 페이지 이동",
        )

    if _looks_like_signin_page_navigation_request(message):
        return _proposal_response(
            action_key="auth.signin.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="로그인 페이지 이동",
        )

    if _looks_like_register_page_navigation_request(message):
        return _proposal_response(
            action_key="auth.register.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="회원가입 페이지 이동",
        )

    if _looks_like_bung_create_navigation_request(message):
        return _proposal_response(
            action_key="bung.create",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.96,
            summary="벙 생성 화면 이동",
        )

    if _looks_like_avatar_page_navigation_request(message):
        return _proposal_response(
            action_key="avatar.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.96,
            summary="아바타 페이지 이동",
        )

    if _looks_like_bung_search_page_navigation_request(message):
        return _proposal_response(
            action_key="bung.search.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="벙 검색 페이지 이동",
        )

    if _looks_like_explore_page_navigation_request(message):
        return _proposal_response(
            action_key="explore.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="벙 탐색/검색 페이지 이동",
        )

    if _looks_like_challenge_completed_navigation_request(message):
        return _proposal_response(
            action_key="challenge.completed.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="완료 도전과제 페이지 이동",
        )

    if _looks_like_challenge_repetitive_navigation_request(message):
        return _proposal_response(
            action_key="challenge.repetitive.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="반복 도전과제 페이지 이동",
        )

    if _looks_like_challenge_general_navigation_request(message):
        return _proposal_response(
            action_key="challenge.general.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="일반 도전과제 페이지 이동",
        )

    if _looks_like_challenge_progress_navigation_request(message):
        return _proposal_response(
            action_key="challenge.progress.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="진행 중 도전과제 페이지 이동",
        )

    if _looks_like_challenge_page_navigation_request(message):
        return _proposal_response(
            action_key="challenge.open_page",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.95,
            summary="도전과제 페이지 이동",
        )

    if _looks_like_bung_manage_members_navigation_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.manage_members.open_page",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="벙 멤버 관리 페이지 이동",
            state_patch=state_patch,
        )

    if _looks_like_bung_delegate_owner_navigation_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.delegate_owner.open_page",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="벙주 위임 페이지 이동",
            state_patch=state_patch,
        )

    if _looks_like_bung_detail_navigation_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.detail.open_page",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="벙 상세 페이지 이동",
            state_patch=state_patch,
        )

    if _looks_like_weather_request(message):
        location = _extract_weather_location(message)
        params: dict[str, Any] = {}
        if location:
            params["location"] = location
        return _proposal_response(
            action_key="read.weather.current",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="지역 현재 날씨 조회",
        )

    if _looks_like_user_profile_summary_request(message):
        return _proposal_response(
            action_key="read.user.profile.summary",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.93,
            summary="내 프로필 요약 조회",
        )

    if _looks_like_user_nickname_request(message):
        return _proposal_response(
            action_key="read.user.profile.nickname",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.93,
            summary="내 닉네임 조회",
        )

    if _looks_like_user_email_request(message):
        return _proposal_response(
            action_key="read.user.profile.email",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.93,
            summary="내 이메일 조회",
        )

    if _looks_like_user_wallet_request(message):
        return _proposal_response(
            action_key="read.user.profile.wallet",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.93,
            summary="내 지갑 주소 조회",
        )

    if _looks_like_running_preferences_request(message):
        return _proposal_response(
            action_key="read.user.running.preferences",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.92,
            summary="내 러닝 설정 조회",
        )

    if _looks_like_suggestions_count_request(message):
        return _proposal_response(
            action_key="read.user.suggestions.count",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.9,
            summary="추천 러너 수 조회",
        )

    if _looks_like_suggestions_list_request(message):
        return _proposal_response(
            action_key="read.user.suggestions.list",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.9,
            summary="추천 러너 목록 조회",
        )

    if _looks_like_explore_bungs_count_request(message):
        return _proposal_response(
            action_key="read.bungs.explore.count",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.88,
            summary="탐색 벙 개수 조회",
        )

    if _looks_like_explore_bungs_list_request(message):
        return _proposal_response(
            action_key="read.bungs.explore.names",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.88,
            summary="탐색 벙 목록 조회",
        )

    if _looks_like_challenge_claimable_count_request(message):
        return _proposal_response(
            action_key="read.challenge.claimable.count",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.91,
            summary="보상 가능 도전과제 개수 조회",
        )

    if _looks_like_challenge_claimable_list_request(message):
        return _proposal_response(
            action_key="read.challenge.claimable.list",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.91,
            summary="보상 가능 도전과제 목록 조회",
        )

    if _looks_like_challenge_completed_count_request(message):
        return _proposal_response(
            action_key="read.challenge.completed.count",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.89,
            summary="완료 도전과제 개수 조회",
        )

    if _looks_like_challenge_completed_list_request(message):
        return _proposal_response(
            action_key="read.challenge.completed.list",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.89,
            summary="완료 도전과제 목록 조회",
        )

    if _looks_like_challenge_progress_summary_request(message):
        return _proposal_response(
            action_key="read.challenge.progress.summary",
            parsed_params={},
            pending_action=pending_action,
            confidence=0.89,
            summary="도전과제 현황 요약 조회",
        )

    if _looks_like_named_challenge_status_request(message):
        challenge_name = _extract_named_challenge_from_message(message)
        if challenge_name:
            return _proposal_response(
                action_key="read.challenge.by_name.status",
                parsed_params={"challengeName": challenge_name},
                pending_action=pending_action,
                confidence=0.9,
                summary="특정 도전과제 상태 조회",
            )

    if _looks_like_bung_modify_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.modify",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="벙 수정 화면 이동",
            state_patch=state_patch,
        )

    if _looks_like_bung_delete_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.delete",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="벙 삭제 실행",
            state_patch=state_patch,
        )

    if _looks_like_delegate_owner_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.delegate_owner",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="벙주 위임 실행",
            state_patch=state_patch,
        )

    if _looks_like_kick_member_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.kick_member",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="멤버 내보내기 실행",
            state_patch=state_patch,
        )

    if _looks_like_invite_members_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.invite_members",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="멤버 초대 요청",
            state_patch=state_patch,
        )

    if _looks_like_bung_join_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.join",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.93,
            summary="벙 참여 실행",
            state_patch=state_patch,
        )

    if _looks_like_bung_leave_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.leave",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.93,
            summary="벙 참여 취소 실행",
            state_patch=state_patch,
        )

    if _looks_like_bung_complete_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.complete",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.93,
            summary="벙 완료 실행",
            state_patch=state_patch,
        )

    if _looks_like_bung_certify_request(message):
        params, state_patch = _build_bung_ref_params(message, conversation_state)
        return _proposal_response(
            action_key="bung.certify",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="벙 참여 인증 실행",
            state_patch=state_patch,
        )

    if _looks_like_my_bung_count_request(message):
        return _proposal_response(
            action_key="read.my_bungs.count",
            parsed_params={"scope": _infer_scope_from_message(message)},
            pending_action=pending_action,
            confidence=0.95,
            summary="내 벙 개수 조회",
        )

    if _looks_like_my_bung_names_request(message):
        return _proposal_response(
            action_key="read.my_bungs.names",
            parsed_params={"scope": _infer_scope_from_message(message)},
            pending_action=pending_action,
            confidence=0.95,
            summary="내 벙 이름 목록 조회",
        )

    inferred_ref = parse_bung_ref_from_message(message, conversation_state)
    has_member_count_intent = _looks_like_my_bung_member_count_request(message) or (
        inferred_ref is not None and any(token in _normalize_text(message) for token in ("몇명", "인원", "참여자수", "멤버수"))
    )

    if has_member_count_intent:
        bung_ref = inferred_ref
        params: dict[str, Any] = {}
        state_patch: dict[str, Any] | None = None
        if bung_ref:
            params["bungRef"] = bung_ref
            resolved = resolve_bung_ref(bung_ref, conversation_state)
            if resolved.get("status") == "resolved" and isinstance(resolved.get("match"), dict):
                match = resolved["match"]
                bung_id = match.get("bungId")
                if isinstance(bung_id, str) and bung_id.strip():
                    params["bungId"] = bung_id
                    state_patch = _build_focus_state_patch(bung_id)

        return _proposal_response(
            action_key="read.my_bung.member_count",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.92,
            summary="내 벙 참여자 수 조회",
            state_patch=state_patch,
        )

    has_member_list_intent = _looks_like_my_bung_members_request(message) or (
        inferred_ref is not None
        and any(token in _normalize_text(message) for token in ("누구", "목록", "리스트", "명단", "참여자", "멤버"))
        and not any(token in _normalize_text(message) for token in ("몇명", "인원", "참여자수", "멤버수"))
    )

    if has_member_list_intent:
        bung_ref = inferred_ref
        params: dict[str, Any] = {}
        state_patch: dict[str, Any] | None = None
        if bung_ref:
            params["bungRef"] = bung_ref
            resolved = resolve_bung_ref(bung_ref, conversation_state)
            if resolved.get("status") == "resolved" and isinstance(resolved.get("match"), dict):
                match = resolved["match"]
                bung_id = match.get("bungId")
                if isinstance(bung_id, str) and bung_id.strip():
                    params["bungId"] = bung_id
                    state_patch = _build_focus_state_patch(bung_id)

        return _proposal_response(
            action_key="read.my_bung.members",
            parsed_params=params,
            pending_action=pending_action,
            confidence=0.9,
            summary="내 벙 참여자 목록 조회",
            state_patch=state_patch,
        )

    fallback = _contextual_read_fallback(message, history, pending_action)
    if fallback is not None:
        return fallback
    return None


def _clarify_action_response(
    question: str | None = None,
    candidate_lanes: list[str] | None = None,
) -> dict[str, Any]:
    default_question = (
        "실행 요청으로 이해했지만 어떤 동작인지 확신이 낮아요. "
        "원하는 동작을 한 번만 더 구체적으로 말해 주세요."
    )
    clarify_question = question or _nlg_reply(
        {
            "mode": "clarify",
            "candidateLanes": candidate_lanes or ["action", "read", "qa"],
        },
        default_question,
    )
    lanes = candidate_lanes or ["action", "read", "qa"]
    patch = {"pendingClarification": _build_pending_clarification(clarify_question, lanes)}
    return {
        "kind": "action_collect",
        "reply": clarify_question,
        "sources": [],
        "proposal": None,
        "statePatch": patch,
    }


def _clarify_from_pending(pending: dict[str, Any]) -> dict[str, Any]:
    question = pending.get("question")
    reply = question if isinstance(question, str) and question.strip() else _nlg_reply(
        {"mode": "clarify_followup"},
        "조금 더 구체적으로 말씀해 주세요.",
    )
    return {
        "kind": "action_collect",
        "reply": reply,
        "sources": [],
        "proposal": None,
        "statePatch": {"pendingClarification": pending},
    }


def _proposal_response(
    action_key: str,
    parsed_params: dict[str, Any] | None,
    pending_action: dict[str, Any] | None,
    confidence: float,
    summary: str | None = None,
    state_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capability = CAPABILITY_CATALOG[action_key]
    params = _normalize_params(parsed_params, pending_action)
    missing_fields = _build_missing_fields(action_key, params)

    action_mode = capability["mode"]
    if action_mode == "execute":
        normalized_kind = "action_collect" if missing_fields else "action_ready"
    elif action_mode == "navigate":
        normalized_kind = "action_navigate"
    elif action_mode == "read":
        normalized_kind = "action_collect" if missing_fields else "qa"
    else:
        normalized_kind = "action_unavailable"

    if not summary:
        summary = f"{action_key} 요청"

    if normalized_kind == "action_collect":
        reply = _fallback_collect_reply(action_key, missing_fields)
    elif normalized_kind == "action_ready":
        reply = _fallback_ready_reply(action_key)
    elif action_mode == "read" and not missing_fields:
        reply = _nlg_reply(
            {"mode": "read_ready", "actionKey": action_key},
            "확인해서 알려드릴게요.",
        )
    else:
        reply = _fallback_navigate_reply(action_key)

    navigation = _default_navigation(action_key, params) if action_mode == "navigate" else None

    return {
        "kind": normalized_kind,
        "reply": reply,
        "sources": [],
        "proposal": {
            "actionKey": action_key,
            "summary": summary,
            "params": params,
            "missingFields": missing_fields,
            "dangerLevel": capability.get("danger", "low"),
            "navigation": navigation,
            "confidence": confidence,
        },
        "statePatch": state_patch,
    }


def _result_lane(result: dict[str, Any]) -> str:
    kind = result.get("kind")
    proposal = result.get("proposal")
    if kind == "chat":
        return "chat"
    if isinstance(proposal, dict):
        action_key = proposal.get("actionKey")
        if isinstance(action_key, str) and action_key.startswith("read."):
            return "read"
        if isinstance(action_key, str):
            return "action"
    if kind in {"action_collect", "action_ready", "action_navigate", "action_unavailable"}:
        return "action"
    return "qa"


def _decorate_response(result: dict[str, Any]) -> dict[str, Any]:
    lane = _result_lane(result)
    sources = result.get("sources")
    show_sources = lane == "qa" and isinstance(sources, list) and len(sources) > 0

    proposal = result.get("proposal")
    kind = result.get("kind")
    show_action_buttons = (
        lane == "action"
        and isinstance(proposal, dict)
        and kind in {"action_collect", "action_ready", "action_navigate", "action_unavailable"}
    )

    decorated = dict(result)
    decorated["lane"] = lane
    decorated["uiHints"] = {
        "showSources": show_sources,
        "showActionButtons": show_action_buttons,
    }
    if "statePatch" not in decorated:
        decorated["statePatch"] = None
    return decorated


def _sanitize_navigation(
    navigation: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(navigation, dict):
        return fallback

    nav_type = navigation.get("type")
    if nav_type not in ("route", "modal"):
        return fallback

    if nav_type == "route":
        href = navigation.get("href")
        if not isinstance(href, str) or not href.startswith("/"):
            return fallback
        return {"type": "route", "href": href}

    modal_key = navigation.get("modalKey")
    if not isinstance(modal_key, str) or modal_key.strip() == "":
        return fallback
    sanitized: dict[str, Any] = {"type": "modal", "modalKey": modal_key}
    prefill = navigation.get("prefill")
    if isinstance(prefill, dict):
        sanitized["prefill"] = prefill
    return sanitized


def _plan_with_llm(
    message: str,
    history: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
    conversation_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    cache_payload = {
        "message": message,
        "history": history[-6:],
        "pending_action": pending_action or {},
        "conversation_state": conversation_state or {},
    }
    cache_key = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)
    cached = _LLM_ROUTER_CACHE.get(cache_key)
    if cached:
        cached_at, cached_value = cached
        if (time.time() - cached_at) <= ROUTER_CACHE_TTL_SECONDS:
            return cached_value
        _LLM_ROUTER_CACHE.pop(cache_key, None)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    chain = PLANNER_PROMPT | llm | StrOutputParser()

    raw = chain.invoke(
        {
            "catalog": json.dumps(_catalog_for_prompt(), ensure_ascii=False),
            "history": _compact_history(history),
            "conversation_state": _compact_conversation_state(conversation_state),
            "pending_action": json.dumps(pending_action or {}, ensure_ascii=False),
            "message": message,
        }
    )
    parsed = _safe_json_loads(raw)
    _LLM_ROUTER_CACHE[cache_key] = (time.time(), parsed)
    return parsed


def _history_text(history: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in history[-8:]:
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return " ".join(parts).lower()


def _looks_like_openrun_domain_text(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(token in normalized for token in OPENRUN_DOMAIN_TOKENS)


def _history_is_openrun_context(history: list[dict[str, Any]]) -> bool:
    return _looks_like_openrun_domain_text(_history_text(history))


def _is_openrun_domain_message(
    message: str,
    history: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
    conversation_state: dict[str, Any] | None,
) -> bool:
    if _looks_like_openrun_domain_text(message):
        return True

    if isinstance(pending_action, dict):
        proposal = pending_action.get("proposal")
        if isinstance(proposal, dict) and isinstance(proposal.get("actionKey"), str):
            return True

    pending = _pending_clarification(conversation_state)
    if isinstance(pending, dict) and not bool(pending.get("resolved")):
        return True

    has_structured_context = False
    if isinstance(conversation_state, dict):
        last_read = conversation_state.get("lastReadResult")
        focus = conversation_state.get("focus")
        entities = conversation_state.get("entities")
        has_structured_context = bool(last_read) or bool(focus) or bool(entities)

    if _history_is_openrun_context(history) and has_structured_context:
        normalized = _normalize_text(message)
        follow_up_tokens = (
            "그거",
            "그걸",
            "그벙",
            "그페이지",
            "방금",
            "아까",
            "1번",
            "첫번째",
            "두번째",
            "이동",
        )
        if any(token in normalized for token in follow_up_tokens):
            return True

    return False


def _general_chat_response(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    cache_payload = {
        "message": message,
        "history": history[-6:],
    }
    cache_key = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)
    cached = _GENERAL_CHAT_CACHE.get(cache_key)
    if cached:
        cached_at, cached_reply = cached
        if (time.time() - cached_at) <= GENERAL_CHAT_CACHE_TTL_SECONDS:
            return {
                "kind": "chat",
                "reply": cached_reply,
                "sources": [],
                "proposal": None,
                "statePatch": None,
            }
        _GENERAL_CHAT_CACHE.pop(cache_key, None)

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.35,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        chain = GENERAL_CHAT_PROMPT | llm | StrOutputParser()
        reply = str(
            chain.invoke(
                {
                    "history": _compact_history(history),
                    "message": message,
                }
            )
        ).strip()
        if not reply:
            reply = "좋은 질문이에요. 조금 더 자세히 말씀해 주시면 더 정확히 답해드릴게요."
    except Exception:
        reply = "네, 들리고 있어요. 말씀하신 내용을 기준으로 계속 도와드릴게요."

    _GENERAL_CHAT_CACHE[cache_key] = (time.time(), reply)
    return {
        "kind": "chat",
        "reply": reply,
        "sources": [],
        "proposal": None,
        "statePatch": None,
    }


def _history_mentions_my_bungs(history: list[dict[str, Any]]) -> bool:
    text = _history_text(history)
    tokens = (
        "참여 중인 벙",
        "참여하고 있는 벙",
        "참여했던 벙",
        "내 벙",
        "벙은 총",
    )
    return any(token in text for token in tokens)


def _history_mentions_challenges(history: list[dict[str, Any]]) -> bool:
    text = _history_text(history)
    tokens = ("도전과제", "챌린지", "보상 받기", "보상받기", "민팅")
    return any(token in text for token in tokens)


def _infer_scope_from_history(history: list[dict[str, Any]]) -> str:
    text = _history_text(history)
    if any(token in text for token in ("지금", "현재", "참여 중인 벙", "참여하고 있는 벙")):
        return "active"
    return "all"


def _looks_like_name_list_followup(message: str) -> bool:
    normalized = message.lower().replace(" ", "")
    name_tokens = ("이름", "목록", "리스트", "각각", "나열", "뭐였")
    return any(token in normalized for token in name_tokens)


def _looks_like_claimable_challenge_followup(message: str) -> tuple[bool, bool]:
    if _looks_like_reward_button_explanatory_qa(message):
        return (False, False)

    normalized = message.lower().replace(" ", "")
    claim_tokens = ("보상받기", "보상받을", "보상을받을", "민팅가능", "완료가능", "수령가능")
    if not any(token in normalized for token in claim_tokens):
        return (False, False)

    list_tokens = ("이름", "목록", "리스트", "뭐야", "뭐가", "어떤", "알려", "보여")
    count_tokens = ("몇개", "개수", "갯수", "얼마나", "총몇")
    wants_list = any(token in normalized for token in list_tokens)
    wants_count = any(token in normalized for token in count_tokens)
    return (wants_count, wants_list)


def _looks_like_completed_challenge_followup(message: str) -> tuple[bool, bool]:
    normalized = message.lower().replace(" ", "")
    completed_tokens = ("완료", "달성", "끝낸")
    if not any(token in normalized for token in completed_tokens):
        return (False, False)

    list_tokens = ("이름", "목록", "리스트", "뭐야", "뭐가", "어떤", "알려", "보여")
    count_tokens = ("몇개", "개수", "갯수", "얼마나", "총몇")
    wants_list = any(token in normalized for token in list_tokens)
    wants_count = any(token in normalized for token in count_tokens)
    return (wants_count, wants_list)


def _contextual_read_fallback(
    message: str,
    history: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _history_mentions_my_bungs(history):
        if _looks_like_name_list_followup(message):
            scope = _infer_scope_from_history(history)
            return _proposal_response(
                action_key="read.my_bungs.names",
                parsed_params={"scope": scope},
                pending_action=pending_action,
                confidence=0.86,
                summary="내 벙 이름 목록 조회",
            )

    if _history_mentions_challenges(history):
        wants_count, wants_list = _looks_like_claimable_challenge_followup(message)
        if wants_list:
            return _proposal_response(
                action_key="read.challenge.claimable.list",
                parsed_params={},
                pending_action=pending_action,
                confidence=0.86,
                summary="보상 가능 도전과제 목록 조회",
            )
        if wants_count:
            return _proposal_response(
                action_key="read.challenge.claimable.count",
                parsed_params={},
                pending_action=pending_action,
                confidence=0.86,
                summary="보상 가능 도전과제 개수 조회",
            )
        completed_count, completed_list = _looks_like_completed_challenge_followup(message)
        if completed_list:
            return _proposal_response(
                action_key="read.challenge.completed.list",
                parsed_params={},
                pending_action=pending_action,
                confidence=0.84,
                summary="완료 도전과제 목록 조회",
            )
        if completed_count:
            return _proposal_response(
                action_key="read.challenge.completed.count",
                parsed_params={},
                pending_action=pending_action,
                confidence=0.84,
                summary="완료 도전과제 개수 조회",
            )

    return None


def _is_rag_not_found_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    not_found_tokens = (
        "제공된문서에서해당정보를찾을수없습니다",
        "문서에서해당정보를찾을수없습니다",
        "문서에없는내용",
    )
    return any(token in normalized for token in not_found_tokens)


def _rewrite_openrun_qa_query(question: str) -> str:
    normalized = _normalize_text(question)

    if "보상받기" in normalized and any(token in normalized for token in ("누르면", "누를", "버튼", "얻게", "받게")):
        return "도전과제에서 보상 받기 버튼을 누르면 어떤 보상(NFT)을 받나요?"

    if any(token in normalized for token in ("일반", "반복")) and any(token in normalized for token in ("차이", "다른", "구분")):
        return "도전과제 일반과 반복 카테고리의 차이는 무엇인가요?"

    return question


def _qa_response(question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result = rag_query(question)
    answer = str(result.get("answer", "") or "").strip()
    if answer and _is_rag_not_found_answer(answer):
        if _looks_like_openrun_domain_text(question):
            rewritten = _rewrite_openrun_qa_query(question)
            if rewritten != question:
                retry_result = rag_query(rewritten)
                retry_answer = str(retry_result.get("answer", "") or "").strip()
                if retry_answer and not _is_rag_not_found_answer(retry_answer):
                    return {
                        "kind": "qa",
                        "reply": retry_result["answer"],
                        "sources": retry_result["sources"],
                        "proposal": None,
                    }

        if _looks_like_openrun_domain_text(question):
            return {
                "kind": "chat",
                "reply": _nlg_reply(
                    {
                        "mode": "domain_not_found",
                        "question": question,
                    },
                    "해당 내용은 현재 연결된 오픈런 문서/데이터에서 확인되지 않아요.",
                ),
                "sources": [],
                "proposal": None,
                "statePatch": None,
            }
        # 비도메인 질의는 일반 대화로 자연스럽게 폴백한다.
        return _general_chat_response(question, history or [])
    return {
        "kind": "qa",
        "reply": result["answer"],
        "sources": result["sources"],
        "proposal": None,
    }


def _plan_assistant_legacy(
    message: str,
    history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or []

    if _looks_like_other_person_data_request(message):
        return {
            "kind": "qa",
            "reply": _nlg_reply(
                {"mode": "privacy_block", "topic": "other_user_bung_data"},
                "개인정보 보호 정책상 다른 사용자의 벙 정보는 조회할 수 없습니다.",
            ),
            "sources": [],
            "proposal": None,
            "statePatch": None,
        }

    if _looks_like_my_bung_count_request(message):
        return _proposal_response(
            action_key="read.my_bungs.count",
            parsed_params={"scope": _infer_scope_from_message(message)},
            pending_action=pending_action,
            confidence=0.95,
            summary="내 벙 개수 조회",
        )

    if _looks_like_my_bung_names_request(message):
        return _proposal_response(
            action_key="read.my_bungs.names",
            parsed_params={"scope": _infer_scope_from_message(message)},
            pending_action=pending_action,
            confidence=0.95,
            summary="내 벙 이름 목록 조회",
        )

    if _looks_like_plain_qa(message):
        fallback = _contextual_read_fallback(message, history, pending_action)
        if fallback is not None:
            return fallback
        return _qa_response(message, history)

    parsed = _plan_with_llm(
        message=message,
        history=history,
        pending_action=pending_action,
        conversation_state=None,
    )

    if not isinstance(parsed, dict):
        fallback = _contextual_read_fallback(message, history, pending_action)
        if fallback is not None:
            return fallback
        if _needs_action_clarification(message):
            return _clarify_action_response()
        return _qa_response(message, history)

    intent = str(parsed.get("intent", "qa") or "qa").strip().lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    key = parsed.get("key")
    action_key = key if isinstance(key, str) else None
    reply = str(parsed.get("reply", "") or "").strip()
    summary = parsed.get("summary")
    summary_value = summary if isinstance(summary, str) and summary.strip() else None
    params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}

    if intent == "refuse":
        return {
            "kind": "qa",
            "reply": reply
            or _nlg_reply(
                {"mode": "privacy_block", "topic": "other_user_data"},
                "개인정보 보호 정책상 다른 사용자의 정보는 조회할 수 없습니다.",
            ),
            "sources": [],
            "proposal": None,
            "statePatch": None,
        }

    if intent == "qa":
        fallback = _contextual_read_fallback(message, history, pending_action)
        if fallback is not None:
            return fallback
        return _qa_response(message, history)

    if intent == "clarify":
        return _clarify_action_response()

    if confidence < CONFIDENCE_THRESHOLD:
        if intent in {"action", "read"} or _needs_action_clarification(message):
            return _clarify_action_response()
        return _qa_response(message, history)

    if intent == "read":
        if action_key not in READ_CATALOG:
            return _clarify_action_response()
        result = _proposal_response(
            action_key=action_key,
            parsed_params=params,
            pending_action=pending_action,
            confidence=confidence,
            summary=summary_value,
        )
        if reply:
            result["reply"] = reply
        return result

    if intent == "action":
        if action_key not in ACTION_CATALOG:
            return _clarify_action_response()
        result = _proposal_response(
            action_key=action_key,
            parsed_params=params,
            pending_action=pending_action,
            confidence=confidence,
            summary=summary_value,
        )
        if reply:
            result["reply"] = reply

        if isinstance(result.get("proposal"), dict):
            current_navigation = result["proposal"].get("navigation")
            candidate_navigation = parsed.get("navigation") if isinstance(parsed.get("navigation"), dict) else None
            result["proposal"]["navigation"] = _sanitize_navigation(candidate_navigation, current_navigation)
        return result

    if _needs_action_clarification(message):
        return _clarify_action_response()

    fallback = _contextual_read_fallback(message, history, pending_action)
    if fallback is not None:
        return fallback
    return _qa_response(message, history)


def _should_shortcut_to_qa(
    message: str,
    history: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
    conversation_state: dict[str, Any] | None,
) -> bool:
    if not _is_openrun_domain_message(message, history, pending_action, conversation_state):
        return False
    if _looks_like_smalltalk(message):
        return False
    normalized = message.lower().replace(" ", "")
    if any(token in normalized for token in ("참여자", "멤버", "몇명", "인원", "1번", "첫번째", "그벙")):
        return False
    return _looks_like_plain_qa(message)


def _plan_assistant_v2(
    message: str,
    history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
    conversation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or []

    deterministic = _deterministic_route(
        message=message,
        history=history,
        pending_action=pending_action,
        conversation_state=conversation_state,
    )
    if deterministic is not None:
        return deterministic

    if not _is_openrun_domain_message(message, history, pending_action, conversation_state):
        return _general_chat_response(message, history)

    if _should_shortcut_to_qa(message, history, pending_action, conversation_state):
        return _qa_response(message, history)

    parsed = _plan_with_llm(
        message=message,
        history=history,
        pending_action=pending_action,
        conversation_state=conversation_state,
    )

    if not isinstance(parsed, dict):
        if _needs_action_clarification(message):
            return _clarify_action_response()
        return _qa_response(message, history)

    intent = str(parsed.get("intent", "qa") or "qa").strip().lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    key = parsed.get("key")
    action_key = key if isinstance(key, str) else None
    reply = str(parsed.get("reply", "") or "").strip()
    summary = parsed.get("summary")
    summary_value = summary if isinstance(summary, str) and summary.strip() else None
    params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}

    if intent == "refuse":
        return {
            "kind": "qa",
            "reply": reply
            or _nlg_reply(
                {"mode": "privacy_block", "topic": "other_user_data"},
                "개인정보 보호 정책상 다른 사용자의 정보는 조회할 수 없습니다.",
            ),
            "sources": [],
            "proposal": None,
            "statePatch": None,
        }

    if intent == "qa":
        if _looks_like_smalltalk(message):
            return _chat_response(message, history)
        return _qa_response(message, history)

    if intent == "clarify":
        return _clarify_action_response()

    if confidence < CONFIDENCE_THRESHOLD:
        if intent in {"action", "read"} or _needs_action_clarification(message):
            return _clarify_action_response()
        return _qa_response(message, history)

    if intent == "read":
        if action_key not in READ_CATALOG:
            return _clarify_action_response()

        parsed_state_patch = parsed.get("statePatch")
        state_patch = parsed_state_patch if isinstance(parsed_state_patch, dict) else None
        result = _proposal_response(
            action_key=action_key,
            parsed_params=params,
            pending_action=pending_action,
            confidence=confidence,
            summary=summary_value,
            state_patch=state_patch,
        )
        if reply:
            result["reply"] = reply
        return result

    if intent == "action":
        if action_key not in ACTION_CATALOG:
            return _clarify_action_response()
        result = _proposal_response(
            action_key=action_key,
            parsed_params=params,
            pending_action=pending_action,
            confidence=confidence,
            summary=summary_value,
        )
        if reply:
            result["reply"] = reply

        if isinstance(result.get("proposal"), dict):
            current_navigation = result["proposal"].get("navigation")
            candidate_navigation = parsed.get("navigation") if isinstance(parsed.get("navigation"), dict) else None
            result["proposal"]["navigation"] = _sanitize_navigation(candidate_navigation, current_navigation)
        return result

    if _needs_action_clarification(message):
        return _clarify_action_response()
    return _qa_response(message, history)


def _router_log_context(result: dict[str, Any]) -> dict[str, Any]:
    proposal = result.get("proposal")
    action_key = proposal.get("actionKey") if isinstance(proposal, dict) else None
    confidence = proposal.get("confidence") if isinstance(proposal, dict) else None
    return {
        "lane": result.get("lane"),
        "kind": result.get("kind"),
        "actionKey": action_key,
        "confidence": confidence,
    }


def plan_assistant(
    message: str,
    history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
    conversation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or []
    mode = ROUTER_MODE if ROUTER_MODE in {"shadow", "active"} else "active"
    pending = _pending_clarification(conversation_state)
    pending_active = isinstance(pending, dict) and not bool(pending.get("resolved"))

    if pending_active and _is_ack_only_message(message):
        return _decorate_response(_clarify_from_pending(pending))

    new_result = _plan_assistant_v2(
        message=message,
        history=history,
        pending_action=pending_action,
        conversation_state=conversation_state,
    )
    if pending_active:
        has_new_pending = (
            new_result.get("kind") == "action_collect"
            and isinstance(new_result.get("statePatch"), dict)
            and isinstance(new_result["statePatch"].get("pendingClarification"), dict)
        )
        if not has_new_pending:
            resolved_patch = _resolve_pending_patch(pending)
            current_patch = new_result.get("statePatch") if isinstance(new_result.get("statePatch"), dict) else None
            new_result["statePatch"] = _merge_state_patch(current_patch, resolved_patch)

    if mode == "shadow":
        legacy_result = _plan_assistant_legacy(
            message=message,
            history=history,
            pending_action=pending_action,
        )
        legacy_result = _decorate_response(legacy_result)
        decorated_new = _decorate_response(new_result)
        ROUTER_LOGGER.info(
            "assistant_router_shadow old=%s new=%s",
            json.dumps(_router_log_context(legacy_result), ensure_ascii=False),
            json.dumps(_router_log_context(decorated_new), ensure_ascii=False),
        )
        return legacy_result

    decorated_result = _decorate_response(new_result)
    ROUTER_LOGGER.info("assistant_router_active result=%s", json.dumps(_router_log_context(decorated_result), ensure_ascii=False))
    return decorated_result
