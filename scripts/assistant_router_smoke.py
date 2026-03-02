#!/usr/bin/env python3
"""
Assistant router smoke test.

Runs a focused set of representative prompts against /rag/assistant and validates
lane/action routing so regressions can be caught before frontend deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Case:
    name: str
    message: str
    expected_lane: str | tuple[str, ...]
    expected_action: str | None | tuple[str | None, ...] = None
    expected_kind: str | tuple[str, ...] | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    conversation_state: dict[str, Any] | None = None


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _as_tuple(value: str | None | tuple[Any, ...]) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _evaluate_case(endpoint: str, case: Case, timeout: float) -> tuple[bool, str]:
    payload = {
        "message": case.message,
        "history": case.history,
        "pendingAction": None,
        "conversationState": case.conversation_state,
    }
    data = _post_json(endpoint, payload, timeout=timeout)

    lane = data.get("lane")
    kind = data.get("kind")
    proposal = data.get("proposal") if isinstance(data.get("proposal"), dict) else {}
    action = proposal.get("actionKey")

    checks: list[str] = []
    ok = True

    expected_lanes = _as_tuple(case.expected_lane)
    if lane not in expected_lanes:
        ok = False
        checks.append(f"lane expected={expected_lanes} actual={lane}")

    if case.expected_kind is not None and kind not in _as_tuple(case.expected_kind):
        ok = False
        checks.append(f"kind expected={_as_tuple(case.expected_kind)} actual={kind}")

    if action not in _as_tuple(case.expected_action):
        ok = False
        checks.append(f"action expected={_as_tuple(case.expected_action)} actual={action}")

    status = "PASS" if ok else "FAIL"
    detail = ", ".join(checks) if checks else f"kind={kind}, lane={lane}, action={action}"
    return ok, f"[{status}] {case.name}: {detail}"


def _default_cases() -> list[Case]:
    followup_state = {
        "entities": {
            "bungs": [
                {"bungId": "b1", "name": "Infinite Query Test Bung 3", "order": 1, "status": "PARTICIPATING"},
                {"bungId": "b2", "name": "Infinite Query Test Bung 4", "order": 2, "status": "PARTICIPATING"},
            ]
        },
        "focus": {"lastBungId": "b1"},
        "lastReadResult": {"type": "my_bungs_list", "scope": "active", "total": 2},
    }
    challenge_context = [{"role": "assistant", "content": "도전과제 페이지에서 보상 받기 탭을 안내했어요."}]

    return [
        # chat lane
        Case(name="chat_smalltalk_heard", message="내 말이 들려?", expected_lane="chat", expected_kind="chat"),
        Case(name="chat_smalltalk_greeting", message="안녕!", expected_lane="chat", expected_kind="chat"),
        Case(name="chat_general_name", message="네 이름이 뭐야?", expected_lane="chat", expected_kind="chat"),
        Case(name="chat_general_thanks", message="고마워", expected_lane="chat", expected_kind="chat"),
        Case(name="chat_general_non_domain", message="점심 뭐 먹을까?", expected_lane="chat", expected_kind="chat"),
        Case(name="chat_general_outside_topic", message="주식 추천해줘", expected_lane="chat", expected_kind="chat"),

        # qa lane
        Case(name="qa_challenge_definition", message="도전과제가 뭐야?", expected_lane="qa", expected_action=None),
        Case(name="qa_bung_definition", message="벙이 뭐야?", expected_lane="qa", expected_action=None),
        Case(name="qa_avatar_explain", message="아바타 기능 설명해줘", expected_lane="qa", expected_action=None),
        Case(name="qa_explore_explain", message="탐색 페이지는 뭐 하는 곳이야?", expected_lane="qa", expected_action=None),
        Case(name="qa_bung_modify_guide", message="벙 수정은 어디서 해?", expected_lane="qa", expected_action=None),
        Case(name="chat_smalltalk", message="내 말이 들려?", expected_lane="chat", expected_kind="chat"),
        Case(name="qa_general_challenge_explain", message="도전과제 일반/반복 차이가 뭐야?", expected_lane="qa", expected_action=None),
        Case(
            name="qa_reward_button_explain",
            message="보상받기 버튼 누르면 어떤 보상을 받아?",
            expected_lane="qa",
            expected_action=None,
            history=challenge_context,
        ),
        Case(
            name="qa_reward_button_explain_variant",
            message="보상받기를 누르면 뭘 얻게 돼?",
            expected_lane="qa",
            expected_action=None,
            history=challenge_context,
        ),

        # read lane - challenge
        Case(
            name="read_claimable_count",
            message="지금 보상 받기 가능한 도전과제가 몇 개야?",
            expected_lane="read",
            expected_action="read.challenge.claimable.count",
        ),
        Case(
            name="read_claimable_list",
            message="보상 받을 수 있는 도전과제 목록 보여줘",
            expected_lane="read",
            expected_action="read.challenge.claimable.list",
        ),
        Case(
            name="read_completed_count",
            message="완료한 도전과제가 몇 개야?",
            expected_lane="read",
            expected_action="read.challenge.completed.count",
        ),
        Case(
            name="read_completed_list",
            message="완료한 도전과제 목록 보여줘",
            expected_lane="read",
            expected_action="read.challenge.completed.list",
        ),
        Case(
            name="read_challenge_progress_summary",
            message="도전과제 현황 요약해줘",
            expected_lane="read",
            expected_action="read.challenge.progress.summary",
        ),
        Case(
            name="read_named_challenge_status",
            message="중에서 '프로필 완성하기' 도전과제를 완료할 수 있는 상태야?",
            expected_lane="read",
            expected_action="read.challenge.by_name.status",
        ),

        # read lane - bungs
        Case(
            name="read_bung_count",
            message="지금 내가 참여하고 있는 벙이 몇 개야?",
            expected_lane="read",
            expected_action="read.my_bungs.count",
        ),
        Case(
            name="read_bung_count_variant",
            message="현재 참여 중인 벙 수 알려줘",
            expected_lane="read",
            expected_action="read.my_bungs.count",
        ),
        Case(
            name="read_bung_names",
            message="지금 참여 중인 벙 이름 알려줘",
            expected_lane="read",
            expected_action="read.my_bungs.names",
        ),
        Case(
            name="read_bung_names_variant",
            message="내가 참여한 벙 목록 보여줘",
            expected_lane="read",
            expected_action="read.my_bungs.names",
        ),
        Case(
            name="read_bung_member_count_by_index",
            message="1번 벙 참여자가 몇 명이야?",
            expected_lane="read",
            expected_action="read.my_bung.member_count",
            conversation_state=followup_state,
        ),
        Case(
            name="read_bung_member_count_by_name",
            message="Infinite Query Test Bung 3 참여자가 몇 명이야?",
            expected_lane="read",
            expected_action="read.my_bung.member_count",
            conversation_state=followup_state,
        ),
        Case(
            name="read_bung_member_count_by_deictic",
            message="그 벙 참여자 수가 몇 명이야?",
            expected_lane="read",
            expected_action="read.my_bung.member_count",
            conversation_state=followup_state,
        ),
        Case(
            name="read_bung_members_by_index",
            message="1번 벙 멤버가 누구야?",
            expected_lane="read",
            expected_action="read.my_bung.members",
            conversation_state=followup_state,
        ),
        Case(
            name="read_bung_members_by_deictic",
            message="방금 말한 벙 참여자 목록 보여줘",
            expected_lane="read",
            expected_action="read.my_bung.members",
            conversation_state=followup_state,
        ),

        # read lane - profile / suggestion / explore / weather
        Case(
            name="read_profile_summary",
            message="내 프로필 요약해줘",
            expected_lane="read",
            expected_action="read.user.profile.summary",
        ),
        Case(
            name="read_profile_nickname",
            message="내 닉네임 알려줘",
            expected_lane="read",
            expected_action="read.user.profile.nickname",
        ),
        Case(
            name="read_profile_email",
            message="내 이메일이 뭐야?",
            expected_lane="read",
            expected_action="read.user.profile.email",
        ),
        Case(
            name="read_profile_wallet",
            message="내 지갑 주소 알려줘",
            expected_lane="read",
            expected_action="read.user.profile.wallet",
        ),
        Case(
            name="read_running_preferences",
            message="내 러닝 페이스랑 빈도 알려줘",
            expected_lane="read",
            expected_action="read.user.running.preferences",
        ),
        Case(
            name="read_suggestions_count",
            message="추천 러너가 몇 명이야?",
            expected_lane="read",
            expected_action="read.user.suggestions.count",
        ),
        Case(
            name="read_suggestions_list",
            message="추천 러너 목록 보여줘",
            expected_lane="read",
            expected_action="read.user.suggestions.list",
        ),
        Case(
            name="read_explore_count",
            message="탐색 가능한 벙이 몇 개야?",
            expected_lane="read",
            expected_action="read.bungs.explore.count",
        ),
        Case(
            name="read_explore_names",
            message="탐색 벙 이름 리스트 알려줘",
            expected_lane="read",
            expected_action="read.bungs.explore.names",
        ),
        Case(
            name="read_explore_names_variant",
            message="탐색 벙 뭐 있어?",
            expected_lane="read",
            expected_action="read.bungs.explore.names",
        ),
        Case(
            name="read_weather_local",
            message="서울시 서대문구 날씨 어때?",
            expected_lane="read",
            expected_action="read.weather.current",
        ),
        Case(
            name="read_weather_without_location_collect",
            message="오늘 기온 어때?",
            expected_lane="read",
            expected_action="read.weather.current",
            expected_kind="action_collect",
        ),
        Case(
            name="read_claimable_existence",
            message="보상 받기 가능한 과제 있니?",
            expected_lane="read",
            expected_action="read.challenge.claimable.count",
        ),

        # action lane - navigate
        Case(
            name="action_navigate_challenge",
            message="도전과제 페이지로 데려가줘",
            expected_lane="action",
            expected_action="challenge.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_home",
            message="홈 화면으로 이동시켜줘",
            expected_lane="action",
            expected_action="home.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_challenge_variant",
            message="도전과제 화면으로 이동해줘",
            expected_lane="action",
            expected_action="challenge.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_challenge_general",
            message="일반 도전과제 탭으로 이동해줘",
            expected_lane="action",
            expected_action="challenge.general.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_challenge_repetitive",
            message="반복 도전과제 탭으로 이동해줘",
            expected_lane="action",
            expected_action="challenge.repetitive.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_challenge_completed",
            message="완료 도전과제 탭으로 이동해줘",
            expected_lane="action",
            expected_action="challenge.completed.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_avatar",
            message="아바타 페이지로 가고 싶어",
            expected_lane="action",
            expected_action="avatar.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_profile",
            message="프로필 페이지로 이동시켜줘",
            expected_lane="action",
            expected_action="profile.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_profile_modify",
            message="프로필 정보 수정 페이지로 이동해줘",
            expected_lane="action",
            expected_action="profile.modify.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_profile_notification",
            message="알림 설정 페이지로 이동해줘",
            expected_lane="action",
            expected_action="profile.notification.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_explore",
            message="벙 탐색 페이지로 이동해줘",
            expected_lane="action",
            expected_action="explore.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_bung_search",
            message="벙 검색 페이지로 이동해줘",
            expected_lane="action",
            expected_action="bung.search.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_signin",
            message="로그인 페이지로 이동해줘",
            expected_lane="action",
            expected_action="auth.signin.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_register",
            message="회원가입 페이지로 이동해줘",
            expected_lane="action",
            expected_action="auth.register.open_page",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_bung_create",
            message="벙 만들기를 하고 싶어",
            expected_lane="action",
            expected_action="bung.create",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_bung_modify_collect",
            message="벙 수정하고 싶어",
            expected_lane="action",
            expected_action="bung.modify",
            expected_kind="action_navigate",
        ),
        Case(
            name="action_navigate_bung_detail",
            message="1번 벙 상세 페이지로 이동해줘",
            expected_lane="action",
            expected_action="bung.detail.open_page",
            expected_kind="action_navigate",
            conversation_state=followup_state,
        ),
        Case(
            name="action_navigate_bung_manage_members",
            message="1번 벙 멤버 관리 페이지로 이동해줘",
            expected_lane="action",
            expected_action="bung.manage_members.open_page",
            expected_kind="action_navigate",
            conversation_state=followup_state,
        ),
        Case(
            name="action_navigate_bung_delegate_owner_page",
            message="1번 벙 벙주 위임 페이지로 이동해줘",
            expected_lane="action",
            expected_action="bung.delegate_owner.open_page",
            expected_kind="action_navigate",
            conversation_state=followup_state,
        ),

        # action lane - execute / collect
        Case(
            name="action_execute_join_collect",
            message="이 벙 참여해줘",
            expected_lane="action",
            expected_action="bung.join",
        ),
        Case(
            name="action_execute_join_desire_collect",
            message="이 벙 들어가고 싶어",
            expected_lane="action",
            expected_action="bung.join",
        ),
        Case(
            name="action_execute_leave_collect",
            message="이 벙 참여 취소해줘",
            expected_lane="action",
            expected_action="bung.leave",
        ),
        Case(
            name="action_execute_complete_collect",
            message="이 벙 완료해줘",
            expected_lane="action",
            expected_action="bung.complete",
        ),
        Case(
            name="action_execute_certify_collect",
            message="참여 인증해줘",
            expected_lane="action",
            expected_action="bung.certify",
        ),
        Case(
            name="action_execute_delete_collect",
            message="이 벙 삭제해줘",
            expected_lane="action",
            expected_action="bung.delete",
        ),
        Case(
            name="action_delegate_owner_collect",
            message="벙주 넘겨줘",
            expected_lane="action",
            expected_action="bung.delegate_owner",
        ),
        Case(
            name="action_kick_member_collect",
            message="저 사람 내보내줘",
            expected_lane="action",
            expected_action="bung.kick_member",
        ),
        Case(
            name="action_invite_unavailable",
            message="유저 초대해줘",
            expected_lane="action",
            expected_action="bung.invite_members",
            expected_kind="action_unavailable",
        ),
        Case(
            name="action_execute_delete_account",
            message="계정 탈퇴해줘",
            expected_lane="action",
            expected_action="user.delete_account",
            expected_kind=("action_ready", "action_collect"),
        ),

        # clarify / policy
        Case(
            name="clarify_ambiguous_action",
            message="그거 해줘",
            expected_lane="action",
            expected_action=None,
            expected_kind="action_collect",
        ),
        Case(
            name="privacy_block_other_user",
            message="민수님 이메일 알려줘",
            expected_lane="qa",
            expected_action=None,
        ),
        Case(
            name="privacy_block_other_user_bung",
            message="다른 사람 벙 목록 알려줘",
            expected_lane="qa",
            expected_action=None,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test assistant routing via /rag/assistant")
    parser.add_argument("--endpoint", default="http://localhost:8000/rag/assistant", help="assistant endpoint URL")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    parser.add_argument("--rounds", type=int, default=1, help="repeat full suite N rounds to catch flaky routing")
    args = parser.parse_args()

    cases = _default_cases()
    passed = 0
    failed = 0

    rounds = max(args.rounds, 1)
    for round_idx in range(rounds):
        if rounds > 1:
            print(f"\n== Round {round_idx + 1}/{rounds} ==")

        for case in cases:
            try:
                ok, line = _evaluate_case(args.endpoint, case, timeout=args.timeout)
            except urllib.error.HTTPError as exc:
                failed += 1
                print(f"[FAIL] {case.name}: HTTP {exc.code} {exc.reason}")
                continue
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {case.name}: {exc}")
                continue

            if ok:
                passed += 1
            else:
                failed += 1
            print(line)

    total = passed + failed
    print(f"\nResult: {passed}/{total} passed, {failed} failed (rounds={rounds})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
