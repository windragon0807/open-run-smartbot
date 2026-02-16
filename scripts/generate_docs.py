"""
AI 문서 자동 업데이트 스크립트

프론트엔드 코드 변경(git diff)을 분석하여
knowledge/ 문서 중 사양이 변경된 부분을 감지하고 업데이트를 제안합니다.

사용법:
  # CI에서 사용 (크로스 리포 — frontend 리포 경로와 커밋 범위 지정)
  python scripts/generate_docs.py --mode ci --apply \
    --frontend-dir /path/to/frontend --before abc123 --after def456

  # 로컬 모노리포에서 사용 (기존 방식 호환)
  python scripts/generate_docs.py --mode ci --apply

  # 수동 분석 (적용 없이 결과만 확인)
  python scripts/generate_docs.py --mode api --from-ref HEAD~5 --to-ref HEAD

출력: JSON 형식으로 변경 제안을 반환합니다.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
BOT_DIR = SCRIPT_DIR.parent
KNOWLEDGE_DIR = BOT_DIR / "knowledge"
REPO_ROOT = BOT_DIR.parent


def get_git_diff(
    from_ref: str = "HEAD~1",
    to_ref: str = "HEAD",
    frontend_dir: str | None = None,
) -> str:
    """프론트엔드 코드의 git diff를 가져옵니다."""
    try:
        if frontend_dir:
            # 크로스 리포: frontend 리포에서 직접 diff (src/ 기준)
            result = subprocess.run(
                [
                    "git", "diff", from_ref, to_ref,
                    "--", "src/",
                    ":!src/**/*.css",
                    ":!src/**/*.test.*",
                    ":!src/**/*.spec.*",
                ],
                capture_output=True,
                text=True,
                cwd=frontend_dir,
            )
        else:
            # 모노리포: 루트에서 frontend/src/ 기준 diff
            result = subprocess.run(
                [
                    "git", "diff", from_ref, to_ref,
                    "--", "frontend/src/",
                    ":!frontend/src/**/*.css",
                    ":!frontend/src/**/*.test.*",
                    ":!frontend/src/**/*.spec.*",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
        return result.stdout
    except Exception as e:
        print(f"git diff 실행 실패: {e}", file=sys.stderr)
        return ""


def get_changed_files(
    from_ref: str = "HEAD~1",
    to_ref: str = "HEAD",
    frontend_dir: str | None = None,
) -> list[str]:
    """변경된 프론트엔드 파일 목록을 가져옵니다."""
    try:
        if frontend_dir:
            # 크로스 리포: frontend 리포에서 직접 조회 (src/ 기준)
            result = subprocess.run(
                [
                    "git", "diff", "--name-only", from_ref, to_ref,
                    "--", "src/",
                ],
                capture_output=True,
                text=True,
                cwd=frontend_dir,
            )
        else:
            # 모노리포: 루트에서 frontend/src/ 기준 조회
            result = subprocess.run(
                [
                    "git", "diff", "--name-only", from_ref, to_ref,
                    "--", "frontend/src/",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []


def load_knowledge_docs() -> dict[str, str]:
    """knowledge/ 폴더의 모든 문서를 읽어옵니다."""
    docs = {}
    for f in sorted(KNOWLEDGE_DIR.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".md"):
            docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def analyze_changes(diff: str, changed_files: list[str], docs: dict[str, str]) -> dict:
    """OpenAI API로 코드 변경이 사양에 영향을 주는지 분석합니다."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 문서 목록 요약
    docs_summary = "\n\n".join([
        f"=== {name} ===\n{content}"
        for name, content in docs.items()
    ])

    prompt = f"""당신은 소프트웨어 문서 관리 전문가입니다.

아래에 프론트엔드 코드의 변경사항(git diff)과 현재 서비스 문서가 있습니다.

## 작업
1. 코드 변경사항을 분석하여 **사용자 경험이나 서비스 사양에 영향을 주는 변경**인지 판단하세요.
2. 영향이 있다면, 어떤 문서를 어떻게 수정해야 하는지 구체적으로 제안하세요.
3. 단순 리팩토링, 스타일 변경, 버그 수정 등 사양에 영향이 없는 변경은 무시하세요.

## 변경된 파일 목록
{chr(10).join(changed_files)}

## Git Diff
```
{diff[:15000]}
```

## 현재 서비스 문서
{docs_summary}

## 응답 형식 (반드시 JSON으로)
{{
  "has_changes": true/false,
  "summary": "변경 요약 (한국어)",
  "updates": [
    {{
      "filename": "문서 파일명",
      "reason": "수정 이유",
      "updated_content": "수정된 전체 문서 내용"
    }}
  ]
}}

사양 변경이 없으면 has_changes를 false로, updates를 빈 배열로 반환하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 정확하고 보수적인 문서 관리자입니다. 확실한 사양 변경만 반영합니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"has_changes": False, "summary": "분석 결과를 파싱할 수 없습니다.", "updates": []}


def apply_updates(updates: list[dict]) -> list[str]:
    """분석 결과를 knowledge/ 파일에 적용합니다."""
    applied = []
    for update in updates:
        file_path = KNOWLEDGE_DIR / update["filename"]
        if file_path.exists() and file_path.suffix in (".txt", ".md"):
            file_path.write_text(update["updated_content"], encoding="utf-8")
            applied.append(update["filename"])
    return applied


def main():
    parser = argparse.ArgumentParser(description="AI 문서 자동 업데이트")
    parser.add_argument("--mode", choices=["ci", "api"], default="ci",
                        help="실행 모드: ci (CI/CD용), api (관리 UI용)")
    parser.add_argument("--from-ref", default="HEAD~1", help="시작 커밋 (기본: HEAD~1)")
    parser.add_argument("--to-ref", default="HEAD", help="끝 커밋 (기본: HEAD)")
    parser.add_argument("--apply", action="store_true", help="변경사항을 파일에 직접 적용")
    parser.add_argument("--frontend-dir", default=None,
                        help="크로스 리포용: frontend 리포의 로컬 경로")
    parser.add_argument("--before", default=None,
                        help="크로스 리포용: 변경 전 커밋 SHA")
    parser.add_argument("--after", default=None,
                        help="크로스 리포용: 변경 후 커밋 SHA")
    args = parser.parse_args()

    # 커밋 범위 결정: --before/--after가 있으면 우선, 없으면 --from-ref/--to-ref 사용
    from_ref = args.before if args.before else args.from_ref
    to_ref = args.after if args.after else args.to_ref

    # 1. git diff 추출
    diff = get_git_diff(from_ref, to_ref, args.frontend_dir)
    changed_files = get_changed_files(from_ref, to_ref, args.frontend_dir)

    if not diff or not changed_files:
        result = {"has_changes": False, "summary": "프론트엔드 변경사항이 없습니다.", "updates": []}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 2. knowledge 문서 로드
    docs = load_knowledge_docs()

    # 3. AI 분석
    result = analyze_changes(diff, changed_files, docs)

    # 4. 적용 (--apply 옵션)
    if args.apply and result.get("has_changes") and result.get("updates"):
        applied = apply_updates(result["updates"])
        result["applied_files"] = applied

    # 5. 결과 출력
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
