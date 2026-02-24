# OpenRun Bot

> OpenRun 서비스의 RAG 기반 AI 챗봇 서버 — 서비스 문서를 이해하고 답변하며, 프론트엔드 코드 변경에 따라 문서를 자동 업데이트합니다.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-FF6F00)
![GCP Cloud Run](https://img.shields.io/badge/Cloud_Run-GCP-4285F4?logo=googlecloud)

---

## 주요 기능

| 기능 | 엔드포인트 | 설명 |
|------|-----------|------|
| **RAG 질문 답변** | `POST /rag/query` | 질문을 임베딩 → ChromaDB에서 관련 청크 3개 검색 → GPT-4o-mini가 답변 생성 |
| **문서 위치 검색** | `POST /rag/locate` | 질문과 관련된 문서/섹션을 JSON으로 반환 |
| **문서 수정 제안** | `POST /rag/edit` | 수정 요청을 받아 AI가 수정안 생성 (원본 + 수정본 반환) |
| **문서 관리 UI** | `GET /manage` | 웹 에디터 + 챗봇 통합 인터페이스 (diff 미리보기, hunk별 수락/거부) |
| **실시간 동기화** | 자동 | `watchfiles`로 knowledge/ 폴더 감시, 변경 시 벡터 DB 자동 반영 |
| **문서 자동 업데이트** | CI/CD | 프론트엔드 코드 변경 → AI 분석 → knowledge 문서 수정 PR 자동 생성 |

---

## 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **서버** | FastAPI, Uvicorn |
| **RAG 파이프라인** | LangChain, LangChain-OpenAI, LangChain-Chroma |
| **벡터 DB** | ChromaDB (PersistentClient) |
| **LLM** | OpenAI GPT-4o-mini |
| **임베딩** | OpenAI text-embedding-3-small |
| **문서 감시** | watchfiles |
| **배포** | Docker + GCP Cloud Run |
| **CI/CD** | GitHub Actions |

---

## 시작하기

### 사전 요구사항

- **Python** 3.13+
- **Docker** (로컬 개발 시 선택)
- **OpenAI API Key**

### 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY를 입력하세요

# 3. 서버 실행
python main.py
```

### Docker로 실행

```bash
docker compose up --build
```

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | — |
| `CHROMA_DIR` | ChromaDB 저장 경로 | `./chroma_db` |

> Cloud Run 배포 시 `CHROMA_DIR=/tmp/chroma_db`로 설정됩니다 (읽기 전용 파일시스템).

### 정상 동작 확인

`http://localhost:8000` 접속 시 상태 응답이 반환됩니다.

```json
{ "status": "running", "message": "Smart Chatbot API 서버가 실행 중입니다." }
```

서버 시작 후 knowledge/ 문서의 초기 동기화가 완료될 때까지 RAG 엔드포인트는 503을 반환합니다.

---

## 프로젝트 구조

```
bot/
├── main.py                    # FastAPI 서버 진입점, 모든 API 엔드포인트
├── schemas.py                 # Pydantic 요청/응답 모델
├── Dockerfile                 # Docker 이미지 빌드 (python:3.13-slim)
├── docker-compose.yml         # 로컬 개발용 (knowledge/ 볼륨 마운트)
├── requirements.txt           # Python 의존성
│
├── rag/                       # RAG 파이프라인 모듈
│   ├── chain.py               #   LangChain RAG 체인 (query, locate, edit)
│   ├── document_loader.py     #   문서 로드 및 청크 분할 (500자, 50자 오버랩)
│   ├── vector_store.py        #   ChromaDB 벡터 저장소 (싱글톤)
│   └── watcher.py             #   knowledge/ 폴더 실시간 감시
│
├── knowledge/                 # RAG 참조 문서 (마크다운 10개)
│   ├── 01_서비스_개요.md
│   ├── 02_회원가입_및_로그인.md
│   ├── ...
│   └── 10_용어_사전.md
│
├── static/manage.html         # 문서 관리 UI (웹 에디터 + 챗봇)
├── scripts/generate_docs.py   # 프론트엔드 변경 분석 → 문서 자동 업데이트
└── .github/workflows/
    ├── bot-deploy.yml         #   GCP Cloud Run 자동 배포
    └── doc-sync.yml           #   프론트엔드 변경 → 문서 동기화 PR 생성
```

---

## API 엔드포인트

### 상태 & 모델

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/` | 서버 상태 확인 |
| `GET` | `/models` | 사용 가능한 GPT 모델 목록 |

### RAG

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/rag/query` | 문서 기반 질문 답변 (출처 포함) |
| `POST` | `/rag/locate` | 질문 관련 문서 위치 검색 |
| `POST` | `/rag/edit` | AI 기반 문서 수정 제안 |

### 문서 관리

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/documents` | knowledge/ 문서 목록 조회 |
| `GET` | `/documents/status` | 동기화 상태 조회 (파일별 상태, DB 건강, 이벤트 로그) |
| `GET` | `/documents/{filename}` | 문서 내용 조회 |
| `PUT` | `/documents/{filename}` | 문서 수정 (저장 → watcher가 벡터 DB 자동 반영) |
| `POST` | `/documents/sync` | 수동 전체 동기화 (`force_reset=True`) |
| `DELETE` | `/documents/reset` | 벡터 DB 초기화 |

### 관리 UI

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/manage` | 웹 기반 문서 편집기 + AI 챗봇 |

---

## 아키텍처

### 요청 흐름

```
[사용자] → [Next.js 프론트엔드] → [/api/chat API Route (프록시)] → [FastAPI /rag/query]
```

### RAG 파이프라인

```
질문 입력
  ↓
Retrieval  : 질문 임베딩 → ChromaDB에서 관련 청크 3개 검색
  ↓
Augmentation : 검색된 문서를 시스템 프롬프트에 삽입
  ↓
Generation : GPT-4o-mini가 문서 기반 답변 생성 → 응답 반환
```

### 프론트엔드 ↔ 봇 자동 연동

```
① 프론트엔드 src/** 변경 → main push
  ↓
② notify-bot-repo.yml → repository_dispatch (type: frontend-changed)
  ↓
③ doc-sync.yml → 프론트엔드 체크아웃 + git diff 추출
  ↓
④ generate_docs.py → AI가 사양 변경 여부 판단
  ↓
⑤ 사양 변경 시 → knowledge 문서 수정 → docs/auto-update PR 자동 생성
  ↓
⑥ 팀원 리뷰 후 머지 → bot-deploy.yml → Cloud Run 자동 배포
```

---

## 배포 & CI/CD

### Cloud Run 배포 (`bot-deploy.yml`)

main 브랜치에 push 시 자동 실행됩니다.

| 항목 | 설정 |
|------|------|
| **베이스 이미지** | `python:3.13-slim` |
| **레지스트리** | GCP Artifact Registry (`asia-northeast3`) |
| **메모리 / CPU** | 1Gi / 1 vCPU |
| **인스턴스** | min 1 ~ max 3 (자동 스케일링) |
| **ChromaDB 경로** | `/tmp/chroma_db` (읽기 전용 파일시스템) |
| **CPU 부스트** | 활성화 (콜드 스타트 시간 단축) |

### 문서 자동 동기화 (`doc-sync.yml`)

프론트엔드 리포에서 `repository_dispatch` 이벤트 수신 시 실행됩니다.

1. 프론트엔드 코드 체크아웃 + `git diff` 추출
2. `generate_docs.py`로 AI 분석 (사양 변경 여부 판단)
3. 변경 시 knowledge 문서 수정 → `docs/auto-update` 브랜치로 PR 생성

---

## 알려진 제약사항

| 항목 | 설명 |
|------|------|
| **ChromaDB 휘발성** | Cloud Run `/tmp`에 저장되므로 인스턴스 종료 시 데이터 소멸. `min-instances=1`로 완화 |
| **동기화 중 503** | 초기 동기화 완료 전까지 RAG 엔드포인트가 503 반환 (증분 동기화로 개선됨) |
| **CORS** | 현재 `allow_origins=["*"]`로 설정되어 있음 |
