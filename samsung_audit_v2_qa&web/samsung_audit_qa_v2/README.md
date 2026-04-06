# 삼성전자 감사보고서 NLP QA 시스템

삼성전자 2014~2024년 감사보고서를 분석하는 **금융 도메인 특화 챗봇**

## 시스템 아키텍처

```
[HTML 감사보고서 10개년]
        ↓
[parser.py] HTML 파싱 & 재무 데이터 추출
        ↓
[ingest.py] SQLite DB 적재 (financial_facts + chunks)
        ↓
[db_schema.py] Task Catalog 등록 (LLM Action Set 사전 정의)
        ↓
[router.py] TF-IDF 기반 Task 라우팅 + 슬롯 추출
        ↓
[tools.py] SQL 조회 + 재무 계산 실행
        ↓
[chatbot.py] Qwen3(Ollama) + 파이프라인 통합 챗봇
```

## 독창적 설계: LLM Action Set 사전 등록

이 시스템의 핵심 독창성은 **LLM이 실행 전 Action Set을 DB에 사전 등록**하는 구조입니다:

```
task_catalog 테이블:
  ┌─────────────────────────┬─────────────┬────────────────────────────┐
  │ task_name               │ task_group  │ execution_plan (JSON)      │
  ├─────────────────────────┼─────────────┼────────────────────────────┤
  │ get_metric_by_year      │ feasible    │ [sql_fetch_fact, format...] │
  │ compare_metric_...      │ feasible    │ [fetch_a, fetch_b, calc...] │
  │ trend_of_metric         │ feasible    │ [fetch_series, stats...]   │
  │ calculate_financial...  │ feasible    │ [fetch_multi, calc_ratio]  │
  │ stock_recommendation    │ infeasible  │ [return_infeasible]        │
  │ ambiguous_year_missing  │ exception   │ [ask_clarification]        │
  └─────────────────────────┴─────────────┴────────────────────────────┘
```

- **Feasible**: DB 데이터로 처리 가능한 질의 → 자동 실행
- **Infeasible**: 처리 불가 질의 → 사유 및 대안 안내
- **Exception**: 모호/범위외 질의 → 명확화 요청

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 전체 초기화 (파싱 + DB 적재 + 테스트)
python run_setup.py

# 챗봇 실행
python chatbot.py

# Qwen3 모델 사전 준비 (Ollama)
ollama pull qwen3:8b
```

## 실행 환경

- Python 3.9+
- Ollama (로컬 LLM): `qwen3:8b` 또는 `qwen3`
- 데이터: `삼성전자_감사보고서_2014_2024/` 폴더의 .htm 파일

## 지원 질의 유형

| 유형 | 예시 |
|------|------|
| 특정 연도 지표 | "2023년 매출액은?" |
| 연도 간 비교 | "2019년과 2023년 영업이익 비교" |
| 추이 분석 | "10년간 자산총계 변화" |
| 극값 탐색 | "영업이익 가장 높았던 해?" |
| 감사의견 | "2022년 감사의견?" |
| 재무비율 | "2023년 부채비율, ROE, ROA" |
| 연도 요약 | "2020년 종합 요약" |
| 키워드 검색 | "반도체 관련 내용" |

## 파일 구조

```
samsung_audit_qa/
├── parser.py        # HTML 파서 (재무 팩트 + 텍스트 청크 추출)
├── db_schema.py     # SQLite 스키마 + Task Catalog 초기화
├── ingest.py        # DB 적재 파이프라인
├── tools.py         # SQL 조회 + 재무 계산 함수
├── router.py        # TF-IDF 기반 Task Router + Executor
├── chatbot.py       # 챗봇 CLI (Qwen3 + 파이프라인 통합)
├── run_setup.py     # 원클릭 초기화 스크립트
├── requirements.txt
└── README.md
```

## 테스트

```bash
python chatbot.py --test
```
