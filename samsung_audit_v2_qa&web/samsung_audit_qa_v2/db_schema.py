"""
db_schema.py - SQLite 스키마 초기화 + Task Catalog 적재
핵심: LLM이 사전에 Action Set을 DB에 등록 → 챗봇 실행 시 로드
"""
import sqlite3
import json
import os


DB_PATH = os.path.join(os.path.dirname(__file__), "samsung_audit.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """스키마 생성"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # ─── 핵심 테이블 ───────────────────────────────────────────
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT NOT NULL DEFAULT '삼성전자',
        report_year  INTEGER NOT NULL UNIQUE,
        source_file  TEXT NOT NULL,
        source_path  TEXT,
        audit_opinion TEXT,
        parsed_at   TEXT,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id      INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        text        TEXT NOT NULL,
        char_count  INTEGER,
        year        INTEGER NOT NULL,
        section_type TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    );

    CREATE TABLE IF NOT EXISTS financial_facts (
        fact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id      INTEGER NOT NULL,
        year        INTEGER NOT NULL,
        statement_type TEXT,
        metric_name_ko TEXT NOT NULL,
        metric_name_norm TEXT NOT NULL,
        value       REAL,
        unit        TEXT DEFAULT '백만원',
        consolidation_scope TEXT DEFAULT 'separate',
        source_row_label TEXT,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    );

    -- ─── Task Catalog: LLM이 사전 등록하는 Action Set ──────────
    CREATE TABLE IF NOT EXISTS task_catalog (
        task_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name    TEXT NOT NULL UNIQUE,
        task_group   TEXT NOT NULL,   -- feasible / infeasible / exception
        description  TEXT NOT NULL,
        required_slots TEXT,          -- JSON: ["metric", "year"]
        execution_plan TEXT,          -- JSON: action 실행 계획
        response_template TEXT,
        example_queries TEXT          -- JSON: 예시 질의 목록
    );

    CREATE TABLE IF NOT EXISTS conversation_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,    -- user / assistant
        content     TEXT NOT NULL,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- ─── 인덱스 ──────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_facts_year_metric
        ON financial_facts(year, metric_name_norm);
    CREATE INDEX IF NOT EXISTS idx_facts_scope
        ON financial_facts(consolidation_scope);
    CREATE INDEX IF NOT EXISTS idx_chunks_year_section
        ON chunks(year, section_type);
    CREATE INDEX IF NOT EXISTS idx_docs_year
        ON documents(report_year);
    """)

    conn.commit()
    conn.close()
    print(f"✅ DB 스키마 초기화 완료: {db_path}")


# ─────────────────────────────────────────────────────────────────
# Task Catalog: LLM Action Set 정의
# - 각 task는 실행 계획(execution_plan)을 JSON으로 보유
# - chatbot 시작 시 LLM이 이 목록을 컨텍스트로 로드
# ─────────────────────────────────────────────────────────────────
TASK_CATALOG = [
    # ═══════════════ FEASIBLE (수행 가능) ═══════════════
    {
        "task_name": "get_metric_by_year",
        "task_group": "feasible",
        "description": "특정 연도의 재무 지표 값 조회",
        "required_slots": ["year", "metric"],
        "execution_plan": [
            {"action": "resolve_metric", "input": "metric"},
            {"action": "sql_fetch_fact", "params": {"year": "{year}", "metric": "{resolved_metric}"}},
            {"action": "format_currency", "input": "value"}
        ],
        "response_template": "{year}년 {metric_ko}은(는) {value_formatted}입니다.",
        "example_queries": [
            "2023년 매출액은 얼마야?",
            "2020년 영업이익 알려줘",
            "2019년 자산총계?",
            "2022년 순이익은?",
            "2018년 부채총계 얼마야",
        ]
    },
    {
        "task_name": "compare_metric_between_years",
        "task_group": "feasible",
        "description": "두 연도 간 재무 지표 비교 및 증감률 계산",
        "required_slots": ["year_a", "year_b", "metric"],
        "execution_plan": [
            {"action": "resolve_metric", "input": "metric"},
            {"action": "sql_fetch_fact", "params": {"year": "{year_a}", "metric": "{resolved_metric}"}},
            {"action": "sql_fetch_fact", "params": {"year": "{year_b}", "metric": "{resolved_metric}"}},
            {"action": "calculate_change_rate", "inputs": ["value_a", "value_b"]}
        ],
        "response_template": "{year_a}년 대비 {year_b}년 {metric_ko}: {value_a_fmt} → {value_b_fmt} (증감률: {change_rate}%)",
        "example_queries": [
            "2019년과 2023년 영업이익을 비교해줘",
            "2020년 대비 2024년 매출 변화",
            "2015년과 2024년 자산 비교",
            "2018년 vs 2022년 순이익",
        ]
    },
    {
        "task_name": "trend_of_metric",
        "task_group": "feasible",
        "description": "특정 재무 지표의 연도별 추이 분석 (전체 또는 범위 지정)",
        "required_slots": ["metric"],
        "execution_plan": [
            {"action": "resolve_metric", "input": "metric"},
            {"action": "sql_fetch_series", "params": {"metric": "{resolved_metric}", "start": "{start_year}", "end": "{end_year}"}},
            {"action": "calculate_trend_stats", "input": "series"}
        ],
        "response_template": "{metric_ko} 연도별 추이:\n{series_table}\n최고: {max_year}년, 최저: {min_year}년",
        "example_queries": [
            "최근 10년 매출액 추이 보여줘",
            "영업이익 연도별 변화",
            "2018년부터 2024년까지 순이익 흐름",
            "자산총계 트렌드",
        ]
    },
    {
        "task_name": "find_best_worst_year",
        "task_group": "feasible",
        "description": "특정 지표가 가장 높거나 낮았던 연도 찾기",
        "required_slots": ["metric", "direction"],
        "execution_plan": [
            {"action": "resolve_metric", "input": "metric"},
            {"action": "sql_find_extreme", "params": {"metric": "{resolved_metric}", "direction": "{direction}"}}
        ],
        "response_template": "{metric_ko}이(가) 가장 {direction_ko} 해는 {best_year}년으로 {best_value}였습니다.",
        "example_queries": [
            "영업이익이 가장 높았던 해는?",
            "매출이 제일 낮았던 연도",
            "순이익 최고 기록 연도",
            "부채가 제일 많았던 해",
        ]
    },
    {
        "task_name": "get_audit_opinion",
        "task_group": "feasible",
        "description": "특정 연도의 감사의견 조회",
        "required_slots": ["year"],
        "execution_plan": [
            {"action": "sql_fetch_opinion", "params": {"year": "{year}"}}
        ],
        "response_template": "{year}년 감사의견: {opinion}",
        "example_queries": [
            "2023년 감사의견은?",
            "2020년 감사 결과",
            "최근 감사의견 알려줘",
            "2018년 감사보고서 의견",
        ]
    },
    {
        "task_name": "search_by_keyword",
        "task_group": "feasible",
        "description": "감사보고서 본문에서 특정 키워드 검색",
        "required_slots": ["keyword"],
        "execution_plan": [
            {"action": "sql_search_chunks", "params": {"keyword": "{keyword}", "year": "{year}"}},
            {"action": "rank_by_relevance", "input": "chunks"}
        ],
        "response_template": "'{keyword}' 관련 내용 ({year_hint}년):\n{results}",
        "example_queries": [
            "반도체 관련 내용 찾아줘",
            "2022년 핵심감사사항",
            "COVID 영향에 대한 언급",
            "환율 위험 관련 주석",
        ]
    },
    {
        "task_name": "calculate_financial_ratio",
        "task_group": "feasible",
        "description": "재무비율 계산 (부채비율, ROE, ROA 등)",
        "required_slots": ["year", "ratio_type"],
        "execution_plan": [
            {"action": "sql_fetch_multi_facts", "params": {"year": "{year}", "metrics": ["total_liabilities", "total_equity", "net_income", "total_assets"]}},
            {"action": "calculate_ratio", "params": {"ratio_type": "{ratio_type}"}}
        ],
        "response_template": "{year}년 {ratio_name}: {ratio_value}%",
        "example_queries": [
            "2023년 부채비율",
            "2022년 ROE 계산해줘",
            "2020년 ROA는?",
            "2019년 자기자본비율",
        ]
    },
    {
        "task_name": "list_available_data",
        "task_group": "feasible",
        "description": "조회 가능한 연도 및 지표 목록 안내",
        "required_slots": [],
        "execution_plan": [
            {"action": "sql_list_years"},
            {"action": "list_available_metrics"}
        ],
        "response_template": "조회 가능한 데이터:\n• 연도: {years}\n• 지표: {metrics}",
        "example_queries": [
            "어떤 데이터 조회 가능해?",
            "무슨 질문할 수 있어?",
            "어떤 연도 데이터 있어?",
        ]
    },
    {
        "task_name": "summarize_year",
        "task_group": "feasible",
        "description": "특정 연도 재무 현황 종합 요약",
        "required_slots": ["year"],
        "execution_plan": [
            {"action": "sql_fetch_all_facts", "params": {"year": "{year}"}},
            {"action": "sql_fetch_opinion", "params": {"year": "{year}"}},
            {"action": "generate_summary", "input": "all_data"}
        ],
        "response_template": "{year}년 삼성전자 재무 요약:\n{summary}",
        "example_queries": [
            "2023년 실적 요약해줘",
            "2020년 종합 분석",
            "2019년 감사보고서 핵심 내용",
        ]
    },

    # ═══════════════ INFEASIBLE (수행 불가) ═══════════════
    {
        "task_name": "stock_recommendation",
        "task_group": "infeasible",
        "description": "주가 기반 투자 추천 (외부 데이터 필요)",
        "required_slots": [],
        "execution_plan": [{"action": "return_infeasible_response"}],
        "response_template": "투자 추천은 현재 시스템 범위를 벗어납니다. 감사보고서 재무 정보만 제공 가능합니다.",
        "example_queries": [
            "삼성전자 주식 사도 돼?",
            "주가 전망이 어때?",
            "투자 추천해줘",
        ]
    },
    {
        "task_name": "future_forecast",
        "task_group": "infeasible",
        "description": "미래 실적 예측 (2024년 이후)",
        "required_slots": [],
        "execution_plan": [{"action": "return_infeasible_response"}],
        "response_template": "미래 실적 예측은 제공 범위를 벗어납니다. 2014~2024년 실적 데이터를 기반으로 질문해 주세요.",
        "example_queries": [
            "2025년 매출 예측해줘",
            "내년 영업이익 전망",
            "앞으로 어떻게 될까?",
        ]
    },
    {
        "task_name": "competitor_comparison",
        "task_group": "infeasible",
        "description": "경쟁사 비교 분석 (삼성전자 단독 데이터만 보유)",
        "required_slots": [],
        "execution_plan": [{"action": "return_infeasible_response"}],
        "response_template": "현재 삼성전자 감사보고서 데이터만 보유하고 있어 경쟁사 비교는 불가합니다.",
        "example_queries": [
            "SK하이닉스와 비교해줘",
            "업계 평균 대비 어때?",
            "경쟁사 대비 실적은?",
        ]
    },

    # ═══════════════ EXCEPTION (예외 처리) ═══════════════
    {
        "task_name": "ambiguous_year_missing",
        "task_group": "exception",
        "description": "연도가 지정되지 않은 모호한 질의",
        "required_slots": [],
        "execution_plan": [{"action": "ask_clarification", "params": {"missing": "year"}}],
        "response_template": "연도를 지정해 주세요. 예: '2023년 매출액은?'\n조회 가능 연도: 2014~2024년",
        "example_queries": [
            "매출액 알려줘",
            "영업이익은?",
            "자산 얼마야?",
        ]
    },
    {
        "task_name": "out_of_range_year",
        "task_group": "exception",
        "description": "데이터 범위 밖 연도 요청",
        "required_slots": [],
        "execution_plan": [{"action": "return_range_error"}],
        "response_template": "요청 연도({year})는 보유 데이터 범위(2014~2024)를 벗어납니다.",
        "example_queries": [
            "2010년 매출 알려줘",
            "2025년 영업이익",
            "2000년대 실적은?",
        ]
    },
]


def populate_task_catalog(db_path: str = DB_PATH) -> None:
    """Task Catalog를 DB에 적재 (LLM Action Set 사전 등록)"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    for task in TASK_CATALOG:
        cur.execute("""
            INSERT OR REPLACE INTO task_catalog
            (task_name, task_group, description, required_slots,
             execution_plan, response_template, example_queries)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            task["task_name"],
            task["task_group"],
            task["description"],
            json.dumps(task.get("required_slots", []), ensure_ascii=False),
            json.dumps(task.get("execution_plan", []), ensure_ascii=False),
            task.get("response_template", ""),
            json.dumps(task.get("example_queries", []), ensure_ascii=False),
        ))

    conn.commit()
    conn.close()
    print(f"✅ Task Catalog {len(TASK_CATALOG)}개 등록 완료")


def setup_database(db_path: str = DB_PATH) -> None:
    """전체 DB 초기화 + Task Catalog 적재"""
    # 기존 DB 삭제 후 재생성
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(db_path)
    populate_task_catalog(db_path)
    print(f"✅ 데이터베이스 설정 완료")


if __name__ == "__main__":
    setup_database()
