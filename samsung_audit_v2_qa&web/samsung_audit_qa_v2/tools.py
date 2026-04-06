"""
tools.py - Tool 함수 집합 (SQL 조회 + 재무 계산)
Router가 Task Execution Plan에 따라 이 함수들을 호출
"""
import sqlite3
import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(__file__))
from db_schema import get_connection, DB_PATH



# ─── QA 질의용 metric 사전 ─────────────────────────────────────
# 주의:
# - 이 사전은 HTML 파서용이 아니라, 사용자 질문 해석용이다.
# - report_parser와 분리해서 여기서 직접 관리한다.



QUERY_METRIC_MAP = {
    # 핵심 지표
    "매출액": "revenue",
    "영업수익": "revenue",
    "수익": "revenue",
    "영업이익": "operating_income",
    "영업이익(손실)": "operating_income",
    "당기순이익": "net_income",
    "당기순이익(손실)": "net_income",
    "순이익": "net_income",
    "자산총계": "total_assets",
    "자산 총계": "total_assets",
    "부채총계": "total_liabilities",
    "부채 총계": "total_liabilities",
    "자본총계": "total_equity",
    "자본 총계": "total_equity",
    "유동자산": "current_assets",
    "비유동자산": "non_current_assets",
    "유동부채": "current_liabilities",
    "비유동부채": "non_current_liabilities",
    "현금및현금성자산": "cash_and_equivalents",
    "현금 및 현금성자산": "cash_and_equivalents",

    # 이번에 추가할 세부 계정
    "매출채권": "trade_receivables",
    "매출채권및기타채권": "trade_receivables",
    "매출채권 및 기타채권": "trade_receivables",
    "매출채권및기타수취채권": "trade_receivables",
    "매출채권 및 기타수취채권": "trade_receivables",
    "재고자산": "inventories",
    "유형자산": "property_plant_equipment",
    "유형자산순액": "property_plant_equipment",
    "유형자산(순)": "property_plant_equipment",
    "충당부채": "provisions",
    "유동충당부채": "provisions",
    "장기충당부채": "long_term_provisions",
    "매출원가": "cost_of_sales",
    "매출 원가": "cost_of_sales",

    # 현금흐름
    "영업활동현금흐름": "operating_cash_flow",
    "영업활동 현금흐름": "operating_cash_flow",
    "투자활동현금흐름": "investing_cash_flow",
    "투자활동 현금흐름": "investing_cash_flow",
    "재무활동현금흐름": "financing_cash_flow",
    "재무활동 현금흐름": "financing_cash_flow",
}

REVERSE_QUERY_METRIC_MAP = {
    "revenue": "매출액",
    "operating_income": "영업이익",
    "net_income": "당기순이익",
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "current_assets": "유동자산",
    "non_current_assets": "비유동자산",
    "current_liabilities": "유동부채",
    "non_current_liabilities": "비유동부채",
    "cash_and_equivalents": "현금및현금성자산",
    "trade_receivables": "매출채권",
    "inventories": "재고자산",
    "property_plant_equipment": "유형자산",
    "provisions": "충당부채",
    "long_term_provisions": "장기충당부채",
    "cost_of_sales": "매출원가",
    "depreciation": "감가상각비",
    "amortization": "무형자산상각비",
    "income_tax": "법인세비용",
    "operating_cash_flow": "영업활동현금흐름",
    "investing_cash_flow": "투자활동현금흐름",
    "financing_cash_flow": "재무활동현금흐름",
}

# 지표별 출처 재무제표 매핑
METRIC_SOURCE_MAP = {
    # 손익계산서
    "revenue": "손익계산서",
    "operating_income": "손익계산서",
    "net_income": "손익계산서",
    "cost_of_sales": "손익계산서",
    "depreciation": "손익계산서",
    "amortization": "손익계산서",
    "income_tax": "손익계산서",
    # 재무상태표
    "total_assets": "재무상태표",
    "total_liabilities": "재무상태표",
    "total_equity": "재무상태표",
    "current_assets": "재무상태표",
    "non_current_assets": "재무상태표",
    "current_liabilities": "재무상태표",
    "non_current_liabilities": "재무상태표",
    "cash_and_equivalents": "재무상태표",
    "trade_receivables": "재무상태표",
    "inventories": "재무상태표",
    "property_plant_equipment": "재무상태표",
    "provisions": "재무상태표",
    "long_term_provisions": "재무상태표",
    # 현금흐름표
    "operating_cash_flow": "현금흐름표",
    "investing_cash_flow": "현금흐름표",
    "financing_cash_flow": "현금흐름표",
}


def get_metric_source(metric_norm: str, ko_name: str = None) -> dict:
    """지표의 출처 재무제표 반환"""
    source = METRIC_SOURCE_MAP.get(metric_norm)
    if source:
        label = ko_name or REVERSE_QUERY_METRIC_MAP.get(metric_norm, metric_norm)
        return {
            "status": "success",
            "metric_norm": metric_norm,
            "source": source,
            "answer": source,
        }
    return {"status": "no_data", "message": f"{metric_norm} 출처 정보 없음"}


# 재무비율 계산 타입
RATIO_TYPES = {
    "debt_ratio": {
        "name": "부채비율",
        "formula": lambda d: round(d.get("total_liabilities", 0) / d.get("total_equity", 1) * 100, 2)
    },
    "roe": {
        "name": "자기자본이익률(ROE)",
        "formula": lambda d: round(d.get("net_income", 0) / d.get("total_equity", 1) * 100, 2)
    },
    "roa": {
        "name": "총자산이익률(ROA)",
        "formula": lambda d: round(d.get("net_income", 0) / d.get("total_assets", 1) * 100, 2)
    },
    "equity_ratio": {
        "name": "자기자본비율",
        "formula": lambda d: round(d.get("total_equity", 0) / d.get("total_assets", 1) * 100, 2)
    },
    "operating_margin": {
        "name": "영업이익률",
        "formula": lambda d: round(d.get("operating_income", 0) / d.get("revenue", 1) * 100, 2)
    },
}

DATA_YEAR_RANGE = (2014, 2024)

def sql_fetch_note_title(year: int, note_number: int, db_path: str = DB_PATH) -> dict:
    """특정 연도와 주석 번호에 해당하는 주석 제목을 반환한다.
    우선순위: 1) note_titles 테이블(HTML 직접 파싱 결과)
              2) chunks 텍스트에서 "N. 제목" 패턴 추출
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    # 1순위: note_titles 테이블 (HTML 파싱으로 구축된 정확한 데이터)
    try:
        cur.execute(
            "SELECT title FROM note_titles WHERE year=? AND note_number=?",
            (year, note_number),
        )
        row = cur.fetchone()
        if row and row["title"]:
            conn.close()
            return {
                "year": year,
                "note_number": note_number,
                "note_title": row["title"],
            }
    except Exception:
        pass  # note_titles 테이블 없으면 fallback

    # 2순위: chunks 텍스트에서 "N. 제목" 패턴 추출
    cur.execute(
        "SELECT text FROM chunks WHERE year=? AND section_type='notes'",
        (year,),
    )
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        text = row["text"] or ""
        m = re.match(rf"^{note_number}[\.\s]+([^\n,]+)", text)
        if m:
            return {
                "year": year,
                "note_number": note_number,
                "note_title": m.group(1).strip(),
            }

    return {"year": year, "note_number": note_number, "note_title": None}


def sql_fetch_fact_value(year: int, metric_norm: str, db_path: str = DB_PATH):
    """단일 연도+metric 값 반환 (내부 계산용)"""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT value FROM financial_facts WHERE year=? AND metric_name_norm=? LIMIT 1",
        (year, metric_norm),
    )
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def calculate_absolute_diff(year_a: int, year_b: int, metric_norm: str,
                            ko_name: str, db_path: str = DB_PATH) -> dict:
    """두 연도 간 절대 증감액을 계산하여 반환한다."""
    val_a = sql_fetch_fact_value(year_a, metric_norm, db_path)
    val_b = sql_fetch_fact_value(year_b, metric_norm, db_path)

    if val_a is None or val_b is None:
        missing_year = year_a if val_a is None else year_b
        return {
            "status": "no_data",
            "message": f"{missing_year}년 {ko_name} 데이터를 찾을 수 없습니다.",
        }

    diff = val_b - val_a
    direction = "증가" if diff > 0 else ("감소" if diff < 0 else "동일")
    abs_diff = abs(diff)

    return {
        "status": "success",
        "year_a": year_a,
        "year_b": year_b,
        "metric_ko": ko_name,
        "value_a": val_a,
        "value_b": val_b,
        "diff": diff,
        "answer": (
            f"{year_a}년 대비 {year_b}년 {ko_name} 증감액은 "
            f"{'+' if diff >= 0 else ''}{diff:,.0f}백만원입니다. "
            f"({year_a}년: {val_a:,.0f} → {year_b}년: {val_b:,.0f}, "
            f"{abs_diff:,.0f}백만원 {direction})"
        ),
    }


def resolve_metric(user_text: str) -> tuple:
    """사용자 질문에서 metric을 찾아 (한글명, canonical metric)을 반환한다."""
    text = user_text or ""
    normalized_text = re.sub(r"\s+", "", text)

    # 긴 표현부터 먼저 검사해야 부분 매칭 오류를 줄일 수 있다.
    metric_items = sorted(
        QUERY_METRIC_MAP.items(),
        key=lambda item: len(re.sub(r"\s+", "", item[0])),
        reverse=True,
    )

    # 한글 metric 탐지
    for ko_name, norm_name in metric_items:
        ko_normalized = re.sub(r"\s+", "", ko_name)
        if ko_name in text or ko_normalized in normalized_text:
            return ko_name, norm_name

    # 영문 canonical metric도 허용
    lowered_text = text.lower()
    for en_name, ko_name in REVERSE_QUERY_METRIC_MAP.items():
        if en_name in lowered_text:
            return ko_name, en_name

    return None, None


def format_value(value: float, unit: str = "백만원") -> str:
    """숫자를 읽기 쉬운 형태로 포맷 (항상 정확한 백만원 단위 반환)"""
    if value is None:
        return "데이터 없음"
    if unit == "백만원":
        return f"{value:,.0f}백만원" + (f" (약 {value/1_000_000:.1f}조)" if abs(value) >= 1_000_000 else "")
    return f"{value:,.0f}"


def sql_fetch_fact(year: int, metric_norm: str,
                   scope: str = None, db_path: str = DB_PATH) -> dict:
    """특정 연도+지표 값 조회"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    query = """
        SELECT f.year, f.metric_name_ko, f.metric_name_norm,
               f.value, f.unit, f.consolidation_scope
        FROM financial_facts f
        WHERE f.year = ? AND f.metric_name_norm = ?
    """
    params = [year, metric_norm]

    if scope:
        query += " AND f.consolidation_scope = ?"
        params.append(scope)

    query += " ORDER BY f.fact_id LIMIT 1"

    cur.execute(query, params)
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "year": row["year"],
            "metric_ko": row["metric_name_ko"],
            "metric_norm": row["metric_name_norm"],
            "value": row["value"],
            "unit": row["unit"],
            "scope": row["consolidation_scope"],
            "value_fmt": format_value(row["value"], row["unit"]),
        }
    return {"year": year, "metric_norm": metric_norm, "value": None, "value_fmt": "데이터 없음"}


def sql_fetch_series(metric_norm: str, start_year: int = 2014,
                     end_year: int = 2024, db_path: str = DB_PATH) -> list:
    """연도별 시계열 데이터 조회"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT year, metric_name_ko, value, unit, consolidation_scope
        FROM financial_facts
        WHERE metric_name_norm = ?
          AND year BETWEEN ? AND ?
        ORDER BY year
    """, (metric_norm, start_year, end_year))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "year": r["year"],
            "metric_ko": r["metric_name_ko"],
            "value": r["value"],
            "unit": r["unit"],
            "scope": r["consolidation_scope"],
            "value_fmt": format_value(r["value"], r["unit"]),
        }
        for r in rows
    ]


def sql_fetch_audit_opinion(year: int, db_path: str = DB_PATH) -> dict:
    """감사의견 조회"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT report_year, audit_opinion
        FROM documents
        WHERE report_year = ?
    """, (year,))

    row = cur.fetchone()
    conn.close()

    if row:
        return {"year": row["report_year"], "opinion": row["audit_opinion"]}
    return {"year": year, "opinion": "데이터 없음"}


def sql_search_chunks(keyword: str, year: int = None,
                      db_path: str = DB_PATH, top_k: int = 3) -> list:
    """키워드 기반 청크 검색"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    if year:
        cur.execute("""
            SELECT year, section_type, text
            FROM chunks
            WHERE year = ? AND text LIKE ?
            ORDER BY chunk_id
            LIMIT ?
        """, (year, f"%{keyword}%", top_k))
    else:
        cur.execute("""
            SELECT year, section_type, text
            FROM chunks
            WHERE text LIKE ?
            ORDER BY year DESC, chunk_id
            LIMIT ?
        """, (f"%{keyword}%", top_k))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "year": r["year"],
            "section_type": r["section_type"],
            "text": r["text"][:300],
        }
        for r in rows
    ]


def sql_find_extreme(metric_norm: str, direction: str = "max",
                     db_path: str = DB_PATH) -> dict:
    """지표의 최고/최저 연도 찾기"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    order = "DESC" if direction == "max" else "ASC"
    cur.execute(f"""
        SELECT year, metric_name_ko, value, unit
        FROM financial_facts
        WHERE metric_name_norm = ? AND value IS NOT NULL
        ORDER BY value {order}
        LIMIT 1
    """, (metric_norm,))

    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "year": row["year"],
            "metric_ko": row["metric_name_ko"],
            "value": row["value"],
            "unit": row["unit"],
            "value_fmt": format_value(row["value"], row["unit"]),
            "direction": "가장 높은" if direction == "max" else "가장 낮은",
        }
    return {}


def sql_fetch_all_facts_for_year(year: int, db_path: str = DB_PATH) -> list:
    """특정 연도 모든 재무 팩트 조회"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT metric_name_ko, metric_name_norm, value, unit, consolidation_scope
        FROM financial_facts
        WHERE year = ?
        ORDER BY fact_id
    """, (year,))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "metric_ko": r["metric_name_ko"],
            "metric_norm": r["metric_name_norm"],
            "value": r["value"],
            "unit": r["unit"],
            "scope": r["consolidation_scope"],
            "value_fmt": format_value(r["value"], r["unit"]),
        }
        for r in rows
    ]


def calculate_change_rate(value_a: float, value_b: float) -> dict:
    """증감률 계산"""
    if value_a is None or value_b is None:
        return {"difference": None, "change_rate": None}
    if value_a == 0:
        return {"difference": value_b, "change_rate": None}

    diff = value_b - value_a
    rate = round((diff / abs(value_a)) * 100, 2)
    return {
        "difference": diff,
        "change_rate": rate,
        "change_label": "증가" if rate > 0 else "감소",
    }


def calculate_ratio(year: int, ratio_type: str, db_path: str = DB_PATH) -> dict:
    """재무비율 계산"""
    # 필요한 팩트 한번에 조회
    facts = sql_fetch_all_facts_for_year(year, db_path)
    data = {f["metric_norm"]: f["value"] for f in facts if f["value"] is not None}

    if ratio_type not in RATIO_TYPES:
        return {"error": f"지원하지 않는 비율: {ratio_type}"}

    try:
        rt = RATIO_TYPES[ratio_type]
        value = rt["formula"](data)
        return {
            "year": year,
            "ratio_type": ratio_type,
            "ratio_name": rt["name"],
            "value": value,
            "value_fmt": f"{value:.2f}%",
        }
    except ZeroDivisionError:
        return {"error": "계산에 필요한 데이터 부족"}


def list_available_data(db_path: str = DB_PATH) -> dict:
    """보유 데이터 목록 반환"""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT report_year FROM documents ORDER BY report_year")
    years = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT metric_name_ko FROM financial_facts ORDER BY metric_name_ko")
    metrics = [r[0] for r in cur.fetchall()]

    conn.close()

    return {
        "years": years,
        "metrics": metrics,
        "year_range": f"{min(years)}~{max(years)}" if years else "없음",
    }


def build_year_summary(year: int, db_path: str = DB_PATH) -> str:
    """특정 연도 종합 요약 텍스트 생성"""
    facts = sql_fetch_all_facts_for_year(year, db_path)
    opinion = sql_fetch_audit_opinion(year, db_path)

    lines = [f"【{year}년 삼성전자 재무 요약】"]
    lines.append(f"• 감사의견: {opinion.get('opinion', 'N/A')}")
    lines.append("")

    priority_metrics = ["revenue", "operating_income", "net_income",
                        "total_assets", "total_liabilities", "total_equity"]

    for pm in priority_metrics:
        for f in facts:
            if f["metric_norm"] == pm:
                lines.append(f"• {f['metric_ko']}: {f['value_fmt']}")
                break

    return "\n".join(lines)


if __name__ == "__main__":
    # 빠른 테스트
    print(list_available_data())
