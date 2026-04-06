"""
ingest.py - 파싱 결과 → SQLite DB 적재 파이프라인
"""
import sqlite3
import json
import os
import sys
from datetime import datetime

# 같은 패키지 import 처리
sys.path.insert(0, os.path.dirname(__file__))
from parser import parse_all_reports
from db_schema import get_connection, DB_PATH, setup_database


def ingest_document(conn: sqlite3.Connection, parsed: dict) -> int:
    """단일 문서 적재 → doc_id 반환"""
    cur = conn.cursor()

    # 이미 있으면 업데이트, 없으면 삽입
    cur.execute("""
        INSERT OR REPLACE INTO documents
        (company_name, report_year, source_file, source_path,
         audit_opinion, parsed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "삼성전자",
        parsed["year"],
        parsed["source_file"],
        parsed.get("source_path", ""),
        parsed.get("audit_opinion", ""),
        parsed.get("parsed_at", datetime.now().isoformat()),
    ))

    doc_id = cur.lastrowid
    conn.commit()
    return doc_id


def ingest_chunks(conn: sqlite3.Connection, doc_id: int, chunks: list) -> int:
    """텍스트 청크 일괄 적재"""
    cur = conn.cursor()
    # 기존 청크 삭제
    cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    for chunk in chunks:
        cur.execute("""
            INSERT INTO chunks
            (doc_id, chunk_index, text, char_count, year, section_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            chunk["chunk_index"],
            chunk["text"],
            chunk.get("char_count", len(chunk["text"])),
            chunk["year"],
            chunk.get("section_type", "other"),
        ))

    conn.commit()
    return len(chunks)


def ingest_financial_facts(conn: sqlite3.Connection, doc_id: int,
                            facts: list) -> int:
    """재무 팩트 일괄 적재"""
    cur = conn.cursor()
    # 기존 팩트 삭제
    cur.execute("DELETE FROM financial_facts WHERE doc_id = ?", (doc_id,))

    inserted = 0
    for fact in facts:
        if fact.get("value") is None:
            continue
        cur.execute("""
            INSERT INTO financial_facts
            (doc_id, year, statement_type, metric_name_ko, metric_name_norm,
             value, unit, consolidation_scope, source_row_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            fact["year"],
            fact.get("statement_type", "other"),
            fact["metric_name_ko"],
            fact["metric_name_norm"],
            fact["value"],
            fact.get("unit", "백만원"),
            fact.get("consolidation_scope", "separate"),
            fact.get("source_row_label", ""),
        ))
        inserted += 1

    conn.commit()
    return inserted


def run_ingestion(data_dir: str, db_path: str = DB_PATH) -> dict:
    """전체 적재 파이프라인 실행"""
    print("=" * 60)
    print("🚀 삼성전자 감사보고서 DB 적재 파이프라인 시작")
    print("=" * 60)

    # 1. 파싱
    print("\n[1/3] HTML 파일 파싱 중...")
    parsed_docs = parse_all_reports(data_dir)

    if not parsed_docs:
        print("❌ 파싱된 문서가 없습니다.")
        return {}

    # 2. DB 연결
    print(f"\n[2/3] DB 적재 중... ({db_path})")
    conn = get_connection(db_path)

    stats = {"documents": 0, "chunks": 0, "facts": 0, "errors": []}

    for parsed in sorted(parsed_docs, key=lambda x: x["year"]):
        try:
            doc_id = ingest_document(conn, parsed)
            n_chunks = ingest_chunks(conn, doc_id, parsed["chunks"])
            n_facts = ingest_financial_facts(conn, doc_id, parsed["financial_facts"])

            stats["documents"] += 1
            stats["chunks"] += n_chunks
            stats["facts"] += n_facts

            print(f"  ✅ {parsed['year']}년: 청크 {n_chunks}건, "
                  f"팩트 {n_facts}건, 감사의견: {parsed['audit_opinion']}")

        except Exception as e:
            stats["errors"].append(f"{parsed.get('year')}: {e}")
            print(f"  ❌ {parsed.get('year')}년 오류: {e}")

    conn.close()

    # 3. 검증
    print(f"\n[3/3] 적재 결과 검증...")
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents")
    db_docs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM financial_facts")
    db_facts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    db_chunks = cur.fetchone()[0]
    cur.execute("SELECT report_year, audit_opinion FROM documents ORDER BY report_year")
    years_opinions = cur.fetchall()

    conn.close()

    print(f"\n{'='*60}")
    print(f"✅ 적재 완료 요약")
    print(f"{'='*60}")
    print(f"  문서: {db_docs}개 | 재무팩트: {db_facts}건 | 청크: {db_chunks}건")
    print(f"\n  연도별 감사의견:")
    for row in years_opinions:
        print(f"    {row[0]}년: {row[1]}")

    if stats["errors"]:
        print(f"\n  ⚠️  오류 {len(stats['errors'])}건:")
        for e in stats["errors"]:
            print(f"    - {e}")

    return stats


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # DB 초기화 + Task Catalog 적재
    setup_database()

    # 파싱 + 적재
    run_ingestion(data_dir)
