import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from report_parser.audit_report_parser import AuditReportParser


def ensure_dir(path: Path) -> None:
    # 디렉토리가 없으면 생성
    path.mkdir(parents=True, exist_ok=True)


def extract_year_from_path(path_str: str) -> Optional[int]:
    # 파일 경로 문자열에서 연도 추출
    match = re.search(r"(20\d{2})", str(path_str))
    if match is None:
        return None
    return int(match.group(1))


def normalize_whitespace(text: str) -> str:
    # 과도한 공백 정리
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def df_to_compact_text(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    # DataFrame을 검색/임베딩용 텍스트로 변환
    working_df = df.copy()

    if max_rows is not None:
        working_df = working_df.head(max_rows)

    # NaN을 빈 문자열로 치환
    working_df = working_df.fillna("")

    # 모든 값을 문자열화
    working_df = working_df.astype(str)

    # 컬럼명 추가
    lines: List[str] = []
    lines.append("컬럼: " + " | ".join(list(working_df.columns)))

    # 각 행을 한 줄씩 추가
    for _, row in working_df.iterrows():
        row_text = " | ".join([str(x).strip() for x in row.tolist()])
        lines.append(row_text)

    return normalize_whitespace("\n".join(lines))


def make_base_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    # 모든 레코드에 공통으로 들어갈 기본 메타데이터 생성
    metadata = result["metadata"]
    document_structure = result["document_structure"]

    source_file = metadata.get("source_file")
    year = extract_year_from_path(str(source_file)) if source_file is not None else None

    return {
        "year": year,
        "source_file": source_file,
        "encoding": metadata.get("encoding"),
        "company_name": metadata.get("company_name"),
        "report_date": metadata.get("report_date"),
        "table_count": document_structure.get("table_count"),
        "text_length": document_structure.get("text_length"),
        "text_only_length": document_structure.get("text_only_length"),
    }


def build_main_section_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # text.main_sections -> corpus record 변환
    base_meta = make_base_metadata(result)
    text_result = result.get("text", {})
    main_sections = text_result.get("main_sections", [])

    records: List[Dict[str, Any]] = []

    for idx, section in enumerate(main_sections):
        title = section.get("title", "")
        section_type = section.get("section_type", "")
        section_text = normalize_whitespace(str(section.get("text", "")))

        record = {
            "doc_id": f"{base_meta['year']}_section_{idx}",
            "source_type": "main_section",
            "title": title,
            "text": section_text,
            "metadata": {
                **base_meta,
                "section_id": section.get("section_id"),
                "section_type": section_type,
                "level": section.get("level"),
                "start": section.get("start"),
                "end": section.get("end"),
                "start_line_index": section.get("start_line_index"),
                "end_line_index": section.get("end_line_index"),
            },
        }
        records.append(record)

    return records


def build_note_block_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # text.note_blocks -> corpus record 변환
    base_meta = make_base_metadata(result)
    text_result = result.get("text", {})
    note_blocks = text_result.get("note_blocks", [])

    records: List[Dict[str, Any]] = []

    for idx, note in enumerate(note_blocks):
        note_number = str(note.get("note_number", ""))
        note_title = str(note.get("note_title", ""))
        title_line = str(note.get("title_line", ""))
        note_text = normalize_whitespace(str(note.get("text", "")))

        record = {
            "doc_id": f"{base_meta['year']}_note_{note_number}_{idx}",
            "source_type": "note_block",
            "title": title_line if title_line else note_title,
            "text": note_text,
            "metadata": {
                **base_meta,
                "note_number": note_number,
                "note_title": note_title,
                "level": note.get("level"),
                "start": note.get("start"),
                "end": note.get("end"),
                "start_line_index": note.get("start_line_index"),
                "end_line_index": note.get("end_line_index"),
                "is_continued": note.get("metadata", {}).get("is_continued", False),
            },
        }
        records.append(record)

    return records


def build_normalized_table_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # tables.normalized -> corpus record 변환
    base_meta = make_base_metadata(result)
    normalized_tables = result.get("tables", {}).get("normalized", {})

    records: List[Dict[str, Any]] = []

    for statement_type, value in normalized_tables.items():
        # 자본변동표는 block list 구조
        if statement_type == "changes_in_equity":
            ce_blocks = value

            for block_idx, block in enumerate(ce_blocks):
                df = block.get("data")
                if not isinstance(df, pd.DataFrame):
                    continue

                table_text = df_to_compact_text(df)
                block_type = block.get("block_type")
                start_label = block.get("start_label")
                end_label = block.get("end_label")

                record = {
                    "doc_id": f"{base_meta['year']}_{statement_type}_{block_type}_{block_idx}",
                    "source_type": "normalized_table",
                    "title": f"{statement_type} - {block_type}",
                    "text": table_text,
                    "metadata": {
                        **base_meta,
                        "statement_type": statement_type,
                        "table_kind": "changes_in_equity_block",
                        "block_index": block_idx,
                        "block_type": block_type,
                        "start_label": start_label,
                        "end_label": end_label,
                        "n_rows": len(df),
                        "n_cols": len(df.columns),
                        "columns": list(df.columns),
                    },
                }
                records.append(record)

        # 나머지 재무제표는 단일 DataFrame 구조
        else:
            df = value
            if not isinstance(df, pd.DataFrame):
                continue

            table_text = df_to_compact_text(df)

            record = {
                "doc_id": f"{base_meta['year']}_{statement_type}",
                "source_type": "normalized_table",
                "title": statement_type,
                "text": table_text,
                "metadata": {
                    **base_meta,
                    "statement_type": statement_type,
                    "table_kind": "statement_table",
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "columns": list(df.columns),
                },
            }
            records.append(record)

    return records


def build_table_summary_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # table summary를 lightweight corpus record로 변환
    base_meta = make_base_metadata(result)
    summary_df = result.get("tables", {}).get("summary")

    if not isinstance(summary_df, pd.DataFrame):
        return []

    records: List[Dict[str, Any]] = []

    for _, row in summary_df.iterrows():
        statement_hits = row.get("statement_hits", [])
        if isinstance(statement_hits, list):
            statement_hits_text = ", ".join(statement_hits)
        else:
            statement_hits_text = str(statement_hits)

        preview = str(row.get("preview", ""))
        text = normalize_whitespace(
            f"table_type: {row.get('table_type', '')}\n"
            f"statement_hits: {statement_hits_text}\n"
            f"preview: {preview}"
        )

        record = {
            "doc_id": f"{base_meta['year']}_table_summary_{int(row['table_index'])}",
            "source_type": "table_summary",
            "title": f"table_summary_{int(row['table_index'])}",
            "text": text,
            "metadata": {
                **base_meta,
                "table_index": int(row["table_index"]),
                "table_type": row.get("table_type"),
                "n_rows": int(row.get("n_rows", 0)),
                "n_cols": int(row.get("n_cols", 0)),
                "numeric_count": int(row.get("numeric_count", 0)),
                "statement_hits": statement_hits,
            },
        }
        records.append(record)

    return records


def build_corpus_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # parser 결과 하나를 unified corpus record list로 변환
    records: List[Dict[str, Any]] = []

    records.extend(build_main_section_records(result))
    records.extend(build_note_block_records(result))
    records.extend(build_normalized_table_records(result))
    records.extend(build_table_summary_records(result))

    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    # JSONL 저장
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    # corpus 요약 CSV 저장
    rows: List[Dict[str, Any]] = []

    for record in records:
        meta = record["metadata"]
        rows.append({
            "doc_id": record["doc_id"],
            "source_type": record["source_type"],
            "title": record["title"],
            "year": meta.get("year"),
            "statement_type": meta.get("statement_type"),
            "note_number": meta.get("note_number"),
            "section_type": meta.get("section_type"),
            "text_length": len(record["text"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def write_type_summary_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    # source_type별 요약 저장
    df = pd.DataFrame(
        [{
            "source_type": record["source_type"],
            "year": record["metadata"].get("year"),
            "text_length": len(record["text"]),
        } for record in records]
    )

    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    summary = (
        df.groupby(["source_type", "year"], dropna=False)
        .agg(record_count=("source_type", "size"), avg_text_length=("text_length", "mean"))
        .reset_index()
    )
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="감사보고서 파싱 결과를 unified corpus JSONL로 변환")
    parser.add_argument("--data-dir", type=str, default="report_parser/data", help="HTML 파일 디렉토리")
    parser.add_argument("--output-dir", type=str, default="corpus_exports", help="출력 디렉토리")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    per_year_dir = output_dir / "per_year_jsonl"

    ensure_dir(output_dir)
    ensure_dir(per_year_dir)

    html_files = sorted(data_dir.glob("*.htm"))

    all_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for html_path in html_files:
        # parser 실행
        report_parser = AuditReportParser(str(html_path))
        result = report_parser.parse()

        # corpus record 생성
        records = build_corpus_from_result(result)
        all_records.extend(records)

        # per-year JSONL 저장
        year = extract_year_from_path(html_path.name)
        per_year_path = per_year_dir / f"corpus_{year}.jsonl"
        write_jsonl(records, per_year_path)

        # 연도별 요약 행 저장
        summary_rows.append({
            "file": html_path.name,
            "year": year,
            "record_count": len(records),
            "main_section_count": sum(1 for r in records if r["source_type"] == "main_section"),
            "note_block_count": sum(1 for r in records if r["source_type"] == "note_block"),
            "normalized_table_count": sum(1 for r in records if r["source_type"] == "normalized_table"),
            "table_summary_count": sum(1 for r in records if r["source_type"] == "table_summary"),
        })

    # 전체 JSONL 저장
    write_jsonl(all_records, output_dir / "unified_corpus.jsonl")

    # 요약 CSV 저장
    pd.DataFrame(summary_rows).to_csv(output_dir / "corpus_build_summary.csv", index=False, encoding="utf-8-sig")
    write_summary_csv(all_records, output_dir / "corpus_document_index.csv")
    write_type_summary_csv(all_records, output_dir / "corpus_type_summary.csv")

    # 완료 로그 출력
    print("=" * 100)
    print("[CORPUS BUILD COMPLETE]")
    print("input files:", len(html_files))
    print("total records:", len(all_records))
    print("output dir:", output_dir)
    print()

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()