# 필요한 라이브러리 import
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 프로젝트 루트를 import 경로에 추가
SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()
IMPORT_CANDIDATES = [CWD, SCRIPT_DIR, CWD.parent, SCRIPT_DIR.parent]

for candidate in IMPORT_CANDIDATES:
    if (candidate / "report_parser").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

# 최상위 parser import
from report_parser.audit_report_parser import AuditReportParser


def extract_year_from_path(path: Path) -> int | None:
    # 파일명에서 연도 추출
    match = re.search(r"(20\d{2})", path.name)
    if match is None:
        return None
    return int(match.group(1))


def ensure_dir(path: Path) -> None:
    # 디렉토리가 없으면 생성
    path.mkdir(parents=True, exist_ok=True)


def build_summary_row(result: Dict[str, Any], html_path: Path) -> Dict[str, Any]:
    # 연도별 텍스트 파싱 요약 행 생성
    text_result = result["text"]
    validation = text_result["validation"]

    return {
        "source_file": str(html_path),
        "file_name": html_path.name,
        "year": extract_year_from_path(html_path),
        "company_name": result["metadata"].get("company_name"),
        "report_date": result["metadata"].get("report_date"),
        "line_count": text_result.get("line_count"),
        "main_section_count": len(text_result.get("main_sections", [])),
        "note_block_count": len(text_result.get("note_blocks", [])),
        "has_auditor_report_title": validation.get("has_auditor_report_title"),
        "has_opinion": validation.get("has_opinion"),
        "has_basis_for_opinion": validation.get("has_basis_for_opinion"),
        "has_key_audit_matters": validation.get("has_key_audit_matters"),
        "has_notes_section": validation.get("has_notes_section"),
        "all_main_section_offsets_valid": validation.get("all_main_section_offsets_valid"),
        "all_note_offsets_valid": validation.get("all_note_offsets_valid"),
    }


def build_main_section_rows(result: Dict[str, Any], html_path: Path) -> List[Dict[str, Any]]:
    # 메인 섹션 flat row 생성
    rows: List[Dict[str, Any]] = []
    text_result = result["text"]
    year = extract_year_from_path(html_path)

    for section in text_result.get("main_sections", []):
        rows.append({
            "source_file": str(html_path),
            "file_name": html_path.name,
            "year": year,
            "section_id": section.get("section_id"),
            "section_type": section.get("section_type"),
            "title": section.get("title"),
            "level": section.get("level"),
            "start": section.get("start"),
            "end": section.get("end"),
            "start_line_index": section.get("start_line_index"),
            "end_line_index": section.get("end_line_index"),
            "text": section.get("text"),
            "metadata_json": json.dumps(section.get("metadata", {}), ensure_ascii=False),
        })

    return rows


def build_note_block_rows(result: Dict[str, Any], html_path: Path) -> List[Dict[str, Any]]:
    # 주석 블록 flat row 생성
    rows: List[Dict[str, Any]] = []
    text_result = result["text"]
    year = extract_year_from_path(html_path)

    for note in text_result.get("note_blocks", []):
        rows.append({
            "source_file": str(html_path),
            "file_name": html_path.name,
            "year": year,
            "note_number": note.get("note_number"),
            "note_title": note.get("note_title"),
            "title_line": note.get("title_line"),
            "level": note.get("level"),
            "start": note.get("start"),
            "end": note.get("end"),
            "start_line_index": note.get("start_line_index"),
            "end_line_index": note.get("end_line_index"),
            "is_continued": note.get("metadata", {}).get("is_continued"),
            "text": note.get("text"),
            "metadata_json": json.dumps(note.get("metadata", {}), ensure_ascii=False),
        })

    return rows


def save_per_year_json(result: Dict[str, Any], html_path: Path, output_dir: Path) -> None:
    # 연도별 전체 텍스트 결과 JSON 저장
    year = extract_year_from_path(html_path)
    out_name = f"text_parse_{year}.json" if year is not None else f"{html_path.stem}.json"
    out_path = output_dir / out_name

    payload = {
        "metadata": result["metadata"],
        "document_structure": result["document_structure"],
        "text": result["text"],
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_text_results(data_dir: Path, output_dir: Path) -> None:
    # 전체 HTML 파일을 순회하며 텍스트 결과 export
    ensure_dir(output_dir)
    ensure_dir(output_dir / "per_year_json")

    html_files = sorted(data_dir.glob("*.htm"))
    if not html_files:
        raise FileNotFoundError(f"HTML 파일이 없습니다: {data_dir}")

    summary_rows: List[Dict[str, Any]] = []
    main_section_rows: List[Dict[str, Any]] = []
    note_block_rows: List[Dict[str, Any]] = []
    jsonl_rows: List[str] = []

    for html_path in html_files:
        # parser 실행
        parser = AuditReportParser(str(html_path))
        result = parser.parse()

        # 연도별 요약 row 추가
        summary_rows.append(build_summary_row(result, html_path))

        # 메인 섹션 row 추가
        main_section_rows.extend(build_main_section_rows(result, html_path))

        # 주석 블록 row 추가
        note_block_rows.extend(build_note_block_rows(result, html_path))

        # 연도별 JSON 저장
        save_per_year_json(result, html_path, output_dir / "per_year_json")

        # JSONL용 한 줄 payload 저장
        jsonl_payload = {
            "metadata": result["metadata"],
            "document_structure": result["document_structure"],
            "text": result["text"],
        }
        jsonl_rows.append(json.dumps(jsonl_payload, ensure_ascii=False))

    # CSV 저장
    pd.DataFrame(summary_rows).to_csv(output_dir / "text_parse_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(main_section_rows).to_csv(output_dir / "text_main_sections.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(note_block_rows).to_csv(output_dir / "text_note_blocks.csv", index=False, encoding="utf-8-sig")

    # JSONL 저장
    (output_dir / "text_parse_all.jsonl").write_text("\n".join(jsonl_rows), encoding="utf-8")

    # 실행 요약 출력
    print("=" * 100)
    print("[TEXT EXPORT DONE]")
    print("data_dir:", data_dir)
    print("output_dir:", output_dir)
    print("file_count:", len(html_files))
    print("summary_rows:", len(summary_rows))
    print("main_section_rows:", len(main_section_rows))
    print("note_block_rows:", len(note_block_rows))
    print("saved files:")
    print("-", output_dir / "text_parse_summary.csv")
    print("-", output_dir / "text_main_sections.csv")
    print("-", output_dir / "text_note_blocks.csv")
    print("-", output_dir / "text_parse_all.jsonl")
    print("-", output_dir / "per_year_json")


def main() -> None:
    # CLI 엔트리포인트
    parser = argparse.ArgumentParser(description="Export parsed text structure from Samsung audit report HTML files.")
    parser.add_argument("--data-dir", type=str, default="report_parser/data", help="HTML 데이터 폴더 경로")
    parser.add_argument("--output-dir", type=str, default="text_exports", help="출력 폴더 경로")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    export_text_results(data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
