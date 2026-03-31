# 필요한 라이브러리 import
from pathlib import Path

# 최상위 parser import
from report_parser.audit_report_parser import AuditReportParser


def run_batch(data_dir: str) -> None:
    # 데이터 디렉토리 내 모든 HTML 파일 순회
    html_files = sorted(Path(data_dir).glob("*.htm"))

    print("=" * 170)
    print("[TEXT PARSE BATCH RESULT]")
    print(
        f"{'file':30} | "
        f"{'main_sections':13} | "
        f"{'note_blocks':11} | "
        f"{'has_opinion':11} | "
        f"{'has_notes':9} | "
        f"{'main_offsets_ok':16} | "
        f"{'note_offsets_ok':15}"
    )
    print("-" * 170)

    for html_path in html_files:
        # parser 실행
        parser = AuditReportParser(str(html_path))
        result = parser.parse()

        # text 결과가 없으면 연결이 안 된 상태
        if "text" not in result:
            print(f"{html_path.name:30} | text_result_missing")
            continue

        text_result = result["text"]
        validation = text_result["validation"]

        # 실제 validator key 기준으로 출력
        main_offsets_ok = validation.get("all_main_section_offsets_valid", False)
        note_offsets_ok = validation.get("all_note_offsets_valid", False)

        print(
            f"{html_path.name:30} | "
            f"{len(text_result['main_sections']):13} | "
            f"{len(text_result['note_blocks']):11} | "
            f"{str(validation.get('has_opinion', False)):11} | "
            f"{str(validation.get('has_notes_section', False)):9} | "
            f"{str(main_offsets_ok):16} | "
            f"{str(note_offsets_ok):15}"
        )


if __name__ == "__main__":
    # 데이터 경로 지정
    run_batch("report_parser/data")