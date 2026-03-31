# run_parse.py

import argparse
from pathlib import Path

from report_parser.audit_report_parser import AuditReportParser
from report_parser.validation import (
    validate_parse_result,
    print_validation_report,
)

def main() -> None:
    # 인자 파서 생성
    parser = argparse.ArgumentParser(description="Parse Samsung audit report HTML file.")
    parser.add_argument("html_path", type=str, help="Path to audit report HTML file")
    args = parser.parse_args()

    # 입력 파일 경로
    html_path = Path(args.html_path)

    # 파일 존재 여부 확인
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # 파서 실행
    audit_parser = AuditReportParser(str(html_path))
    result = audit_parser.parse()

    # 검증 출력
    print(result["metadata"])
    print(result["document_structure"])

    report = validate_parse_result(result)
    print_validation_report(report)

if __name__ == "__main__":
    main()