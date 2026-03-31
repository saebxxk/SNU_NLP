from typing import Any, Dict, Optional
import re
import pandas as pd


def _extract_year_from_source_file(source_file: Optional[str]) -> Optional[int]:
    # source_file 경로에서 연도 추출
    if source_file is None:
        return None

    match = re.search(r"(20\d{2})", str(source_file))
    if match is None:
        return None

    return int(match.group(1))


def _extract_year_from_report_date(report_date: Optional[str]) -> Optional[int]:
    # report_date 문자열에서 연도 추출
    if report_date is None:
        return None

    match = re.search(r"(20\d{2})년", str(report_date))
    if match is None:
        return None

    return int(match.group(1))


def validate_parse_result(result: Dict[str, Any]) -> Dict[str, Any]:
    # 검증 결과 저장용
    report: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. 기본 개수 검증
    # -------------------------------------------------------------------------
    raw_tables = result["tables"]["raw_tables"]
    summary_df = result["tables"]["summary"]
    classified = result["tables"]["classified"]
    packages = result["tables"]["packages"]
    normalized = result["tables"]["normalized"]

    report["table_count_raw"] = len(raw_tables)
    report["table_count_summary"] = len(summary_df)
    report["table_count_classified"] = len(classified)

    report["table_count_match"] = (
        len(raw_tables) == len(summary_df) == len(classified)
    )

    # -------------------------------------------------------------------------
    # 2. type 분포
    # -------------------------------------------------------------------------
    type_counts = summary_df["table_type"].value_counts(dropna=False).to_dict()
    report["type_counts"] = type_counts

    # unknown_table 목록
    unknown_df = summary_df[summary_df["table_type"] == "unknown_table"].copy()
    report["unknown_count"] = len(unknown_df)
    report["unknown_table_indices"] = unknown_df["table_index"].tolist()

    # -------------------------------------------------------------------------
    # 3. package 검증
    # -------------------------------------------------------------------------
    expected_statement_types = {
        "balance_sheet",
        "income_statement",
        "comprehensive_income",
        "changes_in_equity",
        "cash_flow",
    }

    package_statement_types = {pkg["statement_type"] for pkg in packages}

    report["package_count"] = len(packages)
    report["package_statement_types"] = sorted(list(package_statement_types))
    report["package_types_complete"] = (package_statement_types == expected_statement_types)

    # package body index 중복 여부
    body_indices = [pkg["body_index"] for pkg in packages]
    body_index_set = set(body_indices)
    report["package_body_index_unique"] = (len(body_indices) == len(body_index_set))

    # -------------------------------------------------------------------------
    # 4. normalized 결과 존재 여부
    # -------------------------------------------------------------------------
    normalized_keys = set(normalized.keys())
    report["normalized_keys"] = sorted(list(normalized_keys))
    report["normalized_complete"] = (normalized_keys == expected_statement_types)

    # -------------------------------------------------------------------------
    # 4-1. report_date plausibility 검증
    # -------------------------------------------------------------------------
    metadata = result["metadata"]
    source_file = metadata.get("source_file")
    report_date = metadata.get("report_date")

    source_year = _extract_year_from_source_file(source_file)
    report_date_year = _extract_year_from_report_date(report_date)

    # 일반적으로 감사보고서 파일 연도와 감사보고서일 연도는
    # 동일 연도이거나 다음 연도일 가능성이 높다.
    # 예: 2024 재무제표 -> 2025년 2월 감사보고서일
    if source_year is None or report_date_year is None:
        report["report_date_year_check"] = {
            "ok": False,
            "source_year": source_year,
            "report_date_year": report_date_year,
            "reason": "year_parse_failed",
        }
    else:
        ok = report_date_year in {source_year, source_year + 1}
        report["report_date_year_check"] = {
            "ok": ok,
            "source_year": source_year,
            "report_date_year": report_date_year,
            "report_date": report_date,
        }

    # -------------------------------------------------------------------------
    # 4-2. package 밖 추가 본문표 검증
    # -------------------------------------------------------------------------
    packaged_body_indices = {pkg["body_index"] for pkg in packages}

    extra_statement_body_df = summary_df[
        (summary_df["table_type"] == "statement_body_table_type_a")
        & (~summary_df["table_index"].isin(packaged_body_indices))
    ].copy()

    report["extra_statement_body_count"] = len(extra_statement_body_df)
    report["extra_statement_body_indices"] = extra_statement_body_df["table_index"].tolist()
    report["extra_statement_body_df"] = extra_statement_body_df

    # -------------------------------------------------------------------------
    # 5. 타입 A 스키마 검증
    # -------------------------------------------------------------------------
    type_a_names = [
        "balance_sheet",
        "income_statement",
        "comprehensive_income",
        "cash_flow",
    ]

    type_a_schema_results = {}

    for name in type_a_names:
        if name not in normalized:
            type_a_schema_results[name] = {
                "exists": False,
                "ok": False,
                "reason": "missing",
            }
            continue

        df = normalized[name]
        expected_cols = ["과목", "주석", "당기", "전기", "row_type"]

        missing_cols = [c for c in expected_cols if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_cols]

        type_a_schema_results[name] = {
            "exists": True,
            "ok": (len(missing_cols) == 0),
            "missing_cols": missing_cols,
            "extra_cols": extra_cols,
            "n_rows": len(df),
        }

    report["type_a_schema_results"] = type_a_schema_results

    # -------------------------------------------------------------------------
    # 6. 자본변동표 스키마 검증
    # -------------------------------------------------------------------------
    ce_result = {
        "exists": False,
        "ok": False,
    }

    if "changes_in_equity" in normalized:
        ce_blocks = normalized["changes_in_equity"]
        ce_result["exists"] = True
        ce_result["block_count"] = len(ce_blocks)

        expected_block_types = {"prior_period", "current_period"}
        actual_block_types = {blk["block_type"] for blk in ce_blocks}
        ce_result["block_types"] = sorted(list(actual_block_types))
        ce_result["block_types_complete"] = (actual_block_types == expected_block_types)

        block_schema_results = []

        expected_cols = [
            "과목",
            "주석",
            "자본금",
            "주식발행 초과금",
            "이익잉여금",
            "기타자본항목",
            "총계",
            "row_type",
        ]

        for i, blk in enumerate(ce_blocks):
            df = blk["data"]
            missing_cols = [c for c in expected_cols if c not in df.columns]
            extra_cols = [c for c in df.columns if c not in expected_cols]

            block_schema_results.append({
                "block_index": i,
                "block_type": blk["block_type"],
                "start_label": blk["start_label"],
                "end_label": blk["end_label"],
                "ok": (len(missing_cols) == 0),
                "missing_cols": missing_cols,
                "extra_cols": extra_cols,
                "n_rows": len(df),
            })

        ce_result["block_schema_results"] = block_schema_results
        ce_result["ok"] = (
            ce_result["block_count"] == 2
            and ce_result["block_types_complete"]
            and all(x["ok"] for x in block_schema_results)
        )

    report["changes_in_equity_result"] = ce_result

    # -------------------------------------------------------------------------
    # 7. suspicious table rule 기반 점검
    # -------------------------------------------------------------------------
    suspicious_rows = []

    for _, row in summary_df.iterrows():
        table_index = row["table_index"]
        table_type = row["table_type"]
        n_rows = row["n_rows"]
        n_cols = row["n_cols"]
        numeric_count = row["numeric_count"]
        preview = row["preview"]

        # package 밖에서 statement_body_table_type_a가 나오면 재검토 필요
        if table_type == "statement_body_table_type_a":
            if table_index not in body_index_set:
                suspicious_rows.append({
                    "table_index": table_index,
                    "reason": "statement_body_outside_package",
                    "table_type": table_type,
                    "preview": preview,
                })

        # 본문표인데 너무 작음
        if table_type in ["statement_body_table_type_a", "statement_body_table_changes_in_equity"]:
            if n_rows < 5:
                suspicious_rows.append({
                    "table_index": table_index,
                    "reason": "body_table_but_too_small",
                    "table_type": table_type,
                    "preview": preview,
                })

        # 제목표인데 너무 큼
        if table_type == "statement_title_table":
            if n_rows > 10 or n_cols > 4:
                suspicious_rows.append({
                    "table_index": table_index,
                    "reason": "title_table_but_too_large",
                    "table_type": table_type,
                    "preview": preview,
                })

        # unknown인데 숫자가 많으면 재검토 필요
        if table_type == "unknown_table":
            if numeric_count >= 10:
                suspicious_rows.append({
                    "table_index": table_index,
                    "reason": "unknown_but_numeric_heavy",
                    "table_type": table_type,
                    "preview": preview,
                })

    suspicious_df = pd.DataFrame(suspicious_rows)
    report["suspicious_df"] = suspicious_df

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    # 검증 리포트 출력 함수
    print("=" * 80)
    print("[1] 기본 개수 검증")
    print("raw table count      :", report["table_count_raw"])
    print("summary table count  :", report["table_count_summary"])
    print("classified row count :", report["table_count_classified"])
    print("count match          :", report["table_count_match"])
    print()

    print("=" * 80)
    print("[2] table_type 분포")
    for k, v in report["type_counts"].items():
        print(f"{k}: {v}")
    print()

    print("=" * 80)
    print("[3] unknown_table")
    print("unknown count:", report["unknown_count"])
    print("unknown indices:", report["unknown_table_indices"][:50])
    print()

    print("=" * 80)
    print("[4] package 검증")
    print("package count:", report["package_count"])
    print("package types:", report["package_statement_types"])
    print("package types complete:", report["package_types_complete"])
    print("package body index unique:", report["package_body_index_unique"])
    print()

    print("=" * 80)
    print("[5] normalized 결과")
    print("normalized keys:", report["normalized_keys"])
    print("normalized complete:", report["normalized_complete"])
    print()

    print("=" * 80)
    print("[5-1] report_date 검증")
    for k, v in report["report_date_year_check"].items():
        print(f"{k}: {v}")
    print()

    print("=" * 80)
    print("[5-2] package 밖 추가 본문표")
    print("extra statement body count:", report["extra_statement_body_count"])
    print("extra statement body indices:", report["extra_statement_body_indices"])

    extra_statement_body_df = report["extra_statement_body_df"]
    if not extra_statement_body_df.empty:
        print(extra_statement_body_df.to_string())
    print()

    print("=" * 80)
    print("[6] 타입 A 스키마 검증")
    for name, info in report["type_a_schema_results"].items():
        print(f"- {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
    print()

    print("=" * 80)
    print("[7] 자본변동표 검증")
    ce = report["changes_in_equity_result"]
    for k, v in ce.items():
        if k != "block_schema_results":
            print(f"{k}: {v}")

    if "block_schema_results" in ce:
        print("block schema results:")
        for blk in ce["block_schema_results"]:
            print("  ", blk)
    print()

    print("=" * 80)
    print("[8] suspicious table")
    suspicious_df = report["suspicious_df"]

    if suspicious_df.empty:
        print("없음")
    else:
        print(suspicious_df.to_string())


def show_table_type_samples(result: Dict[str, Any], table_type: str, n: int = 5) -> None:
    # table_type별 샘플 확인 함수
    summary_df = result["tables"]["summary"]
    raw_tables = result["tables"]["raw_tables"]

    target = summary_df[summary_df["table_type"] == table_type].head(n)

    for _, row in target.iterrows():
        idx = row["table_index"]
        print("=" * 100)
        print(f"[table_index={idx}] type={table_type}, shape={raw_tables[idx].shape}")
        print(raw_tables[idx].head(10).to_string())
        print()
