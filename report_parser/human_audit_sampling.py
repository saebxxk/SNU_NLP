from __future__ import annotations

# 커맨드라인 인자 처리를 위한 라이브러리
import argparse

# 랜덤 샘플링을 위한 라이브러리
import random

# 시스템 경로 보정을 위한 라이브러리
import sys

# 경로 처리를 위한 라이브러리
from pathlib import Path

# 타입 힌트를 위한 라이브러리
from typing import Dict, List, Optional

# 데이터 처리용 라이브러리
import pandas as pd


# =============================================================================
# 0. 프로젝트 import 경로 보정
# =============================================================================

def bootstrap_import_path() -> None:
    # 현재 작업 디렉터리와 스크립트 디렉터리 후보를 sys.path에 추가
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    candidates = [cwd, script_dir, cwd.parent, script_dir.parent]

    for candidate in candidates:
        if (candidate / "report_parser").exists():
            sys.path.insert(0, str(candidate))
            return


# import 전에 경로 보정 실행
bootstrap_import_path()

# 프로젝트 parser import
from report_parser.audit_report_parser import AuditReportParser
from report_parser.validation import validate_parse_result


# =============================================================================
# 1. 공통 유틸 함수
# =============================================================================

def extract_year_from_filename(path: Path) -> Optional[int]:
    # 파일명에서 연도 추출
    for part in path.stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def build_package_role_map(packages: List[Dict[str, object]]) -> Dict[int, str]:
    # statement package 정보를 table_index -> 역할 문자열로 변환
    role_map: Dict[int, str] = {}

    for pkg in packages:
        statement_type = str(pkg["statement_type"])
        title_index = int(pkg["title_index"])
        body_index = int(pkg["body_index"])

        role_map[title_index] = f"title:{statement_type}"
        role_map[body_index] = f"body:{statement_type}"

        footer_index = pkg.get("footer_index")
        if footer_index is not None:
            role_map[int(footer_index)] = f"footer:{statement_type}"

    return role_map


def compute_review_priority(row: pd.Series) -> int:
    # 사람이 먼저 봐야 할 가능성이 높은 표에 우선순위 점수 부여
    score = 0

    table_type = str(row["table_type"])
    package_role = str(row["package_role"])
    n_rows = int(row["n_rows"])
    n_cols = int(row["n_cols"])
    numeric_count = int(row["numeric_count"])
    report_date_ok = bool(row["report_date_year_check_ok"])

    # package 바깥 본문표는 가장 우선적으로 검토
    if table_type == "statement_body_table_type_a" and package_role == "outside_package":
        score += 100

    # unknown_table은 항상 우선순위 높게 부여
    if table_type == "unknown_table":
        score += 80

    # 분류 경계에 걸리기 쉬운 broad type들에 가중치 부여
    if table_type in {
        "simple_matrix_table",
        "note_like_table",
        "financial_instrument_table",
        "counterparty_table",
        "rollforward_table",
        "audit_related_table",
    }:
        score += 25

    # 핵심 statement package 자체도 일정량 직접 확인할 가치가 있음
    if package_role.startswith("body:"):
        score += 20
    elif package_role.startswith("title:") or package_role.startswith("footer:"):
        score += 10

    # shape가 지나치게 작은/큰 표는 오분류 가능성이 있으므로 가중치 부여
    if n_rows <= 2 or n_cols <= 1:
        score += 10
    if n_rows >= 40:
        score += 8
    if n_cols >= 8:
        score += 8

    # 숫자가 매우 많은 표는 단순 메타표로 분류되면 위험하므로 가중치 부여
    if numeric_count >= 20:
        score += 6

    # report_date 검증 실패 파일은 전체적으로 더 조심해서 봐야 함
    if not report_date_ok:
        score += 5

    return score


# =============================================================================
# 2. 전체 표 population 생성
# =============================================================================

def build_audit_population(data_dir: Path) -> pd.DataFrame:
    # 모든 HTML 파일을 파싱하여 표 단위 population 생성
    rows: List[Dict[str, object]] = []

    html_paths = sorted(data_dir.glob("*.htm"))
    if not html_paths:
        raise FileNotFoundError(f"HTML 파일을 찾지 못했습니다: {data_dir}")

    for html_path in html_paths:
        # parser 실행
        parser = AuditReportParser(str(html_path))
        result = parser.parse()

        # validation 실행
        validation = validate_parse_result(result)

        # table summary와 package 역할 맵 준비
        summary_df = result["tables"]["summary"].copy()
        packages = result["tables"]["packages"]
        role_map = build_package_role_map(packages)

        # 파일 수준 정보
        file_year = extract_year_from_filename(html_path)
        metadata = result["metadata"]
        report_date = metadata.get("report_date")
        report_date_check = validation.get("report_date_year_check", {})

        # 표 단위 행 생성
        for _, row in summary_df.iterrows():
            table_index = int(row["table_index"])
            package_role = role_map.get(table_index, "outside_package")

            rows.append(
                {
                    # 식별자 정보
                    "year": file_year,
                    "source_file": html_path.name,
                    "table_index": table_index,
                    "table_uid": f"{html_path.stem}__{table_index}",

                    # 예측 결과 정보
                    "table_type": row["table_type"],
                    "statement_hits": ", ".join(row["statement_hits"]),
                    "package_role": package_role,

                    # 기본 shape 정보
                    "n_rows": int(row["n_rows"]),
                    "n_cols": int(row["n_cols"]),
                    "numeric_count": int(row["numeric_count"]),

                    # 문서 수준 메타정보
                    "report_date": report_date,
                    "report_date_year_check_ok": report_date_check.get("ok"),
                    "company_name": metadata.get("company_name"),
                    "encoding": metadata.get("encoding"),

                    # preview 정보
                    "preview": row["preview"],

                    # 사람이 채울 컬럼
                    "human_label": "",
                    "is_prediction_correct": "",
                    "human_notes": "",
                }
            )

    # DataFrame 생성
    population_df = pd.DataFrame(rows)

    # 우선순위 점수 계산
    population_df["review_priority"] = population_df.apply(compute_review_priority, axis=1)

    # 보기 좋게 정렬
    population_df = population_df.sort_values(
        by=["year", "table_index"], ascending=[True, True]
    ).reset_index(drop=True)

    return population_df


# =============================================================================
# 3. 샘플링 함수
# =============================================================================

def sample_stratified_by_type(
    population_df: pd.DataFrame,
    n_per_type: int,
    seed: int,
) -> pd.DataFrame:
    # table_type별로 균형 샘플 추출
    sampled_frames: List[pd.DataFrame] = []

    for table_type, group_df in population_df.groupby("table_type", sort=True):
        # review_priority가 높은 표를 우선 후보로 둔 뒤,
        # 동일 우선순위 내부에서 랜덤성을 주기 위해 sample 사용
        group_sorted = group_df.sort_values(
            by=["review_priority", "year", "table_index"],
            ascending=[False, True, True],
        )

        # 표 수가 충분하면 상위 후보군에서 랜덤 샘플링
        if len(group_sorted) > n_per_type:
            # 상위 후보군 크기는 넉넉하게 n_per_type * 3 또는 전체 길이 중 작은 값 사용
            candidate_size = min(len(group_sorted), max(n_per_type * 3, n_per_type))
            candidate_df = group_sorted.head(candidate_size)
            sampled_df = candidate_df.sample(n=n_per_type, random_state=seed)
        else:
            sampled_df = group_sorted.copy()

        sampled_frames.append(sampled_df)

    if not sampled_frames:
        return population_df.head(0).copy()

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    sampled_df = sampled_df.sort_values(
        by=["table_type", "year", "table_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return sampled_df


def sample_priority_rows(
    population_df: pd.DataFrame,
    top_n: int,
    seed: int,
) -> pd.DataFrame:
    # review_priority가 높은 표를 우선적으로 추출
    # 같은 우선순위에서는 랜덤 셔플 후 상위 top_n을 선택
    shuffled_df = population_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled_df = shuffled_df.sort_values(
        by=["review_priority", "year", "table_index"],
        ascending=[False, True, True],
    ).head(top_n)

    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df


def sample_core_statement_rows(population_df: pd.DataFrame) -> pd.DataFrame:
    # 핵심 statement package 표는 전수 확인용으로 별도 추출
    core_df = population_df[
        population_df["package_role"].str.startswith(("title:", "body:", "footer:"))
    ].copy()

    core_df = core_df.sort_values(
        by=["year", "table_index"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return core_df


# =============================================================================
# 4. 요약 리포트 생성
# =============================================================================

def build_summary(population_df: pd.DataFrame) -> pd.DataFrame:
    # table_type별 개수와 기초 통계를 요약
    summary_df = (
        population_df.groupby("table_type", dropna=False)
        .agg(
            table_count=("table_uid", "count"),
            avg_rows=("n_rows", "mean"),
            avg_cols=("n_cols", "mean"),
            avg_numeric_count=("numeric_count", "mean"),
            max_priority=("review_priority", "max"),
        )
        .reset_index()
        .sort_values(by=["table_count", "table_type"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary_df


# =============================================================================
# 5. 저장 함수
# =============================================================================

def save_with_human_columns(df: pd.DataFrame, output_path: Path) -> None:
    # 사람이 엑셀/스프레드시트에서 바로 라벨링할 수 있도록 CSV 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# =============================================================================
# 6. 메인 함수
# =============================================================================

def main() -> None:
    # CLI 인자 정의
    parser = argparse.ArgumentParser(
        description="감사보고서 표 파싱 결과에 대한 human audit용 샘플 CSV 생성"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="report_parser/data",
        help="감사보고서 HTML 파일 디렉터리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_outputs",
        help="출력 CSV 저장 디렉터리 경로",
    )
    parser.add_argument(
        "--n-per-type",
        type=int,
        default=5,
        help="table_type별 stratified sample 개수",
    )
    parser.add_argument(
        "--top-n-priority",
        type=int,
        default=40,
        help="우선 검토 샘플 개수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    # 랜덤 시드 고정
    random.seed(args.seed)

    # 경로 준비
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 전체 population 생성
    population_df = build_audit_population(data_dir)

    # 샘플 생성
    stratified_df = sample_stratified_by_type(
        population_df=population_df,
        n_per_type=args.n_per_type,
        seed=args.seed,
    )
    priority_df = sample_priority_rows(
        population_df=population_df,
        top_n=args.top_n_priority,
        seed=args.seed,
    )
    core_df = sample_core_statement_rows(population_df)
    summary_df = build_summary(population_df)

    # 파일 저장
    save_with_human_columns(population_df, output_dir / "audit_population_all.csv")
    save_with_human_columns(stratified_df, output_dir / "audit_sample_stratified.csv")
    save_with_human_columns(priority_df, output_dir / "audit_sample_priority.csv")
    save_with_human_columns(core_df, output_dir / "audit_sample_core_statements.csv")
    save_with_human_columns(summary_df, output_dir / "audit_summary_by_type.csv")

    # 콘솔 출력
    print("=" * 100)
    print("[HUMAN AUDIT SAMPLE GENERATED]")
    print(f"data_dir                 : {data_dir}")
    print(f"output_dir               : {output_dir}")
    print(f"population_count         : {len(population_df)}")
    print(f"stratified_sample_count  : {len(stratified_df)}")
    print(f"priority_sample_count    : {len(priority_df)}")
    print(f"core_statement_count     : {len(core_df)}")
    print()
    print("saved files:")
    print(f"- {output_dir / 'audit_population_all.csv'}")
    print(f"- {output_dir / 'audit_sample_stratified.csv'}")
    print(f"- {output_dir / 'audit_sample_priority.csv'}")
    print(f"- {output_dir / 'audit_sample_core_statements.csv'}")
    print(f"- {output_dir / 'audit_summary_by_type.csv'}")


# 스크립트 직접 실행 시 main 호출
if __name__ == "__main__":
    main()
