import re
from typing import Any, Dict, List, Optional
import pandas as pd

from .base import (
    StatementRelatedTableParser,
    StatementBodyTableParser,
)

class StatementTitleTableParser(StatementRelatedTableParser):
    # 재무제표 제목표 parser
    table_type = "statement_title_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 재무제표 제목 키워드가 있고, 작고, 숫자가 많지 않고,
        # 요약표/설명표가 아니면 제목표로 판단
        has_any_title = any(features["statement_title_keywords"].values())

        return (
            has_any_title
            and features["n_rows"] <= 6
            and features["numeric_count"] <= 15
            and not features["has_summary_phrase"]
            and not features["has_star_note_like"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        # 제목표 파싱 payload
        return {
            "statement_type": self.infer_statement_type(),
        }


class StatementBodyTypeAParser(StatementBodyTableParser):
    # 재무상태표 / 손익계산서 / 포괄손익계산서 / 현금흐름표 본문 parser
    table_type = "statement_body_table_type_a"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 타입 A 본문표 판별
        #
        # 수정 포인트:
        # 1. 현재 삼성전자 2014~2024 코퍼스에서
        #    실제 4개 본문표(재무상태표/손익계산서/포괄손익계산서/현금흐름표)는
        #    raw shape가 일관되게 6개 컬럼이다.
        # 2. 반면 false positive였던
        #    - 미처분이익잉여금처분계산서 계열은 보통 5개 컬럼
        #    - 이연법인세 변동표 계열은 7개 컬럼
        #    이므로 n_cols == 6 제약이 precision을 크게 높인다.
        # 3. "미처분이익잉여금"은 명시적으로 제외한다.

        text = features["text"]
        col_text = features["col_text"]

        # 명시적 제외 키워드
        has_disqualifying_keyword = (
            ("미처분이익잉여금" in text) or ("미처분이익잉여금" in col_text)
        )

        return (
            features["has_subject_col"]
            and features["has_current_col"]
            and features["has_prior_col"]
            and features["n_rows"] >= 5
            and features["n_cols"] == 6
            and not features["has_equity_columns"]
            and not has_disqualifying_keyword
        )

    def normalize(self) -> pd.DataFrame:
        # 타입 A 본문표 정규화
        out = self.df.copy()

        # 컬럼명 공백 정리
        out.columns = [self.clean_cell(c) for c in out.columns]

        # 과목 컬럼 찾기
        subject_candidates = ["과 목", "과목"]
        subject_col = None
        for c in out.columns:
            if c in subject_candidates:
                subject_col = c
                break

        if subject_col is None:
            raise ValueError("과목 컬럼을 찾지 못했습니다.")

        # 주석 컬럼 찾기
        note_candidates = ["주석", "주 석"]
        note_col = None
        for c in out.columns:
            if c in note_candidates:
                note_col = c
                break

        # 셀 정리
        out = out.map(lambda x: pd.NA if self.clean_cell(x) == "" else self.clean_cell(x))

        # 당기 / 전기 컬럼 찾기
        current_cols = [c for c in out.columns if ("당" in c and "기" in c)]
        prior_cols = [c for c in out.columns if ("전" in c and "기" in c)]

        if len(current_cols) != 2:
            raise ValueError(f"당기 컬럼 2개를 찾지 못했습니다: {current_cols}")
        if len(prior_cols) != 2:
            raise ValueError(f"전기 컬럼 2개를 찾지 못했습니다: {prior_cols}")

        # 기간 컬럼 병합
        out["당기"] = out[current_cols].bfill(axis=1).iloc[:, 0]
        out["전기"] = out[prior_cols].bfill(axis=1).iloc[:, 0]

        # 필요한 컬럼만 유지
        keep_cols = [subject_col]
        if note_col is not None:
            keep_cols.append(note_col)

        result = out[keep_cols].copy()
        result["당기"] = out["당기"]
        result["전기"] = out["전기"]

        # 컬럼명 표준화
        rename_map = {subject_col: "과목"}
        if note_col is not None:
            rename_map[note_col] = "주석"
        result = result.rename(columns=rename_map)

        # 주석 컬럼이 없으면 생성
        if "주석" not in result.columns:
            result["주석"] = pd.NA

        # 필수 컬럼 존재 강제
        if "당기" not in result.columns or "전기" not in result.columns:
            raise ValueError(
                f"[table_index={self.table_index}] normalize 결과에 당기/전기 컬럼이 없습니다. "
                f"columns={list(result.columns)}"
            )

        # 컬럼 순서 고정
        result = result[["과목", "주석", "당기", "전기"]]

        # 숫자 변환
        result["당기"] = result["당기"].apply(self.parse_number)
        result["전기"] = result["전기"].apply(self.parse_number)

        # 주석 정리
        result["주석"] = result["주석"].apply(self.clean_note_value)

        # 완전 빈 행 제거
        result = result.dropna(how="all").reset_index(drop=True)

        return result

    def classify_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # 타입 A 행 유형 분류
        # 학습용 기준:
        # - structure: 대제목 / 섹션 제목
        # - total: 총계 / 이익 / 합계 성격
        # - value: 실제 계정 / 세부 항목

        # 원본 보존
        out = df.copy()

        # 필수 컬럼 검사
        required_cols = ["과목", "당기", "전기"]
        missing_cols = [c for c in required_cols if c not in out.columns]
        if missing_cols:
            raise ValueError(
                f"[table_index={self.table_index}] classify_rows 입력 컬럼 이상: "
                f"missing={missing_cols}, columns={list(out.columns)}"
            )

        # row_type 저장
        row_types = []

        # total 후보 키워드
        total_keywords = [
            "총계",
            "총포괄손익",
            "영업이익",
            "영업손실",
            "매출총이익",
            "법인세비용차감전순이익",
            "당기순이익",
            "자산총계",
            "부채총계",
            "자본총계",
        ]

        # 정확한 대제목 후보
        exact_structure_labels = [
            "자산",
            "부채",
            "자본",
        ]

        # 섹션 제목 후보
        section_keywords = [
            "유동자산",
            "비유동자산",
            "유동부채",
            "비유동부채",
            "영업활동현금흐름",
            "투자활동현금흐름",
            "재무활동현금흐름",
            "기타포괄손익",
        ]

        # 로마숫자 패턴
        roman_section_pattern = re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.")

        # 세부 항목 패턴
        detail_pattern = re.compile(r"^\s*([0-9]+\.|[가나다라마바사아자차카타파하]\.)")

        for _, row in out.iterrows():
            # 과목 정리
            account = self.clean_cell(row["과목"])
            account_no_space = re.sub(r"\s+", "", account)

            # 값 존재 여부
            has_value = not (pd.isna(row["당기"]) and pd.isna(row["전기"]))

            # 1. 값이 없으면 structure
            if not has_value:
                row_types.append("structure")
                continue

            # 2. total 키워드면 total
            if any(kw.replace(" ", "") in account_no_space for kw in total_keywords):
                row_types.append("total")
                continue

            # 3. 번호/가나다 패턴이면 value
            if detail_pattern.match(account):
                row_types.append("value")
                continue

            # 4. 정확한 대제목이면 structure
            if account_no_space in exact_structure_labels:
                row_types.append("structure")
                continue

            # 5. 로마숫자로 시작하는 경우
            if roman_section_pattern.match(account):
                if any(kw in account_no_space for kw in section_keywords):
                    row_types.append("structure")
                else:
                    row_types.append("value")
                continue

            # 6. 나머지는 value
            row_types.append("value")

        # 결과 컬럼 추가
        out["row_type"] = row_types

        return out


class ChangesInEquityParser(StatementBodyTableParser):
    # 자본변동표 본문 parser
    table_type = "statement_body_table_changes_in_equity"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 자본변동표 본문 판별
        return features["has_equity_columns"] and features["n_rows"] >= 5

    def normalize(self) -> pd.DataFrame:
        # 자본변동표 정규화
        out = self.df.copy()

        # 컬럼명 공백 정리
        out.columns = [self.clean_cell(c) for c in out.columns]

        # 셀 정리
        out = out.map(lambda x: pd.NA if self.clean_cell(x) == "" else self.clean_cell(x))

        # 주석 컬럼 정리
        if "주 석" in out.columns:
            out["주 석"] = out["주 석"].apply(self.clean_note_value)

        # 숫자 컬럼 후보
        value_cols = [
            "자본금",
            "주식발행 초과금",
            "이익잉여금",
            "기타자본항목",
            "총 계",
        ]

        # 숫자 변환
        for col in value_cols:
            if col in out.columns:
                out[col] = out[col].apply(self.parse_number)

        # 컬럼명 표준화
        rename_map = {
            "과 목": "과목",
            "주 석": "주석",
            "총 계": "총계",
        }
        out = out.rename(columns=rename_map)

        # 주석 컬럼 보정
        if "주석" not in out.columns:
            out["주석"] = pd.NA

        # 완전 빈 행 제거
        out = out.dropna(how="all").reset_index(drop=True)

        return out

    def classify_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # 자본변동표 행 유형 분류
        out = df.copy()

        # 자본변동표 숫자 컬럼
        value_cols = ["자본금", "주식발행 초과금", "이익잉여금", "기타자본항목", "총계"]

        # 필수 컬럼 검사
        required_cols = ["과목"] + value_cols
        missing_cols = [c for c in required_cols if c not in out.columns]
        if missing_cols:
            raise ValueError(
                f"[table_index={self.table_index}] 자본변동표 classify_rows 입력 컬럼 이상: "
                f"missing={missing_cols}, columns={list(out.columns)}"
            )

        row_types = []

        def is_time_row(account: Any) -> bool:
            # 시점행 여부 판별
            if pd.isna(account):
                return False

            s = str(account)
            keywords = ["전기초", "전기말", "당기초", "당기말"]
            return any(kw in s for kw in keywords)

        def is_section_row(row: pd.Series) -> bool:
            # 숫자 컬럼이 전부 비어 있으면 섹션행
            return row[value_cols].isna().all()

        for _, row in out.iterrows():
            account = row["과목"]

            if is_time_row(account):
                row_types.append("time_row")
            elif is_section_row(row):
                row_types.append("section_row")
            else:
                row_types.append("event_row")

        out["row_type"] = row_types
        return out

    def split_blocks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # 자본변동표를 전기 / 당기 블록으로 분리
        out = df.copy()

        def get_time_row_kind(account: Any) -> Optional[str]:
            # 시점행 종류 추론
            if pd.isna(account):
                return None

            s = str(account)

            if "전기초" in s:
                return "prior_start"
            if "전기말" in s:
                return "prior_end"
            if "당기초" in s:
                return "current_start"
            if "당기말" in s:
                return "current_end"

            return None

        blocks = []
        current_rows = []
        start_label = None
        block_type = None

        for _, row in out.iterrows():
            account = row["과목"]
            kind = get_time_row_kind(account)

            # 블록 시작
            if kind in ["prior_start", "current_start"]:
                # 닫히지 않은 블록이 있으면 우선 저장
                if current_rows:
                    block_df = pd.DataFrame(current_rows).reset_index(drop=True)
                    blocks.append({
                        "block_type": block_type,
                        "start_label": start_label,
                        "end_label": None,
                        "data": block_df,
                    })

                current_rows = [row.to_dict()]
                start_label = account
                block_type = "prior_period" if kind == "prior_start" else "current_period"
                continue

            # 현재 블록에 행 누적
            if current_rows:
                current_rows.append(row.to_dict())

                # 블록 종료
                if kind in ["prior_end", "current_end"]:
                    block_df = pd.DataFrame(current_rows).reset_index(drop=True)
                    blocks.append({
                        "block_type": block_type,
                        "start_label": start_label,
                        "end_label": account,
                        "data": block_df,
                    })

                    current_rows = []
                    start_label = None
                    block_type = None

        # 마지막 블록이 닫히지 않았으면 저장
        if current_rows:
            block_df = pd.DataFrame(current_rows).reset_index(drop=True)
            blocks.append({
                "block_type": block_type,
                "start_label": start_label,
                "end_label": None,
                "data": block_df,
            })

        return blocks

    def parse(self) -> Dict[str, Any]:
        # 자본변동표는 normalized + blocks 둘 다 필요하므로 parse override
        payload = self.build_base_payload()

        normalized = self.normalize()
        classified = self.classify_rows(normalized)
        blocks = self.split_blocks(classified)

        payload.update({
            "normalized": classified,
            "blocks": blocks,
        })
        return payload


class StatementFooterTableParser(StatementRelatedTableParser):
    # 재무제표 footer 표 parser
    table_type = "statement_footer_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # footer 문구가 있으면 footer 표
        return features["has_footer_phrase"]

    def parse_payload(self) -> Dict[str, Any]:
        # footer 파싱 payload
        return {
            "text": self.text,
        }