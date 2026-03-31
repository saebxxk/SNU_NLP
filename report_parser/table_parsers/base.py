import re
from typing import Any, Dict, Optional
import pandas as pd
from report_parser.utils import (
    clean_cell,
    normalize_for_keyword_match,
    table_to_text,
    summarize_table,
    is_numeric_like_string,
    count_numeric_like_cells,
    parse_number,
    clean_note_value,
)

class BaseTableParser:
    # 모든 표 parser의 최상위 공통 부모 클래스

    # 클래스 수준 broad type 이름
    table_type = "unknown_table"

    def __init__(self, df: pd.DataFrame, table_index: int):
        # 원본 표 저장
        self.df = df.copy()

        # 표 인덱스 저장
        self.table_index = table_index

        # 표 전체 텍스트 저장
        self.text = self.table_to_text(self.df)

        # 키워드 탐지용 공백 제거 텍스트 저장
        self.text_norm = self.normalize_for_keyword_match(self.text)

        # preview 저장
        self.preview = self.summarize_table(self.df)

        # shape 저장
        self.n_rows, self.n_cols = self.df.shape

        # 컬럼명 텍스트 저장
        self.col_names = [self.clean_cell(c) for c in self.df.columns.tolist()]
        self.col_text = " | ".join(self.col_names)
        self.col_text_norm = self.normalize_for_keyword_match(self.col_text)

        # 숫자처럼 보이는 셀 개수 저장
        self.numeric_count = self.count_numeric_like_cells(self.df)

    # -------------------------------------------------------------------------
    # 공통 유틸 함수
    # -------------------------------------------------------------------------

    @staticmethod
    def clean_cell(x: Any) -> str:
        # 셀 문자열 정리 함수
        return clean_cell(x)

    @staticmethod
    def normalize_for_keyword_match(text: Any) -> str:
        # 키워드 매칭용으로 공백 제거 및 소문자화
        return normalize_for_keyword_match(text)

    @classmethod
    def table_to_text(cls, df: pd.DataFrame) -> str:
        # 표 전체를 하나의 문자열로 합치는 함수
        return table_to_text(df)

    @classmethod
    def summarize_table(cls, df: pd.DataFrame, n_rows: int = 3, n_cols: int = 6) -> str:
        # 표 preview 생성 함수
        return summarize_table(df, n_rows = n_rows, n_cols = n_cols)

    @classmethod
    def is_numeric_like_string(cls, x: Any) -> bool:
        # 숫자처럼 보이는 문자열인지 판별
        return is_numeric_like_string(x)

    @classmethod
    def count_numeric_like_cells(cls, df: pd.DataFrame) -> int:
        # 숫자처럼 보이는 셀 개수 계산
        return count_numeric_like_cells(df)

    @classmethod
    def parse_number(cls, x: Any) -> Any:
        # 숫자 문자열을 숫자로 변환
        return parse_number(x)

    @classmethod
    def clean_note_value(cls, x: Any) -> Any:
        # 주석 값 문자열 정리
        return clean_note_value(x)

    # -------------------------------------------------------------------------
    # 공통 payload 빌더
    # -------------------------------------------------------------------------

    def build_base_payload(self) -> Dict[str, Any]:
        # 모든 parser가 공통으로 반환할 기본 payload
        return {
            "table_index": self.table_index,
            "table_type": self.table_type,
            "preview": self.preview,
            "shape": self.df.shape,
        }

    def parse_payload(self) -> Dict[str, Any]:
        # 자식 클래스가 override해서 추가 payload만 반환
        return {}

    # -------------------------------------------------------------------------
    # feature / 분류 / 파싱 인터페이스
    # -------------------------------------------------------------------------

    def extract_basic_features(self) -> Dict[str, Any]:
        # 표 기본 feature 추출
        statement_title_keywords = {
            "balance_sheet": "재무상태표" in self.text_norm,
            "income_statement": "손익계산서" in self.text_norm,
            "comprehensive_income": "포괄손익계산서" in self.text_norm,
            "changes_in_equity": "자본변동표" in self.text_norm,
            "cash_flow": "현금흐름표" in self.text_norm,
        }

        # 타입 A 헤더 단서
        has_subject_col = ("과목" in self.col_text_norm) or ("과목" in self.text_norm)
        has_note_col = ("주석" in self.col_text_norm) or ("주석" in self.text_norm)
        has_current_col = (
            ("당기" in self.text_norm)
            or ("당기" in self.col_text_norm)
            or (("당" in self.col_text_norm) and ("기" in self.col_text_norm))
        )
        has_prior_col = (
            ("전기" in self.text_norm)
            or ("전기" in self.col_text_norm)
            or (("전" in self.col_text_norm) and ("기" in self.col_text_norm))
        )

        # 자본변동표 컬럼 단서
        has_equity_columns = all(
            kw in self.col_text_norm
            for kw in ["자본금", "주식발행초과금", "이익잉여금", "기타자본항목", "총계"]
        )

        # footer 문구
        has_footer_phrase = "별첨주석은본재무제표의일부입니다" in self.text_norm

        # 요약표 / 주석성 표 단서
        has_summary_phrase = (
            "요약재무상태표" in self.text_norm
            or "요약손익계산서" in self.text_norm
            or "요약포괄손익계산서" in self.text_norm
        )

        has_star_note_like = "(*)" in self.text
        looks_like_small_text_table = (
            self.n_rows <= 3 and self.n_cols <= 3 and self.numeric_count <= 2
        )

                # ---------------------------------------------------------------------
        # unknown_table 축소용 추가 feature
        # ---------------------------------------------------------------------

        # 롤포워드 표 단서
        rollforward_keywords = [
            "기초", "기말", "취득", "처분", "상각", "환입", "증가", "감소",
            "당기근무원가", "이자원가", "재측정요소", "지급", "설정", "사용"
        ]
        rollforward_hit_count = sum(
            1 for kw in rollforward_keywords if kw in self.text
        )

        # 금융상품 표 단서
        financial_instrument_keywords = [
            "금융자산", "금융부채", "상각후원가", "공정가치", "당기손익",
            "기타포괄손익", "채무상품", "지분상품", "파생상품", "매출채권"
        ]
        financial_instrument_hit_count = sum(
            1 for kw in financial_instrument_keywords if kw in self.text
        )

        # 관계회사 / 종속기업 표 단서
        counterparty_keywords = [
            "종속기업", "관계기업", "공동기업", "지분율", "소재지", "법인명",
            "회사명", "국가", "지역", "소유지분", "의결권"
        ]
        counterparty_hit_count = sum(
            1 for kw in counterparty_keywords if kw in self.text or kw in self.col_text
        )

        # 감사 관련 표 단서
        audit_related_keywords = [
            "감사참여자", "전반감사계획", "감사위원회", "외부감사", "감사시간",
            "업무수행이사", "품질관리검토자", "감사보수", "핵심감사사항"
        ]
        audit_related_hit_count = sum(
            1 for kw in audit_related_keywords if kw in self.text
        )

        # 전체 셀 수
        total_cells = max(self.n_rows * self.n_cols, 1)

        # 숫자 비중
        numeric_ratio = self.numeric_count / total_cells

        # 컬럼 단서
        has_company_like_cols = any(
            kw in self.col_text
            for kw in ["회사명", "법인명", "종속기업", "관계기업", "소재지", "지분율"]
        )

        has_financial_instrument_cols = any(
            kw in self.col_text
            for kw in ["금융자산", "금융부채", "상각후원가", "공정가치"]
        )

        has_rollforward_cols = any(
            kw in self.col_text
            for kw in ["기초", "기말", "취득", "처분", "상각", "증가", "감소"]
        )

        # 금융위험 / 만기분석 / 민감도 표 단서
        financial_risk_keywords = [
            "환율 상승시", "환율 하락시", "usd", "eur", "jpy", "외환차이",
            "이자비용", "만기", "3개월 이내", "~6개월", "~1년", "1~5년", "5년 초과",
            "금융부채", "기타금융부채", "상각후원가 측정 금융부채",
            "기타포괄손익-공정가치금융자산", "당기손익-공정가치금융자산"
        ]
        financial_risk_hit_count = sum(
            1 for kw in financial_risk_keywords
            if kw.lower() in self.text.lower() or kw.lower() in self.col_text.lower()
        )

        # 배당 / 자기주식 관련 표 단서
        dividend_keywords = [
            "배당받을 주식", "배당률", "배당금액", "보통주", "우선주",
            "주식수", "취득가액", "자기주식"
        ]
        dividend_hit_count = sum(
            1 for kw in dividend_keywords
            if kw in self.text or kw in self.col_text
        )

        # 급여채무 / 보험수리 가정 표 단서
        pension_keywords = [
            "당기근무원가", "순이자원가", "확정급여", "할인율",
            "미래임금상승률", "인플레이션", "보험수리적", "급여채무"
        ]
        pension_hit_count = sum(
            1 for kw in pension_keywords
            if kw in self.text or kw in self.col_text
        )

        # 감사 메타표 단서
        audit_meta_keywords = [
            "감사대상 사업연도", "회사명", "감사인", "외부감사", "감사보고서일"
        ]
        audit_meta_hit_count = sum(
            1 for kw in audit_meta_keywords
            if kw in self.text or kw in self.col_text
        )

        # 롤포워드 구조 조합 단서
        has_rollforward_start_end = ("기초" in self.text or "기초" in self.col_text) and (
            "기말" in self.text or "기말" in self.col_text
        )
        rollforward_movement_hit_count = sum(
            1 for kw in ["취득", "처분", "상각", "증가", "감소", "환입", "사용", "설정"]
            if kw in self.text or kw in self.col_text
        )

        # 만기 분석 컬럼 단서
        has_maturity_bucket_cols = any(
            kw in self.col_text
            for kw in ["3개월 이내", "~6개월", "~1년", "1~5년", "5년 초과"]
        )

        # 환율 민감도 컬럼 단서
        has_fx_sensitivity_cols = any(
            kw in self.col_text
            for kw in ["환율 상승시", "환율 하락시", "USD", "EUR", "JPY"]
        )

        # 배당 표 컬럼 단서
        has_dividend_cols = any(
            kw in self.col_text
            for kw in ["배당받을 주식", "배당률", "배당금액", "보통주", "우선주", "주식수", "취득가액"]
        )

        # 급여채무 표 컬럼 단서
        has_pension_cols = any(
            kw in self.col_text
            for kw in ["당기근무원가", "순이자원가", "할인율", "미래임금상승률"]
        )

        # 재고평가 / 충당금 표 단서
        inventory_keywords = [
            "제품 및 상품", "반제품 및 재공품", "원재료 및 저장품",
            "평가전금액", "평가충당금", "장부금액", "미착품"
        ]
        inventory_hit_count = sum(
            1 for kw in inventory_keywords
            if kw in self.text or kw in self.col_text
        )

        # 비용 breakdown 표 단서
        expense_keywords = [
            "판매비와관리비", "급여", "퇴직급여", "지급수수료",
            "감가상각비", "무형자산상각비", "광고선전비",
            "판매촉진비", "운반비", "서비스비"
        ]
        expense_hit_count = sum(
            1 for kw in expense_keywords
            if kw in self.text or kw in self.col_text
        )

        # 차입금 / 회사채 관련 표 단서
        debt_keywords = [
            "단기차입금", "장기차입금", "유동성장기차입금",
            "담보부차입금", "회사채", "차입금", "이자율", "우리은행 외"
        ]
        debt_hit_count = sum(
            1 for kw in debt_keywords
            if kw in self.text or kw in self.col_text
        )

        # 환경 / 배출권 관련 표 단서
        environment_keywords = [
            "배출권", "무상할당 배출권", "배출량 추정치"
        ]
        environment_hit_count = sum(
            1 for kw in environment_keywords
            if kw in self.text or kw in self.col_text
        )

                # 관계회사 숫자 요약 1행표 단서
        has_toshiba_samsung_storage = (
            "toshiba samsung storage technology japan" in self.text.lower()
        )

        # 첫 셀에 영문 회사명처럼 보이는지 여부
        first_cell = ""
        if self.n_rows > 0 and self.n_cols > 0:
            first_cell = self.clean_cell(self.df.iloc[0, 0])

        first_cell_is_company_like = bool(
            re.search(r"[A-Za-z]{3,}", first_cell)
        )

        # 1행 숫자 요약표 여부
        is_one_row_numeric_summary = (
            self.n_rows == 1
            and self.n_cols >= 5
            and self.numeric_count >= 4
            and first_cell_is_company_like
        )

        return {
            "table_index": self.table_index,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "numeric_count": self.numeric_count,
            "text": self.text,
            "text_norm": self.text_norm,
            "col_text": self.col_text,
            "col_text_norm": self.col_text_norm,
            "statement_title_keywords": statement_title_keywords,
            "has_footer_phrase": has_footer_phrase,
            "has_subject_col": has_subject_col,
            "has_note_col": has_note_col,
            "has_current_col": has_current_col,
            "has_prior_col": has_prior_col,
            "has_equity_columns": has_equity_columns,
            "has_summary_phrase": has_summary_phrase,
            "has_star_note_like": has_star_note_like,
            "looks_like_small_text_table": looks_like_small_text_table,
            "preview": self.preview,
            "rollforward_hit_count": rollforward_hit_count,
            "financial_instrument_hit_count": financial_instrument_hit_count,
            "counterparty_hit_count": counterparty_hit_count,
            "audit_related_hit_count": audit_related_hit_count,
            "numeric_ratio": numeric_ratio,
            "has_company_like_cols": has_company_like_cols,
            "has_financial_instrument_cols": has_financial_instrument_cols,
            "has_rollforward_cols": has_rollforward_cols,
            "financial_risk_hit_count": financial_risk_hit_count,
            "dividend_hit_count": dividend_hit_count,
            "pension_hit_count": pension_hit_count,
            "audit_meta_hit_count": audit_meta_hit_count,
            "has_rollforward_start_end": has_rollforward_start_end,
            "rollforward_movement_hit_count": rollforward_movement_hit_count,
            "has_maturity_bucket_cols": has_maturity_bucket_cols,
            "has_fx_sensitivity_cols": has_fx_sensitivity_cols,
            "has_dividend_cols": has_dividend_cols,
            "has_pension_cols": has_pension_cols,
            "inventory_hit_count": inventory_hit_count,
            "expense_hit_count": expense_hit_count,
            "debt_hit_count": debt_hit_count,
            "environment_hit_count": environment_hit_count,
            "has_toshiba_samsung_storage": has_toshiba_samsung_storage,
            "first_cell_is_company_like": first_cell_is_company_like,
            "is_one_row_numeric_summary": is_one_row_numeric_summary,
        }

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 자식 클래스에서 override
        return False

    def parse(self) -> Dict[str, Any]:
        # 공통 parse 조립
        payload = self.build_base_payload()
        payload.update(self.parse_payload())
        return payload


# =============================================================================
# 3. 중간 부모 클래스들
# =============================================================================

class StatementRelatedTableParser(BaseTableParser):
    # 재무제표 관련 표의 공통 부모 클래스

    @staticmethod
    def infer_statement_type_from_text(text_norm: str) -> Optional[str]:
        # 텍스트 기반 재무제표 종류 추론
        if "포괄손익계산서" in text_norm:
            return "comprehensive_income"
        if "재무상태표" in text_norm:
            return "balance_sheet"
        if "자본변동표" in text_norm:
            return "changes_in_equity"
        if "현금흐름표" in text_norm:
            return "cash_flow"
        if "손익계산서" in text_norm:
            return "income_statement"
        return None

    def infer_statement_type(self) -> Optional[str]:
        # 인스턴스 텍스트 기준 재무제표 종류 추론
        return self.infer_statement_type_from_text(self.text_norm)


class StatementBodyTableParser(StatementRelatedTableParser):
    # 재무제표 본문계열의 공통 부모 클래스

    def normalize_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        # 컬럼명 / 셀 공통 정리
        out = df.copy()
        out.columns = [self.clean_cell(c) for c in out.columns]
        out = out.map(lambda x: pd.NA if self.clean_cell(x) == "" else self.clean_cell(x))
        return out

    def normalize(self) -> pd.DataFrame:
        # 자식 클래스에서 필요 시 override
        return self.normalize_cells(self.df)

    def classify_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # 자식 클래스에서 override
        return df

    def parse_payload(self) -> Dict[str, Any]:
        # 본문표 공통 parse 흐름
        normalized = self.normalize()
        classified = self.classify_rows(normalized)
        return {
            "normalized": classified,
        }


class TextLikeTableParser(BaseTableParser):
    # 텍스트 중심 표의 공통 부모 클래스

    def parse_payload(self) -> Dict[str, Any]:
        # 텍스트성 표 기본 payload
        return {
            "text": self.text,
        }

class MatrixLikeTableParser(BaseTableParser):
    # 숫자 / 비교 매트릭스형 표의 공통 부모 클래스

    def normalize_matrix(self) -> pd.DataFrame:
        # 컬럼명 / 셀 최소 정리
        out = self.df.copy()
        out.columns = [self.clean_cell(c) for c in out.columns]
        out = out.map(lambda x: pd.NA if self.clean_cell(x) == "" else self.clean_cell(x))
        return out

    def parse_payload(self) -> Dict[str, Any]:
        # 매트릭스형 표 기본 payload
        return {
            "normalized": self.normalize_matrix(),
            "numeric_ratio": self.extract_basic_features()["numeric_ratio"],
        }