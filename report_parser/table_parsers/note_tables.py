from typing import Any, Dict
from .base import MatrixLikeTableParser

class RollForwardTableParser(MatrixLikeTableParser):
    # 기초 -> 변동 -> 기말 구조의 롤포워드 표 parser
    table_type = "rollforward_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 단순 키워드 2개 hit가 아니라,
        # 기초/기말 구조 또는 변동항목 다중 존재를 요구
        return (
            (
                features["has_rollforward_start_end"]
                or features["rollforward_movement_hit_count"] >= 2
            )
            and features["n_rows"] >= 3
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_subject_col"]
            and not features["has_equity_columns"]
            and features["dividend_hit_count"] == 0
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["rollforward_hit_count"],
        })
        return payload


class FinancialInstrumentTableParser(MatrixLikeTableParser):
    table_type = "financial_instrument_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            (
                features["financial_instrument_hit_count"] >= 1
                or features["financial_risk_hit_count"] >= 2
                or features["has_financial_instrument_cols"]
                or features["has_maturity_bucket_cols"]
                or features["has_fx_sensitivity_cols"]
                or features["debt_hit_count"] >= 1
            )
            and features["n_rows"] >= 1
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_equity_columns"]
            and features["dividend_hit_count"] == 0
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["financial_instrument_hit_count"],
            "risk_hits": self.extract_basic_features()["financial_risk_hit_count"],
        })
        return payload


class CounterpartyTableParser(MatrixLikeTableParser):
    # 종속기업 / 관계기업 / 공동기업 / 지분율 관련 표 parser
    table_type = "counterparty_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            (
                features["counterparty_hit_count"] >= 1
                or features["has_company_like_cols"]
                or ("일본" in features["text"] or "중국" in features["text"] or "미국" in features["text"])
                or features["has_toshiba_samsung_storage"]
                or features["is_one_row_numeric_summary"]
            )
            and features["n_cols"] >= 3
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_equity_columns"]
            and features["debt_hit_count"] == 0
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["counterparty_hit_count"],
        })
        return payload

class DividendTableParser(MatrixLikeTableParser):
    # 배당 / 자기주식 관련 표 parser
    table_type = "dividend_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            (
                features["dividend_hit_count"] >= 2
                or features["has_dividend_cols"]
            )
            and features["n_rows"] >= 2
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["dividend_hit_count"],
        })
        return payload

class PensionTableParser(MatrixLikeTableParser):
    # 확정급여 / 보험수리 가정 관련 표 parser
    table_type = "pension_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            (
                features["pension_hit_count"] >= 1
                or features["has_pension_cols"]
            )
            and features["n_rows"] >= 2
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_equity_columns"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["pension_hit_count"],
        })
        return payload

class InventoryValuationTableParser(MatrixLikeTableParser):
    # 재고자산 평가 / 충당금 관련 표 parser
    table_type = "inventory_valuation_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            features["inventory_hit_count"] >= 3
            and features["n_rows"] >= 3
            and features["n_cols"] >= 3
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["inventory_hit_count"],
        })
        return payload


class ExpenseBreakdownTableParser(MatrixLikeTableParser):
    # 판매비와관리비 등 비용 breakdown 표 parser
    table_type = "expense_breakdown_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            features["expense_hit_count"] >= 3
            and features["n_rows"] >= 3
            and 2 <= features["n_cols"] <= 4
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_equity_columns"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["expense_hit_count"],
        })
        return payload

class EnvironmentalTableParser(MatrixLikeTableParser):
    table_type = "environmental_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            features["environment_hit_count"] >= 1
            and features["n_rows"] >= 2
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
        )