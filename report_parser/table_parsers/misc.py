from typing import Any, Dict
from .base import TextLikeTableParser, MatrixLikeTableParser

class MetadataTableParser(TextLikeTableParser):
    # 작은 메타표 parser
    table_type = "small_metadata_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 정말 작은 텍스트성 메타 표만 허용
        return (
            features["n_rows"] <= 4
            and features["n_cols"] <= 3
            and features["numeric_count"] <= 6
            and features["numeric_ratio"] <= 0.35
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_summary_phrase"]
            and not features["has_subject_col"]
            and not features["has_equity_columns"]
            and features["rollforward_hit_count"] == 0
            and features["financial_instrument_hit_count"] == 0
            and features["counterparty_hit_count"] == 0
            and features["audit_related_hit_count"] == 0
        )

class NoteTableParser(TextLikeTableParser):
    # 주석성 / 요약성 표 parser
    table_type = "note_like_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 요약표, 설명표, 작은 텍스트성 표를 note_like로 분류
        return (
            features["has_summary_phrase"]
            or features["has_star_note_like"]
            or features["looks_like_small_text_table"]
        )

class AuditRelatedTableParser(TextLikeTableParser):
    # 외부감사 관련 관리성 표 parser
    table_type = "audit_related_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            (
                features["audit_related_hit_count"] >= 1
                or features["audit_meta_hit_count"] >= 1
            )
            and features["n_rows"] >= 2
            and features["n_cols"] >= 2
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
        )

    def parse_payload(self) -> Dict[str, Any]:
        payload = super().parse_payload()
        payload.update({
            "keyword_hits": self.extract_basic_features()["audit_related_hit_count"],
            "meta_hits": self.extract_basic_features()["audit_meta_hit_count"],
        })
        return payload

class SimpleMatrixTableParser(MatrixLikeTableParser):
    table_type = "simple_matrix_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        return (
            features["n_rows"] >= 1
            and 2 <= features["n_cols"] <= 6
            and features["numeric_ratio"] >= 0.20
            and not any(features["statement_title_keywords"].values())
            and not features["has_footer_phrase"]
            and not features["has_subject_col"]
            and not features["has_equity_columns"]
            and features["rollforward_movement_hit_count"] < 2
            and features["financial_instrument_hit_count"] < 2
            and features["counterparty_hit_count"] < 2
            and features["audit_related_hit_count"] == 0
            and features["debt_hit_count"] == 0
        )

class UnknownTableParser(TextLikeTableParser):
    # 분류되지 않은 표 parser
    table_type = "unknown_table"

    @classmethod
    def match(cls, features: Dict[str, Any]) -> bool:
        # 마지막 fallback
        return True