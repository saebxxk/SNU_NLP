from dataclasses import dataclass
from typing import Optional
# =============================================================================
# 1. 공통 데이터 클래스
# =============================================================================

@dataclass
class StatementPackage:
    # 재무제표 패키지 정보 저장용 데이터 클래스
    statement_type: str
    title_index: int
    body_index: int
    footer_index: Optional[int] = None