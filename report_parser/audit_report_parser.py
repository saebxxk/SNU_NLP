from __future__ import annotations

from dataclasses import asdict
from report_parser.text_parsers.extractor import AuditReportTextExtractor
from report_parser.text_parsers.sections import TextSectionBuilder, TextSectionValidator
from pathlib import Path
from io import StringIO
import re
from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd
from bs4 import BeautifulSoup
from report_parser.models import StatementPackage
from report_parser.table_parsers import (
    BaseTableParser,
    StatementFooterTableParser,
    StatementTitleTableParser,
    ChangesInEquityParser,
    StatementBodyTypeAParser,
    FinancialInstrumentTableParser,
    DividendTableParser,
    PensionTableParser,
    InventoryValuationTableParser,
    ExpenseBreakdownTableParser,
    RollForwardTableParser,
    CounterpartyTableParser,
    AuditRelatedTableParser,
    EnvironmentalTableParser,
    SimpleMatrixTableParser,
    NoteTableParser,
    MetadataTableParser,
    UnknownTableParser,
)


class AuditReportParser:
    # 감사보고서 전체 parser

    # 표 parser 등록 순서
    TABLE_PARSER_CLASSES: List[Type[BaseTableParser]] = [
        StatementFooterTableParser,
        StatementTitleTableParser,
        ChangesInEquityParser,
        StatementBodyTypeAParser,
        FinancialInstrumentTableParser,
        DividendTableParser,
        PensionTableParser,
        InventoryValuationTableParser,
        ExpenseBreakdownTableParser,
        RollForwardTableParser,
        CounterpartyTableParser,
        AuditRelatedTableParser,
        EnvironmentalTableParser,
        SimpleMatrixTableParser,
        NoteTableParser,
        MetadataTableParser,
        UnknownTableParser,
    ]

    def __init__(self, html_path: str):
        # 파일 경로 저장
        self.html_path = str(html_path)

        # 원시 데이터 저장 변수 초기화
        self.html_text: Optional[str] = None
        self.encoding: Optional[str] = None
        self.soup: Optional[BeautifulSoup] = None
        self.full_text: Optional[str] = None
        self.tables: List[pd.DataFrame] = []
        self.text_only: Optional[str] = None

        # 파싱 결과 저장 변수
        self.table_objects: List[BaseTableParser] = []
        self.table_summary: Optional[pd.DataFrame] = None
        self.packages: List[StatementPackage] = []

    # -------------------------------------------------------------------------
    # 파일 / 문서 처리
    # -------------------------------------------------------------------------

    def read_html_file(
        self,
        encodings: Tuple[str, ...] = ("utf-8", "cp949", "euc-kr"),
    ) -> Tuple[str, str]:
        # HTML 파일을 여러 인코딩 후보로 읽는 함수
        path = Path(self.html_path)

        for enc in encodings:
            try:
                text = path.read_text(encoding=enc)
                return text, enc
            except Exception:
                continue

        raise ValueError(f"파일을 읽을 수 없습니다: {path}")

    def extract_text_without_tables(self) -> str:
        # 표를 제거한 일반 텍스트만 추출
        if self.html_text is None:
            raise ValueError("html_text가 없습니다.")

        soup = BeautifulSoup(self.html_text, "html.parser")

        for table in soup.find_all("table"):
            table.decompose()

        return soup.get_text(separator="\n", strip=True)

    def load(self) -> None:
        # HTML 읽기 및 기본 파싱
        html_text, encoding = self.read_html_file()
        self.html_text = html_text
        self.encoding = encoding

        # BeautifulSoup 파싱
        self.soup = BeautifulSoup(self.html_text, "html.parser")

        # 전체 텍스트 추출
        self.full_text = self.soup.get_text(separator="\n", strip=True)

        # 표 제거 일반 텍스트 추출
        self.text_only = self.extract_text_without_tables()

        # 표 추출
        self.tables = pd.read_html(StringIO(self.html_text))

    # -------------------------------------------------------------------------
    # 문서 메타 추출
    # -------------------------------------------------------------------------

    def extract_company_name(self) -> Optional[str]:
        # 회사명 추출
        if self.full_text is None:
            return None

        lines = [line.strip() for line in self.full_text.splitlines() if line.strip()]

        for line in lines[:50]:
            if "주식회사" in line:
                return line

        return None

    def extract_report_date(self) -> Optional[str]:
        # 감사보고서일 추출
        #
        # 아이디어:
        # - "독립된 감사인의 감사보고서" 이후부터
        # - "(첨부) 재무제표 / 주석 시작" 직전까지를 감사보고서 본문으로 보고
        # - 그 구간의 마지막 날짜를 감사보고서일로 사용한다.
        #
        # 이유:
        # - 기존 1000자 window 방식은 초기 연도(2014~2018)에서
        #   감사보고서일보다 앞의 재무제표 기준일을 잘못 잡을 수 있다.
        # - 2018년처럼 감사보고서일이 표 영역 안에 있는 경우가 있어
        #   text_only보다 full_text를 우선 사용하는 것이 안전하다.

        # full_text 우선 사용
        source_text = self.full_text if self.full_text is not None else self.text_only

        if source_text is None:
            return None

        # 공백 정규화
        text = re.sub(r"\s+", " ", source_text)

        # 감사보고서 시작 anchor
        anchor = "독립된 감사인의 감사보고서"
        start = text.find(anchor)

        if start == -1:
            start = 0

        # 감사보고서 본문 종료 후보
        # - 재무제표 첨부 시작
        # - 주석 시작
        end_candidates = []

        end_markers = [
            "(첨부)재 무 제 표",
            "(첨부) 재 무 제 표",
            "재 무 제 표 주석",
            "재무제표 주석",
            "주석 1. 일반적 사항",
            "1. 일반적 사항",
        ]

        for marker in end_markers:
            pos = text.find(marker, start)
            if pos != -1:
                end_candidates.append(pos)

        # 종료 마커가 있으면 가장 빠른 지점까지,
        # 없으면 anchor 이후 8000자까지를 fallback window로 사용
        if end_candidates:
            end = min(end_candidates)
        else:
            end = min(len(text), start + 8000)

        window = text[start:end]

        # 날짜 패턴
        # - "2017년 2월 27 일"처럼 '일' 앞에 공백이 있는 경우도 허용
        date_pattern = r"\d{4}년\s*\d{1,2}월\s*\d{1,2}\s*일"

        # 1차: 감사보고서 본문 구간의 마지막 날짜
        dates = re.findall(date_pattern, window)
        if dates:
            # "27 일" -> "27일"처럼 마무리 정규화
            return re.sub(r"\s+일$", "일", dates[-1])

        # 2차: text_only가 있으면 동일 로직을 한 번 더 시도
        if self.text_only is not None:
            text_only_norm = re.sub(r"\s+", " ", self.text_only)
            text_only_start = text_only_norm.find(anchor)

            if text_only_start != -1:
                fallback_window = text_only_norm[text_only_start:text_only_start + 8000]
                dates = re.findall(date_pattern, fallback_window)
                if dates:
                    return re.sub(r"\s+일$", "일", dates[-1])

        # 3차: anchor 이후 8000자 내 마지막 날짜
        fallback_window = text[start:start + 8000]
        dates = re.findall(date_pattern, fallback_window)
        if dates:
            return re.sub(r"\s+일$", "일", dates[-1])

        # 4차: 문서 전체 마지막 날짜
        dates = re.findall(date_pattern, text)
        if dates:
            return re.sub(r"\s+일$", "일", dates[-1])

        return None

    def find_section_positions(self, keywords: List[str]) -> Dict[str, int]:
        # 주요 섹션 키워드 위치 찾기
        if self.full_text is None:
            return {}

        positions = {}
        for kw in keywords:
            positions[kw] = self.full_text.find(kw)

        return positions

    def parse_text_sections(self) -> Dict[str, Any]:
        # 텍스트 구조 파싱
        if self.soup is None:
            raise ValueError("load()가 먼저 실행되어야 합니다.")

        # 구조화된 텍스트 추출
        extraction_result = AuditReportTextExtractor.from_soup(self.soup)

        # 헤더 / 섹션 / note block 구성
        structure = TextSectionBuilder.build_text_structure(extraction_result)

        # 최소 유효성 검증
        validation = TextSectionValidator.validate(structure)

        return {
            "line_count": structure["line_count"],
            "main_sections": [asdict(x) for x in structure["main_sections"]],
            "note_blocks": [asdict(x) for x in structure["note_blocks"]],
            "headings": {
                "main_headings": [asdict(x) for x in structure["headings"]["main_headings"]],
                "note_headings": [asdict(x) for x in structure["headings"]["note_headings"]],
                "top_level_note_headings": [
                    asdict(x) for x in structure["headings"]["top_level_note_headings"]
                ],
            },
            "validation": validation,
        }

    # -------------------------------------------------------------------------
    # 표 객체 생성 / 요약
    # -------------------------------------------------------------------------

    def choose_table_parser_class(
        self,
        df: pd.DataFrame,
        table_index: int,
    ) -> Type[BaseTableParser]:
        # 표 feature를 보고 적절한 parser 클래스를 선택
        temp_parser = BaseTableParser(df, table_index)
        features = temp_parser.extract_basic_features()

        for parser_cls in self.TABLE_PARSER_CLASSES:
            if parser_cls.match(features):
                return parser_cls

        return UnknownTableParser

    def build_table_objects(self) -> None:
        # 모든 표에 대해 parser 객체 생성
        self.table_objects = []

        for i, df in enumerate(self.tables):
            parser_cls = self.choose_table_parser_class(df, i)
            parser_obj = parser_cls(df, i)
            self.table_objects.append(parser_obj)

    def build_table_summary(self) -> pd.DataFrame:
        # 전체 표 요약표 생성
        rows = []

        for parser_obj in self.table_objects:
            features = parser_obj.extract_basic_features()

            # 제목표인 경우 statement_type 추론
            statement_hits = []
            for k, v in features["statement_title_keywords"].items():
                if v:
                    statement_hits.append(k)

            rows.append({
                "table_index": parser_obj.table_index,
                "n_rows": parser_obj.n_rows,
                "n_cols": parser_obj.n_cols,
                "numeric_count": parser_obj.numeric_count,
                "table_type": parser_obj.table_type,
                "statement_hits": statement_hits,
                "preview": parser_obj.preview,
            })

        self.table_summary = pd.DataFrame(rows)
        return self.table_summary

    # -------------------------------------------------------------------------
    # statement package 구성
    # -------------------------------------------------------------------------

    def build_statement_packages(self) -> List[StatementPackage]:
        # 제목표 / 본문표 / footer를 묶어서 재무제표 package 생성
        packages: List[StatementPackage] = []

        # table_index -> parser_obj 맵
        table_map = {obj.table_index: obj for obj in self.table_objects}

        for i, parser_obj in table_map.items():
            # 제목표가 아니면 건너뜀
            if parser_obj.table_type != "statement_title_table":
                continue

            # 제목표에서 statement_type 추론
            assert isinstance(parser_obj, StatementTitleTableParser)
            statement_type = parser_obj.infer_statement_type()

            if statement_type is None:
                continue

            # 다음 표가 있어야 함
            if (i + 1) not in table_map:
                continue

            next_parser = table_map[i + 1]

            # 기대하는 본문 타입 결정
            expected_body_type = (
                "statement_body_table_changes_in_equity"
                if statement_type == "changes_in_equity"
                else "statement_body_table_type_a"
            )

            if next_parser.table_type != expected_body_type:
                continue

            # footer 확인
            footer_index = None
            if (i + 2) in table_map:
                if table_map[i + 2].table_type == "statement_footer_table":
                    footer_index = i + 2

            packages.append(
                StatementPackage(
                    statement_type=statement_type,
                    title_index=i,
                    body_index=i + 1,
                    footer_index=footer_index,
                )
            )

        self.packages = packages
        return packages

    # -------------------------------------------------------------------------
    # package 정규화
    # -------------------------------------------------------------------------

    def parse_statement_packages(self) -> Dict[str, Any]:
        # 재무제표 package들을 정규화하여 반환
        normalized: Dict[str, Any] = {}

        table_map = {obj.table_index: obj for obj in self.table_objects}

        for pkg in self.packages:
            body_parser = table_map[pkg.body_index]

            if pkg.statement_type == "changes_in_equity":
                assert isinstance(body_parser, ChangesInEquityParser)
                parsed = body_parser.parse()
                normalized[pkg.statement_type] = parsed["blocks"]
            else:
                assert isinstance(body_parser, StatementBodyTypeAParser)
                parsed = body_parser.parse()
                normalized[pkg.statement_type] = parsed["normalized"]

        return normalized

    # -------------------------------------------------------------------------
    # 전체 파싱
    # -------------------------------------------------------------------------

    def parse(self) -> Dict[str, Any]:
        # 감사보고서 전체 파싱 실행
        self.load()
        self.build_table_objects()
        self.build_table_summary()
        self.build_statement_packages()

        # 문서 구조용 섹션 키워드
        section_keywords = [
            "독립된 감사인의 감사보고서",
            "감사의견",
            "감사의견근거",
            "핵심감사사항",
            "재무상태표",
            "손익계산서",
            "포괄손익계산서",
            "자본변동표",
            "현금흐름표",
            "주석",
        ]

        # 정규화된 재무제표들
        normalized_tables = self.parse_statement_packages()

        # 텍스트 구조 파싱
        text_result = self.parse_text_sections()

        # 개별 표 parse 결과 전부 저장
        classified_tables = []
        for obj in self.table_objects:
            classified_tables.append({
                "table_index": obj.table_index,
                "table_type": obj.table_type,
                "shape": obj.df.shape,
                "preview": obj.preview,
            })

        # 최종 결과 dict
        result = {
            "metadata": {
                "source_file": self.html_path,
                "encoding": self.encoding,
                "company_name": self.extract_company_name(),
                "report_date": self.extract_report_date(),
            },
            "document_structure": {
                "text_length": len(self.full_text) if self.full_text is not None else None,
                "text_only_length": len(self.text_only) if self.text_only is not None else None,
                "table_count": len(self.tables),
                "section_positions": self.find_section_positions(section_keywords),
            },
            "text": text_result,
            "sections": {
                # 지금 단계에서는 위치만 유지
                "section_positions": self.find_section_positions(section_keywords),
            },
            "tables": {
                "summary": self.table_summary,
                "classified": classified_tables,
                "packages": [pkg.__dict__ for pkg in self.packages],
                "normalized": normalized_tables,
                "raw_tables": self.tables,
            },
            "raw": {
                "html_text": self.html_text,
                "full_text": self.full_text,
                "text_only": self.text_only,
            },
        }

        return result