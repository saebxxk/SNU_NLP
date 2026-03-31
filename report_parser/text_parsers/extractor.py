from __future__ import annotations

# 필요한 라이브러리 import
from dataclasses import dataclass, field
from typing import Dict, List
import re

# HTML 파싱
from bs4 import BeautifulSoup


@dataclass
class TextLine:
    # 텍스트 한 줄에 대한 구조 정보
    line_index: int
    text: str
    start: int
    end: int
    normalized: str
    source_line_indices: List[int] = field(default_factory=list)


@dataclass
class TextExtractionResult:
    # 텍스트 추출 결과 전체를 담는 데이터 클래스
    full_text: str
    text_only: str
    full_lines: List[TextLine]
    text_only_lines: List[TextLine]
    metadata: Dict[str, object] = field(default_factory=dict)


class AuditReportTextExtractor:
    # 감사보고서 HTML에서 텍스트를 구조적으로 추출하는 클래스

    @staticmethod
    def clean_text(text: str) -> str:
        # 공백 / 줄바꿈 정규화
        text = text.replace("\xa0", " ")
        text = text.replace("\u3000", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def normalize_line_for_heading_match(text: str) -> str:
        # 헤더 탐지용 정규화
        text = str(text)
        text = text.replace("\xa0", " ")
        text = text.replace("\u3000", " ")
        text = text.strip().lower()
        text = re.sub(r"[:;,·\-–—()\[\]{}]", "", text)
        text = re.sub(r"\s+", "", text)
        return text

    @classmethod
    def extract_full_text(cls, soup: BeautifulSoup) -> str:
        # 표 포함 전체 텍스트 추출
        text = soup.get_text(separator="\n", strip=True)
        return cls.clean_text(text)

    @classmethod
    def extract_text_without_tables(cls, soup: BeautifulSoup) -> str:
        # 표 제거 텍스트 추출
        soup_copy = BeautifulSoup(str(soup), "html.parser")
        for table in soup_copy.find_all("table"):
            # 표는 텍스트 문단 구조를 심하게 오염시키므로 제거
            table.decompose()
        text = soup_copy.get_text(separator="\n", strip=True)
        return cls.clean_text(text)

    @classmethod
    def split_lines(cls, text: str) -> List[TextLine]:
        # 텍스트를 줄 단위 구조로 변환
        lines: List[TextLine] = []
        cursor = 0
        line_index = 0

        for raw_line in text.split("\n"):
            raw_len = len(raw_line)
            stripped = raw_line.strip()

            if stripped:
                leading_ws = len(raw_line) - len(raw_line.lstrip())
                start = cursor + leading_ws
                end = start + len(stripped)
                lines.append(
                    TextLine(
                        line_index=line_index,
                        text=stripped,
                        start=start,
                        end=end,
                        normalized=cls.normalize_line_for_heading_match(stripped),
                        source_line_indices=[line_index],
                    )
                )
                line_index += 1

            cursor += raw_len + 1

        return lines

    @staticmethod
    def _looks_like_orphan_colon(line: TextLine) -> bool:
        # ':' 만 떨어진 줄인지 판별
        return line.text in {":", "："}

    @classmethod
    def merge_broken_colon_lines(cls, lines: List[TextLine]) -> List[TextLine]:
        # 다음 줄의 ':' 를 이전 줄 제목에 붙인다.
        merged: List[TextLine] = []
        i = 0

        while i < len(lines):
            current = lines[i]
            if i + 1 < len(lines) and cls._looks_like_orphan_colon(lines[i + 1]):
                next_line = lines[i + 1]
                new_text = f"{current.text} :"
                merged.append(
                    TextLine(
                        line_index=len(merged),
                        text=new_text,
                        start=current.start,
                        end=next_line.end,
                        normalized=cls.normalize_line_for_heading_match(new_text),
                        source_line_indices=current.source_line_indices + next_line.source_line_indices,
                    )
                )
                i += 2
                continue

            merged.append(
                TextLine(
                    line_index=len(merged),
                    text=current.text,
                    start=current.start,
                    end=current.end,
                    normalized=current.normalized,
                    source_line_indices=list(current.source_line_indices),
                )
            )
            i += 1

        return merged

    @classmethod
    def build_lines(cls, text: str, merge_colon_lines: bool = True) -> List[TextLine]:
        # 줄 구조 생성 + 후처리
        lines = cls.split_lines(text)
        if merge_colon_lines:
            lines = cls.merge_broken_colon_lines(lines)
        return lines

    @classmethod
    def from_html_text(cls, html_text: str) -> TextExtractionResult:
        # HTML 문자열에서 full_text / text_only / line 구조를 모두 생성
        soup = BeautifulSoup(html_text, "html.parser")
        full_text = cls.extract_full_text(soup)
        text_only = cls.extract_text_without_tables(soup)
        full_lines = cls.build_lines(full_text)
        text_only_lines = cls.build_lines(text_only)
        return TextExtractionResult(
            full_text=full_text,
            text_only=text_only,
            full_lines=full_lines,
            text_only_lines=text_only_lines,
            metadata={
                "full_text_length": len(full_text),
                "text_only_length": len(text_only),
                "full_line_count": len(full_lines),
                "text_only_line_count": len(text_only_lines),
            },
        )

    @classmethod
    def from_soup(cls, soup: BeautifulSoup) -> TextExtractionResult:
        # BeautifulSoup 객체에서 텍스트 구조 생성
        return cls.from_html_text(str(soup))
