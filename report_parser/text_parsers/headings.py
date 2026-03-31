from __future__ import annotations

# 필요한 라이브러리 import
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

# 내부 모듈 import
from .extractor import TextLine


@dataclass
class HeadingMatch:
    # 탐지된 헤더 정보
    heading_type: str
    title: str
    line_index: int
    start: int
    end: int
    level: int
    normalized: str
    metadata: Dict[str, object] = field(default_factory=dict)


class HeadingDetector:
    # 감사보고서 텍스트 헤더 탐지기

    MAIN_HEADING_RULES: List[Tuple[str, List[str], int]] = [
        ("auditor_report_title", [r"독립된감사인의감사보고서"], 1),
        ("opinion", [r"^감사의견$"], 2),
        ("basis_for_opinion", [r"^감사의견근거$"], 2),
        ("key_audit_matters", [r"^핵심감사사항$"], 2),
        ("management_responsibility", [r"재무제표에대한경영진의책임", r"재무보고에대한경영진의책임"], 2),
        ("auditor_responsibility", [r"재무제표감사에대한감사인의책임", r"^감사인의책임$"], 2),
        ("other_matters", [r"^기타사항$", r"^강조사항$"], 2),
        ("attached_financial_statements", [r"첨부재무제표", r"첨부별도재무제표"], 1),
        ("notes_section", [r"^주석$", r"재무제표주석", r"별도재무제표주석"], 1),
    ]

    NOTE_HEADER_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*(?::)?\s*$")

    @classmethod
    def detect_main_headings(cls, lines: List[TextLine]) -> List[HeadingMatch]:
        # 메인 헤더 탐지
        matches: List[HeadingMatch] = []
        for line in lines:
            normalized = line.normalized
            for heading_type, patterns, level in cls.MAIN_HEADING_RULES:
                if any(re.search(pattern, normalized) for pattern in patterns):
                    matches.append(
                        HeadingMatch(
                            heading_type=heading_type,
                            title=line.text,
                            line_index=line.line_index,
                            start=line.start,
                            end=line.end,
                            level=level,
                            normalized=normalized,
                            metadata={},
                        )
                    )
                    break
        return matches

    @classmethod
    def parse_note_header(cls, line: TextLine) -> Optional[HeadingMatch]:
        # note 형태 제목인지 판별
        match = cls.NOTE_HEADER_PATTERN.match(line.text)
        if match is None:
            return None

        number = match.group(1).strip()
        title = match.group(2).strip()
        depth = number.count('.') + 1
        is_continued = '계속' in title.replace(' ', '')

        return HeadingMatch(
            heading_type='note_header',
            title=line.text,
            line_index=line.line_index,
            start=line.start,
            end=line.end,
            level=depth,
            normalized=line.normalized,
            metadata={
                'note_number': number,
                'note_title': title,
                'depth': depth,
                'is_continued': is_continued,
            },
        )

    @classmethod
    def detect_note_headings(cls, lines: List[TextLine], top_level_only: bool = False) -> List[HeadingMatch]:
        # note 헤더 탐지
        matches: List[HeadingMatch] = []
        for line in lines:
            parsed = cls.parse_note_header(line)
            if parsed is None:
                continue
            if top_level_only and int(parsed.metadata['depth']) != 1:
                continue
            matches.append(parsed)
        return matches

    @classmethod
    def detect_all_headings(cls, lines: List[TextLine]) -> Dict[str, List[HeadingMatch]]:
        # 메인 헤더 + note 헤더를 한 번에 탐지
        return {
            'main_headings': cls.detect_main_headings(lines),
            'note_headings': cls.detect_note_headings(lines),
            'top_level_note_headings': cls.detect_note_headings(lines, top_level_only=True),
        }
