from __future__ import annotations

# 필요한 라이브러리 import
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 내부 모듈 import
from .extractor import TextLine, TextExtractionResult
from .headings import HeadingDetector, HeadingMatch


@dataclass
class SectionBlock:
    # 일반 텍스트 섹션 블록
    section_id: str
    section_type: str
    title: str
    level: int
    start: int
    end: int
    text: str
    start_line_index: int
    end_line_index: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class NoteBlock:
    # 주석 번호 블록
    note_number: str
    note_title: str
    title_line: str
    level: int
    start: int
    end: int
    text: str
    start_line_index: int
    end_line_index: int
    metadata: Dict[str, object] = field(default_factory=dict)


class TextSectionBuilder:
    # 탐지된 헤더를 실제 section block으로 변환하는 클래스

    MAIN_SECTION_TYPES_FOR_BLOCK = {
        "auditor_report_title",
        "opinion",
        "basis_for_opinion",
        "key_audit_matters",
        "management_responsibility",
        "auditor_responsibility",
        "other_matters",
        "attached_financial_statements",
        "notes_section",
    }

    # 주석 파싱 종료를 유도하는 패턴
    # 주석 본문이 끝난 뒤 나오는 외부감사/내부회계 관련 섹션을 차단한다.
    NOTE_TERMINATION_PATTERNS = [
        "내부회계관리제도감사또는검토의견",
        "독립된감사인의내부회계관리제도감사보고서",
        "내부회계관리제도에대한감사의견",
        "내부회계관리제도감사의견근거",
        "회사의내부회계관리제도운영실태평가보고서",
        "내부회계관리제도감사의견근거",
        "외부감사실시내용",
    ]

    # note 제목 자체 blacklist
    # 여기 나오면 주석이 아니라 감사/내부회계 부속 보고서로 본다.
    DISALLOWED_NOTE_TITLE_PATTERNS = [
        "감사대상업무",
        "감사참여자구분별인원수및감사시간",
        "주요감사실시내용",
        "감사(감사위원회)와의커뮤니케이션",
        "회사의내부회계관리제도운영실태평가보고서",
        "내부회계관리제도운영실태평가보고서",
        "회사의내부회계관리제도운영실태보고서",
        "내부회계관리제도운영실태보고서",
    ]

    @staticmethod
    def lines_to_text(lines: List[TextLine]) -> str:
        # 여러 줄을 섹션 텍스트로 합친다.
        return "\n".join(line.text for line in lines).strip()

    @staticmethod
    def normalize_line_for_match(text: str) -> str:
        # 공백 제거 후 소문자화하여 패턴 탐지 안정화
        return "".join(text.split()).lower()

    @classmethod
    def build_main_sections(
        cls,
        lines: List[TextLine],
        heading_matches: List[HeadingMatch],
    ) -> List[SectionBlock]:
        # 메인 헤더들을 기준으로 섹션 블록 생성
        valid_matches = [m for m in heading_matches if m.heading_type in cls.MAIN_SECTION_TYPES_FOR_BLOCK]
        valid_matches = sorted(valid_matches, key=lambda x: x.line_index)

        blocks: List[SectionBlock] = []
        for i, match in enumerate(valid_matches):
            start_line_index = match.line_index
            end_line_index = valid_matches[i + 1].line_index - 1 if i + 1 < len(valid_matches) else len(lines) - 1

            if start_line_index < 0 or end_line_index < start_line_index:
                continue

            section_lines = lines[start_line_index:end_line_index + 1]
            if not section_lines:
                continue

            blocks.append(
                SectionBlock(
                    section_id=f"section_{match.heading_type}_{len(blocks)}",
                    section_type=match.heading_type,
                    title=match.title,
                    level=match.level,
                    start=section_lines[0].start,
                    end=section_lines[-1].end,
                    text=cls.lines_to_text(section_lines),
                    start_line_index=start_line_index,
                    end_line_index=end_line_index,
                    metadata={
                        "source_heading_line": match.line_index,
                        "source_heading_normalized": match.normalized,
                    },
                )
            )

        return blocks

    @classmethod
    def find_notes_section_start_line(cls, main_sections: List[SectionBlock]) -> Optional[int]:
        # 첫 번째 notes_section 시작 line index 반환
        for block in main_sections:
            if block.section_type == "notes_section":
                return block.start_line_index
        return None

    @classmethod
    def find_note_parsing_end_line(
        cls,
        lines: List[TextLine],
        note_start_line_index: int,
    ) -> int:
        # 주석 파싱 종료 line index 계산
        #
        # 기본값은 문서 끝까지이지만,
        # 내부회계/외부감사 관련 별도 보고서가 시작되면 그 직전까지로 자른다.
        for idx in range(note_start_line_index, len(lines)):
            normalized = cls.normalize_line_for_match(lines[idx].text)

            for pattern in cls.NOTE_TERMINATION_PATTERNS:
                if pattern in normalized:
                    return idx - 1

        return len(lines) - 1

    @classmethod
    def build_note_blocks(
        cls,
        lines: List[TextLine],
        note_heading_matches: List[HeadingMatch],
        note_start_line_index: Optional[int],
        note_end_line_index: Optional[int],
    ) -> List[NoteBlock]:
        # 주석 헤더를 기준으로 note block 생성
        if note_start_line_index is None:
            return []

        if note_end_line_index is None:
            note_end_line_index = len(lines) - 1

        # 최상위 note만 사용
        top_level_matches = [m for m in note_heading_matches if int(m.metadata.get("depth", 0)) == 1]

        # 반드시 notes_section 내부에 있는 note만 사용
        top_level_matches = [
            m for m in top_level_matches
            if note_start_line_index <= m.line_index <= note_end_line_index
        ]
        top_level_matches = sorted(top_level_matches, key=lambda x: x.line_index)

        raw_blocks: List[NoteBlock] = []

        for i, match in enumerate(top_level_matches):
            start_line_index = match.line_index

            # 현재 note 제목 정규화
            normalized_title = cls.normalize_line_for_match(match.title)

            # note 번호 해석
            note_number_raw = str(match.metadata.get("note_number", ""))
            current_note_number = int(note_number_raw) if note_number_raw.isdigit() else None

            # 1) blacklist 제목이면 즉시 note parsing 종료
            if any(pattern in normalized_title for pattern in cls.DISALLOWED_NOTE_TITLE_PATTERNS):
                break

            # 2) 정상적인 주석 번호가 충분히 진행된 뒤(예: 10번 이상),
            #    갑자기 1~5로 번호가 다시 시작하면 외부감사 부속 항목으로 보고 종료
            if raw_blocks and current_note_number is not None:
                previous_note_number_raw = raw_blocks[-1].note_number
                previous_note_number = (
                    int(previous_note_number_raw)
                    if str(previous_note_number_raw).isdigit()
                    else None
                )

                if previous_note_number is not None:
                    if previous_note_number >= 10 and current_note_number <= 5:
                        break

            # 다음 top-level note 직전까지, 단 notes 종료 line을 넘지 않도록 제한
            if i + 1 < len(top_level_matches):
                candidate_end_line_index = top_level_matches[i + 1].line_index - 1
            else:
                candidate_end_line_index = note_end_line_index

            end_line_index = min(candidate_end_line_index, note_end_line_index)

            if start_line_index < 0 or end_line_index < start_line_index:
                continue

            block_lines = lines[start_line_index:end_line_index + 1]
            if not block_lines:
                continue

            raw_blocks.append(
                NoteBlock(
                    note_number=str(match.metadata.get("note_number", "")),
                    note_title=str(match.metadata.get("note_title", "")),
                    title_line=match.title,
                    level=int(match.metadata.get("depth", 1)),
                    start=block_lines[0].start,
                    end=block_lines[-1].end,
                    text=cls.lines_to_text(block_lines),
                    start_line_index=start_line_index,
                    end_line_index=end_line_index,
                    metadata={
                        "is_continued": bool(match.metadata.get("is_continued", False)),
                        "source_heading_line": match.line_index,
                    },
                )
            )

        return cls.merge_continued_note_blocks(raw_blocks)

    @classmethod
    def merge_continued_note_blocks(cls, note_blocks: List[NoteBlock]) -> List[NoteBlock]:
        # '계속' 블록 병합
        if not note_blocks:
            return []

        merged: List[NoteBlock] = []

        for block in note_blocks:
            is_continued = bool(block.metadata.get("is_continued", False))

            if merged and is_continued and merged[-1].note_number == block.note_number:
                previous = merged[-1]
                previous.end = block.end
                previous.end_line_index = block.end_line_index
                previous.text = f"{previous.text}\n{block.text}".strip()

                merged_count = int(previous.metadata.get("merged_continued_count", 0))
                previous.metadata["merged_continued_count"] = merged_count + 1
                continue

            merged.append(block)

        return merged

    @classmethod
    def build_text_structure(cls, extraction_result: TextExtractionResult) -> Dict[str, object]:
        # 전체 텍스트 구조 결과 생성
        lines = extraction_result.text_only_lines
        detected = HeadingDetector.detect_all_headings(lines)

        # 메인 섹션 먼저 생성
        main_sections = cls.build_main_sections(lines, detected["main_headings"])

        # notes_section 범위를 기준으로 note parsing 범위 제한
        note_start_line_index = cls.find_notes_section_start_line(main_sections)
        note_end_line_index = None

        if note_start_line_index is not None:
            note_end_line_index = cls.find_note_parsing_end_line(lines, note_start_line_index)

        note_blocks = cls.build_note_blocks(
            lines=lines,
            note_heading_matches=detected["note_headings"],
            note_start_line_index=note_start_line_index,
            note_end_line_index=note_end_line_index,
        )

        return {
            "main_sections": main_sections,
            "note_blocks": note_blocks,
            "headings": detected,
            "line_count": len(lines),
        }


class TextSectionValidator:
    # 텍스트 구조 결과의 최소 유효성 점검기

    @staticmethod
    def validate(structure: Dict[str, object]) -> Dict[str, object]:
        # 텍스트 구조 검증
        main_sections: List[SectionBlock] = structure.get("main_sections", [])  # type: ignore[assignment]
        note_blocks: List[NoteBlock] = structure.get("note_blocks", [])  # type: ignore[assignment]

        section_types = [block.section_type for block in main_sections]

        return {
            "main_section_count": len(main_sections),
            "note_block_count": len(note_blocks),
            "has_auditor_report_title": "auditor_report_title" in section_types,
            "has_opinion": "opinion" in section_types,
            "has_basis_for_opinion": "basis_for_opinion" in section_types,
            "has_key_audit_matters": "key_audit_matters" in section_types,
            "has_notes_section": "notes_section" in section_types,
            "all_main_section_offsets_valid": all(block.start < block.end for block in main_sections),
            "all_note_offsets_valid": all(block.start < block.end for block in note_blocks),
        }