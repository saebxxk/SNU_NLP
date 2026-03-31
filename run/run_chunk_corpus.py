# 필요한 라이브러리 import
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_dir(path: Path) -> None:
    # 디렉토리가 없으면 생성
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    # JSONL 파일을 전부 읽어서 list[dict]로 반환
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # 빈 줄은 무시
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 실패: {path} / line={line_number} / error={e}") from e

    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    # JSONL 파일로 저장
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    # 과도한 공백 정리
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    # 빈 줄 기준으로 문단 분리
    text = normalize_whitespace(text)
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def split_lines(text: str) -> List[str]:
    # 줄 단위 분리
    return [line.strip() for line in text.split("\n") if line.strip()]


def looks_like_top_level_note_title(line: str) -> bool:
    # 다음 top-level note 제목처럼 보이는지 판별
    #
    # 예:
    # 18. 충당부채:
    # 12. 차입금
    line = line.strip()
    return bool(re.match(r"^\d+\.\s*[^.\n]+:?$", line))


def sanitize_note_block_text(text: str, expected_note_number: Optional[str]) -> str:
    # note_block 안에 다음 note 제목이 섞인 경우 잘라낸다.
    #
    # 현재 block의 첫 줄은 보통 "17. 순확정급여..." 같은 title이고,
    # 그 뒤에 "18. 충당부채:"가 다시 나오면 그 지점에서 끊는다.
    lines = split_lines(text)

    if not lines:
        return text.strip()

    expected_prefix = None
    if expected_note_number is not None and str(expected_note_number).isdigit():
        expected_prefix = f"{int(expected_note_number)}."

    kept_lines: List[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        # 첫 줄은 무조건 유지
        if idx == 0:
            kept_lines.append(stripped)
            continue

        # 다른 top-level note 제목이 나타나면 여기서 중단
        if looks_like_top_level_note_title(stripped):
            if expected_prefix is None:
                break

            # 같은 note 번호 제목 반복은 허용
            if not stripped.startswith(expected_prefix):
                break

        kept_lines.append(stripped)

    return "\n".join(kept_lines).strip()


def smart_chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    # 일반 텍스트를 문단 우선 기준으로 chunk 분할
    text = normalize_whitespace(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = split_paragraphs(text)

    # 문단이 사실상 없으면 줄 단위로 fallback
    if len(paragraphs) <= 1:
        paragraphs = split_lines(text)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 문단 하나가 너무 길면 줄 기준으로 다시 분할
        if len(para) > max_chars:
            sub_lines = split_lines(para)
            if not sub_lines:
                sub_lines = [para]

            for line in sub_lines:
                line = line.strip()
                if not line:
                    continue

                # 줄 하나도 너무 길면 강제 분할
                if len(line) > max_chars:
                    if current_parts:
                        chunks.append("\n\n".join(current_parts).strip())
                        current_parts = []
                        current_len = 0

                    start = 0
                    while start < len(line):
                        end = min(len(line), start + max_chars)
                        piece = line[start:end].strip()
                        if piece:
                            chunks.append(piece)

                        if end == len(line):
                            break

                        start = max(0, end - overlap_chars)
                    continue

                proposed_len = current_len + len(line) + (2 if current_parts else 0)
                if proposed_len <= max_chars:
                    current_parts.append(line)
                    current_len = proposed_len
                else:
                    if current_parts:
                        chunks.append("\n\n".join(current_parts).strip())
                    current_parts = [line]
                    current_len = len(line)

            continue

        proposed_len = current_len + len(para) + (2 if current_parts else 0)
        if proposed_len <= max_chars:
            current_parts.append(para)
            current_len = proposed_len
        else:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
            current_parts = [para]
            current_len = len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    # overlap 보정
    if len(chunks) <= 1:
        return chunks

    overlapped_chunks: List[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            overlapped_chunks.append(chunk)
            continue

        prev_chunk = chunks[idx - 1]
        prefix = prev_chunk[-overlap_chars:].strip()

        if prefix:
            merged = normalize_whitespace(prefix + "\n\n" + chunk)
            overlapped_chunks.append(merged)
        else:
            overlapped_chunks.append(chunk)

    return overlapped_chunks


def chunk_table_text(
    text: str,
    max_chars: int = 1400,
    overlap_chars: int = 100,
) -> List[str]:
    # normalized_table 텍스트를 행(row) 중심으로 chunk 분할
    text = normalize_whitespace(text)

    if len(text) <= max_chars:
        return [text]

    lines = split_lines(text)
    if not lines:
        return [text]

    header = lines[0]
    data_lines = lines[1:] if len(lines) > 1 else []

    chunks: List[str] = []
    current_lines: List[str] = [header]
    current_len = len(header)

    for line in data_lines:
        proposed_len = current_len + len(line) + 1
        if proposed_len <= max_chars:
            current_lines.append(line)
            current_len = proposed_len
        else:
            chunks.append("\n".join(current_lines).strip())

            overlap_block: List[str] = [header]
            overlap_len = len(header)

            for prev_line in reversed(current_lines[1:]):
                if overlap_len + len(prev_line) + 1 > overlap_chars:
                    break
                overlap_block.insert(1, prev_line)
                overlap_len += len(prev_line) + 1

            current_lines = overlap_block + [line]
            current_len = sum(len(x) for x in current_lines) + max(0, len(current_lines) - 1)

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return chunks


def chunk_note_block_text(
    text: str,
    note_number: Optional[str],
    max_chars: int = 1200,
    overlap_chars: int = 120,
) -> List[str]:
    # note_block 전용 chunk 분할
    #
    # 원칙:
    # 1) 먼저 다음 note 제목이 섞였으면 제거
    # 2) 가능하면 한 note = 한 chunk
    # 3) 너무 긴 경우에만 문단 기준으로 분할
    cleaned_text = sanitize_note_block_text(text, expected_note_number=note_number)
    cleaned_text = normalize_whitespace(cleaned_text)

    if len(cleaned_text) <= max_chars:
        return [cleaned_text]

    return smart_chunk_text(
        text=cleaned_text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def should_include_in_dense_index(record: Dict[str, Any]) -> bool:
    # dense index에 넣을 source_type만 통과
    source_type = str(record.get("source_type", ""))

    return source_type in {
        "main_section",
        "note_block",
        "normalized_table",
    }


def chunk_record(
    record: Dict[str, Any],
    max_chars: int,
    overlap_chars: int,
) -> List[Dict[str, Any]]:
    # 단일 corpus record를 chunk record list로 변환
    source_type = str(record.get("source_type", ""))
    doc_id = str(record.get("doc_id", ""))
    title = str(record.get("title", ""))
    text = str(record.get("text", ""))
    metadata = dict(record.get("metadata", {}))

    # source_type별 chunk 전략
    if source_type == "normalized_table":
        chunks = chunk_table_text(
            text=text,
            max_chars=max_chars,
            overlap_chars=min(overlap_chars, 100),
        )
    elif source_type == "note_block":
        chunks = chunk_note_block_text(
            text=text,
            note_number=metadata.get("note_number"),
            max_chars=max_chars,
            overlap_chars=min(overlap_chars, 120),
        )
    else:
        chunks = smart_chunk_text(
            text=text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    chunk_records: List[Dict[str, Any]] = []

    for chunk_index, chunk_text in enumerate(chunks):
        chunk_text = normalize_whitespace(chunk_text)

        # 빈 chunk는 저장하지 않음
        if not chunk_text:
            continue

        chunk_record = {
            "chunk_id": f"{doc_id}_chunk_{chunk_index}",
            "parent_doc_id": doc_id,
            "source_type": source_type,
            "title": title,
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "chunk_text_length": len(chunk_text),
                "parent_title": title,
                "parent_source_type": source_type,
            },
        }
        chunk_records.append(chunk_record)

    return chunk_records


def build_dense_chunks(
    records: List[Dict[str, Any]],
    max_chars: int,
    overlap_chars: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 전체 corpus record를 dense chunk와 skipped record로 분리
    dense_chunks: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    for record in records:
        if not should_include_in_dense_index(record):
            skipped_records.append(record)
            continue

        dense_chunks.extend(
            chunk_record(
                record=record,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return dense_chunks, skipped_records


def write_chunk_summary_csv(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    # chunk 단위 요약 CSV 저장
    fieldnames = [
        "chunk_id",
        "parent_doc_id",
        "source_type",
        "title",
        "year",
        "statement_type",
        "note_number",
        "section_type",
        "chunk_index",
        "chunk_count",
        "chunk_text_length",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            writer.writerow({
                "chunk_id": chunk.get("chunk_id"),
                "parent_doc_id": chunk.get("parent_doc_id"),
                "source_type": chunk.get("source_type"),
                "title": chunk.get("title"),
                "year": meta.get("year"),
                "statement_type": meta.get("statement_type"),
                "note_number": meta.get("note_number"),
                "section_type": meta.get("section_type"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_count": meta.get("chunk_count"),
                "chunk_text_length": meta.get("chunk_text_length"),
            })


def write_chunk_type_summary_csv(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    # source_type x year 집계 CSV 저장
    rows: List[Dict[str, Any]] = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        rows.append({
            "source_type": chunk.get("source_type"),
            "year": meta.get("year"),
            "chunk_text_length": meta.get("chunk_text_length", 0),
        })

    if not rows:
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_type", "year", "chunk_count", "avg_chunk_text_length"])
        return

    summary_map: Dict[Tuple[str, Any], Dict[str, float]] = {}

    for row in rows:
        key = (str(row["source_type"]), row["year"])
        if key not in summary_map:
            summary_map[key] = {
                "chunk_count": 0,
                "sum_text_length": 0.0,
            }

        summary_map[key]["chunk_count"] += 1
        summary_map[key]["sum_text_length"] += float(row["chunk_text_length"])

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_type", "year", "chunk_count", "avg_chunk_text_length"])

        for (source_type, year), agg in sorted(summary_map.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            chunk_count = int(agg["chunk_count"])
            avg_len = agg["sum_text_length"] / chunk_count if chunk_count > 0 else 0.0

            writer.writerow([source_type, year, chunk_count, round(avg_len, 2)])


def write_build_summary_csv(
    dense_chunks: List[Dict[str, Any]],
    skipped_records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    # 전체 chunk build 요약 CSV 저장
    source_counter: Dict[str, int] = {}
    skipped_counter: Dict[str, int] = {}

    for chunk in dense_chunks:
        source_type = str(chunk.get("source_type", ""))
        source_counter[source_type] = source_counter.get(source_type, 0) + 1

    for record in skipped_records:
        source_type = str(record.get("source_type", ""))
        skipped_counter[source_type] = skipped_counter.get(source_type, 0) + 1

    all_source_types = sorted(set(source_counter) | set(skipped_counter))

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_type", "dense_chunk_count", "skipped_record_count"])

        for source_type in all_source_types:
            writer.writerow([
                source_type,
                source_counter.get(source_type, 0),
                skipped_counter.get(source_type, 0),
            ])


def main() -> None:
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="unified_corpus.jsonl을 dense retrieval용 chunk corpus로 변환")
    parser.add_argument(
        "--input-path",
        type=str,
        default="corpus_exports/unified_corpus.jsonl",
        help="입력 unified corpus JSONL 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chunk_exports",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="텍스트 chunk 최대 길이",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=150,
        help="텍스트 chunk overlap 길이",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    per_year_dir = output_dir / "per_year_jsonl"

    ensure_dir(output_dir)
    ensure_dir(per_year_dir)

    # unified corpus 읽기
    records = read_jsonl(input_path)

    # dense chunk 생성
    dense_chunks, skipped_records = build_dense_chunks(
        records=records,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    # 전체 chunk JSONL 저장
    dense_chunk_path = output_dir / "dense_chunk_corpus.jsonl"
    write_jsonl(dense_chunks, dense_chunk_path)

    # skipped record JSONL 저장
    skipped_path = output_dir / "skipped_records.jsonl"
    write_jsonl(skipped_records, skipped_path)

    # 연도별 JSONL 저장
    year_map: Dict[Any, List[Dict[str, Any]]] = {}
    for chunk in dense_chunks:
        year = chunk.get("metadata", {}).get("year")
        year_map.setdefault(year, []).append(chunk)

    for year, year_chunks in year_map.items():
        year_path = per_year_dir / f"dense_chunks_{year}.jsonl"
        write_jsonl(year_chunks, year_path)

    # 요약 CSV 저장
    write_chunk_summary_csv(dense_chunks, output_dir / "dense_chunk_document_index.csv")
    write_chunk_type_summary_csv(dense_chunks, output_dir / "dense_chunk_type_summary.csv")
    write_build_summary_csv(dense_chunks, skipped_records, output_dir / "dense_chunk_build_summary.csv")

    # 완료 로그 출력
    print("=" * 100)
    print("[CHUNK BUILD COMPLETE]")
    print("input records:", len(records))
    print("dense chunks:", len(dense_chunks))
    print("skipped records:", len(skipped_records))
    print("output dir:", output_dir)
    print()

    # source_type별 간단 집계 출력
    chunk_type_counts: Dict[str, int] = {}
    for chunk in dense_chunks:
        source_type = str(chunk.get("source_type", ""))
        chunk_type_counts[source_type] = chunk_type_counts.get(source_type, 0) + 1

    print("[DENSE CHUNK COUNTS BY SOURCE TYPE]")
    for source_type, count in sorted(chunk_type_counts.items()):
        print(f"{source_type}: {count}")

    print()
    skipped_type_counts: Dict[str, int] = {}
    for record in skipped_records:
        source_type = str(record.get("source_type", ""))
        skipped_type_counts[source_type] = skipped_type_counts.get(source_type, 0) + 1

    print("[SKIPPED RECORD COUNTS BY SOURCE TYPE]")
    for source_type, count in sorted(skipped_type_counts.items()):
        print(f"{source_type}: {count}")


if __name__ == "__main__":
    main()