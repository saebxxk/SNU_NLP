# 필요한 라이브러리 import
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# FAISS는 선택적으로 import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


def ensure_dir(path: Path) -> None:
    # 디렉토리가 없으면 생성
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    # JSONL 파일을 list[dict]로 로드
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
                raise ValueError(
                    f"JSONL 파싱 실패: file={path}, line={line_number}, error={e}"
                ) from e

    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    # JSONL 파일 저장
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_embedding_inputs(records: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    # 임베딩할 text 리스트와 저장용 metadata 리스트 생성
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for record in records:
        text = str(record.get("text", "")).strip()

        # 빈 텍스트는 제외
        if not text:
            continue

        texts.append(text)

        metadatas.append({
            "chunk_id": record.get("chunk_id"),
            "parent_doc_id": record.get("parent_doc_id"),
            "source_type": record.get("source_type"),
            "title": record.get("title"),
            "text": text,
            "metadata": record.get("metadata", {}),
        })

    return texts, metadatas


def save_id_map_csv(metadatas: List[Dict[str, Any]], path: Path) -> None:
    # 임베딩 index 순서와 chunk_id를 매핑하는 CSV 저장
    import csv

    fieldnames = [
        "row_id",
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

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_id, item in enumerate(metadatas):
            meta = item.get("metadata", {})
            writer.writerow({
                "row_id": row_id,
                "chunk_id": item.get("chunk_id"),
                "parent_doc_id": item.get("parent_doc_id"),
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "year": meta.get("year"),
                "statement_type": meta.get("statement_type"),
                "note_number": meta.get("note_number"),
                "section_type": meta.get("section_type"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_count": meta.get("chunk_count"),
                "chunk_text_length": meta.get("chunk_text_length"),
            })


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    device: str,
) -> Tuple[np.ndarray, str]:
    # sentence-transformers 모델로 임베딩 생성
    model = SentenceTransformer(model_name, device=device)

    # 임베딩 계산
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    # float32로 고정
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return embeddings, str(model.device)


def build_faiss_index(
    embeddings: np.ndarray,
    normalize_embeddings: bool,
) -> Any:
    # FAISS index 생성
    #
    # normalize_embeddings=True이면 cosine similarity와 같은 효과를 내기 위해
    # inner product index를 사용한다.
    dim = embeddings.shape[1]

    if normalize_embeddings:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


def save_manifest(
    path: Path,
    *,
    input_path: str,
    record_count: int,
    embedding_count: int,
    embedding_dim: int,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    device: str,
    faiss_available: bool,
    faiss_index_written: bool,
    elapsed_seconds: float,
) -> None:
    # 실행 메타정보 저장
    manifest = {
        "input_path": input_path,
        "record_count": record_count,
        "embedding_count": embedding_count,
        "embedding_dim": embedding_dim,
        "model_name": model_name,
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "device": device,
        "faiss_available": faiss_available,
        "faiss_index_written": faiss_index_written,
        "elapsed_seconds": round(elapsed_seconds, 4),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="dense chunk corpus를 임베딩하고 FAISS index를 생성")
    parser.add_argument(
        "--input-path",
        type=str,
        default="chunk_exports/dense_chunk_corpus.jsonl",
        help="입력 dense chunk corpus JSONL 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embed_exports",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="sentence-transformers 모델 이름",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="임베딩 배치 크기",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="임베딩 장치 예: cpu, cuda, mps",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="임베딩을 정규화하여 cosine 유사도 기반으로 사용",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    start_time = time.time()

    # dense chunk corpus 로드
    records = read_jsonl(input_path)

    # 임베딩 입력과 메타데이터 분리
    texts, metadatas = build_embedding_inputs(records)

    if not texts:
        raise ValueError("임베딩할 텍스트가 없습니다. 입력 corpus를 확인하세요.")

    print("=" * 100)
    print("[EMBED START]")
    print("input records:", len(records))
    print("texts to embed:", len(texts))
    print("model:", args.model_name)
    print("device:", args.device)
    print("normalize_embeddings:", args.normalize_embeddings)
    print()

    # 임베딩 계산
    embeddings, actual_device = compute_embeddings(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        device=args.device,
    )

    # embeddings 저장
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    # metadata 저장
    metadata_path = output_dir / "metadata.jsonl"
    write_jsonl(metadatas, metadata_path)

    # id map 저장
    id_map_path = output_dir / "id_map.csv"
    save_id_map_csv(metadatas, id_map_path)

    # FAISS index 저장
    faiss_index_written = False
    if FAISS_AVAILABLE:
        index = build_faiss_index(
            embeddings=embeddings,
            normalize_embeddings=args.normalize_embeddings,
        )
        faiss.write_index(index, str(output_dir / "faiss.index"))
        faiss_index_written = True

    elapsed = time.time() - start_time

    # manifest 저장
    save_manifest(
        output_dir / "manifest.json",
        input_path=str(input_path),
        record_count=len(records),
        embedding_count=len(texts),
        embedding_dim=int(embeddings.shape[1]),
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        device=actual_device,
        faiss_available=FAISS_AVAILABLE,
        faiss_index_written=faiss_index_written,
        elapsed_seconds=elapsed,
    )

    # 완료 로그 출력
    print("=" * 100)
    print("[EMBED COMPLETE]")
    print("records read:", len(records))
    print("embeddings saved:", len(texts))
    print("embedding dim:", embeddings.shape[1])
    print("embeddings path:", embeddings_path)
    print("metadata path:", metadata_path)
    print("id map path:", id_map_path)
    print("faiss available:", FAISS_AVAILABLE)
    print("faiss index written:", faiss_index_written)
    print("output dir:", output_dir)
    print("elapsed seconds:", round(elapsed, 4))


if __name__ == "__main__":
    main()