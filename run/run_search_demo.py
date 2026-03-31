# 필요한 라이브러리 import
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# FAISS import
try:
    import faiss  # type: ignore
except Exception as e:
    raise ImportError(
        "faiss를 import할 수 없습니다. 현재 환경에 faiss-cpu 또는 faiss가 설치되어 있어야 합니다."
    ) from e


# -------------------------------------------------------------------------
# 텍스트 전처리 상수
# -------------------------------------------------------------------------

# 연도 추출 패턴
YEAR_PATTERN = re.compile(r"(20\d{2})\s*년")

# 아주 단순한 한국어 불용어
# 검색 의도에 거의 도움이 안 되는 질의어를 제거한다.
KOREAN_STOPWORDS = {
    "무엇",
    "무엇이야",
    "뭐",
    "뭐야",
    "어떤",
    "어떻게",
    "관련",
    "대한",
    "내용",
    "내용이야",
    "설명",
    "설명해",
    "설명해줘",
    "알려줘",
    "알려주세요",
    "말해줘",
    "보여줘",
    "정리해줘",
    "좀",
    "대해",
    "이다",
    "있는",
    "있어",
    "있나요",
}

# 자주 붙는 조사/어미
# 너무 공격적으로 자르면 오히려 손해라서 보수적으로 넣는다.
JOSA_SUFFIXES = sorted(
    [
        "으로부터",
        "에게서",
        "에서는",
        "으로는",
        "이라고",
        "라는",
        "이다",
        "이며",
        "에서",
        "에게",
        "한테",
        "부터",
        "까지",
        "처럼",
        "으로",
        "로는",
        "로서",
        "로",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "와",
        "과",
        "도",
        "만",
        "야",
        "요",
    ],
    key=len,
    reverse=True,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    # JSONL 파일을 list[dict]로 읽는 함수
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # 빈 줄은 건너뛴다.
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 파싱 실패: file={path}, line={line_number}, error={e}"
                ) from e

    return records


def read_manifest(path: Path) -> Dict[str, Any]:
    # manifest.json을 읽는 함수
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source_types(source_type_text: Optional[str]) -> Optional[List[str]]:
    # 콤마로 들어온 source_type 문자열을 리스트로 정리
    if source_type_text is None:
        return None

    parts = [x.strip() for x in source_type_text.split(",") if x.strip()]
    if not parts:
        return None

    return parts


def load_search_assets(embed_dir: Path) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    # 검색에 필요한 자산을 한 번에 로드
    faiss_path = embed_dir / "faiss.index"
    metadata_path = embed_dir / "metadata.jsonl"
    manifest_path = embed_dir / "manifest.json"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index 파일이 없습니다: {faiss_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl 파일이 없습니다: {metadata_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json 파일이 없습니다: {manifest_path}")

    # FAISS index 로드
    index = faiss.read_index(str(faiss_path))

    # metadata 로드
    metadatas = read_jsonl(metadata_path)

    # manifest 로드
    manifest = read_manifest(manifest_path)

    return index, metadatas, manifest


def build_query_embedding(
    query: str,
    model_name: str,
    normalize_embeddings: bool,
    device: str,
) -> np.ndarray:
    # 질의 텍스트를 임베딩 벡터로 변환
    model = SentenceTransformer(model_name, device=device)

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )

    embedding = np.asarray(embedding, dtype=np.float32)
    return embedding


def metadata_matches_filter(
    item: Dict[str, Any],
    year: Optional[int],
    source_types: Optional[List[str]],
) -> bool:
    # metadata filter 검사
    meta = item.get("metadata", {})

    # year filter
    if year is not None:
        if meta.get("year") != year:
            return False

    # source_type filter
    if source_types is not None:
        if item.get("source_type") not in source_types:
            return False

    return True


def normalize_text_for_match(text: str) -> str:
    # 공백/기호를 줄여 keyword match 안정화
    text = text.lower()
    text = re.sub(r"[\(\)\[\]\{\},:;\"'`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_korean_particle(token: str) -> str:
    # 한국어 조사/어미를 단순 제거
    token = token.strip()

    # 너무 짧으면 유지
    if len(token) <= 1:
        return token

    for suffix in JOSA_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]

    return token


def extract_year_from_query(query: str) -> Optional[int]:
    # 질의에서 연도 추출
    match = YEAR_PATTERN.search(query)
    if match is None:
        return None
    return int(match.group(1))


def remove_year_expression(query: str) -> str:
    # 질의에서 "2024년" 같은 연도 표현 제거
    return YEAR_PATTERN.sub(" ", query)


def extract_query_terms(query: str) -> List[str]:
    # 질의에서 핵심 토큰 추출
    #
    # 개선 포인트:
    # - 연도 제거
    # - 조사 제거
    # - 불용어 제거
    query_wo_year = remove_year_expression(query)
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9]+", query_wo_year.lower())

    seen = set()
    cleaned: List[str] = []

    for raw in raw_tokens:
        token = raw.strip()
        if not token:
            continue

        # 원형과 조사 제거형 둘 다 후보로 본다.
        candidates = [token]
        stripped = strip_korean_particle(token)
        if stripped != token:
            candidates.append(stripped)

        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            if len(cand) < 2:
                continue
            if cand in KOREAN_STOPWORDS:
                continue
            if cand in seen:
                continue

            seen.add(cand)
            cleaned.append(cand)

    return cleaned


def resolve_query_context(query: str, explicit_year: Optional[int]) -> Tuple[Optional[int], List[str]]:
    # 질의에서 연도와 핵심 키워드 추출
    auto_year = extract_year_from_query(query)

    # CLI에서 명시한 연도가 있으면 그 값을 우선
    resolved_year = explicit_year if explicit_year is not None else auto_year
    query_terms = extract_query_terms(query)

    return resolved_year, query_terms


def compute_keyword_features(
    query_terms: List[str],
    title: str,
    text: str,
) -> Dict[str, float]:
    # title/text에 대한 keyword feature 계산
    normalized_title = normalize_text_for_match(title)
    normalized_text = normalize_text_for_match(text)

    title_hits = 0
    text_hits = 0

    for term in query_terms:
        if term in normalized_title:
            title_hits += 1
        if term in normalized_text:
            text_hits += 1

    term_count = max(len(query_terms), 1)

    return {
        "title_hit_ratio": title_hits / term_count,
        "text_hit_ratio": text_hits / term_count,
        "title_hits": float(title_hits),
        "text_hits": float(text_hits),
    }


def source_type_prior(source_type: str) -> float:
    # source_type별 기본 prior
    if source_type == "note_block":
        return 0.05
    if source_type == "main_section":
        return 0.03
    if source_type == "normalized_table":
        return 0.02
    return 0.0


def rerank_score(
    dense_score: float,
    item: Dict[str, Any],
    query_terms: List[str],
    query_year: Optional[int],
) -> Tuple[float, Dict[str, float]]:
    # dense score + lexical/title boost + year boost로 hybrid reranking
    title = str(item.get("title", ""))
    text = str(item.get("text", ""))
    source_type = str(item.get("source_type", ""))
    meta = item.get("metadata", {})

    keyword_features = compute_keyword_features(
        query_terms=query_terms,
        title=title,
        text=text,
    )

    title_hit_ratio = keyword_features["title_hit_ratio"]
    text_hit_ratio = keyword_features["text_hit_ratio"]
    title_hits = keyword_features["title_hits"]
    text_hits = keyword_features["text_hits"]

    normalized_title = normalize_text_for_match(title)

    # title에 query term이 하나라도 직접 들어가면 강하게 boost
    title_term_boost = 0.12 if title_hits > 0 else 0.0

    # 질의 핵심어가 전혀 안 맞으면 약한 penalty
    zero_lexical_penalty = -0.08 if (len(query_terms) > 0 and title_hits == 0 and text_hits == 0) else 0.0

    # 연도 일치 boost
    year_boost = 0.0
    if query_year is not None and meta.get("year") == query_year:
        year_boost = 0.12

    # 전체 phrase가 title에 들어가면 추가 boost
    exact_title_boost = 0.0
    if query_terms:
        joined_query = " ".join(query_terms)
        if joined_query and joined_query in normalized_title:
            exact_title_boost = 0.08

    final_score = (
        dense_score
        + 0.30 * title_hit_ratio
        + 0.12 * text_hit_ratio
        + title_term_boost
        + exact_title_boost
        + year_boost
        + source_type_prior(source_type)
        + zero_lexical_penalty
    )

    debug_features = {
        "dense_score": dense_score,
        "title_hit_ratio": title_hit_ratio,
        "text_hit_ratio": text_hit_ratio,
        "title_term_boost": title_term_boost,
        "exact_title_boost": exact_title_boost,
        "year_boost": year_boost,
        "source_type_prior": source_type_prior(source_type),
        "zero_lexical_penalty": zero_lexical_penalty,
        "final_score": final_score,
    }

    return final_score, debug_features


def search_index(
    index: Any,
    metadatas: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int,
    fetch_k: int,
    year: Optional[int],
    source_types: Optional[List[str]],
    query_terms: List[str],
) -> List[Dict[str, Any]]:
    # FAISS 검색 후 filter 적용 + rerank
    distances, indices = index.search(query_embedding, fetch_k)

    candidate_results: List[Dict[str, Any]] = []

    for score, idx in zip(distances[0], indices[0]):
        # FAISS가 못 찾은 경우 -1이 들어올 수 있다.
        if idx == -1:
            continue

        if idx < 0 or idx >= len(metadatas):
            continue

        item = metadatas[idx]

        if not metadata_matches_filter(item, year=year, source_types=source_types):
            continue

        hybrid_score, debug_features = rerank_score(
            dense_score=float(score),
            item=item,
            query_terms=query_terms,
            query_year=year,
        )

        candidate_results.append({
            "row_id": int(idx),
            "dense_score": float(score),
            "score": float(hybrid_score),
            "chunk_id": item.get("chunk_id"),
            "parent_doc_id": item.get("parent_doc_id"),
            "source_type": item.get("source_type"),
            "title": item.get("title"),
            "text": item.get("text"),
            "metadata": item.get("metadata", {}),
            "debug_features": debug_features,
        })

    # hybrid score 기준 정렬
    candidate_results.sort(key=lambda x: x["score"], reverse=True)

    # parent_doc_id 중복이 너무 많으면 다양성 확보
    diversified: List[Dict[str, Any]] = []
    seen_parent_doc_ids = set()

    for item in candidate_results:
        parent_doc_id = item.get("parent_doc_id")

        if parent_doc_id not in seen_parent_doc_ids:
            diversified.append(item)
            seen_parent_doc_ids.add(parent_doc_id)

        if len(diversified) >= top_k:
            break

    # 다양성으로 부족하면 원래 후보에서 채움
    if len(diversified) < top_k:
        existing_ids = {x.get("chunk_id") for x in diversified}
        for item in candidate_results:
            if item.get("chunk_id") in existing_ids:
                continue
            diversified.append(item)
            existing_ids.add(item.get("chunk_id"))
            if len(diversified) >= top_k:
                break

    return diversified[:top_k]


def make_preview(text: str, max_chars: int = 400) -> str:
    # 긴 본문을 출력용 preview로 자르는 함수
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " ..."


def print_result_item(rank: int, result: Dict[str, Any], show_debug: bool) -> None:
    # 단일 검색 결과 출력
    meta = result.get("metadata", {})
    preview = make_preview(str(result.get("text", "")), max_chars=500)

    print("=" * 100)
    print(f"[RANK {rank}]")
    print(f"score          : {result.get('score'):.6f}")
    print(f"dense_score    : {result.get('dense_score'):.6f}")
    print(f"chunk_id       : {result.get('chunk_id')}")
    print(f"parent_doc_id  : {result.get('parent_doc_id')}")
    print(f"source_type    : {result.get('source_type')}")
    print(f"title          : {result.get('title')}")
    print(f"year           : {meta.get('year')}")
    print(f"statement_type : {meta.get('statement_type')}")
    print(f"note_number    : {meta.get('note_number')}")
    print(f"section_type   : {meta.get('section_type')}")
    print(f"chunk_index    : {meta.get('chunk_index')}")
    print(f"chunk_count    : {meta.get('chunk_count')}")

    if show_debug:
        print(f"debug_features : {result.get('debug_features')}")

    print()
    print("[TEXT PREVIEW]")
    print(preview)
    print()


def print_search_summary(
    query: str,
    embed_dir: Path,
    model_name: str,
    normalize_embeddings: bool,
    year: Optional[int],
    source_types: Optional[List[str]],
    top_k: int,
    fetch_k: int,
    query_terms: List[str],
) -> None:
    # 검색 실행 설정 요약 출력
    print("=" * 100)
    print("[SEARCH CONFIG]")
    print(f"query                 : {query}")
    print(f"query_terms           : {query_terms}")
    print(f"embed_dir             : {embed_dir}")
    print(f"model_name            : {model_name}")
    print(f"normalize_embeddings  : {normalize_embeddings}")
    print(f"year filter           : {year}")
    print(f"source_type filter    : {source_types}")
    print(f"top_k                 : {top_k}")
    print(f"fetch_k               : {fetch_k}")
    print()


def main() -> None:
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="FAISS 기반 hybrid 검색 데모")
    parser.add_argument(
        "--embed-dir",
        type=str,
        default="embed_exports",
        help="임베딩 산출물 디렉토리",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="검색 질의",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="최종 출력할 검색 결과 개수",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=40,
        help="FAISS에서 먼저 넉넉히 가져올 후보 개수",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="연도 filter 예: 2024",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        help="source_type filter 예: note_block,normalized_table",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="질의 임베딩 장치. 생략 시 cpu 사용",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="reranking feature를 함께 출력",
    )
    args = parser.parse_args()

    # 질의가 없으면 인터랙티브 입력
    query = args.query
    if query is None:
        query = input("검색 질의를 입력하세요: ").strip()

    if not query:
        raise ValueError("빈 질의는 검색할 수 없습니다.")

    embed_dir = Path(args.embed_dir)

    # 검색 자산 로드
    index, metadatas, manifest = load_search_assets(embed_dir)

    # manifest에서 임베딩 설정 읽기
    model_name = str(manifest.get("model_name"))
    normalize_embeddings = bool(manifest.get("normalize_embeddings", False))

    # 질의 임베딩 디바이스 결정
    device = args.device if args.device is not None else "cpu"

    source_types = normalize_source_types(args.source_type)

    # 질의에서 연도와 핵심 키워드 추출
    resolved_year, query_terms = resolve_query_context(query, explicit_year=args.year)

    # 검색 설정 출력
    print_search_summary(
        query=query,
        embed_dir=embed_dir,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        year=resolved_year,
        source_types=source_types,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        query_terms=query_terms,
    )

    # 질의 임베딩 생성
    query_embedding = build_query_embedding(
        query=query,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device=device,
    )

    # 검색 수행
    results = search_index(
        index=index,
        metadatas=metadatas,
        query_embedding=query_embedding,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        year=resolved_year,
        source_types=source_types,
        query_terms=query_terms,
    )

    # 결과 출력
    print("=" * 100)
    print("[SEARCH RESULTS]")
    print(f"result_count: {len(results)}")
    print()

    if not results:
        print("검색 결과가 없습니다.")
        sys.exit(0)

    for rank, item in enumerate(results, start=1):
        print_result_item(rank, item, show_debug=args.show_debug)


if __name__ == "__main__":
    main()