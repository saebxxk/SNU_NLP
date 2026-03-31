from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# 데이터 구조
# -----------------------------------------------------------------------------

@dataclass
class Candidate:
    row_id: int
    chunk_id: str
    parent_doc_id: str
    source_type: str
    title: str
    text: str
    metadata: Dict[str, Any]
    dense_score: float
    final_score: float
    debug_features: Dict[str, float]


@dataclass
class PoolResult:
    level_name: str
    pool_size: int
    target_chunk_id: str
    target_raw_score: float
    target_dense_score: float
    target_support_prob: float
    target_rank: int
    is_top1: bool
    strongest_rival_chunk_id: Optional[str]
    strongest_rival_score: Optional[float]
    margin_vs_rival: Optional[float]
    top3_chunk_ids: List[str]


# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 실패: file={path}, line={line_no}, error={e}") from e
    return rows


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    m = np.max(arr)
    exp = np.exp(arr - m)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(arr)
    return exp / denom


def make_preview(text: str, max_chars: int = 140) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


# -----------------------------------------------------------------------------
# 기존 검색 코드 동적 로드
# -----------------------------------------------------------------------------


def load_search_demo_module(search_demo_path: Path):
    if not search_demo_path.exists():
        raise FileNotFoundError(f"run_search_demo.py를 찾을 수 없습니다: {search_demo_path}")

    spec = importlib.util.spec_from_file_location("search_demo_module", search_demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 spec 생성 실패: {search_demo_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["search_demo_module"] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# 점수 계산
# -----------------------------------------------------------------------------


def build_query_embedding_via_existing_code(
    search_demo_module: Any,
    query: str,
    manifest: Dict[str, Any],
    device: str,
) -> np.ndarray:
    model_name = str(manifest["model_name"])
    normalize_embeddings = bool(manifest.get("normalize_embeddings", True))

    embedding = search_demo_module.build_query_embedding(
        query=query,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device=device,
    )

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"query embedding shape가 예상과 다릅니다: {embedding.shape}")

    return np.asarray(embedding[0], dtype=np.float32)



def compute_dense_scores(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    normalize_embeddings: bool,
) -> np.ndarray:
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize_embeddings:
        return embeddings @ query_embedding

    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(embeddings, axis=1)
    denom = np.maximum(doc_norms * max(float(query_norm), 1e-12), 1e-12)
    return (embeddings @ query_embedding) / denom



def build_all_candidates(
    metadatas: List[Dict[str, Any]],
    dense_scores: np.ndarray,
    query_terms: List[str],
    query_year: Optional[int],
    search_demo_module: Any,
) -> List[Candidate]:
    candidates: List[Candidate] = []

    for row_id, item in enumerate(metadatas):
        dense_score = float(dense_scores[row_id])
        final_score, debug_features = search_demo_module.rerank_score(
            dense_score=dense_score,
            item=item,
            query_terms=query_terms,
            query_year=query_year,
        )

        candidates.append(
            Candidate(
                row_id=row_id,
                chunk_id=str(item.get("chunk_id", "")),
                parent_doc_id=str(item.get("parent_doc_id", "")),
                source_type=str(item.get("source_type", "")),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})),
                dense_score=dense_score,
                final_score=float(final_score),
                debug_features=dict(debug_features),
            )
        )

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


# -----------------------------------------------------------------------------
# 후보 분류
# -----------------------------------------------------------------------------


def is_topical_rival(c: Candidate) -> bool:
    title_hit_ratio = float(c.debug_features.get("title_hit_ratio", 0.0))
    text_hit_ratio = float(c.debug_features.get("text_hit_ratio", 0.0))
    return (title_hit_ratio > 0.0) or (text_hit_ratio >= 0.50)



def is_unrelated_distractor(c: Candidate) -> bool:
    title_hit_ratio = float(c.debug_features.get("title_hit_ratio", 0.0))
    text_hit_ratio = float(c.debug_features.get("text_hit_ratio", 0.0))
    return (title_hit_ratio == 0.0) and (text_hit_ratio == 0.0)



def choose_target(candidates: Sequence[Candidate], target_chunk_id: Optional[str]) -> Candidate:
    if target_chunk_id is None:
        return candidates[0]

    for c in candidates:
        if c.chunk_id == target_chunk_id:
            return c

    raise ValueError(f"지정한 target_chunk_id를 후보에서 찾지 못했습니다: {target_chunk_id}")



def build_candidate_buckets(
    candidates: Sequence[Candidate],
    target: Candidate,
    query_year: Optional[int],
) -> Dict[str, List[Candidate]]:
    near_rivals: List[Candidate] = []
    topical_rivals: List[Candidate] = []
    unrelated_distractors: List[Candidate] = []
    year_sensitive_rivals: List[Candidate] = []

    for c in candidates:
        if c.chunk_id == target.chunk_id:
            continue

        if is_unrelated_distractor(c):
            unrelated_distractors.append(c)
            continue

        if is_topical_rival(c):
            topical_rivals.append(c)

            same_title = c.title == target.title
            same_parent_title = c.metadata.get("parent_title") == target.metadata.get("parent_title")
            same_note_number = c.metadata.get("note_number") == target.metadata.get("note_number")
            close_score = abs(c.final_score - target.final_score) <= 0.08

            if same_title or same_parent_title or same_note_number or close_score:
                near_rivals.append(c)

            if query_year is not None and c.metadata.get("year") == query_year:
                year_sensitive_rivals.append(c)

    near_rivals.sort(key=lambda x: x.final_score, reverse=True)
    topical_rivals.sort(key=lambda x: x.final_score, reverse=True)
    unrelated_distractors.sort(key=lambda x: x.final_score, reverse=True)
    year_sensitive_rivals.sort(key=lambda x: x.final_score, reverse=True)

    def dedupe(xs: Iterable[Candidate]) -> List[Candidate]:
        out: List[Candidate] = []
        seen = set()
        for x in xs:
            if x.chunk_id in seen:
                continue
            seen.add(x.chunk_id)
            out.append(x)
        return out

    return {
        "near_rivals": dedupe(near_rivals),
        "topical_rivals": dedupe(topical_rivals),
        "unrelated_distractors": dedupe(unrelated_distractors),
        "year_sensitive_rivals": dedupe(year_sensitive_rivals),
    }


# -----------------------------------------------------------------------------
# level pool 구성
# -----------------------------------------------------------------------------


def take(xs: Sequence[Candidate], k: int) -> List[Candidate]:
    return list(xs[: max(k, 0)])



def unique_candidates(xs: Iterable[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    for x in xs:
        if x.chunk_id in seen:
            continue
        seen.add(x.chunk_id)
        out.append(x)
    return out



def build_level_pools(
    target: Candidate,
    buckets: Dict[str, List[Candidate]],
) -> List[Tuple[str, List[Candidate]]]:
    near = buckets["near_rivals"]
    topical = buckets["topical_rivals"]
    unrelated = buckets["unrelated_distractors"]
    year_sensitive = buckets["year_sensitive_rivals"]

    l0 = unique_candidates([target] + take(unrelated, 5))
    l1 = unique_candidates([target] + take(near, 1) + take(unrelated, 5))
    l2 = unique_candidates([target] + take(topical, 6) + take(unrelated, 3))
    l3 = unique_candidates([target] + take(topical, 6) + take(year_sensitive, 3) + take(unrelated, 2))

    return [
        ("L0_weak_competition", l0),
        ("L1_near_rival", l1),
        ("L2_topical_crowding", l2),
        ("L3_year_sensitive_pressure", l3),
    ]


# -----------------------------------------------------------------------------
# pool 평가
# -----------------------------------------------------------------------------


def evaluate_pool(level_name: str, pool: Sequence[Candidate], target_chunk_id: str) -> PoolResult:
    if not pool:
        raise ValueError(f"pool이 비어 있습니다: {level_name}")

    ranked = sorted(pool, key=lambda x: x.final_score, reverse=True)
    scores = [x.final_score for x in ranked]
    probs = stable_softmax(scores)

    target_idx = None
    for idx, c in enumerate(ranked):
        if c.chunk_id == target_chunk_id:
            target_idx = idx
            break

    if target_idx is None:
        raise ValueError(f"target이 pool에 없습니다: level={level_name}, target={target_chunk_id}")

    target = ranked[target_idx]
    target_rank = target_idx + 1

    rivals = [c for c in ranked if c.chunk_id != target_chunk_id]
    strongest_rival = rivals[0] if rivals else None

    margin = None
    if strongest_rival is not None:
        margin = target.final_score - strongest_rival.final_score

    return PoolResult(
        level_name=level_name,
        pool_size=len(pool),
        target_chunk_id=target.chunk_id,
        target_raw_score=float(target.final_score),
        target_dense_score=float(target.dense_score),
        target_support_prob=float(probs[target_idx]),
        target_rank=target_rank,
        is_top1=(target_rank == 1),
        strongest_rival_chunk_id=(strongest_rival.chunk_id if strongest_rival is not None else None),
        strongest_rival_score=(float(strongest_rival.final_score) if strongest_rival is not None else None),
        margin_vs_rival=margin,
        top3_chunk_ids=[c.chunk_id for c in ranked[:3]],
    )


# -----------------------------------------------------------------------------
# baseline validity / 설명 생성
# -----------------------------------------------------------------------------


def classify_baseline_status(
    baseline: PoolResult,
    explicit_target_was_given: bool,
    valid_rank_threshold: int,
) -> str:
    if not explicit_target_was_given:
        return "normal_competition"
    if baseline.target_rank > valid_rank_threshold:
        return "target_mismatch_diagnostic"
    return "normal_competition"



def margin_label(margin: Optional[float]) -> str:
    if margin is None:
        return "비교 rival 없음"
    if margin > 0.20:
        return "margin이 충분히 큼"
    if margin > 0.05:
        return "margin이 작지만 양수"
    if margin > 0.0:
        return "margin이 거의 0에 가까운 양수"
    if margin > -0.05:
        return "근소하게 밀림"
    return "명확하게 밀림"



def explain_results(
    results: Sequence[PoolResult],
    baseline_status: str,
    explicit_target_was_given: bool,
    valid_rank_threshold: int,
) -> List[str]:
    if not results:
        return ["결과가 없습니다."]

    baseline = results[0]
    comments: List[str] = []

    if baseline_status == "target_mismatch_diagnostic":
        comments.append(
            (
                f"baseline부터 target이 rank {baseline.target_rank}이므로, 현재 query에 대해 이 target은 경쟁 설명 대상이 아니라 "
                f"target mismatch 후보로 보는 편이 맞습니다."
            )
        )
        rival = baseline.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            (
                f"baseline 최강 rival은 {rival}이고, margin은 {baseline.margin_vs_rival:.4f}입니다. "
                f"즉 경쟁이 커져서 무너진 것이 아니라 처음부터 query-target 정합성이 약했습니다."
            )
        )
        comments.append(
            (
                f"같은 target을 유지한 competition 분석을 하려면 query를 고정해야 하고, query가 바뀌었다면 target도 다시 선택하는 것이 적절합니다."
            )
        )
        return comments[:3]

    comments.append(
        (
            f"baseline에서는 target이 rank {baseline.target_rank}이고, strongest rival 대비 margin은 "
            f"{baseline.margin_vs_rival:.4f} ({margin_label(baseline.margin_vs_rival)})입니다."
        )
    )

    reversal_levels = [r for r in results if not r.is_top1]
    if reversal_levels:
        first = reversal_levels[0]
        rival = first.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            (
                f"처음 top-1이 깨지는 지점은 {first.level_name}이며, 이때 strongest rival은 {rival}, "
                f"margin은 {first.margin_vs_rival:.4f}입니다."
            )
        )
    else:
        comments.append("모든 level에서 target이 top-1을 유지했습니다.")

    hardest = min(results, key=lambda r: (r.margin_vs_rival if r.margin_vs_rival is not None else 0.0))
    if hardest.margin_vs_rival is not None and hardest.margin_vs_rival < 0:
        comments.append(
            (
                f"가장 불리한 조건은 {hardest.level_name}이고, 여기서는 target이 {abs(hardest.margin_vs_rival):.4f}만큼 뒤집힙니다. "
                f"따라서 현재 선택은 crowding / year-sensitive rival에 대해 경쟁 민감성이 있습니다."
            )
        )
    else:
        comments.append(
            (
                f"가장 불리한 조건에서도 strongest rival 대비 음수 margin이 나타나지 않아, 현재 선택은 경쟁에 비교적 안정적입니다."
            )
        )

    return comments[:3]


# -----------------------------------------------------------------------------
# 출력
# -----------------------------------------------------------------------------


def print_target_summary(target: Candidate) -> None:
    year = target.metadata.get("year")
    print("=" * 100)
    print("[TARGET]")
    print(f"chunk_id      : {target.chunk_id}")
    print(f"year          : {year}")
    print(f"title         : {target.title}")
    print(f"final_score   : {target.final_score:.6f}")
    print(f"dense_score   : {target.dense_score:.6f}")
    print(f"text_preview  : {make_preview(target.text)}")
    print()



def print_bucket_summary(buckets: Dict[str, List[Candidate]]) -> None:
    print("=" * 100)
    print("[BUCKET SUMMARY]")
    for key, xs in buckets.items():
        print(f"{key:20s}: {len(xs)}")
        for item in xs[:3]:
            print(
                f"  - {item.chunk_id} | year={item.metadata.get('year')} | "
                f"score={item.final_score:.6f} | title={item.title}"
            )
    print()



def print_result_table(results: Sequence[PoolResult]) -> None:
    print("=" * 100)
    print("[RESULT TABLE]")
    header = (
        f"{'level':26s} | {'pool':>4s} | {'raw_score':>10s} | {'support_p':>10s} | "
        f"{'rank':>4s} | {'top1':>4s} | {'margin':>10s} | strongest_rival"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        rival = r.strongest_rival_chunk_id or "-"
        margin_str = "-" if r.margin_vs_rival is None else f"{r.margin_vs_rival:10.6f}"
        print(
            f"{r.level_name:26s} | {r.pool_size:4d} | {r.target_raw_score:10.6f} | {r.target_support_prob:10.6f} | "
            f"{r.target_rank:4d} | {str(r.is_top1):>4s} | {margin_str:>10s} | {rival}"
        )
    print()


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="단일축 retrieval competition explanation prototype (v2)")
    parser.add_argument("--project-root", type=str, required=True, help="report 프로젝트 루트 경로")
    parser.add_argument("--query", type=str, required=True, help="검색 질의")
    parser.add_argument("--target-chunk-id", type=str, default=None, help="고정할 target chunk id. 없으면 top-1 자동 선택")
    parser.add_argument("--device", type=str, default="cpu", help="sentence-transformers device")
    parser.add_argument("--write-json", type=str, default=None, help="결과 JSON 저장 경로")
    parser.add_argument(
        "--valid-rank-threshold",
        type=int,
        default=3,
        help="명시적 target을 썼을 때 baseline 유효성으로 인정할 최대 rank",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    embed_dir = project_root / "embed_exports"
    search_demo_path = project_root / "run" / "run_search_demo.py"

    if not embed_dir.exists():
        raise FileNotFoundError(f"embed_exports 디렉터리를 찾을 수 없습니다: {embed_dir}")

    metadata_path = embed_dir / "metadata.jsonl"
    embeddings_path = embed_dir / "embeddings.npy"
    manifest_path = embed_dir / "manifest.json"

    metadatas = read_jsonl(metadata_path)
    embeddings = np.load(embeddings_path)
    manifest = load_manifest(manifest_path)
    search_demo = load_search_demo_module(search_demo_path)

    query_year, query_terms = search_demo.resolve_query_context(args.query, explicit_year=None)
    query_embedding = build_query_embedding_via_existing_code(
        search_demo_module=search_demo,
        query=args.query,
        manifest=manifest,
        device=args.device,
    )

    dense_scores = compute_dense_scores(
        query_embedding=query_embedding,
        embeddings=embeddings,
        normalize_embeddings=bool(manifest.get("normalize_embeddings", True)),
    )

    candidates = build_all_candidates(
        metadatas=metadatas,
        dense_scores=dense_scores,
        query_terms=query_terms,
        query_year=query_year,
        search_demo_module=search_demo,
    )

    explicit_target_was_given = args.target_chunk_id is not None
    target = choose_target(candidates, args.target_chunk_id)
    buckets = build_candidate_buckets(candidates=candidates, target=target, query_year=query_year)
    level_pools = build_level_pools(target=target, buckets=buckets)
    results = [evaluate_pool(level_name=name, pool=pool, target_chunk_id=target.chunk_id) for name, pool in level_pools]
    baseline_status = classify_baseline_status(
        baseline=results[0],
        explicit_target_was_given=explicit_target_was_given,
        valid_rank_threshold=args.valid_rank_threshold,
    )
    explanations = explain_results(
        results=results,
        baseline_status=baseline_status,
        explicit_target_was_given=explicit_target_was_given,
        valid_rank_threshold=args.valid_rank_threshold,
    )

    print_target_summary(target)
    print_bucket_summary(buckets)
    print_result_table(results)
    print("=" * 100)
    print("[BASELINE STATUS]")
    print(baseline_status)
    print()

    print("=" * 100)
    print("[AUTO EXPLANATION]")
    for idx, line in enumerate(explanations, start=1):
        print(f"{idx}. {line}")
    print()

    if args.write_json is not None:
        out_path = Path(args.write_json)
        payload = {
            "query": args.query,
            "query_terms": query_terms,
            "query_year": query_year,
            "target": {
                "chunk_id": target.chunk_id,
                "year": target.metadata.get("year"),
                "title": target.title,
                "final_score": target.final_score,
                "dense_score": target.dense_score,
            },
            "baseline_status": baseline_status,
            "valid_rank_threshold": args.valid_rank_threshold,
            "results": [
                {
                    "level_name": r.level_name,
                    "pool_size": r.pool_size,
                    "target_raw_score": r.target_raw_score,
                    "target_dense_score": r.target_dense_score,
                    "target_support_prob": r.target_support_prob,
                    "target_rank": r.target_rank,
                    "is_top1": r.is_top1,
                    "strongest_rival_chunk_id": r.strongest_rival_chunk_id,
                    "strongest_rival_score": r.strongest_rival_score,
                    "margin_vs_rival": r.margin_vs_rival,
                    "top3_chunk_ids": r.top3_chunk_ids,
                }
                for r in results
            ],
            "auto_explanation": explanations,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
