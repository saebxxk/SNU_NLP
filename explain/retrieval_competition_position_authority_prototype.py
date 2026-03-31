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
class RetrievalPoolResult:
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


@dataclass
class Reader3AxisResult:
    competition_level: str
    position_level: str
    authority_level: str
    context_size: int
    target_chunk_id: str
    target_reader_logit: float
    target_reader_support_prob: float
    target_reader_rank: int
    is_top1: bool
    strongest_rival_chunk_id: Optional[str]
    strongest_rival_reader_logit: Optional[float]
    margin_vs_rival: Optional[float]
    target_burial_score: float
    target_authority_conflict_score: float
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


def approx_token_len(text: str) -> int:
    tokens = str(text).split()
    return max(len(tokens), 1)


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
# 후보 분류 / retrieval pool 구성
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
# retrieval 단계 평가
# -----------------------------------------------------------------------------


def evaluate_retrieval_pool(level_name: str, pool: Sequence[Candidate], target_chunk_id: str) -> RetrievalPoolResult:
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

    return RetrievalPoolResult(
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



def classify_baseline_status(
    baseline: RetrievalPoolResult,
    explicit_target_was_given: bool,
    valid_rank_threshold: int,
) -> str:
    if not explicit_target_was_given:
        return "normal_competition"
    if baseline.target_rank > valid_rank_threshold:
        return "target_mismatch_diagnostic"
    return "normal_competition"


# -----------------------------------------------------------------------------
# reader proxy 단계: positional accessibility + authority conflict
# -----------------------------------------------------------------------------


def build_position_order(pool: Sequence[Candidate], target_chunk_id: str, position_level: str) -> List[Candidate]:
    ranked = sorted(pool, key=lambda x: x.final_score, reverse=True)
    target = None
    others: List[Candidate] = []
    for c in ranked:
        if c.chunk_id == target_chunk_id:
            target = c
        else:
            others.append(c)

    if target is None:
        raise ValueError(f"target이 position order 구성 pool에 없습니다: {target_chunk_id}")

    if position_level == "front":
        return [target] + others
    if position_level == "back":
        return others + [target]
    if position_level == "middle":
        mid = len(others) // 2
        return others[:mid] + [target] + others[mid:]

    raise ValueError(f"알 수 없는 position_level: {position_level}")



def compute_burial_scores(order: Sequence[Candidate]) -> List[float]:
    lengths = [approx_token_len(c.text) for c in order]
    total = sum(lengths)
    if total <= 0:
        total = len(order)
        lengths = [1 for _ in order]

    burial_scores: List[float] = []
    cursor = 0
    for L in lengths:
        center = cursor + (L / 2.0)
        p = center / total
        burial = 1.0 - 2.0 * abs(p - 0.5)
        burial = max(0.0, min(1.0, burial))
        burial_scores.append(float(burial))
        cursor += L
    return burial_scores



def candidate_year_match_score(c: Candidate, query_year: Optional[int]) -> float:
    if query_year is None:
        return 1.0
    cand_year = c.metadata.get("year")
    if cand_year is None:
        return 0.5
    return 1.0 if cand_year == query_year else 0.0



def candidate_lexical_alignment_score(c: Candidate) -> float:
    title_hit = float(c.debug_features.get("title_hit_ratio", 0.0))
    text_hit = float(c.debug_features.get("text_hit_ratio", 0.0))
    lexical = max(title_hit, text_hit)
    return max(0.0, min(1.0, lexical))



def compute_authority_conflict_scores(order: Sequence[Candidate], query_year: Optional[int]) -> List[float]:
    scores: List[float] = []
    for c in order:
        year_match = candidate_year_match_score(c, query_year)
        lexical_align = candidate_lexical_alignment_score(c)
        privileged_alignment = 0.65 * year_match + 0.35 * lexical_align
        conflict = 1.0 - privileged_alignment
        scores.append(float(max(0.0, min(1.0, conflict))))
    return scores



def compute_reader_proxy_logits(
    order: Sequence[Candidate],
    burial_scores: Sequence[float],
    authority_conflict_scores: Sequence[float],
    retrieval_weight: float,
    position_penalty: float,
    authority_penalty: float,
    score_source: str,
) -> List[float]:
    logits: List[float] = []
    for c, burial, auth_conflict in zip(order, burial_scores, authority_conflict_scores):
        base_score = c.final_score if score_source == "final" else c.dense_score
        logits.append(
            float(
                retrieval_weight * base_score
                - position_penalty * burial
                - authority_penalty * auth_conflict
            )
        )
    return logits



def evaluate_reader_3axis(
    competition_level: str,
    pool: Sequence[Candidate],
    target_chunk_id: str,
    position_level: str,
    authority_level: str,
    retrieval_weight: float,
    position_penalty: float,
    authority_penalty: float,
    score_source: str,
    query_year: Optional[int],
) -> Reader3AxisResult:
    order = build_position_order(pool, target_chunk_id=target_chunk_id, position_level=position_level)
    burial_scores = compute_burial_scores(order)
    authority_conflict_scores = compute_authority_conflict_scores(order, query_year=query_year)
    logits = compute_reader_proxy_logits(
        order=order,
        burial_scores=burial_scores,
        authority_conflict_scores=authority_conflict_scores,
        retrieval_weight=retrieval_weight,
        position_penalty=position_penalty,
        authority_penalty=authority_penalty,
        score_source=score_source,
    )
    probs = stable_softmax(logits)

    ranked_indices = sorted(range(len(order)), key=lambda i: logits[i], reverse=True)
    ranked = [order[i] for i in ranked_indices]
    ranked_logits = [logits[i] for i in ranked_indices]
    ranked_probs = [float(probs[i]) for i in ranked_indices]
    ranked_burial = [float(burial_scores[i]) for i in ranked_indices]
    ranked_conflict = [float(authority_conflict_scores[i]) for i in ranked_indices]

    target_idx = None
    for idx, c in enumerate(ranked):
        if c.chunk_id == target_chunk_id:
            target_idx = idx
            break
    if target_idx is None:
        raise ValueError(f"target이 reader rank에 없습니다: {target_chunk_id}")

    target = ranked[target_idx]
    target_logit = float(ranked_logits[target_idx])
    target_prob = float(ranked_probs[target_idx])
    target_rank = target_idx + 1
    target_burial = float(ranked_burial[target_idx])
    target_conflict = float(ranked_conflict[target_idx])

    rivals = [c for c in ranked if c.chunk_id != target_chunk_id]
    rival_logits = [lg for c, lg in zip(ranked, ranked_logits) if c.chunk_id != target_chunk_id]
    strongest_rival = rivals[0] if rivals else None
    strongest_rival_logit = rival_logits[0] if rival_logits else None
    margin = None
    if strongest_rival_logit is not None:
        margin = target_logit - strongest_rival_logit

    return Reader3AxisResult(
        competition_level=competition_level,
        position_level=position_level,
        authority_level=authority_level,
        context_size=len(pool),
        target_chunk_id=target.chunk_id,
        target_reader_logit=target_logit,
        target_reader_support_prob=target_prob,
        target_reader_rank=target_rank,
        is_top1=(target_rank == 1),
        strongest_rival_chunk_id=(strongest_rival.chunk_id if strongest_rival is not None else None),
        strongest_rival_reader_logit=(float(strongest_rival_logit) if strongest_rival_logit is not None else None),
        margin_vs_rival=(float(margin) if margin is not None else None),
        target_burial_score=target_burial,
        target_authority_conflict_score=target_conflict,
        top3_chunk_ids=[c.chunk_id for c in ranked[:3]],
    )


# -----------------------------------------------------------------------------
# 설명 생성
# -----------------------------------------------------------------------------


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



def explain_retrieval_results(
    results: Sequence[RetrievalPoolResult],
    baseline_status: str,
) -> List[str]:
    if not results:
        return ["retrieval 결과가 없습니다."]

    baseline = results[0]
    comments: List[str] = []
    if baseline_status == "target_mismatch_diagnostic":
        comments.append(
            f"baseline부터 target이 rank {baseline.target_rank}이므로, 현재 query에 대해 이 target은 경쟁 설명 대상이 아니라 target mismatch 후보입니다."
        )
        rival = baseline.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            f"baseline 최강 rival은 {rival}이고, margin은 {baseline.margin_vs_rival:.4f}입니다. 즉 경쟁이 커져서 무너진 것이 아니라 처음부터 query-target 정합성이 약했습니다."
        )
        comments.append(
            "같은 target을 유지한 competition 분석을 하려면 query를 고정해야 하고, query가 바뀌었다면 target도 다시 선택하는 것이 적절합니다."
        )
        return comments[:3]

    comments.append(
        f"retrieval baseline에서는 target이 rank {baseline.target_rank}이고, strongest rival 대비 margin은 {baseline.margin_vs_rival:.4f} ({margin_label(baseline.margin_vs_rival)})입니다."
    )

    reversal_levels = [r for r in results if not r.is_top1]
    if reversal_levels:
        first = reversal_levels[0]
        rival = first.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            f"retrieval top-1이 처음 깨지는 지점은 {first.level_name}이며, 이때 strongest rival은 {rival}, margin은 {first.margin_vs_rival:.4f}입니다."
        )
    else:
        comments.append("모든 retrieval competition level에서 target이 top-1을 유지했습니다.")

    hardest = min(results, key=lambda r: (r.margin_vs_rival if r.margin_vs_rival is not None else 0.0))
    if hardest.margin_vs_rival is not None and hardest.margin_vs_rival < 0:
        comments.append(
            f"가장 불리한 retrieval 조건은 {hardest.level_name}이고, 여기서는 target이 {abs(hardest.margin_vs_rival):.4f}만큼 뒤집힙니다."
        )
    else:
        comments.append("가장 불리한 retrieval 조건에서도 음수 margin이 나타나지 않아 retrieval 경쟁에는 비교적 안정적입니다.")

    return comments[:3]



def explain_position_results(
    reader_results: Sequence[Reader3AxisResult],
    baseline_status: str,
) -> List[str]:
    if baseline_status == "target_mismatch_diagnostic":
        return [
            "position 단계 설명은 생략합니다.",
            "이유는 baseline retrieval 자체가 target mismatch이므로, 현재 target을 reader context에서 앞/중간/뒤로 옮겨도 정상적인 같은-target 경쟁 설명이 되지 않기 때문입니다.",
            "먼저 query에 맞는 target을 다시 선택한 뒤 position 축을 보는 것이 적절합니다.",
        ]

    if not reader_results:
        return ["position 결과가 없습니다."]

    aligned = [r for r in reader_results if r.authority_level == "aligned"]
    grouped: Dict[Tuple[str, str], Reader3AxisResult] = {
        (r.competition_level, r.position_level): r for r in aligned
    }

    comments: List[str] = []
    front = grouped.get(("L0_weak_competition", "front"))
    middle = grouped.get(("L0_weak_competition", "middle"))
    if front and middle:
        drop = front.target_reader_support_prob - middle.target_reader_support_prob
        comments.append(
            f"reader proxy 기준 baseline(L0, authority=aligned)에서 target을 front에서 middle로 옮기면 support가 {drop:.4f} 감소합니다. 즉 기본적으로 middle burial penalty가 존재합니다."
        )
    else:
        comments.append("baseline front/middle position 비교를 위한 결과가 일부 누락되었습니다.")

    ordered_comp_levels = ["L0_weak_competition", "L1_near_rival", "L2_topical_crowding", "L3_year_sensitive_pressure"]
    first_reversal: Optional[Reader3AxisResult] = None
    for comp in ordered_comp_levels:
        for pos in ["front", "middle", "back"]:
            r = grouped.get((comp, pos))
            if r is not None and not r.is_top1:
                first_reversal = r
                break
        if first_reversal is not None:
            break

    if first_reversal is not None:
        rival = first_reversal.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            f"authority=aligned 기준 reader top-1이 처음 깨지는 지점은 competition={first_reversal.competition_level}, position={first_reversal.position_level}이며, strongest rival은 {rival}, margin은 {first_reversal.margin_vs_rival:.4f}입니다."
        )
    else:
        comments.append("authority=aligned 기준 모든 competition × position 조합에서 reader top-1을 유지했습니다.")

    hardest = min(aligned, key=lambda r: (r.margin_vs_rival if r.margin_vs_rival is not None else 0.0)) if aligned else None
    if hardest is not None and hardest.margin_vs_rival is not None and hardest.margin_vs_rival < 0:
        comments.append(
            f"authority=aligned 기준 가장 불리한 reader 조건은 competition={hardest.competition_level}, position={hardest.position_level}이고, 여기서는 target이 {abs(hardest.margin_vs_rival):.4f}만큼 뒤집힙니다."
        )
    else:
        comments.append("authority=aligned 기준 가장 불리한 reader 조건에서도 음수 margin이 나타나지 않아 position 변화에는 비교적 안정적입니다.")

    return comments[:3]



def explain_authority_results(
    reader_results: Sequence[Reader3AxisResult],
    baseline_status: str,
) -> List[str]:
    if baseline_status == "target_mismatch_diagnostic":
        return [
            "authority 단계 설명은 생략합니다.",
            "이유는 baseline retrieval이 이미 target mismatch이므로, privileged-source tension을 높여도 정상적인 같은-target authority 설명이 되지 않기 때문입니다.",
            "먼저 query에 맞는 target을 다시 선택한 뒤 authority 축을 보는 것이 적절합니다.",
        ]

    if not reader_results:
        return ["authority 결과가 없습니다."]

    grouped: Dict[Tuple[str, str, str], Reader3AxisResult] = {
        (r.competition_level, r.position_level, r.authority_level): r for r in reader_results
    }
    comments: List[str] = []

    aligned = grouped.get(("L0_weak_competition", "front", "aligned"))
    strong = grouped.get(("L0_weak_competition", "front", "strong"))
    if aligned and strong:
        drop = aligned.target_reader_support_prob - strong.target_reader_support_prob
        comments.append(
            f"reader proxy 기준 baseline(L0, front)에서 authority를 aligned에서 strong으로 높이면 target support가 {drop:.4f} 감소합니다. 즉 explicit year/topic 우선순위가 강해질수록 현재 target은 불리해집니다."
        )
    else:
        comments.append("baseline authority shift(aligned→strong) 비교를 위한 결과가 일부 누락되었습니다.")

    ordered_comp_levels = ["L0_weak_competition", "L1_near_rival", "L2_topical_crowding", "L3_year_sensitive_pressure"]
    ordered_auth = ["aligned", "weak", "strong"]
    first_reversal: Optional[Reader3AxisResult] = None
    for comp in ordered_comp_levels:
        for auth in ordered_auth:
            r = grouped.get((comp, "front", auth))
            if r is not None and not r.is_top1:
                first_reversal = r
                break
        if first_reversal is not None:
            break

    if first_reversal is not None:
        rival = first_reversal.strongest_rival_chunk_id or "unknown_rival"
        comments.append(
            f"position=front 기준 authority-aware reader top-1이 처음 깨지는 지점은 competition={first_reversal.competition_level}, authority={first_reversal.authority_level}이며, strongest rival은 {rival}, margin은 {first_reversal.margin_vs_rival:.4f}입니다."
        )
    else:
        comments.append("position=front 기준 모든 competition × authority 조합에서 reader top-1을 유지했습니다.")

    strongest_authority = [r for r in reader_results if r.authority_level == "strong"]
    hardest = min(strongest_authority, key=lambda r: (r.margin_vs_rival if r.margin_vs_rival is not None else 0.0)) if strongest_authority else None
    if hardest is not None and hardest.margin_vs_rival is not None and hardest.margin_vs_rival < 0:
        comments.append(
            f"authority=strong 기준 가장 불리한 reader 조건은 competition={hardest.competition_level}, position={hardest.position_level}이고, 여기서는 target이 {abs(hardest.margin_vs_rival):.4f}만큼 뒤집힙니다."
        )
    else:
        comments.append("authority=strong 기준에도 음수 margin이 나타나지 않아 privileged-source tension에는 비교적 안정적입니다.")

    return comments[:3]


# -----------------------------------------------------------------------------
# 출력
# -----------------------------------------------------------------------------


def print_target_summary(target: Candidate) -> None:
    year = target.metadata.get("year")
    print("=" * 110)
    print("[TARGET]")
    print(f"chunk_id      : {target.chunk_id}")
    print(f"year          : {year}")
    print(f"title         : {target.title}")
    print(f"final_score   : {target.final_score:.6f}")
    print(f"dense_score   : {target.dense_score:.6f}")
    print(f"text_preview  : {make_preview(target.text)}")
    print()



def print_bucket_summary(buckets: Dict[str, List[Candidate]]) -> None:
    print("=" * 110)
    print("[BUCKET SUMMARY]")
    for key, xs in buckets.items():
        print(f"{key:20s}: {len(xs)}")
        for item in xs[:3]:
            print(
                f"  - {item.chunk_id} | year={item.metadata.get('year')} | "
                f"score={item.final_score:.6f} | title={item.title}"
            )
    print()



def print_retrieval_table(results: Sequence[RetrievalPoolResult]) -> None:
    print("=" * 110)
    print("[RETRIEVAL RESULT TABLE]")
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



def print_reader_table(results: Sequence[Reader3AxisResult]) -> None:
    print("=" * 110)
    print("[READER 3-AXIS RESULT TABLE]")
    header = (
        f"{'competition':24s} | {'position':8s} | {'authority':8s} | {'ctx':>3s} | {'logit':>10s} | {'support_p':>10s} | "
        f"{'rank':>4s} | {'top1':>4s} | {'burial':>8s} | {'auth_cf':>8s} | {'margin':>10s} | strongest_rival"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        rival = r.strongest_rival_chunk_id or "-"
        margin_str = "-" if r.margin_vs_rival is None else f"{r.margin_vs_rival:10.6f}"
        print(
            f"{r.competition_level:24s} | {r.position_level:8s} | {r.authority_level:8s} | {r.context_size:3d} | "
            f"{r.target_reader_logit:10.6f} | {r.target_reader_support_prob:10.6f} | {r.target_reader_rank:4d} | {str(r.is_top1):>4s} | "
            f"{r.target_burial_score:8.4f} | {r.target_authority_conflict_score:8.4f} | {margin_str:>10s} | {rival}"
        )
    print()


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3축 prototype: retrieval competition × positional accessibility × authority conflict (reader proxy)")
    parser.add_argument("--project-root", type=str, required=True, help="report 프로젝트 루트 경로")
    parser.add_argument("--query", type=str, required=True, help="검색 질의")
    parser.add_argument("--target-chunk-id", type=str, default=None, help="고정할 target chunk id. 없으면 top-1 자동 선택")
    parser.add_argument("--device", type=str, default="cpu", help="sentence-transformers device")
    parser.add_argument("--write-json", type=str, default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--valid-rank-threshold", type=int, default=3, help="명시적 target baseline 유효성으로 인정할 최대 rank")
    parser.add_argument("--reader-retrieval-weight", type=float, default=1.0, help="reader proxy에서 retrieval 점수 가중치")
    parser.add_argument("--reader-position-penalty", type=float, default=0.35, help="reader proxy에서 middle burial penalty 세기")
    parser.add_argument("--authority-penalty-weak", type=float, default=0.15, help="authority=weak penalty")
    parser.add_argument("--authority-penalty-strong", type=float, default=0.35, help="authority=strong penalty")
    parser.add_argument("--reader-score-source", choices=["final", "dense"], default="final", help="reader proxy의 base score source")
    parser.add_argument("--skip-reader-if-mismatch", action="store_true", help="baseline mismatch이면 reader 3축 계산 생략")
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

    retrieval_results = [
        evaluate_retrieval_pool(level_name=name, pool=pool, target_chunk_id=target.chunk_id)
        for name, pool in level_pools
    ]

    baseline_status = classify_baseline_status(
        baseline=retrieval_results[0],
        explicit_target_was_given=explicit_target_was_given,
        valid_rank_threshold=args.valid_rank_threshold,
    )

    authority_penalty_map = {
        "aligned": 0.0,
        "weak": args.authority_penalty_weak,
        "strong": args.authority_penalty_strong,
    }

    reader_results: List[Reader3AxisResult] = []
    if not (args.skip_reader_if_mismatch and baseline_status == "target_mismatch_diagnostic"):
        for level_name, pool in level_pools:
            for position_level in ["front", "middle", "back"]:
                for authority_level in ["aligned", "weak", "strong"]:
                    reader_results.append(
                        evaluate_reader_3axis(
                            competition_level=level_name,
                            pool=pool,
                            target_chunk_id=target.chunk_id,
                            position_level=position_level,
                            authority_level=authority_level,
                            retrieval_weight=args.reader_retrieval_weight,
                            position_penalty=args.reader_position_penalty,
                            authority_penalty=authority_penalty_map[authority_level],
                            score_source=args.reader_score_source,
                            query_year=query_year,
                        )
                    )

    retrieval_explanations = explain_retrieval_results(retrieval_results, baseline_status=baseline_status)
    position_explanations = explain_position_results(reader_results, baseline_status=baseline_status)
    authority_explanations = explain_authority_results(reader_results, baseline_status=baseline_status)

    print_target_summary(target)
    print_bucket_summary(buckets)
    print_retrieval_table(retrieval_results)
    print("=" * 110)
    print("[BASELINE STATUS]")
    print(baseline_status)
    print()
    if reader_results:
        print_reader_table(reader_results)

    print("=" * 110)
    print("[AUTO EXPLANATION: RETRIEVAL]")
    for idx, line in enumerate(retrieval_explanations, start=1):
        print(f"{idx}. {line}")
    print()

    print("=" * 110)
    print("[AUTO EXPLANATION: POSITION]")
    for idx, line in enumerate(position_explanations, start=1):
        print(f"{idx}. {line}")
    print()

    print("=" * 110)
    print("[AUTO EXPLANATION: AUTHORITY]")
    for idx, line in enumerate(authority_explanations, start=1):
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
            "reader_proxy": {
                "retrieval_weight": args.reader_retrieval_weight,
                "position_penalty": args.reader_position_penalty,
                "authority_penalty_weak": args.authority_penalty_weak,
                "authority_penalty_strong": args.authority_penalty_strong,
                "score_source": args.reader_score_source,
                "note": "reader-stage actual LLM score가 아니라 retrieval score, burial penalty, authority conflict penalty를 결합한 proxy입니다.",
            },
            "retrieval_results": [
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
                for r in retrieval_results
            ],
            "reader_results": [
                {
                    "competition_level": r.competition_level,
                    "position_level": r.position_level,
                    "authority_level": r.authority_level,
                    "context_size": r.context_size,
                    "target_reader_logit": r.target_reader_logit,
                    "target_reader_support_prob": r.target_reader_support_prob,
                    "target_reader_rank": r.target_reader_rank,
                    "is_top1": r.is_top1,
                    "strongest_rival_chunk_id": r.strongest_rival_chunk_id,
                    "strongest_rival_reader_logit": r.strongest_rival_reader_logit,
                    "margin_vs_rival": r.margin_vs_rival,
                    "target_burial_score": r.target_burial_score,
                    "target_authority_conflict_score": r.target_authority_conflict_score,
                    "top3_chunk_ids": r.top3_chunk_ids,
                }
                for r in reader_results
            ],
            "auto_explanation": {
                "retrieval": retrieval_explanations,
                "position": position_explanations,
                "authority": authority_explanations,
            },
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
