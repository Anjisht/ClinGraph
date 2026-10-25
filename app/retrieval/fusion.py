"""Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval results."""

from app.schemas.retrieval import ScoredChunk


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    graph_results: list[ScoredChunk] | None = None,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    graph_weight: float = 0.8,
    fused_top_k: int = 15,
) -> list[ScoredChunk]:
    """Fuse dense, sparse, and graph ranked lists using Reciprocal Rank Fusion."""
    combined_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}
    method_scores_map: dict[str, dict[str, float]] = {}

    # Score dense items
    for item in dense_results:
        cid = item.chunk.chunk_id
        chunk_map[cid] = item
        score = dense_weight / (rrf_k + item.rank)
        combined_scores[cid] = combined_scores.get(cid, 0.0) + score

        if cid not in method_scores_map:
            method_scores_map[cid] = {}
        method_scores_map[cid]["dense"] = item.score
        method_scores_map[cid]["dense_rank"] = float(item.rank)

    # Score sparse items
    for item in sparse_results:
        cid = item.chunk.chunk_id
        if cid not in chunk_map:
            chunk_map[cid] = item
        score = sparse_weight / (rrf_k + item.rank)
        combined_scores[cid] = combined_scores.get(cid, 0.0) + score

        if cid not in method_scores_map:
            method_scores_map[cid] = {}
        method_scores_map[cid]["sparse"] = item.score
        method_scores_map[cid]["sparse_rank"] = float(item.rank)

    # Score graph items if provided
    if graph_results:
        for item in graph_results:
            cid = item.chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = item
            score = graph_weight / (rrf_k + item.rank)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + score

            if cid not in method_scores_map:
                method_scores_map[cid] = {}
            method_scores_map[cid]["graph"] = item.score
            method_scores_map[cid]["graph_rank"] = float(item.rank)

    # Sort descending by fused score
    sorted_chunk_ids = sorted(
        combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True
    )

    fused_candidates: list[ScoredChunk] = []
    for rank_idx, cid in enumerate(sorted_chunk_ids[:fused_top_k], start=1):
        base_item = chunk_map[cid]
        fused_scores = method_scores_map[cid]
        fused_scores["rrf"] = combined_scores[cid]

        fused_candidates.append(
            ScoredChunk(
                chunk=base_item.chunk,
                score=combined_scores[cid],
                rank=rank_idx,
                method="fused",
                method_scores=fused_scores,
            )
        )

    return fused_candidates
