"""Conflict detection module distinguishing evidentiary disagreement from missing evidence."""

import re

from app.schemas.retrieval import ScoredChunk
from app.schemas.verification import ClaimVerificationResult, ConflictItem


def _extract_clinical_concepts(text: str) -> set[str]:
    """Extract clinical entities and major keywords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    stopwords = {
        "what",
        "which",
        "when",
        "where",
        "how",
        "does",
        "are",
        "the",
        "and",
        "for",
        "with",
        "about",
        "from",
        "that",
        "this",
        "study",
        "trial",
        "results",
        "patients",
        "effects",
    }
    return {t for t in tokens if t not in stopwords}


class ConflictDetector:
    """Detects evidential disagreement across passages and isolates evidence gaps."""

    def detect_conflicts_and_gaps(
        self,
        query: str,
        candidates: list[ScoredChunk],
        verification_results: list[ClaimVerificationResult],
    ) -> tuple[list[ConflictItem], list[str]]:
        """Identify conflicts across evidence and distinguish from missing evidence."""
        conflicts: list[ConflictItem] = []
        evidence_gaps: list[str] = []

        if not candidates:
            evidence_gaps.append(
                f"No retrieved literature available to address clinical question: '{query}'."
            )
            return conflicts, evidence_gaps

        # 1. Identify Evidence Gaps (Missing Evidence)
        query_concepts = _extract_clinical_concepts(query)
        all_passage_text = " ".join([c.chunk.text.lower() for c in candidates])

        missing_query_concepts = [
            concept for concept in query_concepts if concept not in all_passage_text
        ]
        if missing_query_concepts:
            evidence_gaps.append(
                "Absence of evidence in retrieved literature for query concepts: "
                f"{', '.join(sorted(missing_query_concepts))}."
            )

        # Gaps from unverified claims
        unverified_claims = [r for r in verification_results if r.status == "unverified"]
        for uv in unverified_claims:
            evidence_gaps.append(
                f'Insufficient grounding passage found for claim: "{uv.claim_text}".'
            )

        # 2. Identify Evidentiary Disagreement (Contradictions between passages)
        # Check pairwise candidates for opposing direction on shared concepts
        pos_patterns = [
            r"\b(improved|improves|preserves|preserved|reduces|reduced|superior|effective)\b"
        ]
        neg_patterns = [
            (
                r"\b(failed to reduce|no significant difference|no benefit|"
                r"ineffective|increased risk|worsened|harmful)\b"
            )
        ]

        seen_conflict_pairs = set()

        for i, c1 in enumerate(candidates):
            t1 = c1.chunk.text.lower()
            tokens1 = _extract_clinical_concepts(t1)

            for j in range(i + 1, len(candidates)):
                c2 = candidates[j]
                t2 = c2.chunk.text.lower()
                tokens2 = _extract_clinical_concepts(t2)

                shared_concepts = tokens1.intersection(tokens2)
                if len(shared_concepts) < 2:
                    continue

                c1_pos = any(re.search(p, t1) for p in pos_patterns)
                c1_neg = any(re.search(p, t1) for p in neg_patterns)
                c2_pos = any(re.search(p, t2) for p in pos_patterns)
                c2_neg = any(re.search(p, t2) for p in neg_patterns)

                # Direct opposing outcome on shared clinical concepts
                if (c1_pos and c2_neg) or (c1_neg and c2_pos):
                    pair_key = tuple(sorted([c1.chunk.chunk_id, c2.chunk.chunk_id]))
                    if pair_key not in seen_conflict_pairs:
                        seen_conflict_pairs.add(pair_key)
                        concepts_str = ", ".join(list(shared_concepts)[:3])
                        conflicts.append(
                            ConflictItem(
                                description=(
                                    "Divergent clinical outcomes reported regarding "
                                    f"({concepts_str}) between sources {c1.chunk.source_id} "
                                    f"and {c2.chunk.source_id}."
                                ),
                                competing_chunk_ids=[c1.chunk.chunk_id, c2.chunk.chunk_id],
                                conflict_type="divergent_outcomes",
                            )
                        )

        # 3. Disagreement identified from Claim Verification
        contradicted_results = [r for r in verification_results if r.status == "contradicted"]
        for cr in contradicted_results:
            conflicts.append(
                ConflictItem(
                    description=(
                        f"Claim '{cr.claim_text}' contradicts cited evidence: "
                        f"{cr.contradiction_details or cr.rationale}"
                    ),
                    competing_chunk_ids=[cr.chunk_id] if cr.chunk_id else [],
                    conflict_type="direct_contradiction",
                )
            )

        return conflicts, evidence_gaps
