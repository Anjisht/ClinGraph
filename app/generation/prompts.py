"""Prompt templates for evidence-grounded biomedical synthesis."""

from app.schemas.retrieval import ScoredChunk

SYSTEM_PROMPT = """You are ClinGraph, a biomedical research evidence synthesis assistant.
Your role is to summarize research literature strictly grounded in the provided evidence passages.

CRITICAL SAFETY & GROUNDING RULES:
1. Medical Safety: You are an educational and research exploration tool, NOT a diagnostic
   medical device. Never recommend or prescribe treatments.
2. Grounding Invariant: Base all claims EXCLUSIVELY on the provided passages.
   Do NOT use ungrounded external knowledge.
3. Provenance Requirement: Every material claim MUST explicitly cite one or more
   chunk IDs from the provided evidence.
4. Abstention Rule: If the provided evidence is absent, insufficient, or contradictory,
   state this clearly and set abstained=true.
5. Strict JSON Output: Output MUST be a valid JSON object matching the requested schema.
   Do not include markdown code block backticks."""


def build_synthesis_prompt(query: str, candidates: list[ScoredChunk]) -> str:
    """Format query and retrieved evidence passages into synthesis prompt."""
    passages_text = []
    for c in candidates:
        ch = c.chunk
        header = (
            f"--- [CHUNK_ID: {ch.chunk_id}] "
            f"[SOURCE_ID: {ch.source_id or 'Unknown'}] "
            f"[TITLE: {ch.document_title or 'Unknown'}] ---"
        )
        passages_text.append(f"{header}\n{ch.text}\n")

    context_str = "\n".join(passages_text) if passages_text else "No relevant evidence retrieved."

    prompt = f"""RESEARCH QUESTION: {query}

AVAILABLE EVIDENCE PASSAGES:
{context_str}

TASK:
Synthesize an evidence summary answering the research question based ONLY on the passages above.
Output a JSON object with this exact structure:
{{
  "summary": "High-level summary of findings",
  "claims": [
    {{
      "claim_text": "Atomic statement grounded in evidence",
      "status": "supported",
      "chunk_ids": ["CHUNK_ID_FROM_CONTEXT"],
      "limitations": []
    }}
  ],
  "conflicts": [],
  "evidence_gaps": [],
  "abstained": false
}}

Remember: Every claim MUST cite real CHUNK_IDs from the passages above.
If no evidence answers the question, set abstained=true."""
    return prompt
