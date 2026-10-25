"""Token usage and operational cost estimation utilities.

Note: Pricing figures are benchmark estimates based on common provider tiers,
clearly labeled as estimates per README requirements.
"""

# Price estimates in USD per 1,000 tokens
MODEL_PRICING_ESTIMATES: dict[str, dict[str, float]] = {
    "small": {
        "input_per_1k": 0.0005,  # $0.50 per 1M tokens
        "output_per_1k": 0.0015,  # $1.50 per 1M tokens
    },
    "large": {
        "input_per_1k": 0.0030,  # $3.00 per 1M tokens
        "output_per_1k": 0.0120,  # $12.00 per 1M tokens
    },
    "cached": {
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
    },
}


def estimate_token_count(text: str) -> int:
    """Provide a deterministic approximation of token length for text.

    Uses standard ~4 characters per token heuristic when a tokenizer is absent.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(
    profile: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate estimated request cost in USD based on model profile and token counts."""
    tier = MODEL_PRICING_ESTIMATES.get(profile, MODEL_PRICING_ESTIMATES["small"])
    cost = (input_tokens / 1000.0) * tier["input_per_1k"] + (output_tokens / 1000.0) * tier[
        "output_per_1k"
    ]
    return round(cost, 6)


def estimate_cost_per_thousand(avg_cost_per_request: float) -> float:
    """Project average cost per 1,000 requests in USD."""
    return round(avg_cost_per_request * 1000.0, 4)
