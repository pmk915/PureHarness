def approximate_text_tokens(text: str) -> int:
    """Apply PureHarness's provider-neutral character heuristic."""
    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)
