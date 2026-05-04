"""
build_embed_text.py

Core function for building the text that will be embedded.
This is what embeddings will actually "see" during RAG retrieval.
"""


def build_embed_text(entry: dict) -> str:
    """
    Builds the embedding text from an entry.

    Args:
        entry: Dictionary containing the entry data

    Returns:
        A single string combining all relevant searchable fields
    """
    parts = [
        entry.get("term_arabic", ""),
        entry.get("term_arabizi", ""),
        entry.get("meaning", ""),
        entry.get("example", ""),
        entry.get("usage_context", ""),
    ]

    # Add proverb-specific fields
    if entry.get("type") == "proverb":
        parts.append(entry.get("literal_meaning", ""))
        parts.append(entry.get("real_meaning", ""))
        parts.append(entry.get("when_used", ""))

    # Add expression-specific fields
    if entry.get("type") == "expression":
        parts.append(entry.get("origin", ""))

    # Filter out empty strings and join
    return " ".join([p for p in parts if p])
