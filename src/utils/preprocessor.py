def preprocess_text(text: str) -> str:
    """Preprocess text to normalize it before summarization."""
    text = text.replace("\n", " ").strip()
    # Additional cleaning steps
    return text