from langdetect import detect

def detect_language(text: str) -> str:
    """Detects the language of the given text."""
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "English"