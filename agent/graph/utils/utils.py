import re


def remove_thinking_tags(text: str) -> str:
    # Regex pattern to match <thinking> tags and their contents
    cleaned_text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return cleaned_text
