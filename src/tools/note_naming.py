import re
from typing import Optional

TITLE_PATTERN = re.compile(r"^\*\*Название:\*\*\s*(.+?)\s*$", re.MULTILINE)
INVALID_CHARS = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
DEFAULT_SLUG = "note"
MAX_TITLE_SLUG_LEN = 80


def extract_note_title(text: str) -> Optional[str]:
    match = TITLE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def slugify_title(title: str, max_length: int = MAX_TITLE_SLUG_LEN) -> str:
    slug = title.strip()
    slug = re.sub(r"\s+", " ", slug)
    slug = slug.replace(" ", "_")
    for ch in INVALID_CHARS:
        slug = slug.replace(ch, "")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug or DEFAULT_SLUG


def build_note_filename(note_id: str, text: str) -> str:
    title = extract_note_title(text)
    slug = slugify_title(title) if title else DEFAULT_SLUG
    return f"{slug}_{note_id}.md"
