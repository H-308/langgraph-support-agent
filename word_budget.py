"""
אכיפת אורך תסריט בקוד - לא בפרומפט.

הרעיון: אסור לסמוך על המודל ש"יבטיח" אורך מסוים בפרומפט בלבד
(מודלים חורגים מזה כל הזמן). לכן סופרים מילים בפועל בפייתון,
ומחליטים תוכניתית אם צריך לקצר/להאריך.
"""


def count_words(text: str) -> int:
    """סופר מילים בטקסט (חלוקה לפי רווחים - מספיק לעברית/אנגלית)."""
    return len(text.split())


def clamp_words(text: str, min_words: int, max_words: int) -> str:
    """חותך טקסט שחורג מהמקסימום. לא מוסיף מילים אם קצר מדי (זה תפקיד ה-critique/revise loop)."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def within_budget(text: str, min_words: int, max_words: int) -> bool:
    n = count_words(text)
    return min_words <= n <= max_words
