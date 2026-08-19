"""
שכבת TTS (Text-To-Speech) מאחורי Protocol.

שימוש ב-Protocol (ולא ABC) מאפשר "structural typing" - כל מחלקה
שמממשת synthesize() עם החתימה הנכונה מתאימה, בלי ירושה מפורשת.
זה מאפשר להחליף ספק (ElevenLabs, מוק לבדיקות, ספק אחר בעתיד)
בלי לשנות קוד שמשתמש ב-TTSProvider.
"""

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, text: str, voice: str = "default") -> bytes:
        """ממיר טקסט לאודיו (bytes של קובץ שמע)."""
        ...


class ElevenLabsTTSProvider:
    """ספק TTS אמיתי דרך ElevenLabs API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("חסר ELEVENLABS_API_KEY")

    def synthesize(self, text: str, voice: str = "default") -> bytes:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self.api_key)
        audio_stream = client.text_to_speech.convert(
            voice_id=voice,
            text=text,
            model_id="eleven_multilingual_v2",
        )
        return b"".join(chunk for chunk in audio_stream)


class MockTTSProvider:
    """ספק מדומה - לפיתוח/בדיקות בלי מפתח API אמיתי או תקציב."""

    def synthesize(self, text: str, voice: str = "default") -> bytes:
        # מחזיר "אודיו מזויף" - מאפשר לבדוק את הזרימה המלאה של הגרף
        # בלי לקרוא בפועל לספק חיצוני בתשלום.
        placeholder = f"[MOCK AUDIO | voice={voice} | {len(text.split())} words]".encode()
        return placeholder


def get_tts_provider() -> TTSProvider:
    """Factory - בוחר ספק אמיתי אם יש מפתח, אחרת מוק."""
    if os.getenv("ELEVENLABS_API_KEY"):
        return ElevenLabsTTSProvider()
    return MockTTSProvider()
