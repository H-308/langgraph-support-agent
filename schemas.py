"""
סכמות (Pydantic models) משותפות לגרפים.
"""

from pydantic import BaseModel, Field
from typing import List


class TalkBrief(BaseModel):
    """תקציר להרצאת TED - תוצר של plan_talk."""
    topic: str = Field(description="נושא ההרצאה")
    audience: str = Field(description="קהל היעד")
    goal: str = Field(description="המסר המרכזי שההרצאה צריכה להעביר")
    key_points: List[str] = Field(description="3-5 נקודות מפתח שההרצאה תעבור")
    target_word_count: int = Field(default=1500, description="יעד מספר מילים לתסריט")


class CritiqueResult(BaseModel):
    """תוצר של critique_script / critique_episode."""
    approved: bool = Field(description="האם התסריט מאושר להמשך, ללא צורך בתיקון נוסף")
    feedback: str = Field(description="משוב כללי לכותב")
    issues: List[str] = Field(default_factory=list, description="רשימת בעיות ספציפיות לתיקון")


class SegmentPlan(BaseModel):
    """תכנון קטע בודד בפודקאסט."""
    title: str
    topic: str
    target_words: int
