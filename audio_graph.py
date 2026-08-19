"""
audio_graph.py — graph אודיו נפרד.

מקבל תסריט מאושר (מ-ted_graph או podcast_graph) ומייצר אודיו:
- TED: דובר יחיד
- פודקאסט: רב-דוברים (מפצל לפי קטעים/דוברים ומרכיב קובץ אחד)

יש לו זרימת אישור אנושי *עצמאית* משלו - כלומר גם אחרי שהתסריט
אושר, ניתן לדחות/לאשר את תוצר האודיו הסופי בנפרד.
"""

from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from tts import get_tts_provider


class AudioState(TypedDict):
    script: str
    mode: str  # "single_speaker" | "multi_speaker"
    speakers: Optional[list[dict]]  # למצב multi_speaker: [{"name": "...", "text": "..."}]
    audio_bytes: bytes
    audio_approved: bool


def synthesize_audio(state: AudioState) -> dict:
    provider = get_tts_provider()

    if state["mode"] == "multi_speaker" and state.get("speakers"):
        chunks = [
            provider.synthesize(s["text"], voice=s.get("voice", "default"))
            for s in state["speakers"]
        ]
        audio = b"".join(chunks)
    else:
        audio = provider.synthesize(state["script"], voice="default")

    return {"audio_bytes": audio}


def audio_human_approval(state: AudioState) -> dict:
    decision = interrupt({
        "question": "האודיו מוכן. לאשר את הגרסה הסופית?",
        "audio_size_bytes": len(state["audio_bytes"]),
    })
    return {"audio_approved": bool(decision.get("approve", False))}


def route_after_approval(state: AudioState) -> str:
    # אם נדחה - חוזרים להפקה מחדש (למשל עם קול/פרמטרים אחרים)
    return END if state["audio_approved"] else "synthesize_audio"


def build_audio_graph():
    graph_builder = StateGraph(AudioState)
    graph_builder.add_node("synthesize_audio", synthesize_audio)
    graph_builder.add_node("audio_human_approval", audio_human_approval)

    graph_builder.add_edge(START, "synthesize_audio")
    graph_builder.add_edge("synthesize_audio", "audio_human_approval")
    graph_builder.add_conditional_edges(
        "audio_human_approval", route_after_approval, ["synthesize_audio", END]
    )

    checkpointer = SqliteSaver.from_conn_string("audio_checkpoints.sqlite")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_audio_graph()
    config = {"configurable": {"thread_id": "audio-demo-1"}}
    result = graph.invoke(
        {"script": "זהו תסריט לדוגמה.", "mode": "single_speaker"}, config=config
    )
    print(result.get("__interrupt__", "audio ready"))
