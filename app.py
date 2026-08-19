"""
Backend ב-FastAPI שעוטף את שלושת הגרפים.

Endpoints:
  POST /ted/start        -> מתחיל graph הרצאת TED
  POST /ted/resume       -> ממשיך אחרי interrupt (אישור אנושי)
  POST /podcast/start
  POST /podcast/resume
  POST /audio/start
  POST /audio/resume
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command

from ted_graph import build_ted_graph
from podcast_graph import build_podcast_graph
from audio_graph import build_audio_graph

load_dotenv()

app = FastAPI(title="Studio API")

ted_graph = build_ted_graph()
podcast_graph = build_podcast_graph()
audio_graph = build_audio_graph()


class TedStartRequest(BaseModel):
    thread_id: str
    source_text: str


class PodcastStartRequest(BaseModel):
    thread_id: str
    topic: str


class AudioStartRequest(BaseModel):
    thread_id: str
    script: str
    mode: str = "single_speaker"


class ResumeRequest(BaseModel):
    thread_id: str
    approve: bool


def _serialize(result: dict) -> dict:
    """ממיר את תוצאת הגרף לתשובה - חושף interrupt אם הגרף עצר."""
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"]
        payload = interrupt_obj[0].value if isinstance(interrupt_obj, list) else interrupt_obj
        return {"status": "waiting_for_approval", "payload": payload}
    return {"status": "done", "result": result}


@app.post("/ted/start")
def ted_start(req: TedStartRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = ted_graph.invoke(
        {"source_text": req.source_text, "revision_count": 0}, config=config
    )
    return _serialize(result)


@app.post("/ted/resume")
def ted_resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = ted_graph.invoke(Command(resume={"approve": req.approve}), config=config)
    return _serialize(result)


@app.post("/podcast/start")
def podcast_start(req: PodcastStartRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = podcast_graph.invoke(
        {"topic": req.topic, "segment_drafts": [], "revision_count": 0}, config=config
    )
    return _serialize(result)


@app.post("/podcast/resume")
def podcast_resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = podcast_graph.invoke(Command(resume={"approve": req.approve}), config=config)
    return _serialize(result)


@app.post("/audio/start")
def audio_start(req: AudioStartRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = audio_graph.invoke(
        {"script": req.script, "mode": req.mode}, config=config
    )
    return _serialize(result)


@app.post("/audio/resume")
def audio_resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = audio_graph.invoke(Command(resume={"approve": req.approve}), config=config)
    return _serialize(result)


@app.get("/health")
def health():
    return {"status": "ok"}
