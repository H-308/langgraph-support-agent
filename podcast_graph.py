"""
שלב ד' — Graph פודקאסט.

ארכיטקטורה מנוגדת ל-TED (שהיה לינארי):
- מחקר מקבילי (fan-out) על כל קטע באמצעות Send
- כתיבה לפי קטעים (write_segment)
- תקציב מילים דינמי + clamp בקוד (לא מסתמכים על המודל)
- לולאת critique/revise על הפרק *השלם* (אחרי איחוד הקטעים)
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic

from schemas import SegmentPlan, CritiqueResult
from word_budget import count_words, clamp_words, within_budget

TOTAL_TARGET_WORDS = 3000
MAX_REVISIONS = 3
TOLERANCE = 200


class PodcastState(TypedDict):
    topic: str
    segments_plan: list[dict]
    # reducer operator.add - כל fan-out branch מוסיף לרשימה, לא דורס
    segment_drafts: Annotated[list, operator.add]
    episode_script: str
    critique: Optional[dict]
    revision_count: int
    approved: bool


class SegmentState(TypedDict):
    segment: dict


llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.4)
critic_llm = llm.with_structured_output(CritiqueResult)


def plan_segments(state: PodcastState) -> dict:
    # תקציב מילים דינמי: מחלקים את סה"כ המילים בין קטעים באופן שווה
    # (בפרויקט אמיתי אפשר לתת משקל שונה לכל קטע)
    topics = [
        ("פתיחה וניסוח הבעיה", 0.15),
        ("הצגת הנתונים / המחקר", 0.35),
        ("דוגמאות מהעולם האמיתי", 0.30),
        ("סיכום ומסקנות", 0.20),
    ]
    segments = [
        {
            "title": title,
            "topic": f"{title} בהקשר של: {state['topic']}",
            "target_words": int(TOTAL_TARGET_WORDS * weight),
        }
        for title, weight in topics
    ]
    return {"segments_plan": segments}


def fan_out_research(state: PodcastState) -> list[Send]:
    # Send - מריץ את research_and_write_segment פעם אחת לכל קטע, במקביל
    return [
        Send("research_and_write_segment", {"segment": seg})
        for seg in state["segments_plan"]
    ]


def research_and_write_segment(state: SegmentState) -> dict:
    seg = state["segment"]
    target = seg["target_words"]

    # מחקר (בפועל: אפשר לחבר web_search/RAG אמיתי כאן)
    research_prompt = f"תני 3-4 נקודות עובדתיות רלוונטיות ל: {seg['topic']}"
    research = llm.invoke(research_prompt).content

    write_prompt = (
        f"כתבי קטע פודקאסט בעברית בערך {target} מילים, בסגנון שיחתי, "
        f"על: {seg['topic']}\n\nרקע/מחקר:\n{research}"
    )
    draft = llm.invoke(write_prompt).content

    # clamp בקוד - לא מסתמכים על המודל שיעצור בזמן
    draft = clamp_words(draft, min_words=0, max_words=target + TOLERANCE)

    return {"segment_drafts": [{"title": seg["title"], "text": draft}]}


def combine_segments(state: PodcastState) -> dict:
    # מרכיבים לפי סדר segments_plan (drafts יכולים לחזור בסדר לא מובטח)
    order = [s["title"] for s in state["segments_plan"]]
    drafts_by_title = {d["title"]: d["text"] for d in state["segment_drafts"]}
    episode = "\n\n".join(
        f"## {title}\n{drafts_by_title[title]}" for title in order if title in drafts_by_title
    )
    return {"episode_script": episode}


def critique_episode(state: PodcastState) -> dict:
    n_words = count_words(state["episode_script"])
    length_ok = within_budget(
        state["episode_script"], TOTAL_TARGET_WORDS - TOLERANCE, TOTAL_TARGET_WORDS + TOLERANCE
    )

    result: CritiqueResult = critic_llm.invoke(
        f"בקרי את פרק הפודקאסט השלם. יעד: {TOTAL_TARGET_WORDS} מילים "
        f"(בפועל: {n_words}, {'בתקציב' if length_ok else 'חורג'}). "
        f"בדקי זרימה בין קטעים, חזרות מיותרות, ועקביות טון.\n\n{state['episode_script']}"
    )

    approved = result.approved and length_ok
    issues = list(result.issues)
    if not length_ok:
        issues.append(f"אורך כולל ({n_words}) חורג מהתקציב")

    return {
        "critique": {**result.model_dump(), "approved": approved, "issues": issues},
        "revision_count": state.get("revision_count", 0) + 1,
    }


def revise_episode(state: PodcastState) -> dict:
    critique = state["critique"]
    prompt = (
        f"תקני את פרק הפודקאסט לפי המשוב, שמרי על אורך קרוב ל-{TOTAL_TARGET_WORDS} מילים.\n\n"
        f"משוב: {critique['feedback']}\nבעיות: {', '.join(critique['issues'])}\n\n"
        f"פרק נוכחי:\n{state['episode_script']}"
    )
    response = llm.invoke(prompt)
    script = clamp_words(response.content, min_words=0, max_words=TOTAL_TARGET_WORDS + TOLERANCE)
    return {"episode_script": script}


def human_approval(state: PodcastState) -> dict:
    decision = interrupt({
        "question": "פרק הפודקאסט מוכן. לאשר להמשך להפקת אודיו?",
        "episode_script": state["episode_script"],
        "word_count": count_words(state["episode_script"]),
    })
    return {"approved": bool(decision.get("approve", False))}


def route_after_critique(state: PodcastState) -> str:
    if state["critique"]["approved"] or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "human_approval"
    return "revise_episode"


def build_podcast_graph():
    graph_builder = StateGraph(PodcastState)
    graph_builder.add_node("plan_segments", plan_segments)
    graph_builder.add_node("research_and_write_segment", research_and_write_segment)
    graph_builder.add_node("combine_segments", combine_segments)
    graph_builder.add_node("critique_episode", critique_episode)
    graph_builder.add_node("revise_episode", revise_episode)
    graph_builder.add_node("human_approval", human_approval)

    graph_builder.add_edge(START, "plan_segments")
    # fan-out: plan_segments -> N x research_and_write_segment (מקבילי)
    graph_builder.add_conditional_edges(
        "plan_segments", fan_out_research, ["research_and_write_segment"]
    )
    graph_builder.add_edge("research_and_write_segment", "combine_segments")
    graph_builder.add_edge("combine_segments", "critique_episode")
    graph_builder.add_conditional_edges(
        "critique_episode", route_after_critique, ["human_approval", "revise_episode"]
    )
    graph_builder.add_edge("revise_episode", "critique_episode")
    graph_builder.add_edge("human_approval", END)

    checkpointer = SqliteSaver.from_conn_string("podcast_checkpoints.sqlite")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_podcast_graph()
    config = {"configurable": {"thread_id": "podcast-demo-1"}}
    result = graph.invoke(
        {"topic": "קבלת החלטות תחת אי-ודאות", "segment_drafts": [], "revision_count": 0},
        config=config,
    )
    if "__interrupt__" in result:
        print("הגרף עצר לאישור אנושי.")
    else:
        print(result["episode_script"])
