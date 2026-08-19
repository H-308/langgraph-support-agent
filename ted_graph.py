"""
שלב ג' — Graph הרצאת TED.

מבנה לינארי:
    plan_talk -> gather_context -> write_talk -> critique_script
       -> (מאושר?) -> human_approval (interrupt) -> END
       -> (לא מאושר?) -> revise -> critique_script (לולאה)

עקרונות מרכזיים:
- אכיפת אורך בקוד (count_words), לא רק בפרומפט
- interrupt() לאישור אנושי לפני סיום
- SqliteSaver כ-checkpointer, כדי שהגרף יהיה persistent ואפשר
  לחדש שיחה שנעצרה ב-interrupt
"""

import os
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic

from schemas import TalkBrief, CritiqueResult
from word_budget import count_words, clamp_words, within_budget

WORD_TOLERANCE = 150  # סטייה מותרת מהיעד
MAX_REVISIONS = 3


class TedState(TypedDict):
    source_text: str          # תוכן המקור שהועלה (הרצאה/מאמר)
    brief: Optional[dict]
    context: str
    script: str
    critique: Optional[dict]
    revision_count: int
    approved: bool


llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.4)
planner_llm = llm.with_structured_output(TalkBrief)
critic_llm = llm.with_structured_output(CritiqueResult)


def plan_talk(state: TedState) -> dict:
    brief: TalkBrief = planner_llm.invoke(
        f"בהתבסס על המקור הבא, תכנני הרצאת TED: מהו הנושא, קהל היעד, "
        f"המסר המרכזי, ו-3-5 נקודות מפתח.\n\nמקור:\n{state['source_text']}"
    )
    return {"brief": brief.model_dump()}


def gather_context(state: TedState) -> dict:
    # בפועל: כאן אפשר לחבר חיפוש/RAG על מקורות נוספים.
    # לצורך הפרויקט - מרכזים את נקודות המפתח לכדי context לכתיבה.
    brief = state["brief"]
    context = "נקודות מפתח לפיתוח בהרצאה:\n" + "\n".join(
        f"- {p}" for p in brief["key_points"]
    )
    return {"context": context}


def write_talk(state: TedState) -> dict:
    brief = state["brief"]
    target = brief["target_word_count"]
    prompt = (
        f"כתבי תסריט הרצאת TED בעברית, בסביבות {target} מילים, "
        f"בגוף ראשון, בטון מעורר השראה.\n\n"
        f"נושא: {brief['topic']}\nקהל: {brief['audience']}\nמסר: {brief['goal']}\n\n"
        f"{state['context']}"
    )
    response = llm.invoke(prompt)
    script = clamp_words(response.content, min_words=0, max_words=target + WORD_TOLERANCE)
    return {"script": script}


def critique_script(state: TedState) -> dict:
    brief = state["brief"]
    n_words = count_words(state["script"])
    length_ok = within_budget(state["script"], brief["target_word_count"] - WORD_TOLERANCE,
                               brief["target_word_count"] + WORD_TOLERANCE)

    result: CritiqueResult = critic_llm.invoke(
        f"בקרי את תסריט ההרצאה הבא. יעד אורך: {brief['target_word_count']} מילים "
        f"(אורך בפועל: {n_words} מילים, {'בתקציב' if length_ok else 'חורג מהתקציב'}). "
        f"בדקי בהירות מסר, זרימה, והתאמה לקהל {brief['audience']}.\n\n{state['script']}"
    )

    # אכיפת אורך בקוד: גם אם ה-LLM אישר, אם המילים לא בתקציב - לא מאשרים
    approved = result.approved and length_ok
    issues = list(result.issues)
    if not length_ok:
        issues.append(f"אורך בפועל ({n_words}) חורג מהתקציב המותר")

    return {
        "critique": {**result.model_dump(), "approved": approved, "issues": issues},
        "revision_count": state.get("revision_count", 0) + 1,
    }


def revise(state: TedState) -> dict:
    critique = state["critique"]
    prompt = (
        f"תקני את התסריט הבא לפי המשוב. שמרי על אורך קרוב ל-"
        f"{state['brief']['target_word_count']} מילים.\n\n"
        f"משוב: {critique['feedback']}\nבעיות לתיקון: {', '.join(critique['issues'])}\n\n"
        f"תסריט נוכחי:\n{state['script']}"
    )
    response = llm.invoke(prompt)
    script = clamp_words(
        response.content, min_words=0,
        max_words=state["brief"]["target_word_count"] + WORD_TOLERANCE,
    )
    return {"script": script}


def human_approval(state: TedState) -> dict:
    # עוצר את הגרף ומחכה לתשובת אדם. הריצה מתחדשת עם Command(resume=...)
    decision = interrupt({
        "question": "התסריט מוכן. לאשר להמשך להפקת אודיו?",
        "script": state["script"],
        "word_count": count_words(state["script"]),
    })
    return {"approved": bool(decision.get("approve", False))}


def route_after_critique(state: TedState) -> str:
    if state["critique"]["approved"]:
        return "human_approval"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        # לא להיתקע בלולאה אינסופית - עוברים לאישור אנושי בכל מקרה
        return "human_approval"
    return "revise"


def build_ted_graph():
    graph_builder = StateGraph(TedState)
    graph_builder.add_node("plan_talk", plan_talk)
    graph_builder.add_node("gather_context", gather_context)
    graph_builder.add_node("write_talk", write_talk)
    graph_builder.add_node("critique_script", critique_script)
    graph_builder.add_node("revise", revise)
    graph_builder.add_node("human_approval", human_approval)

    graph_builder.add_edge(START, "plan_talk")
    graph_builder.add_edge("plan_talk", "gather_context")
    graph_builder.add_edge("gather_context", "write_talk")
    graph_builder.add_edge("write_talk", "critique_script")
    graph_builder.add_conditional_edges(
        "critique_script", route_after_critique, ["human_approval", "revise"]
    )
    graph_builder.add_edge("revise", "critique_script")
    graph_builder.add_edge("human_approval", END)

    checkpointer = SqliteSaver.from_conn_string("ted_checkpoints.sqlite")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_ted_graph()
    config = {"configurable": {"thread_id": "ted-demo-1"}}

    source = (
        "פודקאסט/מאמר לדוגמה: כיצד קבלת החלטות תחת אי-ודאות "
        "משפיעה על חדשנות בארגונים, ומדוע 'טעויות זולות ומהירות' "
        "עדיפות על תכנון מושלם שמעולם לא יוצא לפועל."
    )

    result = graph.invoke({"source_text": source, "revision_count": 0}, config=config)

    if "__interrupt__" in result:
        print("הגרף עצר לאישור אנושי:")
        print(result["__interrupt__"])
        print("\nכדי לאשר, הריצי שוב עם Command(resume={'approve': True})")
    else:
        print("תסריט סופי:\n", result["script"])
