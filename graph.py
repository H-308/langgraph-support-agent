

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic

load_dotenv()


# 1. הגדרת ה-State — "הזיכרון" שעובר בין הצמתים (nodes) בגרף.
#    כרגע יש לנו רק רשימת הודעות.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. יצירת מופע של המודל
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


# 3. הגדרת node - פונקציה שמקבלת state ומחזירה עדכון ל-state
def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. בניית הגרף
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  לא נמצא ANTHROPIC_API_KEY. העתיקי את .env.example ל-.env והוסיפי את המפתח שלך.")
        raise SystemExit(1)

    print("צ'אטבוט LangGraph מוכן. הקלידי 'quit' ליציאה.\n")
    while True:
        user_input = input("את/ה: ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break
        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("Bot:", result["messages"][-1].content)