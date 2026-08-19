import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
 
load_dotenv()
 
 
# ----- 1. State -----
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
 
# ----- 2. Tools -----
# "מסד נתונים" מזויף להדגמה
FAKE_ORDERS = {
    "1001": {"item": "אוזניות אלחוטיות", "status": "נשלח", "delivery": "3 ימי עסקים"},
    "1002": {"item": "מקלדת מכנית", "status": "בטיפול", "delivery": "טרם נשלח"},
    "1003": {"item": "עכבר גיימינג", "status": "הגיע ליעד", "delivery": "נמסר"},
}
 
 
@tool
def lookup_order(order_id: str) -> str:
    """מחפש הזמנה לפי מספר הזמנה ומחזיר את הסטטוס שלה.
 
    Args:
        order_id: מספר ההזמנה, למשל "1001"
    """
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return f"לא נמצאה הזמנה עם מספר {order_id}."
    return (
        f"הזמנה {order_id}: {order['item']} — סטטוס: {order['status']}, "
        f"זמן משלוח: {order['delivery']}"
    )
 
 
@tool
def return_policy(product_category: str = "כללי") -> str:
    """מחזיר מידע על מדיניות ההחזרות של החברה.
 
    Args:
        product_category: קטגוריית המוצר (למשל "אלקטרוניקה", "כללי")
    """
    return (
        "ניתן להחזיר מוצרים תוך 14 יום מהרכישה, באריזה מקורית ועם חשבונית. "
        "מוצרי אלקטרוניקה שנפתחו ניתנים להחזרה רק אם יש בהם תקלה."
    )
 
 
tools = [lookup_order, return_policy]
 
# ----- 3. המודל, עם חיבור לכלים -----
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
llm_with_tools = llm.bind_tools(tools)
 
SYSTEM_PROMPT = (
    "את/ה נציג/ת שירות לקוחות אדיב/ה ומקצועי/ת. "
    "כשלקוח שואל על סטטוס הזמנה - השתמש/י בכלי lookup_order. "
    "כשלקוח שואל על מדיניות החזרות - השתמש/י בכלי return_policy. "
    "אחרת, ענה/י ישירות בעברית בנימוס ובקיצור."
)
 
 
# ----- 4. Node ראשי -----
def chatbot(state: State) -> dict:
    messages = state["messages"]
    # מוסיפים system prompt רק אם הוא עוד לא בהיסטוריה
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
 
 
# ----- 5. בניית הגרף -----
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
 
graph_builder.add_edge(START, "chatbot")
 
# ענף מותנה: אם הבוט ביקש להשתמש בכלי -> הולכים ל-tools, אחרת -> מסיימים
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # פונקציה מובנית שבודקת אם יש tool_calls בתשובה
)
graph_builder.add_edge("tools", "chatbot")  # אחרי הכלי, חוזרים לבוט לנסח תשובה
 
# ----- 6. Memory -----
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
 
 
if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  לא נמצא ANTHROPIC_API_KEY. בדקי את קובץ .env")
        raise SystemExit(1)
 
    # thread_id מזהה שיחה אחת, כדי שהזיכרון יעבוד
    config = {"configurable": {"thread_id": "conversation-1"}}
 
    print("סוכן שירות לקוחות מוכן. הקלידי 'quit' ליציאה.")
    print("נסי לשאול: 'מה קורה עם הזמנה 1001?' או 'מה מדיניות ההחזרות שלכם?'\n")
 
    while True:
        user_input = input("את/ה: ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("Bot:", result["messages"][-1].content, "\n")
 