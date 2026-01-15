# ============================================================
# Week 4 Day 2 - LangGraph Demo (Full Python Script)
# ============================================================

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import random

# ------------------------------------------------------------
# Load environment variables (OPENAI_API_KEY, etc.)
# ------------------------------------------------------------
load_dotenv(override=True)

# ------------------------------------------------------------
# Demo data
# ------------------------------------------------------------
nouns = ["Cabbages", "Unicorns", "Toasters", "Penguins", "Bananas", "Zombies", "Rainbows", "Eels", "Pickles", "Muffins"]
adjectives = ["outrageous", "smelly", "pedantic", "existential", "moody", "sparkly", "untrustworthy", "sarcastic", "squishy", "haunted"]

# ------------------------------------------------------------
# Simple Annotated example
# ------------------------------------------------------------
def shout(text: Annotated[str, "something to be shouted"]) -> str:
    print(text.upper())
    return text.upper()

shout("hello")

# ============================================================
# PART 1 — LangGraph without LLM
# ============================================================

# Step 1: Define State
class State(BaseModel):
    messages: Annotated[list, add_messages]

# Step 2: Graph Builder
graph_builder = StateGraph(State)

# Step 3: Node
def our_first_node(old_state: State) -> State:
    reply = f"{random.choice(nouns)} are {random.choice(adjectives)}"
    messages = [{"role": "assistant", "content": reply}]
    return State(messages=messages)

graph_builder.add_node("first_node", our_first_node)

# Step 4: Edges
graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", END)

# Step 5: Compile
graph = graph_builder.compile()

# Visualize graph (optional in Jupyter)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

# Chat UI
def chat_basic(user_input: str, history):
    message = {"role": "user", "content": user_input}
    state = State(messages=[message])
    result = graph.invoke(state)
    print(result)
    return result["messages"][-1].content

# Uncomment to run UI
# gr.ChatInterface(chat_basic, type="messages").launch()


# ============================================================
# PART 2 — LangGraph with LLM (ChatOpenAI)
# ============================================================

# Step 1: Define State
class State(BaseModel):
    messages: Annotated[list, add_messages]

# Step 2: Graph Builder
graph_builder = StateGraph(State)

# Step 3: Node (LLM)
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot_node(old_state: State) -> State:
    response = llm.invoke(old_state.messages)
    return State(messages=[response])

graph_builder.add_node("chatbot", chatbot_node)

# Step 4: Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Step 5: Compile
graph = graph_builder.compile()

# Visualize graph (optional in Jupyter)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

# Chat UI
def chat_llm(user_input: str, history):
    initial_state = State(messages=[{"role": "user", "content": user_input}])
    result = graph.invoke(initial_state)
    return result["messages"][-1].content

# Uncomment to run UI
# gr.ChatInterface(chat_llm, type="messages").launch()


# ============================================================
# END
# ============================================================
