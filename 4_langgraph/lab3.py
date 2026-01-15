from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

import asyncio
import nest_asyncio

# Fix event loop cho Windows + Jupyter
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# nest_asyncio.apply()

load_dotenv(override=True)

# ================= STATE =================

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ================= PUSH TOOL =================

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str):
    requests.post(
        pushover_url,
        data={"token": pushover_token, "user": pushover_user, "message": text}
    )

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="useful for when you want to send a push notification"
)

# ================= PLAYWRIGHT TOOLS =================

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

# Khởi tạo browser async
async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

tool_dict = {tool.name: tool for tool in tools}
navigate_tool = tool_dict.get("navigate_browser")
extract_text_tool = tool_dict.get("extract_text")

# ================= TEST PLAYWRIGHT =================

import asyncio

async def test_playwright():
    await navigate_tool.arun({
        "url": "https://www.cnn.com",
        "wait_until": "domcontentloaded",
        "timeout": 60000,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"
    })

    text = await extract_text_tool.arun({})
    import textwrap
    print(textwrap.fill(text[:2000]))

# asyncio.run(test_playwright())


# ================= LANGGRAPH =================

all_tools = tools + [tool_push]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(all_tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "10"}}

# ================= ASYNC CHAT =================

async def chat(user_input: str, history):
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    return result["messages"][-1].content

gr.ChatInterface(chat, type="messages").launch()
