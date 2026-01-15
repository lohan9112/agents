# ===================== IMPORTS =====================
from io import BytesIO
import requests
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import Image as AGImage
from PIL import Image
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from IPython.display import display, Markdown
from pydantic import BaseModel, Field
from typing import Literal
import textwrap

# Load environment variables (API keys, etc.)
load_dotenv(override=True)


# ===================== IMAGE SETUP =====================

# Image URL
url = "https://edwarddonner.com/wp-content/uploads/2024/10/from-software-engineer-to-AI-DS.jpeg"

# Download image and convert to PIL
pil_image = Image.open(BytesIO(requests.get(url).content))

# Convert PIL image to AutoGen Image
img = AGImage(pil_image)

# Create a multimodal message (text + image)
multi_modal_message = MultiModalMessage(
    content=["Describe the content of this image in detail", img],
    source="User"
)


# ===================== SIMPLE IMAGE DESCRIPTION =====================

async def run_simple_image_description():
    """Call AutoGen agent to describe image in free text"""

    # Init OpenAI client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Create assistant agent
    describer = AssistantAgent(
        name="description_agent",
        model_client=model_client,
        system_message="You are good at describing images",
    )

    # Send message to agent
    response = await describer.on_messages(
        [multi_modal_message],
        cancellation_token=CancellationToken()
    )

    # Extract reply
    reply = response.chat_message.content

    # Render markdown output
    display(Markdown(reply))


# ===================== STRUCTURED IMAGE DESCRIPTION =====================

class ImageDescription(BaseModel):
    scene: str = Field(description="Briefly, the overall scene of the image")
    message: str = Field(description="The point that the image is trying to convey")
    style: str = Field(description="The artistic style of the image")
    orientation: Literal["portrait", "landscape", "square"] = Field(description="The orientation of the image")


async def run_structured_image_description():
    """Call AutoGen agent to describe image with structured schema"""

    # Init OpenAI client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Create assistant agent with output schema
    describer = AssistantAgent(
        name="description_agent",
        model_client=model_client,
        system_message="You are good at describing images in detail",
        output_content_type=ImageDescription,
    )

    # Send message to agent
    response = await describer.on_messages(
        [multi_modal_message],
        cancellation_token=CancellationToken()
    )

    # Extract structured reply
    reply = response.chat_message.content

    # Print raw object
    print(reply)

    # Pretty print each field
    print(f"Scene:\n{textwrap.fill(reply.scene)}\n\n")
    print(f"Message:\n{textwrap.fill(reply.message)}\n\n")
    print(f"Style:\n{textwrap.fill(reply.style)}\n\n")
    print(f"Orientation:\n{textwrap.fill(reply.orientation)}\n\n")


# ===================== TOOL SEARCH + FILE WRITE =====================

from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import Tool


prompt = """Your task is to find a one-way non-stop flight from JFK to LHR in June 2025.
First search online for promising deals.
Next, write all the deals to a file called flights.md with full details.
Finally, select the one you think is best and reply with a short summary.
Reply with the selected flight only, and only after you have written the details to the file."""


async def run_search_and_write_file():
    """Run agent with internet search + file tools"""

    # Create Serper search tool
    serper = GoogleSerperAPIWrapper()
    langchain_serper = Tool(
        name="internet_search",
        func=serper.run,
        description="useful for when you need to search the internet"
    )

    # Wrap LangChain tool into AutoGen tool
    autogen_serper = LangChainToolAdapter(langchain_serper)
    autogen_tools = [autogen_serper]

    # Add file management tools
    langchain_file_management_tools = FileManagementToolkit(root_dir="sandbox").get_tools()
    for tool in langchain_file_management_tools:
        autogen_tools.append(LangChainToolAdapter(tool))

    # Print all registered tools
    for tool in autogen_tools:
        print(tool.name, tool.description)

    # Create model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Create agent with tools
    agent = AssistantAgent(
        name="searcher",
        model_client=model_client,
        tools=autogen_tools,
        reflect_on_tool_use=True
    )

    # Send first task
    message = TextMessage(content=prompt, source="user")
    result = await agent.on_messages([message], cancellation_token=CancellationToken())

    for message in result.inner_messages:
        print(message.content)

    display(Markdown(result.chat_message.content))

    # Send follow-up command to proceed
    message = TextMessage(content="OK proceed", source="user")
    result = await agent.on_messages([message], cancellation_token=CancellationToken())

    for message in result.inner_messages:
        print(message.content)

    display(Markdown(result.chat_message.content))


# ===================== MULTI-AGENT EVALUATION LOOP =====================

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat


async def run_multi_agent_evaluation():
    """Primary agent finds flight, evaluator approves"""

    serper = GoogleSerperAPIWrapper()
    langchain_serper = Tool(
        name="internet_search",
        func=serper.run,
        description="useful for when you need to search the internet"
    )
    autogen_serper = LangChainToolAdapter(langchain_serper)

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    prompt = "Find a one-way non-stop flight from JFK to LHR in June 2025."

    # Primary research agent
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        tools=[autogen_serper],
        system_message="You are a helpful AI research assistant who looks for promising deals on flights. Incorporate any feedback you receive.",
    )

    # Evaluation agent
    evaluation_agent = AssistantAgent(
        "evaluator",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedback is addressed.",
    )

    # Stop condition when evaluator says APPROVE
    text_termination = TextMentionTermination("APPROVE")

    team = RoundRobinGroupChat(
        [primary_agent, evaluation_agent],
        termination_condition=text_termination,
        max_turns=20
    )

    # Run team
    result = await team.run(task=prompt)
    
    for message in result.messages:
        print(f"{message.source}:\n{message.content}\n\n")


# ===================== MCP FETCH TOOL =====================

from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools


async def run_mcp_fetch():
    """Use MCP fetch server to crawl and summarize website"""

    # Setup MCP fetch server
    fetch_mcp_server = StdioServerParams(
        command="uvx",
        args=["mcp-server-fetch"],
        read_timeout_seconds=30
    )

    # Load MCP tools
    fetcher = await mcp_server_tools(fetch_mcp_server)

    # Create agent with fetch tools
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        name="fetcher",
        model_client=model_client,
        tools=fetcher,
        reflect_on_tool_use=True  # type: ignore
    )

    # Run fetch task
    result = await agent.run(
        task="Review edwarddonner.com and summarize what you learn. Reply in Markdown."
    )

    display(Markdown(result.messages[-1].content))


# ===================== MAIN EXECUTION =====================

async def main():
    await run_simple_image_description()
    await run_structured_image_description()
    await run_search_and_write_file()
    await run_multi_agent_evaluation()
    await run_mcp_fetch()


# Call main() in notebook or async runtime:
# await main()
