# ===================== IMPORTS =====================
# Dataclass for defining simple message payload
from dataclasses import dataclass

# Core AutoGen routing primitives
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler

# High-level LLM agent wrapper
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# OpenAI model client
from autogen_ext.models.openai import OpenAIChatCompletionClient

# LangChain tool adapter for AutoGen
from autogen_ext.tools.langchain import LangChainToolAdapter

# Google Serper search utility (LangChain)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

# Notebook rendering helpers
from IPython.display import display, Markdown

# Environment variables loader
from dotenv import load_dotenv

# Load .env (API keys, etc.)
load_dotenv(override=True)

# Flag to control whether all agents run in a single worker
ALL_IN_ONE_WORKER = False


# ===================== MESSAGE MODEL =====================
# Routed message payload between agents
@dataclass
class Message:
    content: str


# ===================== GRPC HOST =====================
# gRPC runtime host that coordinates worker runtimes
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")


# ===================== TOOLING =====================
# Google Serper search wrapped as a LangChain tool
serper = GoogleSerperAPIWrapper()
langchain_serper = Tool(
    name="internet_search",
    func=serper.run,
    description="Useful for when you need to search the internet"
)

# Adapt LangChain tool for AutoGen agents
autogen_serper = LangChainToolAdapter(langchain_serper)


# ===================== PROMPTS =====================
instruction1 = "To help with a decision on whether to use AutoGen in a new AI Agent project, \
please research and briefly respond with reasons in favor of choosing AutoGen; the pros of AutoGen."

instruction2 = "To help with a decision on whether to use AutoGen in a new AI Agent project, \
please research and briefly respond with reasons against choosing AutoGen; the cons of Autogen."

judge = "You must make a decision on whether to use AutoGen for a project. \
Your research team has come up with the following reasons for and against. \
Based purely on the research from your team, please respond with your decision and brief rationale."


# ===================== AGENTS =====================
# Player1 agent: researches the pros of AutoGen
class Player1Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Create OpenAI model client
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        # Delegate LLM work to AssistantAgent with search tool enabled
        self._delegate = AssistantAgent(
            name,
            model_client=model_client,
            tools=[autogen_serper],
            reflect_on_tool_use=True
        )

    # Register this method as a message handler
    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        # Convert routed Message into TextMessage for AssistantAgent
        text_message = TextMessage(content=message.content, source="user")
        # Call the LLM agent
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        # Return result back to runtime
        return Message(content=response.chat_message.content)


# Player2 agent: researches the cons of AutoGen
class Player2Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self._delegate = AssistantAgent(
            name,
            model_client=model_client,
            tools=[autogen_serper],
            reflect_on_tool_use=True
        )

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)


# Judge agent: queries Player1 & Player2 and makes the final decision
class Judge(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare instructions for both players
        message1 = Message(content=instruction1)
        message2 = Message(content=instruction2)

        # Agent IDs for Player1 and Player2
        inner_1 = AgentId("player1", "default")
        inner_2 = AgentId("player2", "default")

        # Send messages to both players via runtime routing
        response1 = await self.send_message(message1, inner_1)
        response2 = await self.send_message(message2, inner_2)

        # Aggregate both responses
        result = (
            f"## Pros of AutoGen:\n{response1.content}\n\n"
            f"## Cons of AutoGen:\n{response2.content}\n\n"
        )

        # Ask LLM to judge based on aggregated research
        judgement = f"{judge}\n{result}Respond with your decision and brief explanation"
        message = TextMessage(content=judgement, source="user")
        response = await self._delegate.on_messages([message], ctx.cancellation_token)

        return Message(content=result + "\n\n## Decision:\n\n" + response.chat_message.content)


# ===================== RUNTIME =====================
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime


# ===================== MAIN (wrap all awaits) =====================
async def main():
    # Start the gRPC host
    host.start()

    if ALL_IN_ONE_WORKER:
        # Single worker hosting all agents
        worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
        await worker.start()

        # Register agents into the runtime
        await Player1Agent.register(worker, "player1", lambda: Player1Agent("player1"))
        await Player2Agent.register(worker, "player2", lambda: Player2Agent("player2"))
        await Judge.register(worker, "judge", lambda: Judge("judge"))

        agent_id = AgentId("judge", "default")

    else:
        # Split agents into separate workers
        worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
        await worker1.start()
        await Player1Agent.register(worker1, "player1", lambda: Player1Agent("player1"))

        worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
        await worker2.start()
        await Player2Agent.register(worker2, "player2", lambda: Player2Agent("player2"))

        worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
        await worker.start()
        await Judge.register(worker, "judge", lambda: Judge("judge"))

        agent_id = AgentId("judge", "default")

    # Kick off the workflow by sending a message to the Judge agent
    response = await worker.send_message(Message(content="Go!"), agent_id)

    # Render the final decision as Markdown
    display(Markdown(response.content))

    # Graceful shutdown of workers and host
    await worker.stop()
    if not ALL_IN_ONE_WORKER:
        await worker1.stop()
        await worker2.stop()

    await host.stop()


# ===================== RUN =====================
# Execute the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
