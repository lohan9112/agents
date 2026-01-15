# ===================== IMPORTS & SETUP =====================

from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)

# ===================== MESSAGE DTO =====================
# Message object dùng để gửi/nhận giữa các agent
@dataclass
class Message:
    content: str


# ===================== SIMPLE AGENT =====================
# Agent cơ bản: nhận message và trả lời cứng
class SimpleAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Simple")

    # message_handler = hàm được AutoGen Core gọi khi có message tới agent
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(
            content=f"This is {self.id.type}-{self.id.key}. You said '{message.content}' and I disagree."
        )


# ===================== LLM AGENT =====================
# Agent dùng LLM để trả lời
class MyLLMAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("LLMAgent")
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        # Delegate sang AssistantAgent để gọi LLM
        self._delegate = AssistantAgent("LLMAgent", model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        print(f"{self.id.type} received message: {message.content}")

        # Convert sang TextMessage cho AssistantAgent
        text_message = TextMessage(content=message.content, source="user")

        # Gọi LLM
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)

        reply = response.chat_message.content
        print(f"{self.id.type} responded: {reply}")
        return Message(content=reply)


# ===================== PLAYER AGENTS (GAME) =====================

from autogen_ext.models.ollama import OllamaChatCompletionClient

# Player 1 dùng OpenAI
class Player1Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)


# Player 2 dùng Ollama
class Player2Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OllamaChatCompletionClient(model="llama3.2", temperature=1.0)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)


# ===================== GAME JUDGE AGENT =====================

JUDGE = "You are judging a game of rock, paper, scissors. The players have made these choices:\n"

class RockPaperScissorsAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        # Instruction cho 2 player
        instruction = (
            "You are playing rock, paper, scissors. "
            "Respond only with the one word, one of the following: rock, paper, or scissors."
        )

        message = Message(content=instruction)

        # Gửi message cho 2 player
        inner_1 = AgentId("player1", "default")
        inner_2 = AgentId("player2", "default")

        response1 = await self.send_message(message, inner_1)
        response2 = await self.send_message(message, inner_2)

        # Tổng hợp kết quả
        result = f"Player 1: {response1.content}\nPlayer 2: {response2.content}\n"
        judgement = f"{JUDGE}{result}Who wins?"

        # Gửi kết quả cho LLM judge
        message = TextMessage(content=judgement, source="user")
        response = await self._delegate.on_messages([message], ctx.cancellation_token)

        return Message(content=result + response.chat_message.content)


# ===================== MAIN RUNTIME WRAPPER =====================
# Wrap toàn bộ await vào 1 hàm async và gọi nó

async def main():
    # ---------- DEMO 1: SimpleAgent ----------
    runtime = SingleThreadedAgentRuntime()
    await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())

    runtime.start()

    agent_id = AgentId("simple_agent", "default")
    response = await runtime.send_message(Message("Well hi there!"), agent_id)
    print(">>>", response.content)

    await runtime.stop()
    await runtime.close()

    # ---------- DEMO 2: SimpleAgent + LLM Agent ----------
    runtime = SingleThreadedAgentRuntime()
    await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())
    await MyLLMAgent.register(runtime, "LLMAgent", lambda: MyLLMAgent())

    runtime.start()

    response = await runtime.send_message(Message("Hi there!"), AgentId("LLMAgent", "default"))
    print(">>>", response.content)

    response = await runtime.send_message(Message(response.content), AgentId("simple_agent", "default"))
    print(">>>", response.content)

    response = await runtime.send_message(Message(response.content), AgentId("LLMAgent", "default"))

    await runtime.stop()
    await runtime.close()

    # ---------- DEMO 3: Rock Paper Scissors ----------
    runtime = SingleThreadedAgentRuntime()
    await Player1Agent.register(runtime, "player1", lambda: Player1Agent("player1"))
    await Player2Agent.register(runtime, "player2", lambda: Player2Agent("player2"))
    await RockPaperScissorsAgent.register(runtime, "rock_paper_scissors",
                                          lambda: RockPaperScissorsAgent("rock_paper_scissors"))

    runtime.start()

    agent_id = AgentId("rock_paper_scissors", "default")
    message = Message(content="go")
    response = await runtime.send_message(message, agent_id)
    print(response.content)

    await runtime.stop()
    await runtime.close()


# ===================== RUN =====================

import asyncio
asyncio.run(main())
