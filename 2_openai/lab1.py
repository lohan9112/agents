# Week 2 Day 1 — OpenAI Agents SDK demo
# Prerequisites:
#   pip install python-dotenv openai-agents
#   export OPENAI_API_KEY=your_key

import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace

# Load environment variables
load_dotenv(override=True)

async def main():
    # Create agent
    agent = Agent(
        name="Jokester",
        instructions="You are a joke teller",
        model="gpt-4o-mini"
    )

    # Run agent with tracing
    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
