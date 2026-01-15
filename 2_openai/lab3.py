import os
import asyncio
from typing import Dict
from dotenv import load_dotenv
from pydantic import BaseModel

import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    trace,
    function_tool,
    OpenAIChatCompletionsModel,
    input_guardrail,
    GuardrailFunctionOutput
)

# ======================================================
# ENV
# ======================================================

load_dotenv(override=True)

openai_api_key   = os.getenv("OPENAI_API_KEY")
google_api_key   = os.getenv("GOOGLE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
groq_api_key     = os.getenv("GROQ_API_KEY")

# ======================================================
# API KEY CHECK
# ======================================================

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")

# ======================================================
# SALES AGENT INSTRUCTIONS
# ======================================================

instructions1 = (
    "You are a sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. "
    "You write professional, serious cold emails."
)

instructions2 = (
    "You are a humorous, engaging sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. "
    "You write witty, engaging cold emails that are likely to get a response."
)

instructions3 = (
    "You are a busy sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. "
    "You write concise, to the point cold emails."
)

# ======================================================
# BASE URL
# ======================================================

GEMINI_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"

# ======================================================
# CLIENTS
# ======================================================

deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
gemini_client   = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
groq_client     = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)

# ======================================================
# MODELS
# ======================================================

deepseek_model = OpenAIChatCompletionsModel(
    model="deepseek-chat",
    openai_client=deepseek_client
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client
)

llama3_3_model = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=groq_client
)

# ======================================================
# SALES AGENTS
# ======================================================

sales_agent1 = Agent(
    name="DeepSeek Sales Agent",
    instructions=instructions1,
    model=deepseek_model
)

sales_agent2 = Agent(
    name="Gemini Sales Agent",
    instructions=instructions2,
    model=gemini_model
)

sales_agent3 = Agent(
    name="Llama3.3 Sales Agent",
    instructions=instructions3,
    model=llama3_3_model
)

# ======================================================
# SALES TOOLS
# ======================================================

description = "Write a cold sales email"

tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

# ======================================================
# SENDGRID TOOL
# ======================================================

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("ed@edwarddonner.com")
    to_email = To("ed.donner@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

# ======================================================
# SUBJECT + HTML AGENTS
# ======================================================

subject_instructions = (
    "You can write a subject for a cold sales email. "
    "You are given a message and you need to write a subject for an email that is likely to get a response."
)

html_instructions = (
    "You can convert a text email body to an HTML email body. "
    "You are given a text email body which might have some markdown "
    "and you need to convert it to an HTML email body with simple, clear, compelling layout and design."
)

subject_writer = Agent(
    name="Email subject writer",
    instructions=subject_instructions,
    model="gpt-4o-mini"
)

subject_tool = subject_writer.as_tool(
    tool_name="subject_writer",
    tool_description="Write a subject for a cold sales email"
)

html_converter = Agent(
    name="HTML email body converter",
    instructions=html_instructions,
    model="gpt-4o-mini"
)

html_tool = html_converter.as_tool(
    tool_name="html_converter",
    tool_description="Convert a text email body to an HTML email body"
)

email_tools = [subject_tool, html_tool, send_html_email]

# ======================================================
# EMAIL MANAGER AGENT
# ======================================================

instructions = (
    "You are an email formatter and sender. You receive the body of an email to be sent. "
    "You first use the subject_writer tool to write a subject for the email, "
    "then use the html_converter tool to convert the body to HTML. "
    "Finally, you use the send_html_email tool to send the email with the subject and HTML body."
)

emailer_agent = Agent(
    name="Email Manager",
    instructions=instructions,
    tools=email_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)

# ======================================================
# SALES MANAGER AGENT
# ======================================================

tools = [tool1, tool2, tool3]
handoffs = [emailer_agent]

sales_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.

Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts.
2. Evaluate and Select: Review the drafts and choose the single best email.
3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent.

Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must hand off exactly ONE email to the Email Manager — never more than one.
"""

sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=handoffs,
    model="gpt-4o-mini"
)

# ======================================================
# GUARDRAIL
# ======================================================

class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str


guardrail_agent = Agent(
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)


@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(
        output_info={"found_name": result.final_output},
        tripwire_triggered=is_name_in_message
    )


careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
    input_guardrails=[guardrail_against_name]
)

# ======================================================
# MAIN
# ======================================================

async def main():
    # Case 1
    message_1 = "Send out a cold sales email addressed to Dear CEO from Alice"
    with trace("Automated SDR"):
        result_1 = await Runner.run(careful_sales_manager, message_1)

    # Case 2
    message_2 = "Send out a cold sales email addressed to Dear CEO from Head of Business Development"
    with trace("Protected Automated SDR"):
        result_2 = await Runner.run(careful_sales_manager, message_2)

    # Case 3 (raw)
    message_3 = "Send out a cold sales email addressed to Dear CEO from Alice"
    with trace("Raw Automated SDR"):
        result_3 = await Runner.run(sales_manager, message_3)


if __name__ == "__main__":
    asyncio.run(main())
