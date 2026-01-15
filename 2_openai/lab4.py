from agents import Agent, WebSearchTool, trace, Runner, gen_trace_id, function_tool
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
from typing import Dict
from IPython.display import display, Markdown

load_dotenv(override=True)

# ======================================================
# SEARCH AGENT
# ======================================================

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly."
)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)

# ======================================================
# PLANNER AGENT
# ======================================================

HOW_MANY_SEARCHES = 3

INSTRUCTIONS = (
    f"You are a helpful research assistant. Given a query, come up with a set of web searches "
    f"to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."
)

class WebSearchItem(BaseModel):
    reason: str = Field(description="Why this search is important")
    query: str = Field(description="Search term")

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)

# ======================================================
# EMAIL TOOL
# ======================================================

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("ed@edwarddonner.com")
    to_email = To("ed.donner@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return "success"

# ======================================================
# EMAIL AGENT
# ======================================================

INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report."""

email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gpt-4o-mini",
)

# ======================================================
# WRITER AGENT
# ======================================================

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "Generate a long markdown report (1000+ words)."
)

class ReportData(BaseModel):
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]

writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,
)

# ======================================================
# ASYNC FUNCTIONS (giữ nguyên)
# ======================================================

async def plan_searches(query: str):
    result = await Runner.run(planner_agent, f"Query: {query}")
    return result.final_output

async def perform_searches(search_plan: WebSearchPlan):
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    return results

async def search(item: WebSearchItem):
    input = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output

async def write_report(query: str, search_results: list[str]):
    input = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, input)
    return result.final_output

async def send_report_email(report: ReportData):
    result = await Runner.run(email_agent, report.markdown_report)
    return report

# ======================================================
# MAIN (gom toàn bộ await vào đây)
# ======================================================

async def main():
    query = "Latest AI Agent frameworks in 2025"

    # Search summary
    with trace("Search"):
        result = await Runner.run(search_agent, query)
        display(Markdown(result.final_output))

    # Planner
    with trace("Search"):
        result = await Runner.run(planner_agent, query)
        print(result.final_output)

    # Full research pipeline
    with trace("Research trace"):
        print("Starting research...")
        search_plan = await plan_searches(query)
        search_results = await perform_searches(search_plan)
        report = await write_report(query, search_results)
        await send_report_email(report)
        print("Hooray!")

# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    asyncio.run(main())
