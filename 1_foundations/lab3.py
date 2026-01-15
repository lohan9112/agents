from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import os
from pathlib import Path
from pydantic import BaseModel

# Load env
load_dotenv(override=True)

# ===== Load LinkedIn PDF =====
BASE_DIR = Path(__file__).resolve().parent
reader = PdfReader(BASE_DIR / "me" / "linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# ===== Load summary =====
with open(BASE_DIR / "me" / "summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# ===== System prompt =====
name = "Ed Donner"
system_prompt = f"""
You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Be professional and engaging.

## Summary:
{summary}

## LinkedIn Profile:
{linkedin}
"""

# ===== OpenAI client =====
openai = OpenAI()

# ===== Basic chat =====
def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": message}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content


# ===== Evaluation model =====
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


evaluator_system_prompt = f"""
You are an evaluator that decides whether a response is acceptable.
The Agent represents {name} on their website.

## Summary:
{summary}

## LinkedIn Profile:
{linkedin}
"""


def evaluator_user_prompt(reply, message, history):
    return f"""
Conversation:
{history}

User message:
{message}

Agent reply:
{reply}
"""


# ===== Gemini evaluator =====
gemini = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def evaluate(reply, message, history) -> Evaluation:
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)}
    ]
    response = gemini.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        messages=messages,
        response_format=Evaluation
    )
    return response.choices[0].message.parsed


# ===== Retry logic =====
def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + f"""
## Previous answer rejected
Attempted answer:
{reply}

Reason:
{feedback}
"""
    messages = [{"role": "system", "content": updated_system_prompt}] + history + [
        {"role": "user", "content": message}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content


# ===== Final chat with evaluation =====
def chat(message, history):
    if "patent" in message:
        system = system_prompt + "\nRespond entirely in Pig Latin."
    else:
        system = system_prompt

    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": message}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, history)
    if not evaluation.is_acceptable:
        reply = rerun(reply, message, history, evaluation.feedback)

    return reply


# ===== UI =====
gr.ChatInterface(chat, type="messages").launch()
