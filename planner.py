import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

PLANNER_PROMPT = """
You are a research planning assistant.
Rephrase the user's question into a single, clear, well-formed search query
that captures exactly what they're asking for. Do not break it into multiple
questions. Just return ONE improved version of the question, nothing else —
no numbering, no extra text.
"""

def plan(query: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    return [text]  # still a list with one item, so researcher_node's loop works unchanged