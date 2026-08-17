import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

PLANNER_PROMPT = """
You are a research planning assistant.
Break the user's question into 2-4 focused sub-questions that together would help answer it thoroughly.
If the question is already simple and specific, just return it as a single sub-question.
Return ONLY the sub-questions, one per line, no numbering, no extra text.

"""

def plan(query:str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content
    sub_questions = [line.strip() for line in text.splitlines() if line.strip()]
    return sub_questions
