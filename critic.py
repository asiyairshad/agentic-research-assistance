import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

CRITIC_PROMPT ="""
You are a research critic.
Given the original question and the search results gathered, decide if the evidence is sufficient to write a thorough, accurate answer.

Respond in exactly this format:
SUFFICIENT: yes or no
MISSING: brief note on what's missing (or "none" if sufficient)
"""

def critique(query: str, results: list[dict]) -> tuple[bool, str]:
    sources = "\n\n".join(
        f"[{i}] {r['title']}\n{r['content']}"
        for i, r in enumerate(results, start=1)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"Original question: {query}\nSearch results: {sources}"}
        ],
        temperature=0,
)

    text = response.choices[0].message.content
    first_line = text.strip().split("\n")[0].lower()
    is_sufficient = "yes" in first_line

    return is_sufficient, text


