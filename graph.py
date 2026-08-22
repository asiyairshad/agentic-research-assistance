from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from critic import critique
from tools import web_search, search_pdfs, search_pubmed,search_arxiv
from planner import plan
from retriever import Retriever
from langchain_openai import ChatOpenAI
retriever = Retriever()

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [web_search, search_pdfs, search_pubmed,search_arxiv]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

class ResearchState(TypedDict):
    query: str
    user_id: str
    session_id: str
    doc_id: Optional[str]
    needs_web_search: bool
    sub_questions: list[str]
    search_results: list[dict]
    critique: str
    is_sufficient: bool
    iteration: int
    final_report: str


# ── ROUTER: decide PDF-only vs PDF + web ─────────────────────────────
def router_node(state: ResearchState) -> ResearchState:
    """Checks if the uploaded PDF alone can answer the query, or if
    the query needs external/current info via web search."""
    router_prompt = f"""You are deciding whether a question can be answered
using ONLY the content of an uploaded document, or whether it needs
external/current web information (news, recent events, facts outside
the document, general knowledge not likely to be in a personal PDF).

Question: {state['query']}

Reply with exactly one word: "pdf" or "web"."""

    decision = retriever.llm.invoke(router_prompt).content.strip().lower()
    needs_web = "web" in decision

    return {"needs_web_search": needs_web}


# ── PLANNER: break query into sub-questions (web + pdf path only) ─────
def planner_node(state: ResearchState) -> ResearchState:
    previous_report = state.get("final_report", "")
    context = f"Previous research report:\n{previous_report}\n\n" if previous_report else ""
    sub_questions = plan(context + state["query"])
    return {"sub_questions": sub_questions}


# ── RESEARCHER: web + pdf search, deduped ──────────────────────────────

def researcher_node(state: ResearchState) -> ResearchState:
    """For each sub-question, lets the LLM decide which tool(s) to call
    (web, PDF, PubMed) instead of calling all three unconditionally."""
    new_results = []

    for sub_q in state["sub_questions"]:
        prompt = f"""Question: {sub_q}
User ID: {state['user_id']}
Document ID: {state.get('doc_id', 'none')}

Decide which tool(s) are relevant to answer this question and call them."""

        ai_msg = llm_with_tools.invoke(prompt)

        for tool_call in ai_msg.tool_calls:
            tool_fn = tool_map[tool_call["name"]]
            args = tool_call["args"]

            if tool_call["name"] == "search_pdfs":
                args.setdefault("user_id", state["user_id"])
                args.setdefault("doc_id", state.get("doc_id"))

            try:
                result = tool_fn.invoke(args)
                new_results.extend(result or [])
            except Exception as e:
                print(f"Tool {tool_call['name']} failed: {e}")
                continue

    all_results = state.get("search_results", []) + new_results

    seen_urls = set()
    seen_content = set()
    deduped_results = []

    for r in all_results:
        url_key = r["url"]
        content_key = r["content"].strip().lower()[:200]

        if url_key in seen_urls or content_key in seen_content:
            continue

        seen_urls.add(url_key)
        seen_content.add(content_key)
        deduped_results.append(r)

    return {
        "search_results": deduped_results,
        "iteration": state.get("iteration", 0) + 1
    }

# ── CRITIC: judge if evidence is sufficient (web + pdf path only) ──────
def critic_node(state: ResearchState) -> ResearchState:
    is_sufficient, critique_text = critique(state["query"], state["search_results"])
    return {"is_sufficient": is_sufficient, "critique": critique_text}


def route_after_critic(state: ResearchState) -> str:
    if state["is_sufficient"] or state.get("iteration", 0) >= 3:
        return "writer"
    return "researcher"


# ── WRITER: generate final answer + save to persistent chat memory ────
def writer_node(state: ResearchState) -> ResearchState:
    context = "\n\n".join(
        f"TITLE: {r['title']}\nCONTENT: {r['content']}"
        for r in state["search_results"]
    )

    prompt = f"""Answer the question using the research below. Refer to sources
by title in your answer text — do not include URLs or links yourself,
they will be added automatically afterward.

Question: {state['query']}

Research:
{context}"""

    answer = retriever.llm.invoke(prompt).content

    # decide whether the user wants actual papers or general info
    wants_papers = any(
        kw in state["query"].lower()
        for kw in ["paper", "research paper", "study", "publication", "arxiv"]
    )

    if wants_papers:
        paper_sources = [r for r in state["search_results"] if r.get("source_type") == "paper"]
        real_sources = paper_sources if paper_sources else []
    else:
        real_sources = [r for r in state["search_results"] if r["url"] != "uploaded_pdf"]

    if real_sources:
        sources_section = "\n\n**Sources:**\n" + "\n".join(
            f"{i+1}. [{r['title']}]({r['url']})" for i, r in enumerate(real_sources)
        )
        answer = answer + sources_section
    elif wants_papers:
        answer += "\n\nI couldn't find specific research papers for this — try rephrasing your query."

    retriever.get_chat_history(state["user_id"], state["session_id"]).add_ai_message(answer)

    return {"final_report": answer}


# ── GRAPH ───────────────────────────────────────────────────────────
builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("critic", critic_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "critic")
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"writer": "writer", "researcher": "researcher"}
)
builder.add_edge("writer", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
