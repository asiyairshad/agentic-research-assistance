import os
import requests
import xml.etree.ElementTree as ET
import feedparser 
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool

from retriever import Retriever

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
ARXIC_API_URL = "http://export.arxiv.org/api/query"

# ── WEB SEARCH ──────────────────────────────────────────────────────────
@tool
def web_search(query: str, max_result: int = 3) -> list[dict]:
    """Searches the web for current information on a given query.
    Use this for general knowledge, news, or facts not likely to be in a personal document."""
    response = tavily_client.search(
        query=query,
        max_results=max_result,
        include_answer=None,
        search_depth='advanced',
        topic='general',
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "source_type": "web",
        })
    return results


# ── PDF SEARCH ──────────────────────────────────────────────────────────
@tool
def search_pdfs(query: str, user_id: str, doc_id: str | None = None) -> list[dict]:
    """Searches the user's uploaded PDF documents for relevant content."""
    retriever = Retriever()
    docs = retriever.query(query, user_id=user_id, session_id="default", doc_id=doc_id)

    return [
        {
            "title": d.metadata.get("filename", "uploaded document"),
            "url": "uploaded_pdf",
            "content": d.page_content,
            "source_type": "pdf"
        }
        for d in docs
    ]


# ── PUBMED SEARCH ───────────────────────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@tool
def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """Searches PubMed for peer-reviewed medical and scientific research articles.
    Use this for academic, clinical, or health-related research questions."""
    search_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    search_resp = requests.get(ESEARCH_URL, params=search_params, timeout=10)
    search_resp.raise_for_status()
    pmids = search_resp.json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=10)
    fetch_resp.raise_for_status()

    root = ET.fromstring(fetch_resp.content)
    results = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="Untitled")
        abstract_parts = article.findall(".//AbstractText")
        content = " ".join(part.text or "" for part in abstract_parts) or "No abstract available."

        results.append({
            "title": title,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "content": content,
            "source_type": "paper",
        })

    return results

import feedparser  # uv add feedparser

ARXIV_API_URL = "https://export.arxiv.org/api/query"


@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Searches arXiv for research papers in CS, AI/ML, physics, and math.
    Use this when the user asks for a specific research paper, wants to find
    papers on a topic, or needs academic literature outside medicine/biology."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=20)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    results = []

    for entry in feed.entries:
        pdf_url = next(
            (link.href for link in entry.links if link.type == "application/pdf"),
            entry.link,
        )
        results.append({
            "title": entry.title.replace("\n", " ").strip(),
            "url": pdf_url,
            "content": entry.summary.replace("\n", " ").strip(),
            "source_type": "paper"
        })

    return results