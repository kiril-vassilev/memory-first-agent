from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.memory_store import MemoryHit, RedisMemoryStore
from app.tools import SearchDocument, TavilySearchService


class AgentState(TypedDict):
    query: str
    query_embedding: list[float]
    route: str
    top_similarity: float
    topic: str
    memory_hits: list[MemoryHit]
    memory_context: str
    documents: list[SearchDocument]
    pages_markdown: list[str]
    ingested_chunk_count: int
    summary: str
    answer: str
    sources: list[dict[str, str]]


def _topic_from_query(query: str) -> str:
    q = query.lower()
    mapping = {
        "health": ["health", "diet", "nutrition", "sleep", "fitness", "exercise", "medical"],
        "technology": ["ai", "software", "python", "langgraph", "redis", "cloud", "programming"],
        "finance": ["stock", "invest", "finance", "tax", "budget", "economy", "crypto"],
        "education": ["learn", "course", "study", "university", "school", "exam"],
        "travel": ["travel", "flight", "hotel", "visa", "destination"],
        "general": [],
    }

    for topic, keywords in mapping.items():
        if any(word in q for word in keywords):
            return topic
    return "general"


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _embed_query_node(state: AgentState, embeddings: AzureOpenAIEmbeddings) -> dict[str, Any]:
    query = state["query"]
    vector = embeddings.embed_query(query)
    topic = _topic_from_query(query)
    return {"query_embedding": vector, "topic": topic}


def _search_memory_node(
    state: AgentState,
    memory_store: RedisMemoryStore,
    threshold: float,
    memory_k: int,
) -> dict[str, Any]:
    hits = memory_store.search(state["query_embedding"], k=memory_k)
    top_similarity = hits[0].similarity if hits else 0.0

    route = "memory_hit" if top_similarity >= threshold else "memory_miss"
    if not hits and top_similarity == 0.0:
        route = "memory_miss"
    context_lines: list[str] = []
    sources: list[dict[str, str]] = []

    if route == "memory_hit":
        for hit in hits:
            context_lines.append(
                f"Title: {hit.title}\nSource: {hit.source_url}\nSimilarity: {hit.similarity:.3f}\n{hit.content}"
            )
            if hit.source_url:
                sources.append({"title": hit.title or "Memory source", "url": hit.source_url})

    return {
        "route": route,
        "top_similarity": top_similarity,
        "memory_hits": hits,
        "memory_context": "\n\n---\n\n".join(context_lines),
        "sources": sources,
    }


def _should_use_memory(state: AgentState) -> str:
    return "answer_from_memory" if state.get("route") == "memory_hit" else "search_web"


def _search_web_node(state: AgentState, search_service: TavilySearchService) -> dict[str, Any]:
    query = state["query"]
    documents = search_service.search_top_documents(query=query, max_results=3)
    sources = [{"title": doc.title, "url": doc.url} for doc in documents if doc.url]
    return {"documents": documents, "sources": sources}


def _fetch_pages_node(state: AgentState, search_service: TavilySearchService) -> dict[str, Any]:
    pages_markdown: list[str] = []
    for doc in state.get("documents", [])[:3]:
        markdown = search_service.fetch_page_as_markdown(doc.url)
        if markdown.strip():
            pages_markdown.append(f"# {doc.title}\nSource: {doc.url}\n\n{markdown}")
        elif doc.content.strip():
            pages_markdown.append(f"# {doc.title}\nSource: {doc.url}\n\n{doc.content}")
    return {"pages_markdown": pages_markdown}


def _summarize_node(state: AgentState, model: AzureChatOpenAI) -> dict[str, str]:
    query = state["query"]
    pages = state.get("pages_markdown", [])

    if not pages:
        return {"summary": "No web content was retrieved."}

    joined_context = "\n\n---\n\n".join(pages)
    prompt = (
        "Summarize the following web content for the user query. "
        "Focus on factual points, avoid speculation, and keep it concise.\n\n"
        f"User query: {query}\n\n"
        f"Web content:\n{joined_context}"
    )

    response = model.invoke(
        [
            SystemMessage(content="You summarize retrieved web pages into grounded notes."),
            HumanMessage(content=prompt),
        ]
    )
    return {"summary": response.content if isinstance(response.content, str) else str(response.content)}


def _ingest_memory_node(
    state: AgentState,
    memory_store: RedisMemoryStore,
    embeddings: AzureOpenAIEmbeddings,
) -> dict[str, int]:
    topic = state.get("topic", "general")
    inserted_total = 0

    for doc, page in zip(state.get("documents", []), state.get("pages_markdown", [])):
        chunks = _chunk_text(page)
        if not chunks:
            continue
        chunk_vectors = embeddings.embed_documents(chunks)
        inserted_total += memory_store.upsert_chunks(
            chunks=chunks,
            embeddings=chunk_vectors,
            title=doc.title,
            source_url=doc.url,
            topic=topic,
        )

    return {"ingested_chunk_count": inserted_total}


def _answer_from_memory_node(state: AgentState, model: AzureChatOpenAI) -> dict[str, str]:
    query = state["query"]
    context = state.get("memory_context", "")

    prompt = (
        "Answer the user question using only memory context below. "
        "If insufficient, explicitly say memory lacks detail.\n\n"
        f"Question: {query}\n\n"
        f"Memory context:\n{context}"
    )

    response = model.invoke(
        [
            SystemMessage(content="You are a grounded assistant that answers from retrieved memory."),
            HumanMessage(content=prompt),
        ]
    )
    return {"answer": response.content if isinstance(response.content, str) else str(response.content)}


def _answer_node(state: AgentState, model: AzureChatOpenAI) -> dict[str, str]:
    query = state["query"]
    summary = state.get("summary", "")

    prompt = (
        "Answer the user question using only the provided summary. "
        "If the summary is insufficient, say what is missing.\n\n"
        f"Question: {query}\n\n"
        f"Summary:\n{summary}"
    )

    response = model.invoke(
        [
            SystemMessage(content="You are a helpful, grounded assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return {"answer": response.content if isinstance(response.content, str) else str(response.content)}


def _log_turn(state: AgentState, log_path: Path = Path("logs/turns.jsonl")) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "route": state.get("route", "memory_miss"),
        "top_similarity": float(state.get("top_similarity", 0.0)),
        "topic": state.get("topic", "general"),
        "memory_hit": state.get("route") == "memory_hit",
        "query": state.get("query", ""),
        "ingested_chunk_count": int(state.get("ingested_chunk_count", 0)),
        "source_count": len(state.get("sources", [])),
        "sources": state.get("sources", []),
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_graph(
    model: AzureChatOpenAI,
    embeddings: AzureOpenAIEmbeddings,
    search_service: TavilySearchService,
    memory_store: RedisMemoryStore,
    memory_similarity_threshold: float,
    memory_k: int = 5,
):
    graph = StateGraph(AgentState)

    graph.add_node("embed_query", lambda s: _embed_query_node(s, embeddings))
    graph.add_node(
        "search_memory",
        lambda s: _search_memory_node(
            s,
            memory_store=memory_store,
            threshold=memory_similarity_threshold,
            memory_k=memory_k,
        ),
    )
    graph.add_node("answer_from_memory", lambda s: _answer_from_memory_node(s, model))
    graph.add_node("search_web", lambda s: _search_web_node(s, search_service))
    graph.add_node("fetch_pages", lambda s: _fetch_pages_node(s, search_service))
    graph.add_node("summarize", lambda s: _summarize_node(s, model))
    graph.add_node("ingest_memory", lambda s: _ingest_memory_node(s, memory_store, embeddings))
    graph.add_node("answer", lambda s: _answer_node(s, model))

    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "search_memory")
    graph.add_conditional_edges("search_memory", _should_use_memory, ["answer_from_memory", "search_web"])
    graph.add_edge("answer_from_memory", END)
    graph.add_edge("search_web", "fetch_pages")
    graph.add_edge("fetch_pages", "summarize")
    graph.add_edge("summarize", "ingest_memory")
    graph.add_edge("ingest_memory", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def run_query(agent, query: str) -> AgentState:
    initial_state: AgentState = {
        "query": query,
        "query_embedding": [],
        "route": "memory_miss",
        "top_similarity": 0.0,
        "topic": "general",
        "memory_hits": [],
        "memory_context": "",
        "documents": [],
        "pages_markdown": [],
        "ingested_chunk_count": 0,
        "summary": "",
        "answer": "",
        "sources": [],
    }
    final_state = agent.invoke(initial_state)
    _log_turn(final_state)
    return final_state
