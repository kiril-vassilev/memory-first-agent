from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from markdownify import markdownify as md
from tavily import TavilyClient


@dataclass
class SearchDocument:
    title: str
    url: str
    content: str


class TavilySearchService:
    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key=api_key)

    def search_top_documents(self, query: str, max_results: int = 3) -> list[SearchDocument]:
        response = self._client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = response.get("results", [])

        docs: list[SearchDocument] = []
        for item in results[:max_results]:
            docs.append(
                SearchDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )
            )
        return docs

    def fetch_page_as_markdown(self, url: str) -> str:
        response: Any = self._client.extract(url)

        # Tavily extract responses can vary by SDK version. This handles common shapes.
        if isinstance(response, dict):
            results = response.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                markdown = first.get("raw_content") or first.get("content")
                if isinstance(markdown, str) and markdown.strip():
                    return self._to_markdown(markdown)

            # Fallback if the SDK returns a direct content payload.
            raw_content = response.get("raw_content") or response.get("content")
            if isinstance(raw_content, str) and raw_content.strip():
                return self._to_markdown(raw_content)

        return ""

    @staticmethod
    def _to_markdown(text: str) -> str:
        candidate = text.strip()
        if "<html" in candidate.lower() or "<body" in candidate.lower() or "<p" in candidate.lower():
            return md(candidate)
        return candidate
