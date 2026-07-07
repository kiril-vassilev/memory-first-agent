from __future__ import annotations

import argparse
import sys

from app.analytics import AnalyticsService
from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from app.config import get_settings
from app.graph import build_graph, run_query
from app.memory_store import RedisMemoryStore
from app.tools import TavilySearchService


def create_app():
    settings = get_settings()

    model = AzureChatOpenAI(  
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        api_key=SecretStr(settings.azure_openai_api_key),
        azure_endpoint=settings.azure_openai_endpoint,
        temperature=0,
    )

    safeguard_model = model
    if settings.azure_openai_safeguard_deployment:
        safeguard_model = AzureChatOpenAI(  
            azure_deployment=settings.azure_openai_safeguard_deployment,
            api_version=settings.azure_openai_api_version,
            api_key=SecretStr(settings.azure_openai_api_key),
            azure_endpoint=settings.azure_openai_endpoint,
            temperature=0,
        )

    embeddings = AzureOpenAIEmbeddings(  
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version,
        api_key=SecretStr(settings.azure_openai_api_key),
        azure_endpoint=settings.azure_openai_endpoint,
    )

    try:
        embedding_dim = len(embeddings.embed_query("dimension probe"))
    except Exception as exc:
        raise ValueError(
            "Embedding deployment check failed. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to an embeddings-capable Azure deployment."
        ) from exc
    memory_store = RedisMemoryStore(
        redis_url=settings.redis_url,
        index_name=settings.redis_index_name,
        embedding_dim=embedding_dim,
    )

    search_service = TavilySearchService(api_key=settings.tavily_api_key)
    graph = build_graph(
        model=model,
        safeguard_model=safeguard_model,
        embeddings=embeddings,
        search_service=search_service,
        memory_store=memory_store,
        memory_similarity_threshold=settings.memory_similarity_threshold,
        memory_k=10,
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory-First Agent: Redis memory-first routing with web fallback"
    )
    parser.add_argument("query", nargs="*", help="User question to search and answer")
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Generate analytics dashboard files from logs/turns.jsonl",
    )
    args = parser.parse_args()

    if args.analytics:
        analytics = AnalyticsService()
        summary_json_path, dashboard_html_path = analytics.write_dashboard_files()
        print("Analytics generated:")
        print(f"- {summary_json_path}")
        print(f"- {dashboard_html_path}")
        return

    if not args.query:
        parser.error("Provide a query or use --analytics")

    query = " ".join(args.query)
    try:
        app = create_app()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    result = run_query(app, query)

    print("\nAnswer:\n")
    print(result.get("answer", ""))
    print(f"\nRoute: {result.get('route', 'memory_miss')}")
    print(f"Top similarity: {result.get('top_similarity', 0.0):.3f}")
    print("\nSources:")
    for source in result.get("sources", []):
        print(f"- {source.get('title', 'Untitled')}: {source.get('url', '')}")


if __name__ == "__main__":
    main()
