from __future__ import annotations

import argparse
import sys

from app.analytics import AnalyticsService
from app.retry_utils import ExternalServiceError, RetryPolicy, run_with_retry
from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from app.config import get_settings
from app.graph import build_graph, run_query
from app.memory_store import RedisMemoryStore
from app.tools import TavilySearchService


def create_app():
    settings = get_settings()
    retry_policy = RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        initial_backoff_seconds=settings.retry_initial_backoff_seconds,
        max_backoff_seconds=settings.retry_max_backoff_seconds,
    )

    model = AzureChatOpenAI(  
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        api_key=SecretStr(settings.azure_openai_api_key),
        azure_endpoint=settings.azure_openai_endpoint,
        temperature=0,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )

    safeguard_model = model
    if settings.azure_openai_safeguard_deployment:
        safeguard_model = AzureChatOpenAI(  
            azure_deployment=settings.azure_openai_safeguard_deployment,
            api_version=settings.azure_openai_api_version,
            api_key=SecretStr(settings.azure_openai_api_key),
            azure_endpoint=settings.azure_openai_endpoint,
            temperature=0,
            timeout=settings.request_timeout_seconds,
            max_retries=0,
        )

    embeddings = AzureOpenAIEmbeddings(  
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version,
        api_key=SecretStr(settings.azure_openai_api_key),
        azure_endpoint=settings.azure_openai_endpoint,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )

    try:
        embedding_dim = len(run_with_retry("OpenAI embedding dimension probe", lambda: embeddings.embed_query("dimension probe"), retry_policy))
    except ExternalServiceError as exc:
        raise ValueError(
            "Embedding deployment check failed. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to an embeddings-capable Azure deployment."
        ) from exc
    memory_store = RedisMemoryStore(
        redis_url=settings.redis_url,
        index_name=settings.redis_index_name,
        embedding_dim=embedding_dim,
        request_timeout_seconds=settings.request_timeout_seconds,
        retry_policy=retry_policy,
    )

    search_service = TavilySearchService(
        api_key=settings.tavily_api_key,
        request_timeout_seconds=settings.request_timeout_seconds,
        retry_policy=retry_policy,
    )
    graph = build_graph(
        model=model,
        safeguard_model=safeguard_model,
        embeddings=embeddings,
        search_service=search_service,
        memory_store=memory_store,
        retry_policy=retry_policy,
        memory_similarity_threshold=settings.memory_similarity_threshold,
        memory_k=settings.memory_k,
        tavily_max_results=settings.tavily_max_results,
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
    except (ValueError, ExternalServiceError) as exc:
        print(f"Execution error: {exc}")
        sys.exit(1)

    try:
        result = run_query(app, query)
    except ExternalServiceError as exc:
        print(f"Execution error: {exc}")
        sys.exit(1)

    print("\nAnswer:\n")
    print(result.get("answer", ""))
    print(f"\nRoute: {result.get('route', 'memory_miss')}")
    print(f"Top similarity: {result.get('top_similarity', 0.0):.3f}")
    print("\nSources:")
    for source in result.get("sources", []):
        print(f"- {source.get('title', 'Untitled')}: {source.get('url', '')}")


if __name__ == "__main__":
    main()
