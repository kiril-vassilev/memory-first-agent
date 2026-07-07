from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, cast
import uuid

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


@dataclass
class MemoryHit:
    id: str
    content: str
    title: str
    source_url: str
    topic: str
    similarity: float


class RedisMemoryStore:
    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding_dim: int,
    ) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=False)
        self._index_name = index_name
        self._prefix = "mem:chunk:"
        self._embedding_dim = embedding_dim
        self._available = False
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._client.ft(self._index_name).info()
            self._available = True
            return
        except Exception:
            pass

        schema = [
            TextField("content"),
            TextField("title"),
            TagField("source_url"),
            TagField("topic"),
            TagField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        ]

        definition = IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH)
        try:
            self._client.ft(self._index_name).create_index(schema, definition=definition)
            self._available = True
        except Exception:
            self._available = False

    @staticmethod
    def _to_vector_bytes(vector: list[float]) -> bytes:
        return np.array(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _stable_chunk_hash(content: str, source_url: str) -> str:
        digest = hashlib.sha256(f"{source_url}::{content}".encode("utf-8")).hexdigest()
        return digest[:20]

    @staticmethod
    def _normalize_topic_filter(topic_filter: str | None) -> str | None:
        if topic_filter is None:
            return None
        normalized = topic_filter.strip().lower()
        if not normalized or normalized == "general":
            return None
        return normalized

    @staticmethod
    def _escape_tag_value(value: str) -> str:
        # Escape RediSearch tag-value special characters used by query syntax.
        special = {",", ".", "<", ">", "{", "}", "[", "]", '"', "'", ":", ";", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "~", "|", " ", "\\"}
        return "".join(f"\\{ch}" if ch in special else ch for ch in value)

    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        title: str,
        source_url: str,
        topic: str,
    ) -> int:
        if not self._available:
            return 0

        inserted = 0
        now = datetime.now(tz=timezone.utc).isoformat()

        for chunk, vector in zip(chunks, embeddings):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_hash = self._stable_chunk_hash(chunk, source_url)
            key = f"{self._prefix}{chunk_hash}"

            mapping: dict[str, str | bytes] = {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "title": title,
                "source_url": source_url,
                "topic": topic,
                "created_at": now,
                "embedding": self._to_vector_bytes(vector),
            }
            self._client.hset(key, mapping=cast(Any, mapping))
            inserted += 1

        return inserted

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        topic_filter: str | None = None,
    ) -> list[MemoryHit]:
        if not self._available:
            return []

        normalized_topic = self._normalize_topic_filter(topic_filter)

        def _run_search(active_topic: str | None) -> Any:
            base = "*"
            if active_topic:
                escaped_topic = self._escape_tag_value(active_topic)
                base = f"@topic:{{{escaped_topic}}}"

            q = (
                Query(f"{base}=>[KNN {k} @embedding $vec AS distance]")
                .sort_by("distance")
                .return_fields("id", "content", "title", "source_url", "topic", "distance")
                .paging(0, k)
                .dialect(2)
            )

            params: dict[str, str | int | float | bytes] = {
                "vec": self._to_vector_bytes(query_embedding)
            }
            return self._client.ft(self._index_name).search(q, query_params=params)

        try:
            results: Any = _run_search(normalized_topic)
            if normalized_topic and not getattr(results, "docs", []):
                results = _run_search(None)
        except Exception:
            self._available = False
            return []

        hits: list[MemoryHit] = []
        for doc in cast(list[Any], getattr(results, "docs", [])):
            # print(f"Memory search hit: {getattr(doc, 'title', '')} - {getattr(doc, 'source_url', '')}; Distance: {getattr(doc, 'distance', 1.0)}")
            distance = float(getattr(doc, "distance", 1.0))
            similarity = max(0.0, min(1.0, 1.0 - distance))
            hits.append(
                MemoryHit(
                    id=(getattr(doc, "id", b"") or b"").decode("utf-8", errors="ignore")
                    if isinstance(getattr(doc, "id", b""), bytes)
                    else str(getattr(doc, "id", "")),
                    content=(getattr(doc, "content", b"") or b"").decode("utf-8", errors="ignore")
                    if isinstance(getattr(doc, "content", b""), bytes)
                    else str(getattr(doc, "content", "")),
                    title=(getattr(doc, "title", b"") or b"").decode("utf-8", errors="ignore")
                    if isinstance(getattr(doc, "title", b""), bytes)
                    else str(getattr(doc, "title", "")),
                    source_url=(getattr(doc, "source_url", b"") or b"").decode("utf-8", errors="ignore")
                    if isinstance(getattr(doc, "source_url", b""), bytes)
                    else str(getattr(doc, "source_url", "")),
                    topic=(getattr(doc, "topic", b"") or b"").decode("utf-8", errors="ignore")
                    if isinstance(getattr(doc, "topic", b""), bytes)
                    else str(getattr(doc, "topic", "")),
                    similarity=similarity,
                )
            )

        return hits
