"""
Shared pipeline logic used by both the sync and streaming profile endpoints.

Yields (stage, pct, result) tuples so the streaming endpoint can emit SSE
events while the sync endpoint simply awaits the final result.
"""
from __future__ import annotations

from typing import AsyncGenerator

from app.cache.redis_cache import ContextCache
from app.connectors.factory import get_connector
from app.llm.summarizer import SemanticEnricher
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.profilers.schema_profiler import DataProfiler


async def run_pipeline(
    source: DataSource,
    cache: ContextCache,
) -> AsyncGenerator[tuple[str, int, object], None]:
    """
    Async generator that drives the full profiling pipeline.

    Yields: (stage_name, pct_complete, payload)
      - payload is None for intermediate stages
      - payload is the ContextObject on the final "complete" stage
    """
    cache_key = ContextCache.generate_key(source)

    # ── 1. Cache check ────────────────────────────────────────────────────────
    yield ("cache_check", 5, None)
    cached = await cache.get_context(cache_key)
    if cached is not None:
        yield ("complete", 100, cached)
        return

    # ── 2. Connect & sample ───────────────────────────────────────────────────
    yield ("connecting", 15, None)
    connector = get_connector(source)
    await connector.connect()

    yield ("sampling", 30, None)
    df = await connector.sample()

    # ── 3. Profile ────────────────────────────────────────────────────────────
    yield ("profiling", 55, None)
    profiler = DataProfiler(df)
    columns = await profiler.profile()

    # ── 4. LLM enrichment ─────────────────────────────────────────────────────
    yield ("enriching", 75, None)
    enricher = SemanticEnricher()
    llm_result = await enricher.enrich_profile(columns)

    # ── 5. Assemble ContextObject ─────────────────────────────────────────────
    yield ("assembling", 90, None)
    context = ContextObject(
        source_id=cache_key,
        source_type=source.type,  # type: ignore[union-attr]
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        metadata={
            "suggested_analyses": llm_result.get("suggested_analyses", []),
            "source": source.model_dump(),  # Include original source for downstream agents
        },
    )

    # ── 6. Cache & return ─────────────────────────────────────────────────────
    yield ("caching", 95, None)
    await cache.set_context(cache_key, context)

    yield ("complete", 100, context)
