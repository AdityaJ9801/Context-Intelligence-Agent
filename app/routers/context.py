"""Context router — cache retrieval and pipeline refresh endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.cache.redis_cache import ContextCache
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.utils.pipeline import run_pipeline

router = APIRouter(tags=["context"])


def get_cache() -> ContextCache:
    """Dependency that provides a ContextCache instance."""
    return ContextCache()


@router.get(
    "/context/{context_id}",
    response_model=ContextObject,
    summary="Retrieve a cached ContextObject",
)
async def get_context(
    context_id: str,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """Return the cached ContextObject for context_id, or 404 if not found."""
    context = await cache.get_context(context_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached context found for id: {context_id!r}",
        )
    return context


class RefreshRequest(BaseModel):
    """Request body for the refresh endpoint."""

    source: DataSource = Field(..., discriminator="type")


@router.post(
    "/refresh/{context_id}",
    response_model=ContextObject,
    summary="Re-profile a data source and refresh the cache",
)
async def refresh_context(
    context_id: str,  # pylint: disable=unused-argument  # kept for URL symmetry
    body: RefreshRequest,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """Re-run the full pipeline for the given source and overwrite the cache entry."""
    context: ContextObject | None = None
    async for _stage, _pct, payload in run_pipeline(body.source, cache):
        if payload is not None:
            context = payload  # type: ignore[assignment]
    if context is None:
        raise HTTPException(status_code=500, detail="Pipeline completed without a result.")
    return context


@router.post(
    "/run",
    response_model=ContextObject,
    summary="Orchestrator-compatible endpoint for retrieving or profiling context",
)
async def run_task(
    payload: dict,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """
    Orchestrator-compatible endpoint that receives task payloads.
    
    Expected payload structure from orchestrator:
    {
        "query": "...",
        "context_id": "...",  # Optional: if provided, retrieve from cache
        "source": { ... DataSource object ... },  # Optional: if provided, profile it
        "_context": { ... upstream dependencies ... }
    }
    
    If context_id is provided, retrieves cached context.
    If source is provided, profiles the source.
    Otherwise returns an error.
    """
    try:
        # Check if we should retrieve cached context
        context_id = payload.get("context_id")
        if context_id:
            context = await cache.get_context(context_id)
            if context:
                return context
            # If not in cache, fall through to try profiling
        
        # Check if we should profile a new source
        source_data = payload.get("source")
        if source_data:
            source = DataSource(**source_data)
            
            # Run the pipeline
            context: ContextObject | None = None
            async for _stage, _pct, result in run_pipeline(source, cache):
                if result is not None:
                    context = result  # type: ignore[assignment]
            
            if context is None:
                raise HTTPException(status_code=500, detail="Pipeline completed without a result.")
            
            return context
        
        # Neither context_id nor source provided
        raise HTTPException(
            status_code=400, 
            detail="Either context_id (to retrieve cached context) or source (to profile) is required"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context error: {str(e)}")
