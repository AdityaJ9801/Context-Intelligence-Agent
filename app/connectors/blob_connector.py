"""Azure Blob Storage connector — downloads blob via SDK (authenticated)."""
from __future__ import annotations

import asyncio
import io
import os
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
from azure.storage.blob import BlobServiceClient

from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import AzureBlobSource
from app.utils.sampler import smart_sample


def _parse_blob_url(url: str) -> tuple[str, str]:
    """Extract (container_name, blob_name) from an Azure Blob URL."""
    # https://<account>.blob.core.windows.net/<container>/<blob_name>
    parsed = urlparse(url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(f"Cannot parse blob URL: {url}")
    return path_parts[0], path_parts[1]


def _get_blob_client(container: str, blob_name: str):
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise ConnectorError("azure_blob", "AZURE_STORAGE_CONNECTION_STRING not set")
    svc = BlobServiceClient.from_connection_string(conn_str)
    return svc.get_blob_client(container=container, blob=blob_name)


class BlobConnector(BaseConnector):
    """Read and adaptively sample an Azure Blob Storage object via SDK."""

    def __init__(self, source: AzureBlobSource) -> None:
        super().__init__(source)
        self._source: AzureBlobSource = source

    async def connect(self) -> None:
        """Verify the blob is accessible using Azure SDK credentials."""
        try:
            container, blob_name = _parse_blob_url(self._source.path)
            client = _get_blob_client(container, blob_name)
            # exists() check — runs in thread to avoid blocking async loop
            exists = await asyncio.to_thread(client.exists)
            if not exists:
                raise ConnectorError("azure_blob", f"Blob not found: {self._source.path}")
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("azure_blob", str(exc)) from exc

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Download and return a sampled DataFrame from the blob."""
        try:
            container, blob_name = _parse_blob_url(self._source.path)
            client = _get_blob_client(container, blob_name)

            def _download() -> bytes:
                downloader = client.download_blob()
                return downloader.readall()

            content = await asyncio.to_thread(_download)
            buf = io.BytesIO(content)

            def _read() -> pd.DataFrame:
                if self._source.format == "parquet":
                    return pd.read_parquet(buf)
                if self._source.format == "json":
                    return pd.read_json(buf)
                return pd.read_csv(buf)

            df = await asyncio.to_thread(_read)
            return await smart_sample(df, target_col)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("azure_blob", str(exc)) from exc

