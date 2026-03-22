"""
Discriminated-union DataSource model.
Each source type carries only the fields relevant to its connector.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field


class LocalFileSource(BaseModel):
    type: Literal["local_file"] = "local_file"
    path: str
    format: Literal["csv", "parquet", "json"] = "csv"


class S3Source(BaseModel):
    type: Literal["s3"] = "s3"
    bucket: str
    key: str
    region: str = "us-east-1"
    # Credentials fall back to Config / IAM role if omitted
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class GCSSource(BaseModel):
    type: Literal["gcs"] = "gcs"
    bucket: str
    blob: str
    # Path to service-account JSON; falls back to GOOGLE_APPLICATION_CREDENTIALS env var
    credentials_path: Optional[str] = None


class SFTPSource(BaseModel):
    type: Literal["sftp"] = "sftp"
    host: str
    port: int = 22
    username: str
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    remote_path: str


class DatabaseSource(BaseModel):
    type: Literal["database"] = "database"
    # Falls back to Config.database_url if omitted
    database_url: Optional[str] = None
    query: str


class KafkaSource(BaseModel):
    type: Literal["kafka"] = "kafka"
    bootstrap_servers: str          # e.g. "broker:9092"
    topic: str
    group_id: str = "cia-consumer"
    max_messages: int = 1000        # cap to avoid unbounded reads


class APISource(BaseModel):
    type: Literal["api"] = "api"
    url: str
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[dict] = None


# ── Discriminated union ───────────────────────────────────────────────────────

DataSource = Annotated[
    Union[
        LocalFileSource,
        S3Source,
        GCSSource,
        SFTPSource,
        DatabaseSource,
        KafkaSource,
        APISource,
    ],
    Field(discriminator="type"),
]
