"""S3-backed lead store — the cloud backend.

Persists the entire lead list as a single JSON object in a bucket. This keeps
the code trivial and always-free, at the cost of read-modify-write races if two
Lambda invocations write concurrently (last write wins). That trade-off is
acceptable for a low-traffic public demo and is documented as a known
limitation; a production system would use per-lead objects or a real database.
"""

from __future__ import annotations

import json

import boto3

from app.storage.base import ListBackedStore


class S3Store(ListBackedStore):
    def __init__(
        self,
        bucket: str,
        key: str = "leads.json",
        client=None,
    ) -> None:
        self._bucket = bucket
        self._key = key
        self._client = client or boto3.client("s3")

    def _load(self) -> list[dict]:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=self._key)
        except self._client.exceptions.NoSuchKey:
            return []
        return json.loads(obj["Body"].read())

    def _save(self, leads: list[dict]) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=json.dumps(leads, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
