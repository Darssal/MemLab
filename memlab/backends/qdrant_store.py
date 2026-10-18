from __future__ import annotations

from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore


class QdrantStore:
    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        vector_size: int = 64,
        shard_number: int = 1,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx package is required for Qdrant operations")
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.vector_size = vector_size
        self.shard_number = max(1, int(shard_number))
        self.client = httpx.AsyncClient(timeout=30.0, trust_env=False)

    async def close(self) -> None:
        await self.client.aclose()

    async def ping(self) -> bool:
        try:
            r = await self.client.get(f"{self.qdrant_url}/collections")
            return r.status_code == 200
        except Exception:
            return False

    async def ensure_collection(self) -> None:
        payload = {
            "vectors": {
                "size": self.vector_size,
                "distance": "Cosine",
            },
            "shard_number": self.shard_number,
        }
        r = await self.client.put(f"{self.qdrant_url}/collections/{self.collection}", json=payload)
        if r.status_code == 409:
            return
        if r.status_code >= 400:
            raise RuntimeError(f"Qdrant ensure_collection failed: HTTP {r.status_code} {r.text}")

    async def upsert(self, point_id: str | int, vector: list[float], payload: dict[str, Any]) -> None:
        body = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }
            ]
        }
        r = await self.client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json=body,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Qdrant upsert failed: HTTP {r.status_code} {r.text}")

    async def upsert_many(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        body = {"points": points}
        r = await self.client.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json=body,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Qdrant upsert_many failed: HTTP {r.status_code} {r.text}")

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if agent_id:
            body["filter"] = {
                "must": [
                    {
                        "key": "agent_id",
                        "match": {"value": agent_id},
                    }
                ]
            }
        r = await self.client.post(
            f"{self.qdrant_url}/collections/{self.collection}/points/search",
            json=body,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Qdrant search failed: HTTP {r.status_code} {r.text}")
        result = r.json().get("result", [])
        if not isinstance(result, list):
            raise RuntimeError("Qdrant search response does not contain result list")
        return result
