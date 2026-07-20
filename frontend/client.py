"""API client for the GraphRAG backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class APIError(Exception):
    """Raised when the API returns an error response."""

    status_code: int
    message: str
    detail: Any = None

    def __str__(self) -> str:
        return f"API error {self.status_code}: {self.message}"


class APIClient:
    """Synchronous HTTP client for the GraphRAG backend API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = base_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            raise APIError(
                status_code=e.response.status_code,
                message=str(e),
                detail=detail,
            ) from e
        except httpx.RequestError as e:
            raise APIError(
                status_code=0,
                message=f"Connection error: {e}",
                detail={"url": str(e.request.url) if e.request else self.base_url},
            ) from e

    def upload_pdf(self, file_path: str) -> dict:
        """Upload a PDF file for ingestion."""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            return self._request("POST", "/documents/upload", files=files)

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return self._request("GET", "/documents")

    def get_document(self, doc_id: str) -> dict:
        """Get metadata for a single document."""
        return self._request("GET", f"/documents/{doc_id}")

    def get_timeline(self, doc_id: str) -> dict:
        """Get execution timeline for a document."""
        return self._request("GET", f"/documents/{doc_id}/timeline")

    def get_graph(self, doc_id: str) -> dict:
        """Get knowledge graph for a document."""
        return self._request("GET", f"/documents/{doc_id}/graph")

    def get_summary(self, doc_id: str) -> dict:
        """Get summary for a document."""
        return self._request("GET", f"/documents/{doc_id}/summary")

    def ask(self, question: str, doc_id: str | None = None, top_k: int = 5) -> dict:
        """Ask a question via the QA endpoint."""
        payload = {"question": question, "top_k": top_k}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        return self._request("POST", "/qa", json=payload)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()