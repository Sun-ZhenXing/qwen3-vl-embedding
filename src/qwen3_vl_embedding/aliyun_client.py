"""Aliyun (DashScope) Embedding API Client.

This module provides a client for the Aliyun DashScope multimodal embedding API,
supporting fusion embeddings that combine text, image, and video.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

import httpx

# API Endpoints
DASHSCOPE_API_URL_CN = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
DASHSCOPE_API_URL_SG = "https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

# Default model
DEFAULT_MODEL = "qwen3-vl-embedding"

# Default dimension
DEFAULT_DIMENSION = 2560

# Valid dimensions for qwen3-vl-embedding
VALID_DIMENSIONS = {256, 512, 768, 1024, 1536, 2048, 2560}


class Region(str, Enum):
    """Aliyun DashScope API regions."""

    CN = "cn"  # China (Beijing)
    SG = "sg"  # Singapore


class AliyunAPIError(Exception):
    """Exception raised for Aliyun API errors.

    Attributes:
        message: Explanation of the error
        status_code: HTTP status code
        error_code: Aliyun error code if available
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

    def __str__(self) -> str:
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


class AliyunEmbeddingClient:
    """Client for Aliyun DashScope multimodal embedding API.

    This client supports generating fusion embeddings that combine text, image,
    and video into a single vector representation using the qwen3-vl-embedding model.

    Example:
        >>> client = AliyunEmbeddingClient(api_key="sk-xxx")
        >>> embedding = client.embed_fusion(
        ...     text="A beautiful sunset",
        ...     image="https://example.com/sunset.jpg",
        ...     dimension=1024
        ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        region: Region | str = Region.CN,
        timeout: float = 60.0,
    ):
        """Initialize the Aliyun embedding client.

        Args:
            api_key: Aliyun DashScope API key. If not provided, will try to read
                from DASHSCOPE_API_KEY environment variable.
            model_name: Model name to use. Defaults to "qwen3-vl-embedding".
            region: API region, either "cn" (China) or "sg" (Singapore).
                Defaults to "cn".
            timeout: Request timeout in seconds. Defaults to 60.0.

        Raises:
            AliyunAPIError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise AliyunAPIError(
                "API key is required. Provide it as an argument or set DASHSCOPE_API_KEY environment variable."
            )

        self.model_name = model_name
        self.region = Region(region) if isinstance(region, str) else region
        self.timeout = timeout

        # Set API URL based on region
        if self.region == Region.CN:
            self.api_url = DASHSCOPE_API_URL_CN
        else:
            self.api_url = DASHSCOPE_API_URL_SG

        # Initialize HTTP client
        self._client = httpx.Client(timeout=self.timeout)

    def _prepare_request_body(
        self,
        text: str | None = None,
        image: str | None = None,
        video: str | None = None,
        dimension: int | None = None,
    ) -> dict[str, Any]:
        """Prepare the request body for the embedding API.

        Args:
            text: Text content to embed
            image: Image URL to embed
            video: Video URL to embed
            dimension: Desired embedding dimension

        Returns:
            Request body dictionary
        """
        content: dict[str, Any] = {}

        if text:
            content["text"] = text
        if image:
            content["image"] = image
        if video:
            content["video"] = video

        body: dict[str, Any] = {
            "model": self.model_name,
            "input": {
                "contents": [content],
            },
        }

        if dimension is not None:
            if dimension not in VALID_DIMENSIONS:
                raise ValueError(
                    f"Invalid dimension {dimension}. Must be one of: {VALID_DIMENSIONS}"
                )
            body["parameters"] = {"dimension": dimension}

        return body

    def _handle_response(self, response: httpx.Response) -> list[float]:
        """Handle the API response and extract embedding.

        Args:
            response: HTTP response from the API

        Returns:
            Embedding vector as a list of floats

        Raises:
            AliyunAPIError: If the API returns an error
        """
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("output", {}).get("embeddings", [])
            if embeddings:
                return embeddings[0].get("embedding", [])
            raise AliyunAPIError("No embeddings found in response")

        # Handle error responses
        error_msg = "Unknown error"
        error_code = None
        try:
            error_data = response.json()
            error_msg = error_data.get("message", error_msg)
            error_code = error_data.get("code")
        except Exception:
            error_msg = response.text or error_msg

        if response.status_code == 401:
            raise AliyunAPIError(
                "Authentication failed: Invalid API key",
                status_code=401,
                error_code=error_code,
            )
        elif response.status_code == 429:
            raise AliyunAPIError(
                "Rate limit exceeded",
                status_code=429,
                error_code=error_code,
            )
        else:
            raise AliyunAPIError(
                error_msg,
                status_code=response.status_code,
                error_code=error_code,
            )

    def embed_fusion(
        self,
        text: str | None = None,
        image: str | None = None,
        video: str | None = None,
        dimension: int | None = None,
    ) -> list[float]:
        """Generate a fusion embedding combining multiple modalities.

        This method creates a single embedding vector that fuses the provided
        text, image, and/or video content. At least one modality must be provided.

        Args:
            text: Text content to embed
            image: Image URL to embed (must be publicly accessible)
            video: Video URL to embed (must be publicly accessible)
            dimension: Desired embedding dimension. Valid values: 256, 512, 768,
                1024, 1536, 2048, 2560. Defaults to 2560.

        Returns:
            A list of floats representing the fusion embedding vector

        Raises:
            ValueError: If no content is provided or invalid dimension
            AliyunAPIError: If the API request fails

        Example:
            >>> # Text only
            >>> embedding = client.embed_fusion(text="A beautiful sunset")
            >>>
            >>> # Text + Image
            >>> embedding = client.embed_fusion(
            ...     text="A cat playing",
            ...     image="https://example.com/cat.jpg",
            ...     dimension=1024
            ... )
            >>>
            >>> # All modalities
            >>> embedding = client.embed_fusion(
            ...     text="Tutorial video",
            ...     image="https://example.com/thumbnail.jpg",
            ...     video="https://example.com/video.mp4"
            ... )
        """
        if not any([text, image, video]):
            raise ValueError("At least one of text, image, or video must be provided")

        body = self._prepare_request_body(text, image, video, dimension)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(
            self.api_url,
            headers=headers,
            json=body,
        )

        return self._handle_response(response)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> AliyunEmbeddingClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
