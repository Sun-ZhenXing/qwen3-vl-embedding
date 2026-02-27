"""LlamaIndex integration for Aliyun DashScope Embedding API."""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding

from ..aliyun_client import AliyunEmbeddingClient, Region


class AliyunEmbedding(BaseEmbedding):
    """LlamaIndex-compatible embedding class for Aliyun DashScope API.

    This class provides a LlamaIndex-compatible interface for generating
    multimodal fusion embeddings using the Aliyun DashScope API.

    Example:
        >>> from qwen3_vl_embedding.llama_index import AliyunEmbedding
        >>> embedding = AliyunEmbedding(api_key="sk-xxx")
        >>> text_embedding = embedding.get_text_embedding("Hello")
        >>> image_embedding = embedding.get_image_embedding("https://example.com/image.jpg")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "qwen3-vl-embedding",
        region: str = "cn",
        dimension: int | None = None,
        embed_batch_size: int = 10,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize the Aliyun embedding wrapper.

        Args:
            api_key: Aliyun DashScope API key. If not provided, will try to read
                from DASHSCOPE_API_KEY environment variable.
            model_name: Model name to use. Defaults to "qwen3-vl-embedding".
            region: API region, either "cn" (China) or "sg" (Singapore).
                Defaults to "cn".
            dimension: Desired embedding dimension. Valid values: 256, 512, 768,
                1024, 1536, 2048, 2560. Defaults to None (model default).
            embed_batch_size: Batch size for embedding requests. Defaults to 10.
            timeout: Request timeout in seconds. Defaults to 60.0.
            **kwargs: Additional arguments passed to BaseEmbedding.
        """
        super().__init__(
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._client = AliyunEmbeddingClient(
            api_key=api_key,
            model_name=model_name,
            region=Region(region),
            timeout=timeout,
        )
        self._model_name = model_name
        self._region = region
        self._dimension = dimension

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self._client.embed_fusion(text=query, dimension=self._dimension)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self._client.embed_fusion(text=text, dimension=self._dimension)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [
            self._client.embed_fusion(text=text, dimension=self._dimension)
            for text in texts
        ]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get embedding for a query string.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return await asyncio.to_thread(
            self._client.embed_fusion, text=query, dimension=self._dimension
        )

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await asyncio.to_thread(
            self._client.embed_fusion, text=text, dimension=self._dimension
        )

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await asyncio.gather(
            *[
                asyncio.to_thread(
                    self._client.embed_fusion, text=text, dimension=self._dimension
                )
                for text in texts
            ]
        )

    def get_image_embedding(self, image_url: str) -> list[float]:
        """Get fusion embedding for an image.

        Args:
            image_url: URL of the image to embed

        Returns:
            Fusion embedding vector for the image (using empty text)
        """
        return self._client.embed_fusion(
            text="", image=image_url, dimension=self._dimension
        )

    def get_fusion_embedding(
        self,
        text: str | None = None,
        image: str | None = None,
        video: str | None = None,
    ) -> list[float]:
        """Generate a fusion embedding combining multiple modalities.

        Args:
            text: Text content to embed
            image: Image URL to embed
            video: Video URL to embed

        Returns:
            Fusion embedding vector
        """
        return self._client.embed_fusion(
            text=text,
            image=image,
            video=video,
            dimension=self._dimension,
        )
