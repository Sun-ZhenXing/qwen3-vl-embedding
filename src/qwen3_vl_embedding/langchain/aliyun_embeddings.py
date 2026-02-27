"""LangChain integration for Aliyun DashScope Embedding API."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from ..aliyun_client import AliyunEmbeddingClient, Region


class AliyunEmbeddings(Embeddings):
    """LangChain-compatible embeddings class for Aliyun DashScope API.

    This class provides a LangChain-compatible interface for generating
    multimodal fusion embeddings using the Aliyun DashScope API.

    Example:
        >>> from qwen3_vl_embedding.langchain import AliyunEmbeddings
        >>> embeddings = AliyunEmbeddings(api_key="sk-xxx")
        >>> text_embeddings = embeddings.embed_documents(["Hello", "World"])
        >>> query_embedding = embeddings.embed_query("Hello")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen3-vl-embedding",
        region: str = "cn",
        dimension: int | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize the Aliyun embeddings wrapper.

        Args:
            api_key: Aliyun DashScope API key. If not provided, will try to read
                from DASHSCOPE_API_KEY environment variable.
            model: Model name to use. Defaults to "qwen3-vl-embedding".
            region: API region, either "cn" (China) or "sg" (Singapore).
                Defaults to "cn".
            dimension: Desired embedding dimension. Valid values: 256, 512, 768,
                1024, 1536, 2048, 2560. Defaults to None (model default).
            timeout: Request timeout in seconds. Defaults to 60.0.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        self._client = AliyunEmbeddingClient(
            api_key=api_key,
            model_name=model,
            region=Region(region),
            timeout=timeout,
        )
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        return [
            self._client.embed_fusion(text=text, dimension=self.dimension)
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """
        return self._client.embed_fusion(text=text, dimension=self.dimension)

    def embed_image(self, image_url: str) -> list[float]:
        """Embed an image.

        Args:
            image_url: URL of the image to embed

        Returns:
            Fusion embedding vector for the image (using empty text)
        """
        return self._client.embed_fusion(
            text="", image=image_url, dimension=self.dimension
        )

    def embed_fusion(
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
            dimension=self.dimension,
        )
