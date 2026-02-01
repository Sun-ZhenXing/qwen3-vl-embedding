import asyncio
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from qwen3_vl_embedding.client import EmbeddingResponse, HttpxEmbeddingClient
from qwen3_vl_embedding.types import EmbeddingContentPart

DEFAULT_QWEN3_VL_INSTRUCTION = "Represent the user's input."


class Qwen3VLEmbedding(BaseEmbedding):
    """Qwen3 VL Embedding model using httpx client."""

    _client: HttpxEmbeddingClient = PrivateAttr()

    # input parameters
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen3-VL-Embedding-2B"
    instruction: str = DEFAULT_QWEN3_VL_INSTRUCTION
    api_key: str = "fake"
    timeout: int = 30

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        instruction: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize the Qwen3VLEmbedding.

        Args:
            base_url: Base URL for the embedding API
            model_name: Model name to use for embeddings
            instruction: System instruction for the embedding model
            api_key: API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for BaseEmbedding
        """
        super().__init__(**kwargs)
        if base_url is not None:
            self.base_url = base_url
        if model_name is not None:
            self.model_name = model_name
        if instruction is not None:
            self.instruction = instruction
        if api_key is not None:
            self.api_key = api_key
        if timeout is not None:
            self.timeout = timeout

        self._client = HttpxEmbeddingClient(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _create_chat_embeddings(
        self,
        messages: List[Dict[str, Any]],
        encoding_format: str = "float",
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
    ) -> EmbeddingResponse:
        """Create chat embeddings using vLLM's Chat Embeddings API.

        This is a convenience function for accessing vLLM's Chat Embeddings API,
        which is an extension of OpenAI's existing Embeddings API.

        Args:
            messages: List of messages for chat embeddings
            encoding_format: Format of the embedding (base64 or float)
            continue_final_message: Whether to continue final message
            add_special_tokens: Whether to add special tokens

        Returns:
            Embedding response
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "messages": messages,
            "model": self.model_name,
            "encoding_format": encoding_format,
            "continue_final_message": continue_final_message,
            "add_special_tokens": add_special_tokens,
        }

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    async def _acreate_chat_embeddings(
        self,
        messages: List[Dict[str, Any]],
        encoding_format: str = "float",
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
    ) -> EmbeddingResponse:
        """Async version of _create_chat_embeddings."""
        return await asyncio.to_thread(
            self._create_chat_embeddings,
            messages,
            encoding_format,
            continue_final_message,
            add_special_tokens,
        )

    def _get_embedding_from_content(
        self, content: List[EmbeddingContentPart]
    ) -> List[float]:
        """Get embedding from multimodal content.

        Args:
            content: List of content items (text, images, videos)

        Returns:
            Embedding vector
        """
        response = self._create_chat_embeddings(
            messages=[
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
            encoding_format="float",
            continue_final_message=True,
            add_special_tokens=True,
        )
        return response["data"][0]["embedding"]

    async def _aget_embedding_from_content(
        self, content: List[EmbeddingContentPart]
    ) -> List[float]:
        """Async version of _get_embedding_from_content."""
        return await asyncio.to_thread(self._get_embedding_from_content, content)

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding from text.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        content: List[EmbeddingContentPart] = [
            {"type": "text", "text": query},
        ]
        return self._get_embedding_from_content(content)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async version of _get_query_embedding."""
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        content: List[EmbeddingContentPart] = [
            {"type": "text", "text": text},
        ]
        return self._get_embedding_from_content(content)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async version of _get_text_embedding."""
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_image_embedding(self, image_url: str) -> Embedding:
        """Get image embedding from URL.

        Args:
            image_url: URL or file path of the image

        Returns:
            Embedding vector
        """
        content: List[EmbeddingContentPart] = [
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        return self._get_embedding_from_content(content)

    async def _aget_image_embedding(self, image_url: str) -> Embedding:
        """Async version of _get_image_embedding."""
        return await asyncio.to_thread(self._get_image_embedding, image_url)

    def _get_video_embedding(self, video_url: str) -> Embedding:
        """Get video embedding from URL.

        Args:
            video_url: URL of the video

        Returns:
            Embedding vector
        """
        content: List[EmbeddingContentPart] = [
            {"type": "video_url", "video_url": {"url": video_url}}
        ]
        return self._get_embedding_from_content(content)

    async def _aget_video_embedding(self, video_url: str) -> Embedding:
        """Async version of _get_video_embedding."""
        return await asyncio.to_thread(self._get_video_embedding, video_url)

    def _get_multimodal_embedding(
        self, content: List[EmbeddingContentPart]
    ) -> Embedding:
        """Get embedding from multimodal content (text, images, videos).

        Args:
            content: List of content items with type and data

        Returns:
            Embedding vector
        """
        return self._get_embedding_from_content(content)

    async def _aget_multimodal_embedding(
        self, content: List[EmbeddingContentPart]
    ) -> Embedding:
        """Async version of _get_multimodal_embedding."""
        return await asyncio.to_thread(self._get_embedding_from_content, content)
