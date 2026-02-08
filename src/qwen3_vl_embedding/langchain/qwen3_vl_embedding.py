import asyncio
from typing import Any, Dict, List

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from qwen3_vl_embedding.client import EmbeddingResponse, HttpxEmbeddingClient
from qwen3_vl_embedding.types import EmbeddingContentPart

DEFAULT_QWEN3_VL_INSTRUCTION = "Represent the user's input."


class Qwen3VLEmbeddings(BaseModel, Embeddings):
    """Qwen3 VL Embedding model using httpx client for LangChain.

    Example:
        ```python
        from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

        embeddings = Qwen3VLEmbeddings(
            base_url="http://localhost:8000/v1",
            model_name="Qwen3-VL-Embedding-2B",
        )

        # Embed text
        text_embedding = embeddings.embed_query("Hello, world!")

        # Embed multiple documents
        doc_embeddings = embeddings.embed_documents(["Doc 1", "Doc 2"])
        ```
    """

    # Configuration for Pydantic
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    _client: HttpxEmbeddingClient = PrivateAttr()

    # Public parameters
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the embedding API",
    )
    model_name: str = Field(
        default="Qwen3-VL-Embedding-2B",
        description="Model name to use for embeddings",
    )
    instruction: str = Field(
        default=DEFAULT_QWEN3_VL_INSTRUCTION,
        description="System instruction for the embedding model",
    )
    api_key: str = Field(
        default="fake",
        description="API key for authentication",
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
    )

    def __init__(self, **kwargs: Any):
        """Initialize the Qwen3VLEmbeddings.

        Args:
            **kwargs: Keyword arguments for configuration
        """
        super().__init__(**kwargs)
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

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as a list of floats
        """
        content: List[EmbeddingContentPart] = [
            {"type": "text", "text": text},
        ]
        return self._get_embedding_from_content(content)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return await asyncio.to_thread(self.embed_query, text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            content: List[EmbeddingContentPart] = [
                {"type": "text", "text": text},
            ]
            embedding = self._get_embedding_from_content(content)
            embeddings.append(embedding)
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_image(self, image_url: str) -> List[float]:
        """Embed an image from URL or file path.

        Args:
            image_url: URL or file path of the image

        Returns:
            Embedding vector as a list of floats
        """
        content: List[EmbeddingContentPart] = [
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        return self._get_embedding_from_content(content)

    async def aembed_image(self, image_url: str) -> List[float]:
        """Async version of embed_image."""
        return await asyncio.to_thread(self.embed_image, image_url)

    def embed_images(self, image_urls: List[str]) -> List[List[float]]:
        """Embed multiple images.

        Args:
            image_urls: List of image URLs or file paths

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for image_url in image_urls:
            embedding = self.embed_image(image_url)
            embeddings.append(embedding)
        return embeddings

    async def aembed_images(self, image_urls: List[str]) -> List[List[float]]:
        """Async version of embed_images."""
        return await asyncio.to_thread(self.embed_images, image_urls)

    def embed_video(self, video_url: str) -> List[float]:
        """Embed a video from URL.

        Args:
            video_url: URL of the video

        Returns:
            Embedding vector as a list of floats
        """
        content: List[EmbeddingContentPart] = [
            {"type": "video_url", "video_url": {"url": video_url}}
        ]
        return self._get_embedding_from_content(content)

    async def aembed_video(self, video_url: str) -> List[float]:
        """Async version of embed_video."""
        return await asyncio.to_thread(self.embed_video, video_url)

    def embed_videos(self, video_urls: List[str]) -> List[List[float]]:
        """Embed multiple videos.

        Args:
            video_urls: List of video URLs

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for video_url in video_urls:
            embedding = self.embed_video(video_url)
            embeddings.append(embedding)
        return embeddings

    async def aembed_videos(self, video_urls: List[str]) -> List[List[float]]:
        """Async version of embed_videos."""
        return await asyncio.to_thread(self.embed_videos, video_urls)

    def embed_multimodal(self, content: List[EmbeddingContentPart]) -> List[float]:
        """Embed multimodal content (text, images, videos).

        Args:
            content: List of content items with type and data.
                Example:
                    [
                        {"type": "text", "text": "Hello"},
                        {"type": "image_url", "image_url": {"url": "file://path/to/image.jpg"}},
                    ]

        Returns:
            Embedding vector as a list of floats
        """
        return self._get_embedding_from_content(content)

    async def aembed_multimodal(
        self, content: List[EmbeddingContentPart]
    ) -> List[float]:
        """Async version of embed_multimodal."""
        return await asyncio.to_thread(self._get_embedding_from_content, content)
