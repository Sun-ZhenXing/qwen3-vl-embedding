from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NotRequired,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import filetype
import httpx

from qwen3_vl_embedding.types import EmbeddingContentPart

DEFAULT_IMAGE_MIME_TYPE = "image/png"


class ChatCompletionMessage(TypedDict):
    """A message in the chat completion request/response."""

    role: str
    content: str
    name: NotRequired[str]
    refusal: NotRequired[str]
    tool_calls: NotRequired[List[Dict[str, Any]]]


class ChatCompletionChoice(TypedDict):
    """A choice in the chat completion response."""

    index: int
    message: ChatCompletionMessage
    logprobs: NotRequired[Dict[str, Any]]
    finish_reason: NotRequired[str]


class ChatCompletion(TypedDict):
    """Chat completion response from LLM."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: NotRequired[Dict[str, int]]  # may not be present in all responses


class EmbeddingData(TypedDict):
    """An embedding data item in the embedding response."""

    object: str
    embedding: List[float]
    index: int


class EmbeddingResponse(TypedDict):
    """Embedding response from embedding model."""

    object: str
    data: List[EmbeddingData]
    model: str
    usage: NotRequired[Dict[str, int]]


class RerankerDocument(TypedDict):
    """A document in the reranker response."""

    index: int
    relevance_score: float
    document: NotRequired[Dict[str, Any]]


class RerankerResponse(TypedDict):
    """Reranker response."""

    results: List[RerankerDocument]
    model: str
    usage: NotRequired[Dict[str, int]]


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        logprobs: Optional[int] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion:
        """Get chat completion from LLM."""
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for Embedding clients."""

    def embed(
        self,
        texts: List[str],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """Get embeddings from embedding model."""
        ...


@runtime_checkable
class RerankerClient(Protocol):
    """Protocol for Reranker clients."""

    def rerank(
        self,
        query: str,
        documents: List[EmbeddingContentPart],
        model: str,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> RerankerResponse:
        """Rerank documents based on query relevance."""
        ...


# ============================================================================
# Httpx Implementation
# ============================================================================


class HttpxEmbeddingClient:
    """httpx-based embedding client implementation."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            base_url: Base URL for the embedding API
            api_key: Optional API key for authentication
            default_headers: Optional default headers to include in requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def embed(
        self,
        texts: List[str],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """Get embeddings from embedding model."""
        headers = {**self.default_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        body: Dict[str, Any] = {
            "input": texts,
            "model": model,
        }
        if encoding_format is not None:
            body["encoding_format"] = encoding_format
        if dimensions is not None:
            body["dimensions"] = dimensions
        if extra_body:
            body.update(extra_body)

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()


class HttpxRerankerClient:
    """Httpx-based reranker client implementation."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the reranker client.

        Args:
            base_url: Base URL for the reranker API
            api_key: Optional API key for authentication
            default_headers: Optional default headers to include in requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def rerank(
        self,
        query: str,
        documents: List[EmbeddingContentPart],
        model: str,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> RerankerResponse:
        """Rerank documents based on query relevance."""
        headers = {**self.default_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        body: Dict[str, Any] = {
            "query": query,
            "documents": {
                "content": documents,
            },
            "model": model,
        }
        if top_n is not None:
            body["top_n"] = top_n
        if return_documents is not None:
            body["return_documents"] = return_documents
        if extra_body:
            body.update(extra_body)

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/rerank",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()


def get_image_url(
    *,
    data: Optional[bytes | str] = None,
    path: Optional[str | Path] = None,
    mime_type: Optional[str] = None,
) -> str:
    """Convert image bytes to data URL."""
    import base64

    if path is not None:
        with open(path, "rb") as f:
            data = f.read()
        extension = Path(path).suffix.replace(".", "")
        if mime_type is None:
            file_type = filetype.get_type(ext=extension)
            if file_type is not None:
                mime_type = file_type.mime
            else:
                mime_type = DEFAULT_IMAGE_MIME_TYPE

    if isinstance(data, str):
        encoded = data
    else:
        if data is None:
            raise ValueError("Either data or path must be provided.")
        encoded = base64.b64encode(data).decode("utf-8")

    if mime_type is None:
        mime_type = DEFAULT_IMAGE_MIME_TYPE
    return f"data:{mime_type};base64,{encoded}"
