import asyncio
import logging
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field, PrivateAttr

from qwen3_vl_embedding.client import HttpxRerankerClient, RerankerResponse
from qwen3_vl_embedding.types import DocumentList, EmbeddingContentPart

logger = logging.getLogger(__name__)


class Qwen3VLReranker(BaseDocumentCompressor):
    """Qwen3 VL Reranker using httpx client for LangChain.

    This reranker supports both text and multimodal (image/video) documents.

    Example:
        ```python
        from langchain_core.documents import Document
        from qwen3_vl_embedding.langchain import Qwen3VLReranker

        reranker = Qwen3VLReranker(
            base_url="http://localhost:8000/v1",
            model_name="Qwen3-VL-Reranker-2B",
            top_n=5,
        )

        # Rerank text documents
        documents = [
            Document(page_content="Python is a programming language."),
            Document(page_content="Machine learning is a subset of AI."),
        ]
        reranked = reranker.compress_documents(documents, "What is Python?")
        ```
    """

    # Configuration for Pydantic
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    _client: HttpxRerankerClient = PrivateAttr()

    # Public parameters
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the rerank API",
    )
    model_name: str = Field(
        default="Qwen3-VL-Reranker-2B",
        description="Model name to use for reranking",
    )
    top_n: int = Field(
        default=5,
        description="Number of top results to return after reranking",
    )
    api_key: str = Field(
        default="fake",
        description="API key for authentication",
    )
    timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
    )
    reraise: bool = Field(
        default=True,
        description="Whether to reraise exceptions on API errors",
    )

    def __init__(self, **kwargs: Any):
        """Initialize Qwen3VLReranker.

        Args:
            **kwargs: Keyword arguments for configuration
        """
        super().__init__(**kwargs)
        self._client = HttpxRerankerClient(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _prepare_documents_from_langchain(self, documents: Sequence[Document]) -> list:
        """Prepare documents from LangChain documents for the reranker API.

        Each document is converted to either a plain string (text-only) or a
        ScoreMultiModalParam dict (hybrid multimodal with mixed content types).

        Args:
            documents: Sequence of LangChain Document objects

        Returns:
            List of documents, each being a str or ScoreMultiModalParam
        """
        from qwen3_vl_embedding.client import get_image_url

        result: list = []
        for doc in documents:
            parts: List[EmbeddingContentPart] = []

            # Check for predefined multimodal content first
            if "multimodal_content" in doc.metadata:
                content = doc.metadata["multimodal_content"]
                if isinstance(content, list):
                    parts.extend(content)
                else:
                    parts.append(content)
            else:
                # Add text content if present
                if doc.page_content:
                    parts.append({"type": "text", "text": doc.page_content})

                # Add image content if present (non-exclusive with text)
                if "image_url" in doc.metadata:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": doc.metadata["image_url"]},
                        }
                    )
                elif "image_path" in doc.metadata:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_url(path=doc.metadata["image_path"])
                            },
                        }
                    )

                # Add video content if present (non-exclusive with text/image)
                if "video_url" in doc.metadata:
                    parts.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": doc.metadata["video_url"]},
                        }
                    )
                elif "video_path" in doc.metadata:
                    parts.append(
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"file://{doc.metadata['video_path']}"
                            },
                        }
                    )

            # Determine document type for the reranker API
            if len(parts) == 1 and parts[0].get("type") == "text":
                # Text-only document: use plain string
                text_part = parts[0]
                result.append(text_part["text"])  # type: ignore[typeddict-item]
            elif parts:
                # Multimodal or hybrid document: wrap in ScoreMultiModalParam
                result.append({"content": parts})
            else:
                # Fallback: empty text
                result.append(doc.page_content or "")

        return result

    def _rerank_documents(
        self,
        query: str,
        documents: DocumentList,
    ) -> RerankerResponse:
        """Rerank documents based on query relevance.

        Args:
            query: Query string
            documents: List of documents (str or ScoreMultiModalParam)

        Returns:
            Reranker response
        """
        return self._client.rerank(
            query=query,
            documents=documents,
            model=self.model_name,
            top_n=self.top_n,
            timeout=self.timeout,
        )

    async def _arerank_documents(
        self,
        query: str,
        documents: DocumentList,
    ) -> RerankerResponse:
        """Async version of _rerank_documents."""
        return await asyncio.to_thread(
            self._rerank_documents,
            query,
            documents,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents by reranking them based on query relevance.

        Args:
            documents: Sequence of documents to rerank
            query: Query string for relevance ranking
            callbacks: Optional callbacks (not used in this implementation)

        Returns:
            Reranked sequence of documents with updated relevance scores
        """
        if not documents:
            return []

        # Prepare documents for reranker API
        prepared_docs = self._prepare_documents_from_langchain(documents)

        try:
            # Call reranker API
            result = self._rerank_documents(
                query=query,
                documents=prepared_docs,
            )
        except Exception as e:
            logger.error(f"Error calling rerank API: {e}")
            if self.reraise:
                raise
            return documents[: self.top_n]

        if "results" not in result:
            logger.warning("No results in rerank response")
            if self.reraise:
                raise ValueError("Invalid rerank response: missing 'results'")
            return documents[: self.top_n]

        # Build reranked documents
        reranked_docs: List[Document] = []
        for item in result["results"][: self.top_n]:
            index = item["index"]
            relevance_score = item["relevance_score"]
            if 0 <= index < len(documents):
                doc = documents[index]
                # Add relevance score to metadata
                new_metadata = {**doc.metadata, "relevance_score": relevance_score}
                reranked_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata,
                )
                reranked_docs.append(reranked_doc)

        return reranked_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Async version of compress_documents."""
        return await asyncio.to_thread(
            self.compress_documents,
            documents,
            query,
            callbacks,
        )

    def rerank(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """Rerank documents based on query relevance.

        This is an alias for compress_documents for convenience.

        Args:
            documents: Sequence of documents to rerank
            query: Query string for relevance ranking

        Returns:
            Reranked sequence of documents
        """
        return self.compress_documents(documents, query)

    async def arerank(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """Async version of rerank."""
        return await self.acompress_documents(documents, query)
