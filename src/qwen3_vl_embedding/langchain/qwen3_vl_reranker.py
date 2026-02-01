import asyncio
import logging
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field, PrivateAttr

from qwen3_vl_embedding.client import HttpxRerankerClient, RerankerResponse
from qwen3_vl_embedding.types import EmbeddingContentPart

logger = logging.getLogger(__name__)


class Qwen3VLReranker(BaseDocumentCompressor):
    """Qwen3 VL Reranker using httpx client for LangChain.

    This reranker supports both text and multimodal (image/video) documents.

    Example:
        ```python
        from langchain_core.documents import Document
        from python_qwen3_vl.langchain import Qwen3VLReranker

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

    def _prepare_documents_from_langchain(
        self, documents: Sequence[Document]
    ) -> List[EmbeddingContentPart]:
        """Prepare document strings from LangChain documents.

        Args:
            documents: Sequence of LangChain Document objects

        Returns:
            List of document strings (JSON serialized for multimodal content)
        """
        # TODO: Implement multimodal content handling
        ...
        # import json

        # doc_strings: List[str] = []
        # for doc in documents:
        #     # Check if document has multimodal content in metadata
        #     if "image_url" in doc.metadata:
        #         # Create multimodal content for images
        #         doc_content = {
        #             "type": "image_url",
        #             "image_url": {"url": doc.metadata["image_url"]},
        #         }
        #         doc_strings.append(json.dumps(doc_content))
        #     elif "image_path" in doc.metadata:
        #         # Create multimodal content for local images
        #         doc_content = {
        #             "type": "image_url",
        #             "image_url": {"url": f"file://{doc.metadata['image_path']}"},
        #         }
        #         doc_strings.append(json.dumps(doc_content))
        #     elif "video_url" in doc.metadata:
        #         # Create multimodal content for videos
        #         doc_content = {
        #             "type": "video_url",
        #             "video_url": {"url": doc.metadata["video_url"]},
        #         }
        #         doc_strings.append(json.dumps(doc_content))
        #     elif "video_path" in doc.metadata:
        #         # Create multimodal content for local videos
        #         doc_content = {
        #             "type": "video_url",
        #             "video_url": {"url": f"file://{doc.metadata['video_path']}"},
        #         }
        #         doc_strings.append(json.dumps(doc_content))
        #     elif "multimodal_content" in doc.metadata:
        #         # Use predefined multimodal content
        #         doc_strings.append(json.dumps(doc.metadata["multimodal_content"]))
        #     else:
        #         # Use plain text content
        #         doc_strings.append(doc.page_content)
        # return doc_strings

    def _rerank_documents(
        self,
        query: str,
        documents: List[EmbeddingContentPart],
    ) -> RerankerResponse:
        """Rerank documents based on query relevance.

        Args:
            query: Query string
            documents: List of document strings

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
        documents: List[EmbeddingContentPart],
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

        # Prepare documents
        doc_strings = self._prepare_documents_from_langchain(documents)

        try:
            # Call reranker API
            result = self._rerank_documents(
                query=query,
                documents=doc_strings,
            )
        except Exception as e:
            logger.error(f"Error calling rerank API: {e}")
            if self.reraise:
                raise e
            return documents[: self.top_n]

        if "results" not in result:
            logger.warning("No results in rerank response")
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
