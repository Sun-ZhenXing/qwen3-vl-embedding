import asyncio
import logging
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    ImageNode,
    Node,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from pydantic import PrivateAttr

from qwen3_vl_embedding.client import (
    HttpxRerankerClient,
    RerankerResponse,
    get_image_url,
)
from qwen3_vl_embedding.types import DocumentList, EmbeddingContentPart, QueryType

logger = logging.getLogger(__name__)


class Qwen3VLReranker(BaseNodePostprocessor):
    """Qwen3 VL Reranker using httpx client."""

    _client: HttpxRerankerClient = PrivateAttr()

    # input parameters
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen3-VL-Reranker-2B"
    top_n: int = 5

    # as default values
    api_key: str = "fake"
    timeout: int = 30
    reraise: bool = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None,
        timeout: Optional[int] = None,
        reraise: bool = True,
    ):
        """Initialize Qwen3VLReranker.

        Args:
            base_url: Base URL of the rerank API
            model_name: Model name to use for reranking
            top_n: Number of top results to return after reranking
            timeout: Timeout for API requests in seconds
            reraise: Whether to reraise exceptions on API errors
        """
        super().__init__()
        self.base_url = base_url or self.base_url
        self.model_name = model_name or self.model_name
        self.top_n = top_n or self.top_n
        self.timeout = timeout or self.timeout
        self.reraise = reraise
        self._client = HttpxRerankerClient(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _prepare_documents_from_nodes(self, nodes: List[NodeWithScore]) -> list:
        """Prepare documents from nodes for the reranker API.

        Each node is converted to either a plain string (text-only) or a
        ScoreMultiModalParam dict (hybrid multimodal with mixed content types).

        Args:
            nodes: List of nodes with scores

        Returns:
            List of documents, each being a str or ScoreMultiModalParam
        """
        documents: list = []
        for node in nodes:
            parts: List[EmbeddingContentPart] = []
            if isinstance(node.node, ImageNode):
                # ImageNode extends TextNode, include text for hybrid support
                text_content = node.node.get_content()
                if text_content:
                    parts.append({"type": "text", "text": text_content})
                if node.node.image:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_url(data=node.node.image),
                            },
                        }
                    )
                if node.node.image_url:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": node.node.image_url},
                        }
                    )
                elif node.node.image_path:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_url(path=node.node.image_path)
                            },
                        }
                    )
            elif isinstance(node.node, TextNode):
                text_content = node.node.get_content()
                if text_content:
                    parts.append({"type": "text", "text": text_content})
            elif isinstance(node.node, Node):
                if node.node.text_resource:
                    text_content = node.node.text_resource.text
                    if text_content:
                        parts.append({"type": "text", "text": text_content})
                if node.node.image_resource:
                    if node.node.image_resource.url:
                        image_url = node.node.image_resource.url.encoded_string()
                    else:
                        image_url = get_image_url(
                            data=node.node.image_resource.data,
                            path=node.node.image_resource.path,
                        )
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                    )
                if node.node.video_resource:
                    if node.node.video_resource.url:
                        video_url = node.node.video_resource.url.encoded_string()
                    else:
                        video_url = get_image_url(
                            data=node.node.video_resource.data,
                            path=node.node.video_resource.path,
                        )
                    parts.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                        }
                    )
            else:  # if other Node types, fallback to text content
                text_content = node.node.get_content()
                if text_content:
                    parts.append({"type": "text", "text": text_content})

            # Convert to appropriate document type
            if len(parts) == 1 and parts[0].get("type") == "text":
                # Text-only document: use plain string
                text_part = parts[0]
                documents.append(text_part["text"])  # type: ignore[typeddict-item]
            elif parts:
                # Multimodal or hybrid document: wrap in ScoreMultiModalParam
                documents.append({"content": parts})
            else:
                # Fallback: empty text
                documents.append("")

        assert len(documents) == len(nodes), "Mismatch in documents and nodes length"
        return documents

    def _rerank_documents(
        self,
        query: QueryType,
        documents: DocumentList,
    ) -> RerankerResponse:
        """Rerank documents based on query relevance.

        Args:
            query: Query string
            documents: List of document contents

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
        query: QueryType,
        documents: DocumentList,
    ) -> RerankerResponse:
        """Async version of _rerank_documents."""
        return await asyncio.to_thread(
            self._rerank_documents,
            query,
            documents,
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes by reranking them.

        Args:
            nodes: List of nodes to rerank
            query_bundle: Query bundle with query string

        Returns:
            Reranked list of nodes
        """
        if query_bundle is None:
            return nodes

        if not nodes:
            return []

        # Prepare documents from nodes
        documents = self._prepare_documents_from_nodes(nodes)

        try:
            # Call reranker API
            result = self._rerank_documents(
                query=query_bundle.query_str,
                documents=documents,
            )
        except Exception as e:
            logger.error(f"Error calling rerank API: {e}")
            if self.reraise:
                raise e
            return nodes[: self.top_n]

        if "results" not in result:
            logger.warning("No results in rerank response")
            return nodes[: self.top_n]

        # Build reranked nodes
        reranked_nodes: List[NodeWithScore] = []
        for item in result["results"][: self.top_n]:
            index = item["index"]
            relevance_score = item["relevance_score"]
            if 0 <= index < len(nodes):
                node_with_score = nodes[index]
                node_with_score.score = relevance_score
                reranked_nodes.append(node_with_score)

        return reranked_nodes

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:
        """Async version of _postprocess_nodes."""
        return await asyncio.to_thread(
            self._postprocess_nodes,
            nodes,
            query_bundle,
        )
