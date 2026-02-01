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
from qwen3_vl_embedding.types import EmbeddingContentPart

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

    def _prepare_documents_from_nodes(
        self, nodes: List[NodeWithScore]
    ) -> List[EmbeddingContentPart]:
        """Prepare document strings from nodes.

        Args:
            nodes: List of nodes with scores

        Returns:
            List of document contents
        """

        documents: List[EmbeddingContentPart] = []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                if node.node.image:
                    documents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_url(data=node.node.image),
                            },
                        }
                    )
                if node.node.image_url:
                    documents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": node.node.image_url},
                        }
                    )
                elif node.node.image_path:
                    documents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_url(path=node.node.image_path)
                            },
                        }
                    )
            elif isinstance(node.node, TextNode):
                text_content = node.node.get_content()
                documents.append({"type": "text", "text": text_content})
                continue
            elif isinstance(node.node, Node):
                if node.node.text_resource:
                    text_content = node.node.text_resource.text
                    if text_content:
                        documents.append({"type": "text", "text": text_content})
                if node.node.image_resource:
                    if node.node.image_resource.url:
                        image_url = node.node.image_resource.url.encoded_string()
                    else:
                        image_url = get_image_url(
                            data=node.node.image_resource.data,
                            path=node.node.image_resource.path,
                        )
                    documents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                    )
                continue
            text_content = node.node.get_content()
            if text_content:
                documents.append({"type": "text", "text": text_content})
        assert len(documents) == len(nodes), "Mismatch in documents and nodes length"
        return documents

    def _rerank_documents(
        self,
        query: str,
        documents: List[EmbeddingContentPart],
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
        query: str,
        documents: List[EmbeddingContentPart],
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
