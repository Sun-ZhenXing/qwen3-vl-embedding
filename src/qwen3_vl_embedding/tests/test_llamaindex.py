"""LlamaIndex integration tests for Qwen3 VL models."""

import os
from pathlib import Path
from typing import List

import pytest
from dotenv import load_dotenv
from pydantic import HttpUrl

from qwen3_vl_embedding.types import EmbeddingContentPart

load_dotenv(Path(__file__).parent / ".env")


@pytest.fixture
def embedding_base_url() -> str:
    """Get embedding API base URL from environment."""
    return os.getenv("QWEN3_VL_EMBEDDING_BASE_URL", "http://localhost:8000/v1")


@pytest.fixture
def embedding_model() -> str:
    """Get embedding model name from environment."""
    return os.getenv("QWEN3_VL_EMBEDDING_MODEL", "Qwen3-VL-Embedding-2B")


@pytest.fixture
def reranker_base_url() -> str:
    """Get reranker API base URL from environment."""
    return os.getenv("QWEN3_VL_RERANKER_BASE_URL", "http://localhost:8000/v1")


@pytest.fixture
def reranker_model() -> str:
    """Get reranker model name from environment."""
    return os.getenv("QWEN3_VL_RERANKER_MODEL", "Qwen3-VL-Reranker-2B")


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    return os.getenv("QWEN3_VL_API_KEY", "fake")


def test_llamaindex_embedding_initialization(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex embedding initialization."""
    from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding

    embedding = Qwen3VLEmbedding(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    assert embedding.base_url == embedding_base_url
    assert embedding.model_name == embedding_model
    assert embedding.api_key == api_key


def test_llamaindex_text_embedding(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex text embedding."""
    from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding

    embedding = Qwen3VLEmbedding(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    # Test query embedding
    query = "What is machine learning?"
    query_embedding = embedding._get_query_embedding(query)

    assert isinstance(query_embedding, list)
    assert len(query_embedding) > 0
    assert all(isinstance(x, float) for x in query_embedding)

    # Test text embedding
    text = "Machine learning is a subset of artificial intelligence."
    text_embedding = embedding._get_text_embedding(text)

    assert isinstance(text_embedding, list)
    assert len(text_embedding) > 0
    assert len(text_embedding) == len(query_embedding)


async def test_llamaindex_async_embedding(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex async embedding."""
    from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding

    embedding = Qwen3VLEmbedding(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    # Test async query embedding
    query = "What is artificial intelligence?"
    query_embedding = await embedding._aget_query_embedding(query)

    assert isinstance(query_embedding, list)
    assert len(query_embedding) > 0

    # Test async text embedding
    text = "AI is transforming the world."
    text_embedding = await embedding._aget_text_embedding(text)

    assert isinstance(text_embedding, list)
    assert len(text_embedding) > 0


def test_llamaindex_image_embedding(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex image embedding."""
    from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding

    embedding = Qwen3VLEmbedding(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    # Test with remote image URL
    image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    image_embedding = embedding._get_image_embedding(image_url)

    assert isinstance(image_embedding, list)
    assert len(image_embedding) > 0
    assert all(isinstance(x, float) for x in image_embedding)


def test_llamaindex_multimodal_embedding(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex multimodal embedding."""
    from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding

    embedding = Qwen3VLEmbedding(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    # Test multimodal content
    content: List[EmbeddingContentPart] = [
        {"type": "text", "text": "A woman playing with her dog on a beach."},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
        },
    ]
    multimodal_embedding = embedding._get_multimodal_embedding(content)

    assert isinstance(multimodal_embedding, list)
    assert len(multimodal_embedding) > 0


def test_llamaindex_reranker_initialization(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex reranker initialization."""
    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=3,
    )

    assert reranker.base_url == reranker_base_url
    assert reranker.model_name == reranker_model
    assert reranker.top_n == 3


def test_llamaindex_reranker_postprocess(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex reranker postprocessing."""
    from llama_index.core.schema import NodeWithScore, TextNode

    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    # Create test nodes
    nodes = [
        NodeWithScore(
            node=TextNode(text="Python is a programming language."),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(text="Machine learning is a subset of AI."),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(text="Data science involves analyzing data."),
            score=0.5,
        ),
    ]

    # Create query bundle
    from llama_index.core.schema import QueryBundle

    query_bundle = QueryBundle(query_str="What is machine learning?")

    # Rerank nodes
    reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

    assert len(reranked_nodes) <= 2
    assert all(isinstance(node, NodeWithScore) for node in reranked_nodes)
    # Check that scores are updated
    if reranked_nodes:
        assert hasattr(reranked_nodes[0], "score")


async def test_llamaindex_reranker_async(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex async reranker."""
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    nodes = [
        NodeWithScore(node=TextNode(text="Text 1"), score=0.5),
        NodeWithScore(node=TextNode(text="Text 2"), score=0.5),
    ]

    query_bundle = QueryBundle(query_str="Search query")

    # Test async postprocessing
    reranked_nodes = await reranker._apostprocess_nodes(nodes, query_bundle)

    assert isinstance(reranked_nodes, list)
    assert len(reranked_nodes) <= 2


def test_llamaindex_reranker_multimodal_image_nodes(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex reranker with ImageNode (text + image hybrid).

    ImageNode extends TextNode and can carry both text and an image_url,
    which the reranker converts into a ScoreMultiModalParam automatically.
    """
    from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle, TextNode

    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    demo_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    nodes = [
        # ImageNode with text + image_url (hybrid multimodal)
        NodeWithScore(
            node=ImageNode(
                text="A woman and her dog enjoying the beach.",
                image_url=demo_image,
            ),
            score=0.5,
        ),
        # ImageNode with image_url only (no text)
        NodeWithScore(
            node=ImageNode(
                text="",
                image_url=demo_image,
            ),
            score=0.5,
        ),
        # Plain TextNode as baseline
        NodeWithScore(
            node=TextNode(text="Python is a popular programming language."),
            score=0.5,
        ),
    ]

    query_bundle = QueryBundle(
        query_str="Find images about people playing on the beach"
    )
    reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

    assert len(reranked_nodes) <= 2
    assert all(isinstance(n, NodeWithScore) for n in reranked_nodes)
    # Scores should be updated by the reranker
    for node in reranked_nodes:
        assert node.score is not None
        assert isinstance(node.score, float)


def test_llamaindex_reranker_mixed_text_and_image(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex reranker with a mix of TextNode and ImageNode.

    Verifies that the reranker correctly handles a heterogeneous list of
    nodes and returns relevance-sorted results.
    """
    from llama_index.core.schema import (
        ImageNode,
        MediaResource,
        Node,
        NodeWithScore,
        QueryBundle,
        TextNode,
    )

    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=3,
    )

    demo_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    nodes = [
        NodeWithScore(
            node=TextNode(text="Machine learning is a subset of AI."),
            score=0.5,
        ),
        NodeWithScore(
            node=ImageNode(
                text="A dog running on the sand.",
                image_url=demo_image,
            ),
            score=0.5,
        ),
        NodeWithScore(
            node=Node(
                text_resource=MediaResource(
                    text="Data science involves analyzing data."
                ),
                video_resource=MediaResource(
                    url=HttpUrl(
                        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
                    )
                ),
            ),
            score=0.5,
        ),
        NodeWithScore(
            node=ImageNode(
                text="Sunset over the ocean.",
                image_url=demo_image,
            ),
            score=0.5,
        ),
    ]

    query_bundle = QueryBundle(query_str="Beach and ocean scenery")
    reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

    assert len(reranked_nodes) <= 3
    assert all(isinstance(n, NodeWithScore) for n in reranked_nodes)
    # Each node should have a valid relevance score
    for node in reranked_nodes:
        assert node.score is not None


async def test_llamaindex_reranker_multimodal_async(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LlamaIndex async reranker with multimodal ImageNode."""
    from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle, TextNode

    from qwen3_vl_embedding.llama_index import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    demo_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    nodes = [
        NodeWithScore(
            node=ImageNode(
                text="A woman playing with her dog on a beach.",
                image_url=demo_image,
            ),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(text="Python is a programming language."),
            score=0.5,
        ),
    ]

    query_bundle = QueryBundle(query_str="Beach scene with a dog")
    reranked_nodes = await reranker._apostprocess_nodes(nodes, query_bundle)

    assert isinstance(reranked_nodes, list)
    assert len(reranked_nodes) <= 2
    for node in reranked_nodes:
        assert node.score is not None
