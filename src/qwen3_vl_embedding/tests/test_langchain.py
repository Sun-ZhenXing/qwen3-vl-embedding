"""LangChain integration tests for Qwen3 VL models."""

import os
from pathlib import Path
from typing import List

import pytest
from dotenv import load_dotenv

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


def test_langchain_embedding_initialization(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain embedding initialization."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    assert embeddings.base_url == embedding_base_url
    assert embeddings.model_name == embedding_model
    assert embeddings.api_key == api_key


def test_langchain_embed_query(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
):
    """Test LangChain embed_query method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)

    assert isinstance(query_embedding, list)
    assert len(query_embedding) > 0
    assert all(isinstance(x, float) for x in query_embedding)


def test_langchain_embed_documents(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain embed_documents method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Data science involves analyzing data.",
    ]
    doc_embeddings = embeddings.embed_documents(documents)

    assert isinstance(doc_embeddings, list)
    assert len(doc_embeddings) == len(documents)
    assert all(isinstance(emb, list) for emb in doc_embeddings)
    assert all(len(emb) > 0 for emb in doc_embeddings)
    # Check all embeddings have the same dimension
    assert len(set(len(emb) for emb in doc_embeddings)) == 1


async def test_langchain_async_embed_query(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain async embed_query method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    query = "What is artificial intelligence?"
    query_embedding = await embeddings.aembed_query(query)

    assert isinstance(query_embedding, list)
    assert len(query_embedding) > 0


async def test_langchain_async_embed_documents(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain async embed_documents method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    documents = ["Doc 1", "Doc 2", "Doc 3"]
    doc_embeddings = await embeddings.aembed_documents(documents)

    assert isinstance(doc_embeddings, list)
    assert len(doc_embeddings) == len(documents)


def test_langchain_embed_image(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain embed_image method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    # Test with remote image URL
    image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    image_embedding = embeddings.embed_image(image_url)

    assert isinstance(image_embedding, list)
    assert len(image_embedding) > 0
    assert all(isinstance(x, float) for x in image_embedding)


def test_langchain_embed_images(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain embed_images method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    image_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    ]
    image_embeddings = embeddings.embed_images(image_urls)

    assert isinstance(image_embeddings, list)
    assert len(image_embeddings) == len(image_urls)
    assert all(isinstance(emb, list) for emb in image_embeddings)


def test_langchain_embed_multimodal(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain embed_multimodal method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
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
    multimodal_embedding = embeddings.embed_multimodal(content)

    assert isinstance(multimodal_embedding, list)
    assert len(multimodal_embedding) > 0


async def test_langchain_async_embed_multimodal(
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
) -> None:
    """Test LangChain async embed_multimodal method."""
    from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings

    embeddings = Qwen3VLEmbeddings(
        base_url=embedding_base_url,
        model_name=embedding_model,
        api_key=api_key,
    )

    content: List[EmbeddingContentPart] = [
        {"type": "text", "text": "Test content"},
    ]
    multimodal_embedding = await embeddings.aembed_multimodal(content)

    assert isinstance(multimodal_embedding, list)
    assert len(multimodal_embedding) > 0


def test_langchain_reranker_initialization(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain reranker initialization."""
    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=3,
    )

    assert reranker.base_url == reranker_base_url
    assert reranker.model_name == reranker_model
    assert reranker.top_n == 3


def test_langchain_compress_documents(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain compress_documents method."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    # Create test documents
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="Machine learning is a subset of AI."),
        Document(page_content="Data science involves analyzing data."),
    ]

    query = "What is machine learning?"

    # Compress documents
    compressed_docs = reranker.compress_documents(documents, query)

    assert isinstance(compressed_docs, list)
    assert len(compressed_docs) <= 2
    assert all(isinstance(doc, Document) for doc in compressed_docs)
    # Check that relevance scores are added
    if compressed_docs:
        assert "relevance_score" in compressed_docs[0].metadata


def test_langchain_rerank(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain rerank method (alias for compress_documents)."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    documents = [
        Document(page_content="Text 1"),
        Document(page_content="Text 2"),
        Document(page_content="Text 3"),
    ]

    query = "Search query"
    reranked_docs = reranker.rerank(documents, query)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) <= 2


async def test_langchain_async_compress_documents(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain async compress_documents method."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    documents = [
        Document(page_content="Doc 1"),
        Document(page_content="Doc 2"),
    ]

    query = "Test query"
    compressed_docs = await reranker.acompress_documents(documents, query)

    assert isinstance(compressed_docs, list)
    assert len(compressed_docs) <= 2


async def test_langchain_async_rerank(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain async rerank method."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    documents = [
        Document(page_content="Doc 1"),
        Document(page_content="Doc 2"),
    ]

    query = "Test query"
    reranked_docs = await reranker.arerank(documents, query)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) <= 2


def test_langchain_reranker_with_image_metadata(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain reranker with image metadata."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    # Documents with image metadata
    documents = [
        Document(
            page_content="Image 1 description",
            metadata={
                "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
        ),
        Document(page_content="Plain text document"),
        Document(
            page_content="Image 2 description",
            metadata={"image_path": "/path/to/image.jpg"},
        ),
    ]

    query = "Find images"
    reranked_docs = reranker.rerank(documents, query)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) <= 2


def test_langchain_reranker_empty_documents(
    reranker_base_url: str,
    reranker_model: str,
    api_key: str,
) -> None:
    """Test LangChain reranker with empty documents."""
    from langchain_core.documents import Document

    from qwen3_vl_embedding.langchain import Qwen3VLReranker

    reranker = Qwen3VLReranker(
        base_url=reranker_base_url,
        model_name=reranker_model,
        top_n=2,
    )

    documents: List[Document] = []
    query = "Test query"
    reranked_docs = reranker.compress_documents(documents, query)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == 0
