"""Tests for Aliyun LangChain integration."""

import os
from unittest.mock import Mock, patch

import pytest

from qwen3_vl_embedding.langchain import AliyunEmbeddings

# Check if real API key is available for integration tests
REAL_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
HAS_REAL_API_KEY = REAL_API_KEY is not None and REAL_API_KEY.startswith("sk-")


class TestAliyunEmbeddings:
    """Test cases for AliyunEmbeddings LangChain wrapper."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(api_key="sk-test-key")
            assert embeddings.model == "qwen3-vl-embedding"
            assert embeddings.dimension is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(
                api_key="sk-test-key",
                model="custom-model",
                region="sg",
                dimension=1024,
                timeout=30.0,
            )
            assert embeddings.model == "custom-model"
            assert embeddings.dimension == 1024

    def test_embed_documents(self):
        """Test embed_documents method."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.side_effect = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(api_key="sk-test-key")
            results = embeddings.embed_documents(["Hello", "World"])

            assert len(results) == 2
            assert results[0] == [0.1, 0.2, 0.3]
            assert results[1] == [0.4, 0.5, 0.6]
            assert mock_instance.embed_fusion.call_count == 2

    def test_embed_query(self):
        """Test embed_query method."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(api_key="sk-test-key")
            result = embeddings.embed_query("Hello")

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="Hello", dimension=None
            )

    def test_embed_image(self):
        """Test embed_image method."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(api_key="sk-test-key")
            result = embeddings.embed_image("https://example.com/image.jpg")

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="", image="https://example.com/image.jpg", dimension=None
            )

    def test_embed_fusion(self):
        """Test embed_fusion method."""
        with patch(
            "qwen3_vl_embedding.langchain.aliyun_embeddings.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embeddings = AliyunEmbeddings(api_key="sk-test-key", dimension=512)
            result = embeddings.embed_fusion(
                text="A cat",
                image="https://example.com/cat.jpg",
                video="https://example.com/cat.mp4",
            )

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="A cat",
                image="https://example.com/cat.jpg",
                video="https://example.com/cat.mp4",
                dimension=512,
            )


@pytest.mark.skipif(
    not HAS_REAL_API_KEY,
    reason="DASHSCOPE_API_KEY environment variable not set or invalid",
)
class TestRealLangChainIntegration:
    """Integration tests with real Aliyun DashScope API for LangChain."""

    def test_real_embed_documents(self):
        """Test real API call for embedding documents."""
        embeddings = AliyunEmbeddings()
        results = embeddings.embed_documents(["Hello", "World"])

        assert len(results) == 2
        assert all(isinstance(emb, list) for emb in results)
        assert all(len(emb) == 2560 for emb in results)

    def test_real_embed_query(self):
        """Test real API call for embedding query."""
        embeddings = AliyunEmbeddings()
        result = embeddings.embed_query("Test query")

        assert isinstance(result, list)
        assert len(result) == 2560

    def test_real_embed_image(self):
        """Test real API call for embedding image."""
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embeddings = AliyunEmbeddings()
        result = embeddings.embed_image(image_url)

        assert isinstance(result, list)
        assert len(result) == 2560

    def test_real_embed_fusion(self):
        """Test real API call for fusion embedding."""
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embeddings = AliyunEmbeddings(dimension=1024)
        result = embeddings.embed_fusion(
            text="A beautiful image",
            image=image_url,
        )

        assert isinstance(result, list)
        assert len(result) == 1024
