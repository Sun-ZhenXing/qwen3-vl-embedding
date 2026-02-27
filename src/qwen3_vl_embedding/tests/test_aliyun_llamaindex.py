"""Tests for Aliyun LlamaIndex integration."""

import os
from unittest.mock import Mock, patch

import pytest

from qwen3_vl_embedding.llama_index import AliyunEmbedding

# Check if real API key is available for integration tests
REAL_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
HAS_REAL_API_KEY = REAL_API_KEY is not None and REAL_API_KEY.startswith("sk-")


class TestAliyunEmbedding:
    """Test cases for AliyunEmbedding LlamaIndex wrapper."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            assert embedding._model_name == "qwen3-vl-embedding"
            assert embedding._dimension is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(
                api_key="sk-test-key",
                model_name="custom-model",
                region="sg",
                dimension=1024,
                embed_batch_size=5,
                timeout=30.0,
            )
            assert embedding._model_name == "custom-model"
            assert embedding._dimension == 1024
            assert embedding._region == "sg"

    def test_get_text_embedding(self):
        """Test get_text_embedding method."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            result = embedding.get_text_embedding("Hello")

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="Hello", dimension=None
            )

    def test_get_text_embeddings_batch(self):
        """Test _get_text_embeddings method (batch)."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.side_effect = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            results = embedding._get_text_embeddings(["Hello", "World"])

            assert len(results) == 2
            assert results[0] == [0.1, 0.2, 0.3]
            assert results[1] == [0.4, 0.5, 0.6]

    def test_get_query_embedding(self):
        """Test get_query_embedding method."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            result = embedding.get_query_embedding("Query")

            assert result == [0.1, 0.2, 0.3]

    def test_get_image_embedding(self):
        """Test get_image_embedding method."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            result = embedding.get_image_embedding("https://example.com/image.jpg")

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="", image="https://example.com/image.jpg", dimension=None
            )

    def test_get_fusion_embedding(self):
        """Test get_fusion_embedding method."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key", dimension=512)
            result = embedding.get_fusion_embedding(
                text="A cat",
                image="https://example.com/cat.jpg",
            )

            assert result == [0.1, 0.2, 0.3]
            mock_instance.embed_fusion.assert_called_once_with(
                text="A cat",
                image="https://example.com/cat.jpg",
                video=None,
                dimension=512,
            )

    @pytest.mark.asyncio
    async def test_aget_text_embedding(self):
        """Test async get_text_embedding method."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.return_value = [0.1, 0.2, 0.3]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            result = await embedding.aget_text_embedding("Hello")

            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aget_text_embeddings_batch(self):
        """Test async _aget_text_embeddings method (batch)."""
        with patch(
            "qwen3_vl_embedding.llama_index.aliyun_embedding.AliyunEmbeddingClient"
        ) as mock_client:
            mock_instance = Mock()
            mock_instance.embed_fusion.side_effect = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
            mock_client.return_value = mock_instance

            embedding = AliyunEmbedding(api_key="sk-test-key")
            results = await embedding._aget_text_embeddings(["Hello", "World"])

            assert len(results) == 2
            assert results[0] == [0.1, 0.2, 0.3]


@pytest.mark.skipif(
    not HAS_REAL_API_KEY,
    reason="DASHSCOPE_API_KEY environment variable not set or invalid",
)
class TestRealLlamaIndexIntegration:
    """Integration tests with real Aliyun DashScope API for LlamaIndex."""

    def test_real_get_text_embedding(self):
        """Test real API call for text embedding."""
        embedding = AliyunEmbedding()
        result = embedding.get_text_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 2560

    def test_real_get_text_embeddings_batch(self):
        """Test real API call for batch text embeddings."""
        embedding = AliyunEmbedding()
        results = embedding._get_text_embeddings(["Hello", "World"])

        assert len(results) == 2
        assert all(len(emb) == 2560 for emb in results)

    def test_real_get_query_embedding(self):
        """Test real API call for query embedding."""
        embedding = AliyunEmbedding()
        result = embedding.get_query_embedding("Test query")

        assert isinstance(result, list)
        assert len(result) == 2560

    def test_real_get_image_embedding(self):
        """Test real API call for image embedding."""
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embedding = AliyunEmbedding()
        result = embedding.get_image_embedding(image_url)

        assert isinstance(result, list)
        assert len(result) == 2560

    def test_real_get_fusion_embedding(self):
        """Test real API call for fusion embedding."""
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embedding = AliyunEmbedding(dimension=1024)
        result = embedding.get_fusion_embedding(
            text="A beautiful image",
            image=image_url,
        )

        assert isinstance(result, list)
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_real_aget_text_embedding(self):
        """Test real async API call for text embedding."""
        embedding = AliyunEmbedding()
        result = await embedding.aget_text_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 2560

    @pytest.mark.asyncio
    async def test_real_aget_text_embeddings_batch(self):
        """Test real async API call for batch text embeddings."""
        embedding = AliyunEmbedding()
        results = await embedding._aget_text_embeddings(["Hello", "World"])

        assert len(results) == 2
        assert all(len(emb) == 2560 for emb in results)
