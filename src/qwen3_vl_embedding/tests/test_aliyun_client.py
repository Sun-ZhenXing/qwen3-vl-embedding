"""Tests for Aliyun DashScope Embedding API Client."""

import os
from unittest.mock import Mock, patch

import pytest

from qwen3_vl_embedding.aliyun_client import (
    VALID_DIMENSIONS,
    AliyunAPIError,
    AliyunEmbeddingClient,
    Region,
)


class TestAliyunEmbeddingClient:
    """Test cases for AliyunEmbeddingClient."""

    def test_init_with_api_key(self):
        """Test client initialization with explicit API key."""
        client = AliyunEmbeddingClient(api_key="sk-test-key")
        assert client.api_key == "sk-test-key"
        assert client.model_name == "qwen3-vl-embedding"
        assert client.region == Region.CN
        assert client.timeout == 60.0
        client.close()

    def test_init_with_custom_model(self):
        """Test client initialization with custom model name."""
        client = AliyunEmbeddingClient(
            api_key="sk-test-key",
            model_name="custom-model",
        )
        assert client.model_name == "custom-model"
        client.close()

    def test_init_with_region_sg(self):
        """Test client initialization with Singapore region."""
        client = AliyunEmbeddingClient(
            api_key="sk-test-key",
            region="sg",
        )
        assert client.region == Region.SG
        assert "dashscope-intl" in client.api_url
        client.close()

    def test_init_with_region_enum(self):
        """Test client initialization with Region enum."""
        client = AliyunEmbeddingClient(
            api_key="sk-test-key",
            region=Region.SG,
        )
        assert client.region == Region.SG
        client.close()

    def test_init_with_timeout(self):
        """Test client initialization with custom timeout."""
        client = AliyunEmbeddingClient(
            api_key="sk-test-key",
            timeout=30.0,
        )
        assert client.timeout == 30.0
        client.close()

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(AliyunAPIError) as exc_info:
            AliyunEmbeddingClient()
        assert "API key is required" in str(exc_info.value)

    def test_init_with_env_var(self, monkeypatch):
        """Test client initialization with environment variable."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env-key")
        client = AliyunEmbeddingClient()
        assert client.api_key == "sk-env-key"
        client.close()

    def test_context_manager(self):
        """Test client as context manager."""
        with AliyunEmbeddingClient(api_key="sk-test-key") as client:
            assert client.api_key == "sk-test-key"


class TestEmbedFusion:
    """Test cases for embed_fusion method."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        client = AliyunEmbeddingClient(api_key="sk-test-key")
        yield client
        client.close()

    def test_embed_fusion_text_only(self, client):
        """Test fusion embedding with text only."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        }

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed_fusion(text="Hello world")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_fusion_text_and_image(self, client):
        """Test fusion embedding with text and image."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.4, 0.5, 0.6]}]}
        }

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed_fusion(
                text="A cat",
                image="https://example.com/cat.jpg",
            )
            assert result == [0.4, 0.5, 0.6]

    def test_embed_fusion_all_modalities(self, client):
        """Test fusion embedding with all modalities."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.7, 0.8, 0.9]}]}
        }

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed_fusion(
                text="Tutorial",
                image="https://example.com/thumb.jpg",
                video="https://example.com/video.mp4",
            )
            assert result == [0.7, 0.8, 0.9]

    def test_embed_fusion_with_dimension(self, client):
        """Test fusion embedding with custom dimension."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.1] * 1024}]}
        }

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed_fusion(text="Hello", dimension=1024)
            assert len(result) == 1024

    @pytest.mark.parametrize("dim", VALID_DIMENSIONS)
    def test_valid_dimensions(self, client, dim):
        """Test that all valid dimensions are accepted."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.1] * dim}]}
        }

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed_fusion(text="Hello", dimension=dim)
            assert len(result) == dim

    def test_invalid_dimension_raises(self, client):
        """Test that invalid dimension raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            client.embed_fusion(text="Hello", dimension=100)
        assert "Invalid dimension" in str(exc_info.value)

    def test_no_content_raises(self, client):
        """Test that calling embed_fusion without content raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            client.embed_fusion()
        assert "At least one of text, image, or video must be provided" in str(
            exc_info.value
        )


class TestErrorHandling:
    """Test cases for error handling."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        client = AliyunEmbeddingClient(api_key="sk-test-key")
        yield client
        client.close()

    def test_authentication_error(self, client):
        """Test handling of 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "message": "Unauthorized",
            "code": "InvalidApiKey",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(AliyunAPIError) as exc_info:
                client.embed_fusion(text="Hello")
            assert exc_info.value.status_code == 401
            assert "Authentication failed" in str(exc_info.value)

    def test_rate_limit_error(self, client):
        """Test handling of 429 rate limit error."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "message": "Too many requests",
            "code": "RateLimit",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(AliyunAPIError) as exc_info:
                client.embed_fusion(text="Hello")
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_generic_error(self, client):
        """Test handling of generic API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "message": "Internal server error",
            "code": "InternalError",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(AliyunAPIError) as exc_info:
                client.embed_fusion(text="Hello")
            assert exc_info.value.status_code == 500
            assert "Internal server error" in str(exc_info.value)

    def test_empty_embeddings_response(self, client):
        """Test handling of empty embeddings in response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": {"embeddings": []}}

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(AliyunAPIError) as exc_info:
                client.embed_fusion(text="Hello")
            assert "No embeddings found" in str(exc_info.value)


class TestRegionEnum:
    """Test cases for Region enum."""

    def test_region_values(self):
        """Test Region enum values."""
        assert Region.CN.value == "cn"
        assert Region.SG.value == "sg"

    def test_region_from_string(self):
        """Test creating Region from string."""
        assert Region("cn") == Region.CN
        assert Region("sg") == Region.SG


# Check if real API key is available for integration tests
REAL_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
HAS_REAL_API_KEY = REAL_API_KEY is not None and REAL_API_KEY.startswith("sk-")


@pytest.mark.skipif(
    not HAS_REAL_API_KEY,
    reason="DASHSCOPE_API_KEY environment variable not set or invalid",
)
class TestRealAPIIntegration:
    """Integration tests with real Aliyun DashScope API.

    These tests make actual API calls and require a valid DASHSCOPE_API_KEY
    environment variable to be set.
    """

    @pytest.fixture
    def real_client(self):
        """Create a client with real API key."""
        client = AliyunEmbeddingClient()
        yield client
        client.close()

    def test_real_embed_fusion_text_only(self, real_client):
        """Test real API call with text only."""
        embedding = real_client.embed_fusion(text="Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == 2560  # Default dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_real_embed_fusion_with_dimension(self, real_client):
        """Test real API call with custom dimension."""
        embedding = real_client.embed_fusion(
            text="Test with custom dimension",
            dimension=1024,
        )
        assert isinstance(embedding, list)
        assert len(embedding) == 1024

    @pytest.mark.parametrize("dim", [256, 512, 768, 1024, 1536, 2048, 2560])
    def test_real_embed_fusion_all_dimensions(self, real_client, dim):
        """Test real API call with all valid dimensions."""
        embedding = real_client.embed_fusion(
            text=f"Test with dimension {dim}",
            dimension=dim,
        )
        assert len(embedding) == dim

    def test_real_embed_fusion_text_and_image(self, real_client):
        """Test real API call with text and image."""
        # Using a public test image
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embedding = real_client.embed_fusion(
            text="A beautiful image",
            image=image_url,
        )
        assert isinstance(embedding, list)
        assert len(embedding) == 2560

    def test_real_embed_fusion_image_only(self, real_client):
        """Test real API call with image only."""
        image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
        embedding = real_client.embed_fusion(image=image_url)
        assert isinstance(embedding, list)
        assert len(embedding) == 2560
