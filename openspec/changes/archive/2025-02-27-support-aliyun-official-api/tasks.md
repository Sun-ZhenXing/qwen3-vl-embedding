# Implementation Tasks: Support Aliyun Official API

## 1. Core API Client Implementation

- [x] 1.1 Create `src/qwen3_vl_embedding/aliyun_client.py` with `AliyunEmbeddingClient` class
- [x] 1.2 Implement client initialization with API key, model_name and region configuration
- [x] 1.3 Implement `embed_fusion()` method for multimodal fusion embeddings (text, image, video)
- [x] 1.4 Create custom exception class `AliyunAPIError` for error handling
- [x] 1.5 Add timeout configuration support

## 2. Types and Data Models

- [x] 2.1 Define request/response type definitions in `types.py` or `aliyun_types.py`
- [x] 2.2 Define model name constant (default: qwen3-vl-embedding)
- [x] 2.3 Define region enum (cn, sg)
- [x] 2.4 Add type hints for all public methods

## 3. LangChain Integration

- [x] 3.1 Create `src/qwen3_vl_embedding/langchain/aliyun_embeddings.py`
- [x] 3.2 Implement `AliyunEmbeddings` class extending `Embeddings` base class
- [x] 3.3 Implement `embed_documents()` method
- [x] 3.4 Implement `embed_query()` method
- [x] 3.5 Add multimodal methods: `embed_image()`, `embed_fusion()`
- [x] 3.6 Update `src/qwen3_vl_embedding/langchain/__init__.py` exports

## 4. LlamaIndex Integration

- [x] 4.1 Create `src/qwen3_vl_embedding/llama_index/aliyun_embedding.py`
- [x] 4.2 Implement `AliyunEmbedding` class extending `BaseEmbedding`
- [x] 4.3 Implement `get_text_embedding()` method
- [x] 4.4 Implement `get_text_embeddings()` method
- [x] 4.5 Implement `get_query_embedding()` method
- [x] 4.6 Implement async methods: `aget_text_embedding()`, `aget_text_embeddings()`
- [x] 4.7 Add multimodal methods: `get_image_embedding()`, `get_fusion_embedding()`
- [x] 4.8 Update `src/qwen3_vl_embedding/llama_index/__init__.py` exports

## 5. Testing

- [x] 5.1 Create `src/qwen3_vl_embedding/tests/test_aliyun_client.py`
- [x] 5.2 Add unit tests for client initialization
- [x] 5.3 Add mocked tests for `embed_fusion()` method with various input combinations
- [x] 5.4 Add error handling tests
- [x] 5.5 Add tests for LangChain integration
- [x] 5.6 Add tests for LlamaIndex integration

## 6. Documentation

- [x] 6.1 Update main `README.md` with Aliyun API usage examples
- [x] 6.2 Add API reference documentation for `AliyunEmbeddingClient`
- [x] 6.3 Add LangChain integration documentation
- [x] 6.4 Add LlamaIndex integration documentation
- [x] 6.5 Add configuration guide (API key, region, timeout)

## 7. Dependencies and Configuration

- [x] 7.1 Add `httpx` to project dependencies in `pyproject.toml`
- [x] 7.2 Verify optional dependencies for LangChain and LlamaIndex
- [x] 7.3 Update `__init__.py` to export new classes
