## ADDED Requirements

### Requirement: LlamaIndex Embedding interface implementation
The system SHALL provide a LlamaIndex-compatible embedding class that implements the standard `BaseEmbedding` interface.

#### Scenario: Initialize LlamaIndex embedding with default model
- **WHEN** user creates `AliyunEmbedding(api_key="sk-xxx")`
- **THEN** the class SHALL be initialized with qwen3-vl-embedding model
- **AND** the class SHALL be ready for LlamaIndex use

#### Scenario: Initialize LlamaIndex embedding with custom model
- **WHEN** user creates `AliyunEmbedding(api_key="sk-xxx", model_name="qwen3-vl-embedding")`
- **THEN** the class SHALL be initialized with the specified model

#### Scenario: Get text embedding
- **WHEN** user calls `get_text_embedding("text to embed")`
- **THEN** the system SHALL return a list of floats representing the embedding

#### Scenario: Get text embeddings batch
- **WHEN** user calls `get_text_embeddings(["text1", "text2", "text3"])`
- **THEN** the system SHALL return a list of embedding vectors for all input texts

#### Scenario: Get query embedding
- **WHEN** user calls `get_query_embedding("search query")`
- **THEN** the system SHALL return an embedding vector optimized for query matching

### Requirement: Support multimodal inputs in LlamaIndex
The system SHALL extend the base LlamaIndex interface to support multimodal inputs.

#### Scenario: Get image embedding
- **WHEN** user calls `get_image_embedding("https://example.com/image.png")`
- **THEN** the system SHALL return a fusion embedding vector for the image (using empty text)

#### Scenario: Get fusion embedding
- **WHEN** user calls `get_fusion_embedding(text="query", image="https://example.com/image.png")`
- **THEN** the system SHALL return a fusion embedding vector combining text and image

### Requirement: Async support for LlamaIndex
The system SHALL provide async methods for LlamaIndex integration.

#### Scenario: Async text embedding
- **WHEN** user calls `aget_text_embedding("text to embed")`
- **THEN** the system SHALL return a coroutine that resolves to the embedding vector

#### Scenario: Async text embeddings batch
- **WHEN** user calls `aget_text_embeddings(["text1", "text2"])`
- **THEN** the system SHALL return a coroutine that resolves to a list of embedding vectors

### Requirement: Configuration and model selection
The system SHALL support flexible configuration for different use cases.

#### Scenario: Configure dimension and batch size
- **WHEN** user creates `AliyunEmbedding(api_key="sk-xxx", embed_batch_size=10, dimension=1024)`
- **THEN** the system SHALL use 1024 dimensions for all embeddings
- **AND** the batch size SHALL be set to 10

#### Scenario: Configure region
- **WHEN** user creates `AliyunEmbedding(api_key="sk-xxx", region="sg")`
- **THEN** the system SHALL use Singapore region endpoint for all requests
