## ADDED Requirements

### Requirement: LangChain Embeddings interface implementation
The system SHALL provide a LangChain-compatible embeddings class that implements the standard `Embeddings` interface.

#### Scenario: Initialize LangChain embeddings with default model
- **WHEN** user creates `AliyunEmbeddings(api_key="sk-xxx")`
- **THEN** the class SHALL be initialized with qwen3-vl-embedding model
- **AND** the class SHALL be ready for LangChain use

#### Scenario: Initialize LangChain embeddings with custom model
- **WHEN** user creates `AliyunEmbeddings(api_key="sk-xxx", model="qwen3-vl-embedding")`
- **THEN** the class SHALL be initialized with the specified model

#### Scenario: Embed documents for LangChain
- **WHEN** user calls `embed_documents(["doc1", "doc2"])`
- **THEN** the system SHALL return a list of embedding vectors for the input texts
- **AND** the return type SHALL be `List[List[float]]`

#### Scenario: Embed query for LangChain
- **WHEN** user calls `embed_query("search query")`
- **THEN** the system SHALL return a single embedding vector for the query
- **AND** the return type SHALL be `List[float]`

### Requirement: Support multimodal inputs in LangChain
The system SHALL extend the base LangChain interface to support multimodal inputs (text and image).

#### Scenario: Embed image for LangChain
- **WHEN** user calls `embed_image("https://example.com/image.png")`
- **THEN** the system SHALL return a fusion embedding vector for the image (using empty text)

#### Scenario: Embed fusion for LangChain
- **WHEN** user calls `embed_fusion(text="query", image="https://example.com/image.png")`
- **THEN** the system SHALL return a fusion embedding vector combining text and image

### Requirement: Configuration through environment variables
The system SHALL support configuration through standard LangChain environment variable patterns.

#### Scenario: Configure via DASHSCOPE_API_KEY
- **WHEN** environment variable `DASHSCOPE_API_KEY` is set
- **AND** user creates `AliyunEmbeddings()` without explicit api_key
- **THEN** the system SHALL use the API key from environment variable

#### Scenario: Configure model and dimension via parameter
- **WHEN** user creates `AliyunEmbeddings(model="qwen3-vl-embedding", dimension=1024)`
- **THEN** the system SHALL use the specified model
- **AND** the system SHALL use 1024 dimensions for all embedding operations
