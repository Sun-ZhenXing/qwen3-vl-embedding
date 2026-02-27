## ADDED Requirements

### Requirement: Client initialization with API key, model and region
The system SHALL provide a client class `AliyunEmbeddingClient` that can be initialized with an API key, optional model name, and optional region configuration. The client SHALL use qwen3-vl-embedding model by default.

#### Scenario: Initialize with default model
- **WHEN** user creates `AliyunEmbeddingClient(api_key="sk-xxx")`
- **THEN** the client SHALL be initialized with the provided API key
- **AND** the model SHALL be set to qwen3-vl-embedding

#### Scenario: Initialize with custom model
- **WHEN** user creates `AliyunEmbeddingClient(api_key="sk-xxx", model_name="qwen3-vl-embedding")`
- **THEN** the client SHALL be initialized with the specified model

#### Scenario: Initialize with environment variable
- **WHEN** user creates `AliyunEmbeddingClient()` without api_key parameter
- **AND** environment variable `DASHSCOPE_API_KEY` is set
- **THEN** the client SHALL read the API key from environment variable

#### Scenario: Initialize with region selection
- **WHEN** user creates `AliyunEmbeddingClient(api_key="sk-xxx", region="sg")`
- **THEN** the client SHALL use Singapore region endpoint

### Requirement: Generate multimodal fusion embedding
The system SHALL support generating fusion embeddings that combine text, image, and video into a single vector representation using the configured model.

#### Scenario: Fusion embedding with text only
- **WHEN** user calls `embed_fusion(text="query")`
- **THEN** the system SHALL return a single embedding vector for the text
- **AND** the vector dimension SHALL match the configured dimension (default 2560)

#### Scenario: Fusion embedding with text and image
- **WHEN** user calls `embed_fusion(text="query", image="https://example.com/image.png")`
- **THEN** the system SHALL return a single embedding vector combining text and image
- **AND** the vector dimension SHALL match the configured dimension

#### Scenario: Fusion embedding with all modalities
- **WHEN** user calls `embed_fusion(text="query", image="https://example.com/image.png", video="https://example.com/video.mp4")`
- **THEN** the system SHALL return a single embedding vector combining all modalities

#### Scenario: Fusion embedding with custom dimension
- **WHEN** user calls `embed_fusion(text="query", dimension=1024)`
- **THEN** the system SHALL return an embedding vector with 1024 dimensions

### Requirement: Error handling
The system SHALL provide clear error messages for API failures.

#### Scenario: Handle authentication error
- **WHEN** API returns 401 status code
- **THEN** the system SHALL raise `AliyunAPIError` with message "Authentication failed: Invalid API key"

#### Scenario: Handle rate limit error
- **WHEN** API returns 429 status code
- **THEN** the system SHALL raise `AliyunAPIError` with message "Rate limit exceeded"

#### Scenario: Handle invalid input error
- **WHEN** API returns 400 status code with invalid input
- **THEN** the system SHALL raise `AliyunAPIError` with the error details from API response

### Requirement: Timeout configuration
The system SHALL support configurable timeout for API requests.

#### Scenario: Set custom timeout
- **WHEN** user initializes client with `timeout=30`
- **THEN** the system SHALL use 30 seconds timeout for all API requests

#### Scenario: Default timeout
- **WHEN** user initializes client without timeout parameter
- **THEN** the system SHALL use default timeout of 60 seconds
