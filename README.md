# Qwen3 VL Embedding

一个面向 Qwen3 VL 的 Python 集成库，提供 HTTPx 客户端以及 LlamaIndex / LangChain 的嵌入与重排序适配。

## 功能概览

- **HTTPx 客户端**：OpenAI 风格的 `/embeddings` 与 `/rerank` 端点访问
- **LlamaIndex 集成**：`Qwen3VLEmbedding`、`Qwen3VLReranker`
- **LangChain 集成**：`Qwen3VLEmbeddings`、`Qwen3VLReranker`
- **多模态嵌入**：支持文本 / 图片 / 视频内容的混合输入（基于聊天式 embeddings 请求）

> 说明：HTTPx 嵌入客户端仅支持文本输入；多模态嵌入通过框架适配类实现。

## 安装

```bash
# 基础
pip install qwen3-vl-embedding

# LlamaIndex 集成
pip install qwen3-vl-embedding[llama]

# LangChain 集成
pip install qwen3-vl-embedding[langchain]
```

## 快速开始

### LlamaIndex

```python
from qwen3_vl_embedding.llama_index import Qwen3VLEmbedding, Qwen3VLReranker

embedding = Qwen3VLEmbedding(
    base_url="http://localhost:8000/v1",
    model_name="Qwen3-VL-Embedding-2B",
)

# 文本嵌入
query_embedding = embedding.get_query_embedding("What is Python?")
text_embeddings = embedding.get_text_embedding_batch([
    "Python is a programming language",
    "Machine learning is powerful",
])

# 多模态嵌入（受保护方法）
multimodal_embedding = embedding._get_multimodal_embedding([
    {"type": "text", "text": "Describe this image:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
])

reranker = Qwen3VLReranker(
    base_url="http://localhost:8000/v1",
    model_name="Qwen3-VL-Reranker-2B",
    top_n=3,
)
```

### LangChain

```python
from qwen3_vl_embedding.langchain import Qwen3VLEmbeddings, Qwen3VLReranker

embeddings = Qwen3VLEmbeddings(
    base_url="http://localhost:8000/v1",
    model_name="Qwen3-VL-Embedding-2B",
)

query_embedding = embeddings.embed_query("What is AI?")
doc_embeddings = embeddings.embed_documents(["Doc 1", "Doc 2"])

image_embedding = embeddings.embed_image("https://example.com/a.jpg")
multimodal_embedding = embeddings.embed_multimodal([
    {"type": "text", "text": "Describe this image:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
])

reranker = Qwen3VLReranker(
    base_url="http://localhost:8000/v1",
    model_name="Qwen3-VL-Reranker-2B",
    top_n=3,
)
```

### 原始客户端

```python
from qwen3_vl_embedding.client import HttpxEmbeddingClient, HttpxRerankerClient

embedding_client = HttpxEmbeddingClient(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
)

# 文本嵌入
embedding_response = embedding_client.embed(
    texts=["Hello", "World"],
    model="Qwen3-VL-Embedding-2B",
)

rerank_client = HttpxRerankerClient(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
)

rerank_response = rerank_client.rerank(
    query="What is Python?",
    documents=["Python is a programming language", "Machine learning is powerful"],
    model="Qwen3-VL-Reranker-2B",
    top_n=2,
)
```

## 配置参数（与源码一致）

### Qwen3VLEmbedding（LlamaIndex）

- `base_url`（默认 `http://localhost:8000/v1`）
- `model_name`（默认 `Qwen3-VL-Embedding-2B`）
- `instruction`（默认 `"Represent the user's input."`）
- `api_key`（默认 `fake`）
- `timeout`（默认 `30` 秒）

### Qwen3VLEmbeddings（LangChain）

参数与 LlamaIndex 版本一致。

### Qwen3VLReranker（LlamaIndex）

- `base_url`（默认 `http://localhost:8000/v1`）
- `model_name`（默认 `Qwen3-VL-Reranker-2B`）
- `top_n`（默认 `5`）
- `timeout`（默认 `30` 秒）
- `reraise`（默认 `True`）

> 说明：当前构造函数未暴露 `api_key` 参数。

### Qwen3VLReranker（LangChain）

- `base_url`（默认 `http://localhost:8000/v1`）
- `model_name`（默认 `Qwen3-VL-Reranker-2B`）
- `top_n`（默认 `5`）
- `api_key`（默认 `fake`）
- `timeout`（默认 `30` 秒）
- `reraise`（默认 `True`）

### HTTPx 客户端

- `HttpxEmbeddingClient(base_url, api_key=None, default_headers=None)`
- `HttpxRerankerClient(base_url, api_key=None, default_headers=None)`

## 多模态内容格式

多模态嵌入使用 `EmbeddingContentPart`，支持 `text` / `image_url` / `video_url`。

```python
content = [
    {"type": "text", "text": "Describe this image:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
]
```

> 本库不会自动将本地路径转为 `file://`，如需本地文件请自行提供完整 URL。

## 测试

```bash
uv run pytest
```

测试用 `.env` 示例（字段名与测试一致）：

```bash
QWEN3_VL_EMBEDDING_BASE_URL=http://localhost:8000/v1
QWEN3_VL_EMBEDDING_MODEL=Qwen3-VL-Embedding-2B
QWEN3_VL_RERANKER_BASE_URL=http://localhost:8000/v1
QWEN3_VL_RERANKER_MODEL=Qwen3-VL-Reranker-2B
QWEN3_VL_API_KEY=fake
```

## 许可证

MIT License
