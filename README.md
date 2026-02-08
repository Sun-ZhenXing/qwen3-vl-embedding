# Qwen3 VL Embedding

一个面向 Qwen3 VL 的 Python 集成库，提供 HTTPx 客户端以及 LlamaIndex / LangChain 的嵌入与重排序适配。

## 功能概览

- **HTTPx 客户端**：OpenAI 风格的 `/embeddings` 与 `/rerank` 端点访问
- **LlamaIndex 集成**：`Qwen3VLEmbedding`、`Qwen3VLReranker`
- **LangChain 集成**：`Qwen3VLEmbeddings`、`Qwen3VLReranker`
- **多模态嵌入**：支持文本 / 图片 / 视频内容的混合输入（基于聊天式 embeddings 请求）
- **Docker 部署**：提供 docker-compose 一键部署 vLLM 推理服务

## 安装

```bash
# 基础（仅 HTTPx 客户端）
pip install qwen3-vl-embedding

# LlamaIndex 集成
pip install qwen3-vl-embedding[llama-index]

# LangChain 集成
pip install qwen3-vl-embedding[langchain]

# CLI 工具
pip install qwen3-vl-embedding[cli]
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

# 图片嵌入
image_embedding = embedding._get_image_embedding(
    "https://example.com/a.jpg"
)

# 视频嵌入
video_embedding = embedding._get_video_embedding(
    "https://example.com/v.mp4"
)

# 多模态嵌入
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

# 文本嵌入
query_embedding = embeddings.embed_query("What is AI?")
doc_embeddings = embeddings.embed_documents(["Doc 1", "Doc 2"])

# 图片嵌入
image_embedding = embeddings.embed_image("https://example.com/a.jpg")
image_embeddings = embeddings.embed_images([
    "https://example.com/a.jpg",
    "https://example.com/b.jpg",
])

# 视频嵌入
video_embedding = embeddings.embed_video("https://example.com/v.mp4")
video_embeddings = embeddings.embed_videos([
    "https://example.com/v1.mp4",
    "https://example.com/v2.mp4",
])

# 多模态嵌入
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

## 配置参数

### Qwen3VLEmbedding（LlamaIndex）

| 参数 | 类型 | 默认值 | 说明 |
| ------ | ------ | -------- | ------ |
| `base_url` | `str` | `http://localhost:8000/v1` | 嵌入 API 基础地址 |
| `model_name` | `str` | `Qwen3-VL-Embedding-2B` | 模型名称 |
| `instruction` | `str` | `Represent the user's input.` | 系统指令 |
| `api_key` | `str` | `fake` | API 认证密钥 |
| `timeout` | `int` | `30` | 请求超时（秒） |

### Qwen3VLEmbeddings（LangChain）

参数与 LlamaIndex 版本一致。

### Qwen3VLReranker（LlamaIndex / LangChain）

| 参数 | 类型 | 默认值 | 说明 |
| ------ | ------ | -------- | ------ |
| `base_url` | `str` | `http://localhost:8000/v1` | 重排序 API 基础地址 |
| `model_name` | `str` | `Qwen3-VL-Reranker-2B` | 模型名称 |
| `top_n` | `int` | `5` | 返回的最高排名数量 |
| `api_key` | `str` | `fake` | API 认证密钥 |
| `timeout` | `int` | `30` | 请求超时（秒） |
| `reraise` | `bool` | `True` | 是否在 API 错误时重新抛出异常 |

### HTTPx 客户端

- `HttpxEmbeddingClient(base_url, api_key=None, default_headers=None)`
- `HttpxRerankerClient(base_url, api_key=None, default_headers=None)`

## 多模态内容格式

多模态嵌入使用 `EmbeddingContentPart`，支持 `text` / `image_url` / `video_url` 三种类型：

```python
content = [
    {"type": "text", "text": "Describe this image:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
]
```

> 本库不会自动将本地路径转为 `file://`，如需本地文件请自行提供完整 URL。

## Docker 部署

项目提供 `docker/` 目录用于通过 vLLM 一键部署推理服务：

```bash
cd docker

# 启动嵌入模型服务（端口 8001）
docker compose --profile embedding up -d

# 启动重排序模型服务（端口 8002）
docker compose --profile reranker up -d

# 同时启动两个服务
docker compose --profile embedding --profile reranker up -d
```

默认端口映射：嵌入服务 → `8001`，重排序服务 → `8002`。可通过 `.env` 文件中的 `EMBEDDING_PORT_OVERRIDE` 和 `RERANKER_PORT_OVERRIDE` 自定义。

## 测试

```bash
uv run pytest
```

测试用 `.env` 示例（放置于 `src/qwen3_vl_embedding/tests/.env`）：

```bash
QWEN3_VL_EMBEDDING_BASE_URL=http://localhost:8001/v1
QWEN3_VL_EMBEDDING_MODEL=Qwen3-VL-Embedding-2B
QWEN3_VL_RERANKER_BASE_URL=http://localhost:8002/v1
QWEN3_VL_RERANKER_MODEL=Qwen3-VL-Reranker-2B
QWEN3_VL_API_KEY=fake
```

## 许可证

MIT License
