# Design: Support Aliyun Official API

## Context

当前 `qwen3-vl-embedding` 库仅支持本地模型推理，通过 transformers 加载 Qwen3-VL-Embedding 模型。这种方式需要：

- 下载数 GB 的模型文件
- 配置 GPU/CPU 推理环境
- 管理模型版本和更新

阿里云百炼平台提供了托管的多模态 Embedding API 服务：

- **多模态融合向量**：将文本、图片、视频融合成统一向量（qwen3-vl-embedding）

API 端点：

- 北京地域：`https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`
- 新加坡地域：`https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`

## Goals / Non-Goals

**Goals:**

- 提供 `AliyunEmbeddingClient` 类支持调用阿里云 qwen3-vl-embedding API
- 支持多模态融合向量生成（文本+图片+视频）
- 支持自定义向量维度
- 提供与现有本地模型类兼容的接口
- 提供 LangChain 和 LlamaIndex 集成适配器
- 支持北京和新加坡两个地域

**Non-Goals:**

- 不支持其他多模态模型（如 tongyi-embedding-vision-plus）
- 不支持独立向量生成（仅支持融合向量）
- 不支持阿里云文本向量 API（text-embedding-v4 等）
- 不实现自动重试和限流处理（由用户通过 HTTP 客户端配置）
- 不支持批量异步处理（Batch API）

## Decisions

### 1. 使用 HTTP 客户端而非 DashScope SDK

**决策**: 使用 `httpx` 直接调用 REST API，而非依赖 `dashscope` SDK。

**理由**:

- 减少外部依赖，保持库轻量
- 更好的类型提示和 IDE 支持
- 更容易 mock 和测试
- 与现有代码风格一致

**替代方案**: 使用 `dashscope` SDK - 提供了更高级的抽象，但增加了依赖。

### 2. 专注融合向量接口

**决策**: `AliyunEmbeddingClient` 主要提供 `embed_fusion()` 方法用于生成多模态融合向量，支持通过 `model_name` 参数配置模型（默认 qwen3-vl-embedding）。

**理由**:

- qwen3-vl-embedding 的核心优势是融合向量能力
- 保留 model_name 参数为未来兼容其他融合向量模型预留空间
- 简化 API 设计，降低使用复杂度
- 融合向量适用于跨模态检索、图搜等核心场景

### 3. 支持同步调用模式

**决策**: 仅实现同步调用，异步模式后续按需添加。

**理由**:

- 大多数 Embedding 使用场景是同步的（索引构建、查询）
- 保持 API 简单
- 用户可以通过 `asyncio.to_thread` 自行包装

### 4. 错误处理策略

**决策**: 定义自定义异常类 `AliyunAPIError`，包含 HTTP 状态码、错误码和错误信息。

**理由**:

- 统一的错误处理接口
- 便于用户根据错误类型做不同处理
- 隐藏底层 HTTP 细节

### 5. 地域配置方式

**决策**: 通过 `region` 参数（`"cn"` 或 `"sg"`）选择地域，而非直接配置 URL。

**理由**:

- 简化用户配置
- 阿里云官方推荐的地域标识
- 便于后续扩展其他地域

## Risks / Trade-offs

| 风险                     | 缓解措施                                                                 |
| :----------------------- | :----------------------------------------------------------------------- |
| API 响应延迟高于本地推理 | 提供超时配置；建议用户根据场景选择（API 适合快速启动，本地适合高频调用） |
| API 限流导致请求失败     | 文档说明限流策略；建议用户实现客户端重试逻辑                             |
| API 密钥泄露风险         | 支持环境变量配置；文档强调安全最佳实践                                   |
| 阿里云 API 变更          | 关注官方文档更新；保持接口封装层，便于适配变更                           |
| 网络不稳定               | 提供可配置的超时和重试参数                                               |

## Migration Plan

此变更为纯新增功能，无需迁移：

1. 用户可选择继续使用本地模型
2. 新用户可直接使用 API 客户端
3. 现有用户可按需迁移，无需修改已有代码

## Open Questions

1. 是否需要支持图片/视频的 Base64 编码上传？（当前计划优先支持 URL）
2. 是否需要提供请求/响应的详细日志？
3. 是否需要支持自定义 HTTP 客户端（如使用 `requests` 替代 `httpx`）？
