# Proposal: Support Aliyun Official API

## Why

当前库仅支持本地加载 Qwen3-VL-Embedding 模型进行推理，需要用户自行下载和管理模型文件，对资源要求较高。阿里云百炼平台提供了官方的多模态 Embedding API 服务（qwen3-vl-embedding），支持文本、图片、视频的融合向量生成。通过支持官方 API，用户无需本地部署模型即可使用高性能的多模态嵌入服务，降低使用门槛和运维成本。

## What Changes

- 新增 `AliyunEmbeddingClient` 类，支持调用阿里云百炼 qwen3-vl-embedding API
- 支持多模态融合向量：将文本、图片、视频融合成统一向量（适用于跨模态检索、图搜等场景）
- 支持自定义向量维度（256, 512, 768, 1024, 1536, 2048, 2560）
- 支持北京和新加坡两个地域的 API 端点
- 提供与现有 `Qwen3VLEmbedding` 类兼容的接口设计
- 新增 LangChain 和 LlamaIndex 的集成适配器
- 添加完整的单元测试和文档

## Capabilities

### New Capabilities

- `aliyun-api-client`: 阿里云百炼 qwen3-vl-embedding API 客户端，支持多模态融合向量生成
- `aliyun-langchain-integration`: LangChain 集成适配器，提供 `AliyunEmbeddings` 类
- `aliyun-llamaindex-integration`: LlamaIndex 集成适配器，提供 `AliyunEmbedding` 类

### Modified Capabilities

- 无（此变更为纯新增功能，不修改现有接口）

## Impact

- **新增依赖**: `dashscope` SDK 或 `httpx` 用于 HTTP 请求
- **新增配置项**: `DASHSCOPE_API_KEY` 环境变量或显式 API Key 配置
- **API 变更**: 无破坏性变更，纯新增功能
- **文档**: 需要更新 README 添加阿里云 API 使用说明
- **测试**: 需要添加单元测试和集成测试（可能需要 mock API 响应）
