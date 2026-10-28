# RAG 文档问答系统

基于 FastAPI、ChromaDB 和大语言模型的检索增强问答项目。支持 PDF、DOCX、HTML、Markdown 文档解析、向量检索、普通问答与 SSE 流式问答；可使用本地 Sentence Transformers 模型或 OpenAI 兼容的在线嵌入服务。

## 功能

- 创建和管理知识库集合。
- 上传文档并解析、分块、向量化；支持 PDF、DOCX、HTML、Markdown。
- 按问题检索相关文档片段并生成回答。
- 支持普通 JSON 问答和 Server-Sent Events（SSE）流式问答。
- 复用嵌入模型、LLM 和 Chroma 客户端；检索结果带有限容量和过期时间的缓存。
- 上传到知识库时按文件内容 SHA-256 去重。同一集合内的相同文件不会重复解析或嵌入。

## 环境要求

- Python `>=3.11,<3.14`
- [uv](https://docs.astral.sh/uv/)
- 使用 CUDA 时需安装与本机 CUDA 环境匹配的 PyTorch；CPU 模式无需自行配置 CUDA。

## 安装与启动

在仓库根目录执行：

```powershell
uv sync
Copy-Item .env.example .env
```

按需编辑 `.env` 并填入模型服务凭据，然后启动 API：

```powershell
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

访问地址：

- Web 页面：<http://127.0.0.1:8000/>
- Swagger API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health/>

首次使用本地嵌入模型时，Sentence Transformers 会下载配置的模型。`main.py` 是项目初始化示例，不是 API 服务启动入口。

## 环境变量

复制 `.env.example` 为 `.env` 后，按需配置以下变量。请勿把真实密钥提交到 Git。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 空 | DashScope LLM 凭据；启用在线嵌入时也可作为 `EMBEDDING_API_KEY` 的备用密钥 |
| `LLM_BASE_URL` | 兼容 DashScope 的默认地址 | LLM OpenAI 兼容 API 地址 |
| `LLM_MODEL` | `qwen3.7-flash` | LLM 模型名称 |
| `DEFAULT_MODEL` | `LLM_MODEL` 的值 | 请求未指定 `model` 时使用的模型 |
| `EMBEDDING_PROVIDER` | `local` | `local` 使用本地模型；`openai` 使用 OpenAI 兼容的在线嵌入服务 |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地 Sentence Transformers 模型 |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | 本地模型运行设备，例如 `cpu` 或 `cuda` |
| `EMBEDDING_API_KEY` | `DASHSCOPE_API_KEY` | 在线嵌入服务密钥 |
| `EMBEDDING_BASE_URL` | DashScope 兼容地址 | 在线嵌入 API 地址 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 在线嵌入模型名称 |
| `EMBEDDING_DIMENSION` | 模型默认维度 | 可选的在线嵌入维度；留空使用模型默认值 |
| `VECTOR_STORE_DIRECTORY` | `./data/chroma_db` | Chroma 持久化目录 |
| `VECTOR_DISTANCE` | `cosine` | 向量距离类型 |
| `VECTOR_HNSW_SEARCH_EF` | `100` | HNSW 搜索精度/延迟参数 |
| `VECTOR_HNSW_CONSTRUCTION_EF` | `100` | HNSW 建索引精度/耗时参数 |
| `VECTOR_HNSW_M` | `16` | HNSW 图连接数 |
| `RETRIEVAL_TOP_K` | `5` | 每次检索最多返回的片段数 |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `30` | 检索缓存有效期；设为 `0` 禁用 |
| `RETRIEVAL_CACHE_MAX_SIZE` | `256` | 检索缓存最大条目数；设为 `0` 禁用 |

### 模型与向量库注意事项

- 默认使用本地嵌入模型；启用在线嵌入时设置 `EMBEDDING_PROVIDER=openai`，并配置在线嵌入密钥、地址和模型。
- LLM 和在线嵌入服务都必须配置有效的服务凭据。仓库不包含可用密钥。
- 一个 Chroma 集合中的向量维度必须一致。切换嵌入模型或维度后，不要将新向量写入使用旧模型的集合；请创建新集合并重新导入文档。
- HNSW 配置在创建集合时应用，不会自动重建或修改已有集合的索引参数。
- 检索缓存为进程内缓存；新写入的文档会使该进程内对应检索服务的缓存版本失效。
- 上传去重范围是“同一集合内相同文件内容”。不指定 `collection_name` 时只解析文件，不写入知识库。

## API 使用

请求和响应结构也可在 `/docs` 中查看。

### 创建知识库

```http
POST /api/collections
Content-Type: application/json
```

```json
{"collection_name": "documents"}
```

### 查看知识库和支持的文件类型

```http
GET /api/collections
GET /api/supported-extensions
```

### 上传并写入知识库

```http
POST /api/upload?collection_name=documents
Content-Type: multipart/form-data
```

上传表单字段名为 `file`。支持 `.pdf`、`.docx`、`.html`、`.htm`、`.md` 和 `.markdown`；单个文件最大 50 MB。上传不带 `collection_name` 时只解析文件，不存储向量。

### 普通问答

```http
POST /api/chat/
Content-Type: application/json
```

```json
{
  "query": "文档中的主要结论是什么？",
  "collection_name": "documents"
}
```

可选字段 `model` 用于指定已配置的 LLM 模型。响应包含答案、引用来源、Token 数和总耗时。

### 流式问答

```http
POST /api/chat/stream
Content-Type: application/json
```

请求 JSON 与普通问答接口相同。响应为 `text/event-stream`，每条事件以 `data:` 开头：

```text
data: {"type": "token", "content": "回答片段"}

data: {"type": "done", "query": "...", "sources": [...]}
```

生成出错时会发送 `{"type":"error","message":"流式生成失败"}` 事件。

## 测试

运行完整测试套件：

```powershell
uv run pytest -q
```

部分测试会调用已配置的在线嵌入 API；未设置相应密钥时，真实 API 测试会跳过。PDF 解析测试会生成或更新 `tests/test_pdf.md` 测试产物；如果希望工作区保持干净，测试后请检查 Git 状态。