# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

双源 RAG 知识库系统，支持文档（PDF/TXT/MD）和代码（多语言源码）的向量检索与 AI 问答。通过 MCP 协议暴露给 Claude Code 调用，也可本地交互式对话。

## Key Commands

```bash
# 启动 MCP HTTP 服务（Claude Code 通过 .mcp.json 连接）
cd langchain_dual_rag
.\venv\Scripts\python.exe mcp_server.py --http 8001

# 构建/重建向量索引
.\venv\Scripts\python.exe build_index.py           # 增量构建
.\venv\Scripts\python.exe build_index.py --rebuild  # 强制重建

# 本地交互式对话
.\venv\Scripts\python.exe chat.py                   # 交互模式
.\venv\Scripts\python.exe chat.py -q "问题文本"     # 单次问答
```

## Architecture

```
mcp_server.py          MCP 入口（stdio 或 HTTP），零依赖纯 stdlib JSON-RPC 2.0
    └─ _ensure_rag()   惰性加载：首次 tools/call 才初始化 Ollama + 向量库
       └─ chat.answer() 核心流水线：分类 → 检索 → LLM 生成
```

### RAG Pipeline (chat.py)

1. **路由** → `classify_question()`：关键词匹配（零 LLM 调用），同时命中文档+代码关键词或都不命中时搜两边；`SKIP_ROUTER=True` 时直接搜两边
2. **检索** → `retrieve_from_db()` → `SimpleVectorStore.similarity_search_with_relevance_scores()`：余弦相似度，score 越低越相关；双库结果去重后按 score 升序排列
3. **生成** → `answer()`：拼接检索到的文档片段作为 context，送入 Ollama LLM 生成中文回答

### index building (build_index.py)

- 文档：PDF（PyPDFLoader）、TXT/MD（TextLoader）→ `RecursiveCharacterTextSplitter`
- 代码：按 `CODE_LANGUAGE_MAP` 中语言用 `RecursiveCharacterTextSplitter.from_language()` 按语法切片
- 入库：`_build_store()` 优先 `embed_documents()` 批量编码，失败则逐条 `embed_query()` 兜底；通过 `SimpleVectorStore._from_vectors()` 写入

### SimpleVectorStore (simple_store.py)

- numpy 数组存向量（`embeddings.npy`）+ pickle 存 Document 列表（`documents.pkl`）
- API 兼容 `langchain_community.vectorstores.Chroma` 常用方法
- `_from_vectors(documents, vectors, persist_directory)` 用于批量编码后直接写入，跳过重复编码

## Configuration (config.py)

- 模型：`EMBED_MODEL = "nomic-embed-text"`，`LLM_MODEL = "qwen2.5:0.5b"`（Ollama 本地）
- 检索：`RETRIEVE_TOP_K=3`（单库），`FUSION_TOP_K=2`（双库时每库）
- 目录：`docs/` 放文档，`codes/` 放代码，索引持久化到 `chroma_doc_db/` 和 `chroma_code_db/`
- `SKIP_ROUTER=True`：跳过关键词路由，始终在两个库同时检索（小规模知识库推荐）

## MCP Server (mcp_server.py)

- 纯 Python 标准库实现 JSON-RPC 2.0，无第三方依赖
- 两种传输模式：`--http 8001`（HTTP，推荐）或默认（stdio）
- Windows 上使用二进制 I/O（`sys.stdin.buffer` / `os.write`）绕过文本模式 `\n→\r\n` 转换破坏 MCP 帧格式
- 提供 3 个工具：`rag_search`（检索+生成）、`rag_retrieve`（仅检索）、`rag_sources`（列出源文件）
- 惰性加载：initialize / tools/list 毫秒级返回，首次 tools/call 才加载 Ollama

## Adding Documents or Code

1. 将文件放入 `docs/` 或 `codes/`
2. 运行 `python build_index.py --rebuild`
3. MCP 服务自动读取新索引（无需重启）

## Dependencies

- Ollama（本地运行，端口 11434）
- Python venv 中的 langchain-ollama, langchain-core, langchain-community, numpy, pypdf
- MCP SDK 不需要（mcp_server.py 纯 stdlib）
