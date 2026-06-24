# RAG 知识库 — Windows 一键部署指南

> **适用场景**：你已经在本机构建好 Docker 镜像 `rag_building-rag-server`，现在要部署到另一台 Windows PC 上开箱即用。
>
> **目标 PC 只需装 Docker Desktop，不需要装 Python、Ollama 或任何依赖。**

---

## 目录

1. [导出镜像](#1-导出镜像)
2. [传输到目标 PC](#2-传输到目标-pc)
3. [目标 PC 加载镜像](#3-目标-pc-加载镜像)
4. [安装 Ollama（目标 PC）](#4-安装-ollama目标-pc)
5. [拉取模型（目标 PC）](#5-拉取模型目标-pc)
6. [构建向量索引（目标 PC）](#6-构建向量索引目标-pc)
7. [启动 RAG 服务（目标 PC）](#7-启动-rag-服务目标-pc)
8. [验证服务是否正常](#8-验证服务是否正常)
9. [队友连接到你的 MCP](#9-队友连接到你的-mcp)

---

## 1. 导出镜像

**在本机（构建了镜像的电脑）执行：**

```bash
docker save rag_building-rag-server:latest -o rag-server.tar
gzip rag-server.tar
```

| 参数 | 含义 |
|---|---|
| `rag_building-rag-server:latest` | 你刚才构建成功的镜像名称 |
| `-o rag-server.tar` | 导出成 tar 文件 |
| `gzip` | 压缩 tar 文件，体积从 ~450MB 降到 ~150MB |

**产物**：`rag-server.tar.gz`（约 150MB）

---

## 2. 传输到目标 PC

用 U 盘、局域网共享、微信文件传输等方式将 `rag-server.tar.gz` 拷贝到目标 PC 的 `D:\rag-server\` 目录下。

---

## 3. 目标 PC 加载镜像

**在目标 PC 上执行（首次只需一次）：**

```bash
cd D:\rag-server
gunzip rag-server.tar.gz
docker load -i rag-server.tar
```

| 参数 | 含义 |
|---|---|
| `gunzip` | 解压 gz 压缩文件，还原为 tar |
| `docker load -i rag-server.tar` | 将 tar 中的 Docker 镜像导入到本机 Docker 中 |

**验证**：
```bash
docker images rag_building-rag-server
```
应该看到 `rag_building-rag-server:latest` 条目。

---

## 4. 安装 Ollama（目标 PC）

Ollama **不要用 Docker 运行**，直接在 Windows 上安装原生版，这样可以利用 GPU 加速（如果有 NVIDIA 显卡）。

1. 访问 https://ollama.com/download/windows
2. 下载并安装 Ollama for Windows
3. 安装完成后，Ollama 会自动在后台运行（托盘区有小羊驼图标）

**验证**：
```bash
ollama --version
```
应显示版本号。

> **为什么不用 Docker 装 Ollama？** 因为 Docker 内的 Ollama 无法访问 Windows 的 GPU（NVIDIA Docker 配置复杂），纯 CPU 推理速度很慢。Windows 原生 Ollama 能自动利用 GPU 加速。

---

## 5. 拉取模型（目标 PC）

**在目标 PC 上执行（首次只需一次）：**

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

| 命令 | 含义 |
|---|---|
| `ollama pull nomic-embed-text` | 下载嵌入向量模型（274MB），用于将文本转换为向量 |
| `ollama pull qwen2.5:0.5b` | 下载问答模型（0.5GB），用于生成回答 |

**验证**：
```bash
ollama list
```
应看到两个模型都已列出。

---

## 6. 构建向量索引（目标 PC）

**在目标 PC 上执行（首次只需一次）：**

先启动一个临时容器，运行 `build_index.py` 来构建知识库索引：

```bash
docker run -it --rm ^
  --name rag-init ^
  -e OLLAMA_HOST=http://host.docker.internal:11434 ^
  -v rag_data:/app/langchain_dual_rag/chroma_doc_db ^
  -v rag_data:/app/langchain_dual_rag/chroma_code_db ^
  rag_building-rag-server ^
  python langchain_dual_rag/build_index.py
```

| 参数 | 含义 |
|---|---|
| `-it` | 交互模式，让你能看到构建进度 |
| `--rm` | 容器运行完毕后自动删除（不留垃圾容器） |
| `--name rag-init` | 给容器起个名字，方便识别 |
| `-e OLLAMA_HOST=http://host.docker.internal:11434` | 告诉容器内的 RAG 代码：Ollama 在宿主机的 11434 端口；`host.docker.internal` 是 Docker 自动解析的宿主机地址 |
| `-v rag_data:/app/.../chroma_doc_db` | 创建名为 `rag_data` 的 Docker Volume，用于持久化索引数据（容器删除后索引不丢失） |
| `-v rag_data:/app/.../chroma_code_db` | 同上，代码向量库也存到同一个 Volume |
| `rag_building-rag-server` | 使用你刚才导入的镜像 |
| `python langchain_dual_rag/build_index.py` | 容器启动后执行的命令：运行构建脚本 |

**构建时间**：取决于知识库大小。纯 CPU 下，每 10 个文档切片约需 1-2 分钟。

---

## 7. 启动 RAG 服务（目标 PC）

**在目标 PC 上执行（每次开机只需执行一次）：**

```bash
docker run -d ^
  --name rag-server ^
  -p 8001:8001 ^
  -e OLLAMA_HOST=http://host.docker.internal:11434 ^
  -v rag_data:/app/langchain_dual_rag/chroma_doc_db ^
  -v rag_data:/app/langchain_dual_rag/chroma_code_db ^
  --restart unless-stopped ^
  rag_building-rag-server
```

| 参数 | 含义 |
|---|---|
| `-d` | 后台运行（detached），不会占用终端窗口 |
| `--name rag-server` | 给容器命名 |
| `-p 8001:8001` | 将容器的 8001 端口映射到宿主机的 8001 端口，这样 Claude Code 才能访问到 |
| `-e OLLAMA_HOST=...` | 告诉容器 Ollama 在宿主机上 |
| `-v rag_data:...` | 挂载之前构建好的索引 Volume |
| `--restart unless-stopped` | 如果 PC 重启或 Docker 重启，自动拉起容器（开机自启） |
| `rag_building-rag-server` | 镜像名称 |

---

## 8. 验证服务是否正常

```bash
docker logs rag-server
```

应看到类似输出：
```
[MCP] RAG MCP Server 已启动
```

也可以测试 MCP 协议是否正常（任选一项）：

```bash
# 方法 A：直接 curl
curl -X POST http://localhost:8001/mcp -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"

# 方法 B：浏览器打开
# http://localhost:8001/mcp 不会返回任何内容（MCP 只接受 POST），但至少证明端口在监听
```

---

## 9. 队友连接到你的 MCP

找到目标 PC 的局域网 IP：

```bash
ipconfig
```
找到 `IPv4 Address`，比如 `192.168.1.100`。

队友在他们的 Claude Code 配置文件（项目根目录 `.mcp.json` 或 `C:\Users\用户名\.claude.json`）中添加：

```json
{
  "mcpServers": {
    "rag-kb": {
      "type": "http",
      "url": "http://192.168.1.100:8001/mcp"
    }
  }
}
```

> 把 `192.168.1.100` 换成目标 PC 的实际 IP。

**队友不需要装任何东西**（不需要 Docker、Python、Ollama、模型），只需配一行 IP 即可。

---

## 知识库更新

如果你的 `docs/` 或 `codes/` 内容有更新，需要重新构建索引：

```bash
docker run -it --rm --name rag-init ^
  -e OLLAMA_HOST=http://host.docker.internal:11434 ^
  -v rag_data:/app/langchain_dual_rag/chroma_doc_db ^
  -v rag_data:/app/langchain_dual_rag/chroma_code_db ^
  rag_building-rag-server ^
  python langchain_dual_rag/build_index.py --rebuild
```

容器重启后生效：

```bash
docker restart rag-server
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `host.docker.internal` 无法连接 | Docker Desktop 版本过低 | 升级到最新版；或改用宿主机实际 IP |
| 构建索引超慢 | 纯 CPU 推理 | 确保 Ollama 是 Windows 原生版（不是 Docker 版），它会自动利用 GPU |
| 无法从外部 PC 连接 | Windows 防火墙拦截 8001 端口 | 在防火墙中添加入站规则，放行 TCP 8001 端口 |
| `docker save` 文件太大 | 基础镜像未压缩 | 用 `gzip` 压缩后传输 |
