#!/bin/bash
# RAG 知识库 — 服务器一键部署脚本
# 用法: chmod +x deploy_server.sh && ./deploy_server.sh

set -e

REPO_URL="https://github.com/23ChosenDJ/e--RAG-building.git"
INSTALL_DIR="/opt/rag-kb"

echo "===== 3. Clone 项目 ====="
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "===== 4. 构建镜像并启动服务 ====="
docker compose up -d

echo "===== 5. 拉取 Ollama 模型（首次慢，后续秒开） ====="
sleep 10
docker exec rag-server-ollama-1 ollama pull nomic-embed-text
docker exec rag-server-ollama-1 ollama pull qwen2.5:0.5b

echo "===== 6. 构建知识库索引 ====="
docker exec rag-server-rag-server-1 python langchain_dual_rag/build_index.py

echo "===== 7. 验证 ====="
docker compose logs --tail=20 rag-server

echo ""
echo "部署完成！"
echo "MCP 地址: http://$(curl -s ifconfig.me):8001/mcp"
echo "队友的 .mcp.json 配置:"
echo '{"mcpServers":{"rag-kb":{"type":"http","url":"http://服务器公网IP:8001/mcp"}}}'
