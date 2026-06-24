#!/bin/bash
# RAG 知识库 — 服务器一键部署脚本
# 用法: chmod +x deploy_server.sh && ./deploy_server.sh

set -e

mkdir -p knowledgeBase && cd knowledgeBase
REPO_URL="git@github.com:23ChosenDJ/Dual_Knowledge_Base.git"
INSTALL_DIR=$(pwd)

echo "===== 1. Clone 项目 ====="
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "===== 2. 构建镜像并启动服务 ====="
docker compose up -d --build

echo "===== 3. 拉取 Ollama 模型（首次慢，后续秒开） ====="
sleep 10
docker exec knowledgeBase-ollama-1(根据docker compose来的) ollama pull nomic-embed-text
docker exec knowledgeBase-ollama-1(根据docker compose来的) ollama pull qwen2.5:7b

echo "===== 4. 构建知识库索引 ====="
docker exec knowledgeBase-rag-server-1 python langchain_dual_rag/build_index.py --rebuild

echo "===== 5. 用户添加.mcp.json配置 ====="
echo ""
echo "---------------部署完成---------------"
echo "将下面内容放置于项目根目录, 打开claude code, 对话框中输入/mcp即可:"
echo '{"mcpServers":{"rag-kb":{"type":"http","url":"http://服务器IP:8001/mcp"}}}'
