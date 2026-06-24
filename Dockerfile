FROM python:3.9-slim

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY langchain_dual_rag/ /app/langchain_dual_rag/

# 暴露 HTTP MCP 端口
EXPOSE 8001

# 启动 MCP HTTP 服务
CMD ["python", "langchain_dual_rag/mcp_server.py", "--http", "8001"]
