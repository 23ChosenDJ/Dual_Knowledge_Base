#!/usr/bin/env python
"""RAG MCP Server — 双源 RAG 知识库 MCP 工具"""
import sys, os, json

# ---- env ----
os.environ.pop("SSLKEYLOGFILE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_rag_ready = False

# ---- raw binary io (bypass Windows text-mode \n->\r\n) ----
_stdin = sys.stdin.buffer
_stdout_fd = sys.stdout.fileno()
os.set_inheritable(_stdout_fd, True)


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _read_msg():
    try:
        line = _stdin.readline().decode()
        if not line: return None
        cl = 0
        while line.strip():
            if line.lower().startswith("content-length:"):
                cl = int(line.split(":")[1].strip())
            line = _stdin.readline().decode()
        if cl <= 0: return None
        return json.loads(_stdin.read(cl).decode())
    except Exception:
        return None


def _send_msg(data):
    body = json.dumps(data, ensure_ascii=False)
    raw = f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n{body}"
    os.write(_stdout_fd, raw.encode("utf-8"))


# ---- lazy init ----
def _ensure_rag():
    global _rag_ready
    if _rag_ready: return

    import time, chat
    t0 = time.time()
    _log("[MCP] 加载中...")

    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from simple_store import SimpleVectorStore
    import config

    emb = OllamaEmbeddings(model=config.EMBED_MODEL)
    llm = OllamaLLM(model=config.LLM_MODEL, temperature=0, timeout=config.LLM_TIMEOUT)
    try: llm.invoke("1")
    except Exception: pass

    doc_db = SimpleVectorStore(emb, config.DOC_VECTOR_DB_DIR) if os.path.isdir(config.DOC_VECTOR_DB_DIR) else None
    code_db = SimpleVectorStore(emb, config.CODE_VECTOR_DB_DIR) if os.path.isdir(config.CODE_VECTOR_DB_DIR) else None
    if not doc_db and not code_db: raise RuntimeError("未找到向量库，先运行 build_index.py")

    chat.embeddings = emb; chat.llm = llm
    chat.doc_db = doc_db; chat.code_db = code_db
    _rag_ready = True
    _log(f"[MCP] 就绪 ({doc_db._collection_count if doc_db else 0}d+{code_db._collection_count if code_db else 0}c) "
         f"({time.time() - t0:.1f}s)")


# ---- dispatch ----
def _dispatch(msg):
    rid, method = msg.get("id"), msg.get("method", "")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "rag-kb", "version": "1.0.0"}}}

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "rag_search", "description": "搜索RAG知识库（文档+代码），返回AI回答",
             "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "rag_retrieve", "description": "只检索不生成，返回原始文本片段",
             "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}}, "required": ["query"]}},
            {"name": "rag_sources", "description": "列出知识库源文件",
             "inputSchema": {"type": "object", "properties": {}}},
        ]}}

    if method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        try:
            if name == "rag_search":
                _ensure_rag(); import chat
                r = chat.answer(args["query"])
                text = json.dumps({"answer": r["answer"], "route": r["route"], "sources": [
                    {"source": d.metadata.get("source", ""), "type": d.metadata.get("source_type", ""),
                     "score": d.metadata.get("score", 0)} for d in r["sources"]
                ]}, ensure_ascii=False, indent=2)

            elif name == "rag_retrieve":
                _ensure_rag(); import chat
                docs = chat.retrieve(args["query"], chat.classify_question(args["query"]))
                k = args.get("k", 3)
                text = json.dumps([{"content": d.page_content, "source": d.metadata.get("source", ""),
                                    "score": d.metadata.get("score", 0)} for d in docs[:k]],
                                  ensure_ascii=False, indent=2)

            elif name == "rag_sources":
                from config import DOC_DIR, CODE_DIR
                r = {"docs": [], "codes": []}
                for d, k in [(DOC_DIR, "docs"), (CODE_DIR, "codes")]:
                    if os.path.isdir(d):
                        for root, _, files in os.walk(d):
                            for f in files:
                                r[k].append(os.path.relpath(os.path.join(root, f), d))
                text = json.dumps(r, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"未知工具:{name}")

            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}

        except Exception as e:
            _log(f"[MCP] 错误:{e}")
            import traceback; _log(traceback.format_exc())
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": f"错误:{e}"}], "isError": True}}

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"不支持:{method}"}}


# ---- entry ----
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--http":
        from http.server import HTTPServer, BaseHTTPRequestHandler
        port = int(sys.argv[2])

        class H(BaseHTTPRequestHandler):
            timeout = 600  # 10分钟超时，给 LLM 足够时间
            def do_POST(s):
                body = s.rfile.read(int(s.headers.get("Content-Length", 0)))
                resp = _dispatch(json.loads(body))
                s.send_response(200); s.send_header("Content-Type", "application/json"); s.end_headers()
                if resp: s.wfile.write(json.dumps(resp, ensure_ascii=False).encode())
            def log_message(s, *a): pass

        HTTPServer(("0.0.0.0", port), H).serve_forever()
    else:
        _log("[MCP] RAG MCP Server 已启动")
        while True:
            msg = _read_msg()
            if msg is None: break
            resp = _dispatch(msg)
            if resp is not None: _send_msg(resp)
