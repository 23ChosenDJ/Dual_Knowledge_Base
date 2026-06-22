"""
双源 RAG 对话入口 —— 智能路由：自动判断问题属于「文档」还是「代码」并检索回答

用法：
    python chat.py                 # 交互式对话
    python chat.py -q "问题"       # 单次提问（CMD 友好，无 emoji）

前置条件：先运行 build_index.py 构建好索引。
"""

import os
import sys
import argparse
import time
import textwrap

# Windows CMD 环境下强制 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 必须在所有 langchain/ollama 导入之前清理 SSL 环境变量
os.environ.pop("SSLKEYLOGFILE", None)

from simple_store import SimpleVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from config import *


# ============================================================
#  初始化：加载模型 & 向量库
# ============================================================

t0 = time.time()
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL, temperature=0, timeout=LLM_TIMEOUT)

# 预热：加载模型到内存，避免首次查询等太久
try:
    _ = llm.invoke("1")
    print(f"[OK] LLM 预热完成")
except Exception:
    pass

doc_db = None
code_db = None

if os.path.exists(DOC_VECTOR_DB_DIR):
    doc_db = SimpleVectorStore(
        embedding_function=embeddings,
        persist_directory=DOC_VECTOR_DB_DIR,
    )
    print(f"[OK] 文档向量库已加载: {DOC_VECTOR_DB_DIR}  ({doc_db._collection_count} 条)")

if os.path.exists(CODE_VECTOR_DB_DIR):
    code_db = SimpleVectorStore(
        embedding_function=embeddings,
        persist_directory=CODE_VECTOR_DB_DIR,
    )
    print(f"[OK] 代码向量库已加载: {CODE_VECTOR_DB_DIR}  ({code_db._collection_count} 条)")

if not doc_db and not code_db:
    print("[X] 未找到任何向量库！请先运行 build_index.py 构建索引。")
    sys.exit(1)

print(f"[OK] 初始化完成 ({time.time()-t0:.1f}s)")


# ============================================================
#  1. 问题分类 -- 基于关键词的快速路由（零 LLM 调用）
# ============================================================

CODE_KEYWORDS = [
    "代码", "函数", "class", "def ", "import", "变量", "bug",
    "报错", "异常", "调试", "算法", "语法", "API", "接口",
    "python", "java", "js", "cpp", "go", "rust", "编程",
    "code", "function", "error", "debug",
]

DOC_KEYWORDS = [
    "文档", "手册", "说明", "指南", "安装", "配置", "概念",
    "步骤", "教程", "介绍", "产品", "参数", "规格", "波特率",
    "doc", "manual", "guide", "config", "setup", "tutorial",
    "readme",
]


def classify_question(question: str) -> str:
    """基于关键词的快速路由，零 LLM 调用；不确定时搜两边保证准确率"""
    if SKIP_ROUTER:
        return "both"
    if doc_db and not code_db:
        return "doc"
    if code_db and not doc_db:
        return "code"

    q_lower = question.lower()

    has_code = any(kw in q_lower for kw in CODE_KEYWORDS)
    has_doc = any(kw in q_lower for kw in DOC_KEYWORDS)

    if has_code and not has_doc:
        return "code"
    if has_doc and not has_code:
        return "doc"
    # 同时命中或都不命中 → 搜两边
    return "both"


# ============================================================
#  2. 检索器
# ============================================================

def retrieve_from_db(db, question: str, top_k: int) -> list:
    """从单个向量库检索，返回 (document, score) 列表"""
    return db.similarity_search_with_relevance_scores(question, k=top_k)


def _dedup_docs(docs: list) -> list:
    """按 page_content 去重，保留 score 最低的"""
    seen = {}
    for doc in docs:
        key = doc.page_content[:120]
        score = doc.metadata.get("score", 0)
        if key not in seen or score < seen[key].metadata.get("score", 1):
            seen[key] = doc
    return list(seen.values())


def retrieve(question: str, route: str) -> list:
    """
    根据路由从对应向量库检索，返回合并后的文档列表。
    每条结果附带来源标记。
    """
    all_docs = []

    if route in ("doc", "both") and doc_db:
        results = retrieve_from_db(doc_db, question, FUSION_TOP_K if route == "both" else RETRIEVE_TOP_K)
        for doc, score in results:
            doc.metadata["source_type"] = "[DOC]"
            doc.metadata["score"] = round(score, 4)
            all_docs.append(doc)

    if route in ("code", "both") and code_db:
        results = retrieve_from_db(code_db, question, FUSION_TOP_K if route == "both" else RETRIEVE_TOP_K)
        for doc, score in results:
            doc.metadata["source_type"] = "[CODE]"
            doc.metadata["score"] = round(score, 4)
            all_docs.append(doc)

    # 去重 + 按相关性分数排序（越低越相关）
    all_docs = _dedup_docs(all_docs)
    all_docs.sort(key=lambda d: d.metadata.get("score", 0))
    return all_docs


# ============================================================
#  3. 上下文拼接 & 回答生成
# ============================================================

def format_context(docs: list) -> str:
    """将检索到的文档拼接成上下文，每个 chunk 截断以控制总长度"""
    if not docs:
        return "（无参考资料）"

    MAX_PER_CHUNK = 1500  # 每个 chunk 最多取前 1500 chars (保证语义 + 控制总长)

    parts = []
    for i, doc in enumerate(docs, 1):
        src_type = doc.metadata.get("source_type", "[?]")
        src_path = doc.metadata.get("source", "unknown")
        src_name = os.path.basename(src_path) if src_path else "unknown"
        content = doc.page_content[:MAX_PER_CHUNK]
        parts.append(
            f"[参考资料 {i}] {src_type} {src_name}\n{content}"
        )
    return "\n\n---\n\n".join(parts)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""\
    你是项目专属知识库助手，请严格遵守以下规则：

    1. **仅根据**下方【参考资料】的内容回答问题，不要使用你自己的知识
    2. 如果参考资料中找不到相关答案，请明确回复：「抱歉，当前知识库中没有相关资料」
    3. 回答要简洁清晰。涉及代码的问题标注关键逻辑
    4. 引用资料时注明出处（如"根据参考资料1..."）

    【参考资料】
    {context}""")),
    ("human", "{question}"),
])


def answer(question: str) -> dict:
    """
    完整 RAG 流水线：分类 -> 检索 -> 生成
    返回 {"answer": str, "sources": list, "route": str}
    """
    # Step 1: 路由（关键词，毫秒级）
    route = classify_question(question)

    if route == "none":
        return {
            "answer": "抱歉，您的问题与知识库内容无关。请提出与项目文档或代码相关的问题。",
            "sources": [],
            "route": "none",
        }

    # Step 2: 检索
    docs = retrieve(question, route)

    if not docs:
        return {
            "answer": "抱歉，当前知识库中没有相关资料。",
            "sources": [],
            "route": route,
        }

    # Step 3: 生成
    context = format_context(docs)
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    t_gen = time.time()
    try:
        response = llm.invoke(prompt_text)
    except Exception:
        response = "抱歉，生成回答超时，请重试或简化问题。"
    elapsed = time.time() - t_gen
    if elapsed > 5:
        print(f"  [LLM 耗时 {elapsed:.1f}s]", flush=True)

    return {
        "answer": response,
        "sources": docs,
        "route": route,
    }


# ============================================================
#  4. 交互界面 (CMD 友好，纯 ASCII)
# ============================================================

# 分隔线
SEP = "-" * 56
SEP_DOUBLE = "=" * 56
SEP_BOLD = "#" * 56

ROUTE_LABELS = {
    "doc":  "    [DOC] 文档库",
    "code": "    [CODE] 代码库",
    "both": "    [DOC+CODE] 双库融合",
    "none": "    [SKIP] 超出范围",
}


def show_sources(sources: list):
    """显示引用来源"""
    if not sources:
        return
    print("\n  [*] 引用来源:")
    seen = set()
    for doc in sources:
        src = os.path.basename(doc.metadata.get("source", ""))
        src_type = doc.metadata.get("source_type", "")
        key = (src, src_type)
        if key not in seen:
            seen.add(key)
            print(f"      {src_type}  {src}")


def print_banner():
    """启动横幅"""
    doc_status = "[OK] 已加载" if doc_db else "[X] 未加载"
    code_status = "[OK] 已加载" if code_db else "[X] 未加载"
    print(f"""
{SEP_DOUBLE}
   RAG 知识库助手 (双源)
{SEP_DOUBLE}
   LLM      : {LLM_MODEL}
   文档库   : {doc_status}
   代码库   : {code_status}
{SEP_DOUBLE}
   命令: exit=退出  sources=显示来源
{SEP}
""")


def interactive():
    """交互式对话循环"""
    print_banner()
    last_sources = []

    while True:
        try:
            user_q = input("  [Q] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break

        if not user_q:
            continue

        if user_q.lower() == "exit":
            print("  再见!")
            break

        if user_q.lower() == "sources":
            show_sources(last_sources)
            continue

        # 显示思考状态
        sys.stdout.write("  [...] 思考中")
        sys.stdout.flush()

        result = answer(user_q)
        last_sources = result["sources"]

        # 清除思考状态行
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

        # 路由信息
        route = result["route"]
        print(ROUTE_LABELS.get(route, f"    [{route}]"))

        # 回答
        print(f"\n  [A] {result['answer']}\n")

        show_sources(result["sources"])
        print(f"\n{SEP}")


def single_question(question: str):
    """单次问答"""
    t_start = time.time()
    print(f"  [Q] {question}")
    result = answer(question)
    route = result["route"]
    print(ROUTE_LABELS.get(route, f"    [{route}]"))
    print(f"\n  [A] {result['answer']}")
    show_sources(result["sources"])
    total = time.time() - t_start
    print(f"\n  [耗时 {total:.1f}s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双源 RAG 对话工具")
    parser.add_argument("--question", "-q", type=str, help="单次提问（不进入交互模式）")
    args = parser.parse_args()

    if args.question:
        single_question(args.question)
    else:
        interactive()
