"""
双源索引构建脚本 -- 一次性构建「文档向量库」+「代码向量库」

用法：
    python build_index.py              # 构建全部（文档 + 代码）
    python build_index.py --doc-only   # 仅构建文档
    python build_index.py --code-only  # 仅构建代码
    python build_index.py --rebuild    # 清空旧库重新构建

每次运行会自动跳过已存在的库，除非指定 --rebuild。
"""

import os
import sys
import argparse
from pathlib import Path

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 必须在所有 langchain/ollama 导入之前清理 SSL 环境变量
os.environ.pop("SSLKEYLOGFILE", None)

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from simple_store import SimpleVectorStore
from langchain_ollama import OllamaEmbeddings

from config import *

SEP = "=" * 60


def _build_store(documents: list, embeddings, persist_dir: str, label: str):
    """统一入库：优先批量 embedding，失败则逐条兜底"""
    print(f"\n  [{label}] 正在编码 {len(documents)} 条文本...")
    texts = [d.page_content for d in documents]

    # 尝试批量编码（Ollama 单次 API 调用）
    try:
        vectors = embeddings.embed_documents(texts)
        print(f"       [OK] 批量编码完成 ({len(vectors)} 条)")
    except Exception:
        print(f"       [!] 批量编码失败，切换到逐条编码...")
        vectors = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"       ... {i + 1}/{len(texts)}")
            vectors.append(embeddings.embed_query(text))

    print(f"  [{label}] 写入 {persist_dir} ...")
    SimpleVectorStore._from_vectors(
        documents=documents,
        vectors=vectors,
        persist_directory=persist_dir,
    )


# ============================================================
#  文档索引构建
# ============================================================

def build_doc_index(rebuild: bool = False):
    """构建文档向量库：PDF + TXT + MD"""
    print(f"\n{SEP}")
    print("  [DOC] 开始构建文档向量库")
    print(SEP)

    # 检查是否已存在
    if os.path.exists(DOC_VECTOR_DB_DIR) and not rebuild:
        print(f"  [!] 文档向量库已存在: {DOC_VECTOR_DB_DIR}")
        print("      跳过构建（如需重建请加 --rebuild）")
        return

    if not os.path.isdir(DOC_DIR):
        print(f"  [X] 文档目录不存在: {DOC_DIR}")
        return

    all_docs = []

    # --- PDF ---
    pdf_files = list(Path(DOC_DIR).rglob("*.pdf"))
    if pdf_files:
        print(f"\n  [PDF] 发现 {len(pdf_files)} 个文件")
        loader = DirectoryLoader(
            DOC_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            silent_errors=True,
        )
        pdf_docs = loader.load()
        all_docs.extend(pdf_docs)
        print(f"       [OK] 加载完成: {len(pdf_docs)} 页")
    else:
        print(f"\n  [PDF] 未发现 PDF 文件，跳过")

    # --- TXT / Markdown ---
    for pattern in ["**/*.txt", "**/*.md"]:
        txt_files = list(Path(DOC_DIR).rglob(pattern.split("/")[-1]))
        if txt_files:
            ext = pattern.split(".")[-1]
            print(f"\n  [{ext.upper()}] 发现 {len(txt_files)} 个文件")
            loader = DirectoryLoader(
                DOC_DIR,
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
                silent_errors=True,
            )
            txt_docs = loader.load()
            all_docs.extend(txt_docs)
            print(f"       [OK] 加载完成: {len(txt_docs)} 个文档")
        else:
            ext = pattern.split(".")[-1]
            print(f"\n  [{ext.upper()}] 未发现文件，跳过")

    if not all_docs:
        print("\n  [X] 没有找到任何文档，请将文件放入 docs/ 目录")
        return

    # 切片
    print(f"\n  [>>] 正在切片 (chunk_size={DOC_CHUNK_SIZE}, overlap={DOC_CHUNK_OVERLAP})...")
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOC_CHUNK_SIZE,
        chunk_overlap=DOC_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    split_docs = doc_splitter.split_documents(all_docs)
    print(f"       [OK] {len(all_docs)} 原始 -> {len(split_docs)} 片段")

    # 入库
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    _build_store(split_docs, embeddings, DOC_VECTOR_DB_DIR, "DOC")
    print(f"       [OK] 文档向量库构建完成 ({len(split_docs)} 条向量)")


# ============================================================
#  代码索引构建
# ============================================================

def build_code_index(rebuild: bool = False):
    """构建代码向量库：多语言按语法切片"""
    print(f"\n{SEP}")
    print("  [CODE] 开始构建代码向量库")
    print(SEP)

    if os.path.exists(CODE_VECTOR_DB_DIR) and not rebuild:
        print(f"  [!] 代码向量库已存在: {CODE_VECTOR_DB_DIR}")
        print("      跳过构建（如需重建请加 --rebuild）")
        return

    if not os.path.isdir(CODE_DIR):
        print(f"  [X] 代码目录不存在: {CODE_DIR}")
        return

    all_split_codes = []

    for ext, (language, glob_pattern) in CODE_LANGUAGE_MAP.items():
        files = list(Path(CODE_DIR).rglob(f"*.{ext}"))
        if not files:
            continue

        print(f"\n  [{ext.upper()}] {len(files)} 个文件")

        loader = DirectoryLoader(
            CODE_DIR,
            glob=f"**/*.{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            silent_errors=True,
        )
        raw_docs = loader.load()

        if not raw_docs:
            print(f"       [!] 加载为空，跳过")
            continue

        raw_docs = [d for d in raw_docs if d.page_content.strip()]
        if not raw_docs:
            print(f"       [!] 所有文件为空，跳过")
            continue

        print(f"       [LOAD] {len(raw_docs)} 个文件")

        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=CODE_CHUNK_SIZE,
            chunk_overlap=CODE_CHUNK_OVERLAP,
        )
        split_docs = code_splitter.split_documents(raw_docs)
        print(f"       [>>] {len(split_docs)} 个片段")
        all_split_codes.extend(split_docs)

    if not all_split_codes:
        print("\n  [X] 没有找到任何代码文件，请将文件放入 codes/ 目录")
        return

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    _build_store(all_split_codes, embeddings, CODE_VECTOR_DB_DIR, "CODE")
    print(f"       [OK] 代码向量库构建完成 ({len(all_split_codes)} 条向量)")


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双源 RAG 索引构建工具")
    parser.add_argument("--doc-only", action="store_true", help="仅构建文档索引")
    parser.add_argument("--code-only", action="store_true", help="仅构建代码索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建已存在的索引")
    args = parser.parse_args()

    build_docs = not args.code_only
    build_codes = not args.doc_only

    print(f"\n{SEP}")
    print("  RAG 索引构建工具")
    print(SEP)
    print(f"  嵌入模型  : {EMBED_MODEL}")
    print(f"  文档目录  : {DOC_DIR}")
    print(f"  代码目录  : {CODE_DIR}")
    print(f"  重建模式  : {'是' if args.rebuild else '否（已存在的库会跳过）'}")

    if build_docs:
        build_doc_index(rebuild=args.rebuild)

    if build_codes:
        build_code_index(rebuild=args.rebuild)

    print(f"\n{SEP}")
    print("  [DONE] 全部构建完成! 运行 chat.py 开始对话")
    print(SEP)
