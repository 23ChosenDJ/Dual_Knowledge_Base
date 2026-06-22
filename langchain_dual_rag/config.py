"""
全局配置 —— 双源 RAG（文档 + 代码）统一参数管理

使用方式：
    from config import *
    或者按需导入单个变量。
"""

import os

# ==================== 路径配置 ====================
# 项目根目录（config.py 所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_DIR = os.path.join(ROOT_DIR, "docs")
CODE_DIR = os.path.join(ROOT_DIR, "codes")
DOC_VECTOR_DB_DIR = os.path.join(ROOT_DIR, "chroma_doc_db")
CODE_VECTOR_DB_DIR = os.path.join(ROOT_DIR, "chroma_code_db")

# ==================== 切片参数 ====================
# 文档切片（适配手册、长文本、Markdown）
DOC_CHUNK_SIZE = 4000
DOC_CHUNK_OVERLAP = 200

# 每个 chunk 最大嵌入字符数（nomic-embed-text context=2048 tokens ≈ 2500 汉字）
# 设 2500 保证 99% chunk 完整嵌入（超过才截断尾部）
MAX_EMBED_CHARS = 2500

# 代码切片（适配源码，尽量不切断函数/类）
CODE_CHUNK_SIZE = 600
CODE_CHUNK_OVERLAP = 100

# ==================== 检索参数 ====================
RETRIEVE_TOP_K = 3           # 单库检索返回数量
FUSION_TOP_K = 2             # 双库合并时每库返回数量

# ==================== 模型配置（Ollama 本地） ====================
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"

# LLM 请求超时（秒），避免长时间卡住
# qwen2.5:0.5b 纯 CPU 推理，RAG context 4000 chars 时需 ~60s
# 设 300s 给足余量，避免复杂查询超时
LLM_TIMEOUT = 300

# 是否跳过 LLM 路由分类（小规模知识库建议 True，直接在两个库检索）
SKIP_ROUTER = True

# ==================== 文档类型支持 ====================
# 文档文件后缀 → loader 类型映射
DOC_GLOB_PATTERNS = ["*.pdf", "*.txt", "*.md"]

# ==================== 代码语言映射 ====================
# 后缀 → Language 枚举 & 匹配 glob
from langchain.text_splitter import Language

CODE_LANGUAGE_MAP = {
    "py":   (Language.PYTHON, "*.py"),
    "java": (Language.JAVA,   "*.java"),
    "js":   (Language.JS,     "*.js"),
    "ts":   (Language.JS,     "*.ts"),
    "cpp":  (Language.CPP,    "*.cpp"),
    "c":    (Language.CPP,    "*.c"),
    "h":    (Language.CPP,    "*.h"),
    "hpp":  (Language.CPP,    "*.hpp"),
    "go":   (Language.GO,     "*.go"),
    "rs":   (Language.RUST,   "*.rs"),
}

# ==================== 环境修正 ====================
# 屏蔽无效 SSL 代理变量
os.environ.pop("SSLKEYLOGFILE", None)
