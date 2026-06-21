"""
轻量持久化向量库 —— 基于 numpy + pickle，零网络依赖

用法：
    # 构建新库
    store = SimpleVectorStore.from_documents(docs, embeddings, persist_directory="./db")

    # 加载已有库
    store = SimpleVectorStore(embedding_function=embeddings, persist_directory="./db")

    # 检索
    docs = store.similarity_search("query", k=3)
    docs_with_scores = store.similarity_search_with_relevance_scores("query", k=3)

API 完全兼容 langchain_community.vectorstores.Chroma 常用接口。
"""

import os
import pickle
import numpy as np


class SimpleVectorStore:
    """基于 numpy 的本地持久化向量库，API 兼容 Chroma 常用方法"""

    def __init__(self, embedding_function, persist_directory: str):
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._documents: list = []
        self._embeddings: np.ndarray | None = None

        # 如果目录已存在，自动加载
        if os.path.isdir(persist_directory):
            self._load()

    # ─── 持久化 ───────────────────────────────────────────

    def _persist(self):
        """保存向量和文档到磁盘"""
        os.makedirs(self._persist_directory, exist_ok=True)
        emb_path = os.path.join(self._persist_directory, "embeddings.npy")
        doc_path = os.path.join(self._persist_directory, "documents.pkl")
        np.save(emb_path, self._embeddings)
        with open(doc_path, "wb") as f:
            pickle.dump(self._documents, f)

    def _load(self):
        """从磁盘加载向量和文档"""
        emb_path = os.path.join(self._persist_directory, "embeddings.npy")
        doc_path = os.path.join(self._persist_directory, "documents.pkl")
        if os.path.exists(emb_path) and os.path.exists(doc_path):
            self._embeddings = np.load(emb_path)
            with open(doc_path, "rb") as f:
                self._documents = pickle.load(f)

    @property
    def _collection_count(self) -> int:
        """返回已存储文档数"""
        return len(self._documents)

    # ─── 构建 ─────────────────────────────────────────────

    @classmethod
    def _from_vectors(cls, documents: list, vectors: list, persist_directory: str):
        """从已编码的向量列表构建并持久化（批量编码的入口）"""
        import numpy as np
        store = cls(embedding_function=None, persist_directory="")
        store._persist_directory = persist_directory
        store._documents = documents
        store._embeddings = np.array(vectors, dtype=np.float32)
        store._persist()
        return store

    @classmethod
    def from_documents(cls, documents: list, embedding, persist_directory: str):
        """从文档列表构建向量库并持久化"""
        store = cls(embedding_function=embedding, persist_directory="")
        store._persist_directory = persist_directory
        store._documents = documents

        # 逐条编码（比批量更稳，避免 Ollama 接口限制）
        texts = [doc.page_content for doc in documents]
        vectors = []
        for i, text in enumerate(texts):
            vec = embedding.embed_query(text)
            vectors.append(vec)

        store._embeddings = np.array(vectors, dtype=np.float32)
        store._persist()
        return store

    # ─── 检索 ─────────────────────────────────────────────

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """计算查询向量与所有文档向量的余弦相似度"""
        if self._embeddings is None or len(self._embeddings) == 0:
            return np.array([])

        # L2 归一化
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)

        return np.dot(doc_norms, query_norm)

    def similarity_search(self, query: str, k: int = 4) -> list:
        """检索最相似的 k 个文档"""
        results = self.similarity_search_with_relevance_scores(query, k)
        return [doc for doc, _ in results]

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list:
        """检索最相似的 k 个文档，返回 (Document, score) 列表"""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_vec = self._embedding_function.embed_query(query)
        query_vec = np.array(query_vec, dtype=np.float32)

        scores = self._cosine_similarity(query_vec)

        # 取 top-k
        if len(scores) <= k:
            top_indices = range(len(scores))
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            doc = self._documents[int(idx)]
            # 余弦相似度越高越相关，这里转为"距离"(越低越相关) 以兼容 Chroma 的 score 语义
            score = 1.0 - float(scores[idx])
            results.append((doc, score))

        return results
