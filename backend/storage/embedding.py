"""
文本向量化服务 - 支持密集向量和稀疏向量（BM25）
核心能力：
1. 密集向量：使用本地 BGE-M3 模型生成语义向量
2. 稀疏向量：自研 BM25 算法生成关键词权重向量
3. 词表、文档频率、文档长度 全部持久化存储
4. 支持增量添加/删除文档，实时更新统计
5. 线程安全，支持高并发
用途：给 Milvus 混合检索提供双向量
"""
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

# 离线模式，不联网下载模型（生产环境必备）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# BM25 状态文件保存路径
_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    """创建密集向量模型（BGE-M3）"""
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingService:
    """
    向量化服务：
    - 密集向量：本地模型语义向量
    - 稀疏向量：BM25 关键词向量
    - 状态持久化、增量更新、线程安全
    """

    def __init__(self, state_path: Path | str | None = None):
        # 模型延迟加载 + 线程锁
        self._embedder = None
        self._embedder_lock = threading.Lock()

        # BM25 状态文件路径
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()  # 全局线程锁

        # BM25 核心参数
        self.k1 = 1.5
        self.b = 0.75

        # BM25 内部状态
        self._vocab: dict[str, int] = {}          # 词 → 索引
        self._vocab_counter = 0                  # 词表自增ID
        self._doc_freq: Counter[str] = Counter()  # 词在多少篇文档出现
        self._total_docs = 0                     # 总文档数
        self._sum_token_len = 0                  # 所有文档总长度
        self._avg_doc_len = 1.0                  # 平均文档长度

        # 从文件加载历史状态
        self._load_state()

    def _recompute_avg_len(self) -> None:
        """重新计算平均文档长度"""
        self._avg_doc_len = self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0

    def _load_state(self) -> None:
        """从 json 文件加载 BM25 词表、文档频率、统计信息"""
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return

        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        self._vocab_counter = max(self._vocab.values()) + 1 if self._vocab else 0
        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        """无锁版：将 BM25 状态写入文件"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)  # 原子替换（生产必备）

    def _persist(self) -> None:
        """加锁持久化"""
        with self._lock:
            self._persist_unlocked()

    def increment_add_documents(self, texts: list[str]) -> None:
        """增量添加文档，更新 BM25 统计（生产级核心）"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """增量删除文档，对称更新统计"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    def _get_embedder(self) -> HuggingFaceEmbeddings:
        """延迟加载密集向量模型（第一次使用才加载）"""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = _create_dense_embedder()
        return self._embedder

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取密集向量"""
        if not texts:
            return []
        embedder = self._get_embedder()
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"模型调用失败: {e}") from e

    def warmup(self) -> None:
        """预热模型"""

    def tokenize(self, text: str) -> list[str]:
        """中英文混合分词：中文单字，英文整词"""
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        english_pattern = re.compile(r"[a-zA-Z]+")
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        """计算 BM25 稀疏向量（无锁）"""
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)

            # BM25 公式
            if df == 0:
                idf = math.log((n + 1) / 2)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator

            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        """获取单条文本的稀疏向量"""
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """批量获取稀疏向量"""
        if not texts:
            return []
        with self._lock:
            out = []
            any_new = False
            for text in texts:
                vec, changed = self._sparse_vector_for_text_unlocked(text)
                out.append(vec)
                any_new = any_new or changed
            if any_new:
                self._persist_unlocked()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """一次性获取：密集向量 + 稀疏向量"""
        dense = self.get_embeddings(texts)
        sparse = self.get_sparse_embeddings(texts)
        return dense, sparse


# 全局单例（整个进程共用一套词表，生产级标准）
embedding_service = EmbeddingService()