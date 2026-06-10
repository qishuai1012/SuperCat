"""
Milvus 客户端 - 支持密集向量+稀疏向量混合检索
功能说明：
1. 管理 Milvus 连接（自动重连、保活机制）
2. 创建集合（包含：密集向量 + 稀疏向量 + 分层分块字段 + 文件元数据）
3. 数据插入、删除、查询
4. 【核心】混合检索 = 密集向量检索 + 稀疏向量检索 + RRF 结果融合
5. 支持 Auto-Merging 自动合并检索（通过 parent_chunk_id 关联父块）
6. 降级方案：纯密集向量检索（当稀疏向量不可用时兜底）
"""
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

load_dotenv()

# RRF 融合算法参数（控制结果排序权重）
RRF_K = int(os.getenv("RRF_K", "60"))

# Milvus 单次查询最大限制（超过会报错，代码中做安全保护）
QUERY_MAX_LIMIT = 16384


class MilvusManager:
    """Milvus 连接、集合管理、检索操作封装（生产级客户端）"""

    def __init__(self):
        # 从环境变量读取配置（不硬编码，符合生产规范）
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self.uri = f"http://{self.host}:{self.port}"
        
        # 延迟初始化客户端（启动快，不阻塞服务）
        self.client = None

    def _get_client(self) -> MilvusClient:
        """延迟创建 Milvus 客户端（第一次使用时才初始化）"""
        if self.client is None:
            self.client = MilvusClient(uri=self.uri)
        return self.client

    def _ensure_connection(self) -> MilvusClient:
        """
        【生产级核心】连接保活 + 自动重连
        每次操作前检查连接是否有效，失效则自动重建
        避免网络波动、Milvus 重启导致服务崩溃
        """
        try:
            client = self._get_client()
            # 轻量级操作：检查集合是否存在，验证连接有效性
            client.has_collection(self.collection_name)
            return client
        except Exception:
            # 连接失效，重新创建客户端
            self.client = MilvusClient(uri=self.uri)
            return self.client

    def init_collection(self, dense_dim: int | None = None):
        """
        初始化 Milvus 集合（表结构）
        包含：
        1. 密集向量（语义向量）
        2. 稀疏向量（BM25 关键词向量）
        3. 文件元数据（文件名、类型、MD5等）
        4. Auto-Merging 分层分块字段（父子块关联）
        """
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))  # bge-m3 默认1024维

        client = self._ensure_connection()

        # 如果集合不存在，则创建
        if not client.has_collection(self.collection_name):
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

            # 主键 ID（自动生成）
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

            # ===================== 双向量核心字段 =====================
            # 密集向量：来自 Embedding 模型（bge-m3）
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            # 稀疏向量：来自 BM25 算法（关键词权重）
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

            # ===================== 文本与文件元数据 =====================
            schema.add_field("text", DataType.VARCHAR, max_length=2000)         # 分块文本
            schema.add_field("filename", DataType.VARCHAR, max_length=255)     # 文件名
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)     # 文件类型
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)   # 文件路径
            schema.add_field("page_number", DataType.INT64)                    # 页码
            schema.add_field("chunk_idx", DataType.INT64)                      # 分块序号

            # ===================== Auto-Merging 分层分块字段 =====================
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)         # 当前分块ID
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)  # 父分块ID
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)    # 根分块ID
            schema.add_field("chunk_level", DataType.INT64)                        # 分块层级（1/2/3）

            # ===================== 文件去重字段 =====================
            schema.add_field("file_md5", DataType.VARCHAR, max_length=32)

            # ===================== 为双向量创建索引（检索加速） =====================
            index_params = client.prepare_index_params()

            # 密集向量索引：HNSW（高性能、适合混合检索）
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",  # 内积相似度
                params={"M": 16, "efConstruction": 256}
            )

            # 稀疏向量索引：倒排索引（专门用于稀疏向量）
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建集合
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

    def insert(self, data: list[dict]):
        """批量插入分块数据（包含文本、双向量、元数据）"""
        return self._ensure_connection().insert(self.collection_name, data)

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        """普通条件查询（带 limit 安全保护，不超过 Milvus 最大限制）"""
        return self._ensure_connection().query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["filename", "file_type"],
            limit=min(limit, QUERY_MAX_LIMIT),
            offset=offset,
        )

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """
        自动分页查询全部数据
        解决 Milvus 单次查询最多 16384 条的限制
        生产环境获取全量文档列表必备
        """
        fields = output_fields or ["filename", "file_type"]
        out = []
        offset = 0
        while True:
            batch = self._ensure_connection().query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=fields,
                limit=QUERY_MAX_LIMIT,
                offset=offset,
            )
            if not batch:
                break
            out.extend(batch)
            if len(batch) < QUERY_MAX_LIMIT:
                break
            offset += len(batch)
        return out

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        根据 chunk_id 批量查询分块内容
        【用于 Auto-Merging】
        检索到叶子块后，通过 parent_chunk_id 查询父块内容
        """
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"
        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text", "filename", "file_type", "page_number",
                "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "chunk_idx"
            ],
            limit=len(ids),
        )

    # ===================== 【核心】混合检索 + RRF 融合 =====================
    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = RRF_K,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        混合检索（企业级 RAG 标准方案）
        检索流程：
        1. 同时发起 密集向量检索 + 稀疏向量检索
        2. 使用 RRF 算法对两路结果进行融合排序
        3. 返回最相关的 top_k 条结果
        """
        # 需要返回的字段（包含 Auto-Merging 所需父子块ID）
        output_fields = [
            "text", "filename", "file_type", "page_number",
            "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "chunk_idx"
        ]

        # ========== 1. 构造【密集向量】检索请求 ==========
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
        )

        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
        )

        # ========== 3. 创建 RRF 融合器（结果融合核心！） ==========
        # RRF = 将两个检索结果公平投票、合并排序
        reranker = RRFRanker(k=rrf_k)

        # ========== 4. 执行混合检索 + 自动 RRF 融合排序 ==========
        search_kwargs = dict(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            ranker=reranker,
            limit=top_k,
            output_fields=output_fields,
        )
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        results = self._ensure_connection().hybrid_search(**search_kwargs)

        # 格式化结果，返回给上层使用
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),
                    "score_type": "rrf",  # 标记结果来自 RRF 融合
                })

        return formatted_results

    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        纯密集向量检索（降级方案）
        当稀疏向量不可用时，保证服务仍能正常使用
        """
        results = self._ensure_connection().search(
            collection_name=self.collection_name,
            data=[dense_embedding],
            anns_field="dense_embedding",
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=[
                "text", "filename", "file_type", "page_number",
                "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "chunk_idx"
            ],
            filter=filter_expr,
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": entity.get("text", ""),
                    "filename": entity.get("filename", ""),
                    "file_type": entity.get("file_type", ""),
                    "page_number": entity.get("page_number", 0),
                    "chunk_id": entity.get("chunk_id", ""),
                    "parent_chunk_id": entity.get("parent_chunk_id", ""),
                    "root_chunk_id": entity.get("root_chunk_id", ""),
                    "chunk_level": entity.get("chunk_level", 0),
                    "chunk_idx": entity.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),
                    "score_type": "raw_similarity",
                })
        return formatted_results

    def delete(self, filter_expr: str):
        """按条件删除数据（如：根据文件名删除所有分块）"""
        return self._ensure_connection().delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self._ensure_connection().has_collection(self.collection_name)

    def drop_collection(self):
        """删除整个集合（用于重建表结构）"""
        client = self._ensure_connection()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)