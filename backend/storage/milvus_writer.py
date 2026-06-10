"""
文档向量化并写入 Milvus - 支持密集+稀疏向量
功能：将分块后的文档 → 生成向量 → 批量写入向量数据库
支持混合检索（密集向量语义搜索 + 稀疏向量关键词搜索）
"""
from storage.embedding import EmbeddingService, embedding_service as _default_embedding_service
from storage.milvus_client import MilvusManager


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        """
        初始化向量写入工具
        :param embedding_service: 向量化服务（生成密集/稀疏向量）
        :param milvus_manager: Milvus 数据库操作客户端
        """
        # 如果外部没传入向量服务，就用默认的全局服务
        self.embedding_service = embedding_service or _default_embedding_service
        # 如果外部没传入 Milvus 客户端，就新建一个
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表（就是 DocumentLoader 切出来的三级 chunk）
        :param batch_size: 批次大小，一次写入50条，防止内存溢出/数据库压力过大
        """
        # 如果没有要处理的文档，直接返回
        if not documents:
            return

        # 初始化 Milvus 集合（如果表不存在则创建，存在则跳过）
        self.milvus_manager.init_collection()

        # 把所有文档的文本提取出来，用于稀疏向量（BM25）构建词表
        all_texts = [doc["text"] for doc in documents]
        # 让稀疏向量模型统计全量词频，提升检索精度
        self.embedding_service.increment_add_documents(all_texts)

        # 获取文档总数，开始分批处理
        total = len(documents)
        for i in range(0, total, batch_size):
            # 截取当前批次的数据
            batch = documents[i:i + batch_size]
            # 取出当前批次所有文本内容
            texts = [doc["text"] for doc in batch]
            
            # ===================== 核心步骤 =====================
            # 调用向量服务，同时生成【密集向量】和【稀疏向量】
            # dense_embeddings: 语义向量（用于语义相似度搜索）
            # sparse_embeddings: 关键词向量（用于精确词匹配）
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            # ===================== 组装要入库的数据 =====================
            insert_data = [
                {
                    # 密集向量（语义向量）
                    "dense_embedding": dense_emb,
                    # 稀疏向量（关键词向量）
                    "sparse_embedding": sparse_emb,
                    # 分块后的文本内容
                    "text": doc["text"],
                    # 文件名
                    "filename": doc["filename"],
                    # 文件类型（PDF/Word/Excel）
                    "file_type": doc["file_type"],
                    # 文件路径
                    "file_path": doc.get("file_path", ""),
                    # 页码（从切分模块带过来的）
                    "page_number": doc.get("page_number", 0),
                    # 块全局序号
                    "chunk_idx": doc.get("chunk_idx", 0),
                    # 块唯一ID
                    "chunk_id": doc.get("chunk_id", ""),
                    # 父块ID（Auto-Merging 合并用）
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    # 根块ID（Auto-Merging 合并用）
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    # 块层级（1/2/3 三级分块）
                    "chunk_level": doc.get("chunk_level", 0),
                }
                # 循环把文档、密集向量、稀疏向量一一配对
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            # ===================== 批量插入 Milvus 向量库 =====================
            self.milvus_manager.insert(insert_data)