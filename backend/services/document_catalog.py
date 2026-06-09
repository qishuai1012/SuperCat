import logging

from parent_chunk_store import ParentChunkStore
from milvus_client import MilvusManager
from schemas import DocumentDeleteResponse, DocumentInfo, DocumentListResponse

# 生产级必备：日志记录器，用于排查线上问题
logger = logging.getLogger(__name__)


class DocumentCatalogService:
    """
    文档目录管理服务（Service 层）
    核心功能：
    1. 查询当前系统中已入库的所有文档列表（带分块数量统计）
    2. 删除指定文档的全部向量数据、父块数据、检索索引
    适用场景：后台管理界面 - 文档列表展示、文档删除操作
    """

    def __init__(self, milvus_manager: MilvusManager, parent_chunk_store: ParentChunkStore, remove_bm25_stats):
        """
        构造函数：依赖注入（生产级标准写法，松耦合、可测试）
        :param milvus_manager: Milvus 向量数据库客户端（操作向量数据）
        :param parent_chunk_store: 父块存储客户端（操作 PG + Redis）
        :param remove_bm25_stats: 清理 BM25 稀疏检索词频统计的方法
        """
        self.milvus_manager = milvus_manager
        self.parent_chunk_store = parent_chunk_store
        self.remove_bm25_stats = remove_bm25_stats

    def list_documents(self) -> DocumentListResponse:
        """
        获取【所有已上传文档】的统计列表
        从 Milvus 中查询所有 chunk 元数据，按文件名分组统计
        返回给前端展示：文件名、文件类型、MD5、分块总数
        """
        # 确保 Milvus 集合（表）已初始化/存在
        self.milvus_manager.init_collection()

        # 从 Milvus 查询所有文档的基础元数据（不需要向量，只查字段）
        results = self.milvus_manager.query_all(output_fields=["filename", "file_type", "file_md5"])

        # 临时字典：用于按 filename 分组，统计每个文件的 chunk 数量
        file_stats = {}

        # 遍历所有 chunk，按文件分组
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            file_md5 = item.get("file_md5", "")

            # 如果该文件第一次出现，初始化统计信息
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "file_md5": file_md5,
                    "chunk_count": 0,  # 分块数量初始化为 0
                }

            # 每遍历一个 chunk，当前文件的分块数 +1
            file_stats[filename]["chunk_count"] += 1

        # 将字典转为标准响应模型列表（Pydantic 模型）
        documents = [DocumentInfo(**stats) for stats in file_stats.values()]

        # 返回标准结构的响应数据给接口层
        return DocumentListResponse(documents=documents)

    def delete_document(self, filename: str) -> DocumentDeleteResponse:
        """
        【核心方法】删除一个文档的**全部相关数据**
        删除范围（保证数据一致性）：
        1. Milvus 向量库中的所有分块
        2. PostgreSQL + Redis 中的父块数据
        3. BM25 稀疏检索的词频统计数据
        异常安全：删除失败只打警告日志，不抛错导致系统崩溃
        """
        # 确保 Milvus 集合存在
        self.milvus_manager.init_collection()

        # 构造 Milvus 删除表达式：删除 filename 等于当前文件的所有 chunk
        delete_expr = f'filename == "{filename}"'

        # 第一步：清理 BM25 稀疏检索的统计数据
        self.remove_bm25_stats(filename)

        # 第二步：删除 Milvus 向量数据（异常捕获，生产必备）
        try:
            result = self.milvus_manager.delete(delete_expr)
        except Exception as e:
            # 记录警告日志，不中断流程
            logger.warning(f"删除 Milvus 向量失败 filename={filename}: {e}")
            result = {"delete_count": 0}

        # 第三步：删除 PostgreSQL + Redis 中的父块数据
        self.parent_chunk_store.delete_by_filename(filename)

        # 第四步：返回标准删除响应
        return DocumentDeleteResponse(
            filename=filename,
            # 安全获取删除数量，兼容不同返回格式
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )