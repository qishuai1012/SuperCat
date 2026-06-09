import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

# 权限校验：管理员才能操作文档
from auth import require_admin
# 文档加载器：读取PDF、Word、TXT等文件内容
from document_loader import DocumentLoader
# 向量嵌入服务：将文本转为向量
from embedding import embedding_service
# Milvus向量数据库客户端：负责查询、删除、管理数据
from milvus_client import MilvusManager
# Milvus写入器：负责将分块文本+向量入库
from milvus_writer import MilvusWriter
# 用户模型
from models import User
# 父文本块存储：用于RAG长上下文召回
from parent_chunk_store import ParentChunkStore
# 文档服务：文档列表查询、删除
from services.document_catalog import DocumentCatalogService
# 文档入库服务：上传、解析、分块、向量化、入库全流程
from services.document_ingestion import DocumentIngestionService
# 接口返回格式定义（Pydantic模型）
from schemas import DocumentDeleteResponse, DocumentListResponse, DocumentUploadResponse

# ====================== 路径配置 ======================
# 获取项目根目录，用于定位文件存储路径
BASE_DIR = Path(__file__).resolve().parent.parent
# 文档上传后保存的目录：data/documents/
UPLOAD_DIR = BASE_DIR.parent / "data" / "documents"

# ====================== 路由与日志 ======================
# 创建文档管理相关API路由
router = APIRouter()
# 获取当前模块日志器，用于记录上传、删除、异常信息
logger = logging.getLogger(__name__)

# ====================== 全局基础组件初始化 ======================
# 文档加载器：解析各种格式文件
loader = DocumentLoader()
# 父块存储：用于RAG多级分块的父块管理
parent_chunk_store = ParentChunkStore()
# Milvus客户端：管理数据库连接、查询、删除
milvus_manager = MilvusManager()
# Milvus写入工具：将文本分块生成向量并写入数据库
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

# ====================== 工具函数 ======================
def _remove_bm25_stats_for_filename(filename: str) -> None:
    """
    删除文档时，同步清理BM25关键词检索的统计数据
    保证混合检索（向量+BM25）的数据一致性
    """
    try:
        # 根据文件名从Milvus查询所有相关文本块
        rows = milvus_manager.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        # 提取文本内容
        texts = [r.get("text") or "" for r in rows]
        # 从BM25统计中移除这些文档
        embedding_service.increment_remove_documents(texts)
    except Exception as e:
        # 记录警告日志，不中断主流程
        logger.warning(f"更新 BM25 统计失败 filename={filename}: {e}")

# ====================== 业务服务初始化 ======================
# 文档目录服务：负责文档列表查询、删除
catalog_service = DocumentCatalogService(
    milvus_manager=milvus_manager,
    parent_chunk_store=parent_chunk_store,
    remove_bm25_stats=_remove_bm25_stats_for_filename,
)

# 文档摄入服务：负责文件上传 → 解析 → 分块 → 入库全流程
ingestion_service = DocumentIngestionService(
    upload_dir=UPLOAD_DIR,
    loader=loader,
    milvus_manager=milvus_manager,
    milvus_writer=milvus_writer,
    parent_chunk_store=parent_chunk_store,
    remove_bm25_stats=_remove_bm25_stats_for_filename,
)

# ====================== API 接口 ======================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """
    获取已上传的所有文档列表
    仅管理员可访问
    """
    try:
        return catalog_service.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """
    上传文档并自动入库
    支持：PDF / DOCX / TXT / MD / Excel等
    流程：保存文件 → 解析文本 → 分块 → 向量化 → 写入Milvus
    仅管理员可上传
    """
    return await ingestion_service.upload_document(file)


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """
    根据文件名删除文档
    会同时删除：文件系统文件 + Milvus向量数据 + BM25统计数据 + 父块数据
    仅管理员可删除
    """
    try:
        return catalog_service.delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")