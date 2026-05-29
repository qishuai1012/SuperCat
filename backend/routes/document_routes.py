import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from auth import require_admin
from document_loader import DocumentLoader
from embedding import embedding_service
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from models import User
from parent_chunk_store import ParentChunkStore
from services.document_catalog import DocumentCatalogService
from services.document_ingestion import DocumentIngestionService
from schemas import DocumentDeleteResponse, DocumentListResponse, DocumentUploadResponse

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR.parent / "data" / "documents"

router = APIRouter()
logger = logging.getLogger(__name__)

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)


def _remove_bm25_stats_for_filename(filename: str) -> None:
    try:
        rows = milvus_manager.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        texts = [r.get("text") or "" for r in rows]
        embedding_service.increment_remove_documents(texts)
    except Exception as e:
        logger.warning(f"更新 BM25 统计失败 filename={filename}: {e}")


catalog_service = DocumentCatalogService(
    milvus_manager=milvus_manager,
    parent_chunk_store=parent_chunk_store,
    remove_bm25_stats=_remove_bm25_stats_for_filename,
)

ingestion_service = DocumentIngestionService(
    upload_dir=UPLOAD_DIR,
    loader=loader,
    milvus_manager=milvus_manager,
    milvus_writer=milvus_writer,
    parent_chunk_store=parent_chunk_store,
    remove_bm25_stats=_remove_bm25_stats_for_filename,
)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    try:
        return catalog_service.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    return await ingestion_service.upload_document(file)


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    try:
        return catalog_service.delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
