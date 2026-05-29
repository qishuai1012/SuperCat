import logging

from parent_chunk_store import ParentChunkStore
from milvus_client import MilvusManager
from schemas import DocumentDeleteResponse, DocumentInfo, DocumentListResponse

logger = logging.getLogger(__name__)


class DocumentCatalogService:
    def __init__(self, milvus_manager: MilvusManager, parent_chunk_store: ParentChunkStore, remove_bm25_stats):
        self.milvus_manager = milvus_manager
        self.parent_chunk_store = parent_chunk_store
        self.remove_bm25_stats = remove_bm25_stats

    def list_documents(self) -> DocumentListResponse:
        self.milvus_manager.init_collection()
        results = self.milvus_manager.query_all(output_fields=["filename", "file_type", "file_md5"])

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            file_md5 = item.get("file_md5", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "file_md5": file_md5,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)

    def delete_document(self, filename: str) -> DocumentDeleteResponse:
        self.milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        self.remove_bm25_stats(filename)

        try:
            result = self.milvus_manager.delete(delete_expr)
        except Exception as e:
            logger.warning(f"删除 Milvus 向量失败 filename={filename}: {e}")
            result = {"delete_count": 0}

        self.parent_chunk_store.delete_by_filename(filename)
        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
