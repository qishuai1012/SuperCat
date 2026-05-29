import hashlib
import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from document_loader import DocumentLoader
from file_utils import calculate_file_md5
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from parent_chunk_store import ParentChunkStore
from schemas import DocumentUploadResponse


class DocumentIngestionService:
    def __init__(
        self,
        upload_dir: Path,
        loader: DocumentLoader,
        milvus_manager: MilvusManager,
        milvus_writer: MilvusWriter,
        parent_chunk_store: ParentChunkStore,
        remove_bm25_stats,
    ):
        self.upload_dir = Path(upload_dir)
        self.loader = loader
        self.milvus_manager = milvus_manager
        self.milvus_writer = milvus_writer
        self.parent_chunk_store = parent_chunk_store
        self.remove_bm25_stats = remove_bm25_stats

    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        os.makedirs(self.upload_dir, exist_ok=True)
        self.milvus_manager.init_collection()

        safe_filename = f"{int(hashlib.md5(filename.encode()).hexdigest(), 16)}_{filename}"
        file_path = self.upload_dir / safe_filename

        try:
            await self._save_upload_to_disk(file, file_path)
            file_md5 = calculate_file_md5(file_path)

            existing_md5_expr = f'file_md5 == "{file_md5}"'
            existing_docs = self.milvus_manager.query(existing_md5_expr, limit=1)
            if existing_docs:
                existing_filename = existing_docs[0].get("filename", "未知文件")
                raise HTTPException(
                    status_code=409,
                    detail=f"文件内容已存在，与文件 '{existing_filename}' 内容相同，无需重复上传",
                )

            delete_expr = f'filename == "{filename}"'
            self.remove_bm25_stats(filename)
            try:
                self.milvus_manager.delete(delete_expr)
            except Exception:
                pass
            try:
                self.parent_chunk_store.delete_by_filename(filename)
            except Exception:
                pass

            new_docs = self.loader.load_document(str(file_path), filename)
            if not new_docs:
                raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

            for doc in new_docs:
                doc["file_md5"] = file_md5

            parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
            leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
            if not leaf_docs:
                raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

            self.parent_chunk_store.upsert_documents(parent_docs)
            self.milvus_writer.write_documents(leaf_docs)

            return DocumentUploadResponse(
                filename=filename,
                chunks_processed=len(leaf_docs),
                file_md5=file_md5,
                message=(
                    f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                    f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
                ),
            )
        except HTTPException:
            if file_path.exists():
                file_path.unlink()
            raise
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")
        finally:
            await file.close()

    async def _save_upload_to_disk(self, file: UploadFile, file_path: Path, chunk_size: int = 1024 * 1024) -> None:
        with open(file_path, "wb") as target:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                target.write(chunk)
