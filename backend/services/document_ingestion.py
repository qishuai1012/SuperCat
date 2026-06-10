import hashlib
import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from document_loader import DocumentLoader
from services.file_utils import calculate_file_md5
from storage.milvus_client import MilvusManager
from storage.milvus_writer import MilvusWriter
from storage.parent_chunk_store import ParentChunkStore
from schemas import DocumentUploadResponse


class DocumentIngestionService:
    """
    【核心服务】文档接入服务（上传 → 解析 → 分块 → 向量化 → 入库）
    职责：处理用户上传的文件，完成全流程 RAG 数据构建
    """

    def __init__(
        self,
        upload_dir: Path,            # 文件上传保存目录
        loader: DocumentLoader,      # 文档加载+分块工具
        milvus_manager: MilvusManager,  # Milvus 客户端
        milvus_writer: MilvusWriter,    # 向量写入工具
        parent_chunk_store: ParentChunkStore,  # 父块存储工具
        remove_bm25_stats,          # 删除 BM25 统计方法
    ):
        # 依赖注入（生产级标准，不硬编码，方便测试）
        self.upload_dir = Path(upload_dir)
        self.loader = loader
        self.milvus_manager = milvus_manager
        self.milvus_writer = milvus_writer
        self.parent_chunk_store = parent_chunk_store
        self.remove_bm25_stats = remove_bm25_stats

    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        """
        上传文档主流程（异步，生产级必备）
        """
        filename = file.filename or ""
        file_lower = filename.lower()

        # ===================== 1. 参数校验 =====================
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        # 只允许 PDF / Word / Excel
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
        # 确保 Milvus 表存在
        self.milvus_manager.init_collection()

        # 生成安全文件名（MD5 + 原名，防止重名覆盖）
        safe_filename = f"{int(hashlib.md5(filename.encode()).hexdigest(), 16)}_{filename}"
        file_path = self.upload_dir / safe_filename

        try:
            # ===================== 2. 保存文件到磁盘 =====================
            await self._save_upload_to_disk(file, file_path)

            # ===================== 3. 计算文件 MD5（唯一标识，用于去重）=====================
            file_md5 = calculate_file_md5(file_path)

            # ===================== 4. MD5 去重：检查文件内容是否已存在 =====================
            existing_md5_expr = f'file_md5 == "{file_md5}"'
            existing_docs = self.milvus_manager.query(existing_md5_expr, limit=1)
            if existing_docs:
                existing_filename = existing_docs[0].get("filename", "未知文件")
                raise HTTPException(
                    status_code=409,
                    detail=f"文件内容已存在，与文件 '{existing_filename}' 内容相同，无需重复上传",
                )

            # ===================== 5. 清理旧数据（同名文件覆盖更新）=====================
            delete_expr = f'filename == "{filename}"'
            self.remove_bm25_stats(filename)  # 清理 BM25 统计
            try:
                self.milvus_manager.delete(delete_expr)  # 删除 Milvus 旧向量
            except Exception:
                pass
            try:
                self.parent_chunk_store.delete_by_filename(filename)  # 删除 PG 父块
            except Exception:
                pass

            # ===================== 6. 核心：加载文档 + 三级分块 =====================
            new_docs = self.loader.load_document(str(file_path), filename)
            if not new_docs:
                raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

            # 给所有块打上文件 MD5 标记
            for doc in new_docs:
                doc["file_md5"] = file_md5

            # ===================== 7. 拆分父块 + 叶子块 =====================
            # 父块（Level1 + Level2）→ 存 PG + Redis
            parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
            # 叶子块（Level3）→ 存 Milvus 向量库
            leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]

            if not leaf_docs:
                raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

            # ===================== 8. 分别入库 =====================
            self.parent_chunk_store.upsert_documents(parent_docs)  # 父块入库
            self.milvus_writer.write_documents(leaf_docs)          # 叶子块向量化+入库

            # ===================== 9. 返回标准响应 =====================
            return DocumentUploadResponse(
                filename=filename,
                chunks_processed=len(leaf_docs),
                file_md5=file_md5,
                message=(
                    f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                    f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
                ),
            )

        # 异常处理：出错则删除临时文件
        except HTTPException:
            if file_path.exists():
                file_path.unlink()
            raise
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")
        finally:
            # 确保关闭上传文件
            await file.close()

    async def _save_upload_to_disk(self, file: UploadFile, file_path: Path, chunk_size: int = 1024 * 1024) -> None:
        """
        异步分块保存大文件（生产级标准，不会内存溢出）
        """
        with open(file_path, "wb") as target:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                target.write(chunk)