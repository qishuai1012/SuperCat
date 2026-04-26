"""父级分块文档存储（用于 Auto-merging Retriever）"""
from datetime import datetime
from typing import List

from cache import cache
from database import SessionLocal
from models import ParentChunk

# ---------------------父块文档的存储工具-----------------
# 作用：
# 把大的父块文档存到 PostgreSQL
# 同时缓存到 Redis
# 提供：增、查、删 功能
# 专门给 Auto-Merge 合并用

class ParentChunkStore:
    """基于 PostgreSQL + Redis 的父级分块存储。"""

    #工具方法：数据库对象 → 字典
    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    #生成 Redis 缓存 key
    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    #写入 / 更新父块
    # 功能：
    # 接收一堆父块 → 存到数据库 + 存到 Redis
    # 流程：
    # 遍历每个块
    # 按 chunk_id 查询是否已存在
    # 存在 → 更新
    # 不存在 → 新增
    # 同时写入 Redis 缓存
    # 返回成功写入条数
    # 作用：
    # 把解析好的大父块，存起来，等合并时用！
    def upsert_documents(self, docs: List[dict]) -> int:
        """写入/更新父级分块，返回写入条数。"""
        if not docs:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                payload = {
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    "text": payload["text"],
                    "filename": payload["filename"],
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": payload["page_number"],
                    "parent_chunk_id": payload["parent_chunk_id"],
                    "root_chunk_id": payload["root_chunk_id"],
                    "chunk_level": payload["chunk_level"],
                    "chunk_idx": payload["chunk_idx"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                cache.set_json(self._cache_key(chunk_id), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    #核心：批量按 ID 查父块（给合并用）
    #功能：
    # 给一堆父块 ID → 返回完整内容
    # 流程：
    # 先去 Redis 查（快）
    # Redis 没有 → 去数据库查
    # 查到后 → 塞进 Redis 缓存
    # 按传入顺序返回结果
    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = cache.get_json(self._cache_key(key))
            if cached:
                ordered_results[key] = cached
            else:
                missing_ids.append(key)

        if missing_ids:
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    # 按文件名删除
    # 删除数据库中该文件的所有父块
    # 同时删除
    # Redis
    # Redis
    # 缓存
    # 返回删除条数
    # 作用：
    # 删除文件时，父块一起删掉！
    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()
