"""
父级分块文档存储（用于 Auto-merging Retriever）
核心功能：
1. 存储 Level1、Level2 父块（不存入向量库，专门用于检索后自动合并）
2. 存储介质：PostgreSQL（持久化） + Redis（缓存加速）
3. 提供：新增/更新、按ID批量查询、按文件名删除
4. 为 Auto-Merging RAG 提供完整上下文支撑，解决小块语义残缺问题
"""
from datetime import datetime
from typing import List

from storage.cache import cache
from database import SessionLocal
from models import ParentChunk


class ParentChunkStore:
    """
    父级分块存储服务
    存储策略：PostgreSQL 做持久化存储，Redis 做高速缓存
    作用：检索到三级小块后，快速查询对应的父块/根块，完成上下文合并
    """

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        """
        工具方法：将数据库 ORM 对象转换为字典
        目的：方便返回给业务使用，同时存入 Redis
        """
        return {
            "text": item.text,                        # 分块文本内容
            "filename": item.filename,                # 所属文件名
            "file_type": item.file_type,              # 文件类型（PDF/Word/Excel）
            "file_path": item.file_path,              # 文件路径
            "page_number": item.page_number,          # 页码
            "chunk_id": item.chunk_id,                # 分块唯一ID
            "parent_chunk_id": item.parent_chunk_id,  # 父块ID
            "root_chunk_id": item.root_chunk_id,      # 根块ID（最上层一级块）
            "chunk_level": item.chunk_level,          # 分块层级（1/2）
            "chunk_idx": item.chunk_idx,              # 全局序号
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        """
        工具方法：生成 Redis 缓存唯一 Key
        格式：parent_chunk:具体chunk_id
        """
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[dict]) -> int:
        """
        【核心写入方法】批量新增或更新父级分块（存在则更新，不存在则插入）
        :param docs: 父级分块数据列表（Level1 + Level2）
        :return: 成功写入/更新的条数
        写入流程：
            1. 遍历每个分块
            2. 根据 chunk_id 查询数据库是否存在
            3. 存在 → 更新数据
            4. 不存在 → 插入新数据
            5. 同时同步更新 Redis 缓存
            6. 提交数据库事务
        """
        # 无数据直接返回
        if not docs:
            return 0

        # 获取数据库连接
        db = SessionLocal()
        upserted = 0  # 统计成功写入条数

        try:
            # 遍历所有父块数据
            for doc in docs:
                # 获取当前块唯一ID，为空则跳过
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                # 根据 chunk_id 查询数据库中是否已有记录
                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()

                # 构造要写入数据库的字段数据
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
                    "updated_at": datetime.utcnow(),  # 更新时间
                }

                # 构造要写入 Redis 缓存的数据
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

                # 数据库记录已存在 → 执行更新
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                # 数据库记录不存在 → 执行新增
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                # 同步写入 Redis 缓存（保证下次查询高速读取）
                cache.set_json(self._cache_key(chunk_id), cache_payload)
                upserted += 1  # 计数+1

            # 提交数据库事务
            db.commit()

        finally:
            # 无论成功失败，都关闭数据库连接
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """
        【核心查询方法】根据 chunk_id 批量查询父块数据
        检索策略：
            1. 先查 Redis 缓存（速度最快）
            2. 缓存未命中 → 查 PostgreSQL
            3. 数据库查到后 → 回写 Redis
            4. 按原始 ID 顺序返回结果
        用途：Auto-Merging 检索时，根据子块的 parent_id 快速获取完整父块上下文
        """
        # 无ID直接返回空
        if not chunk_ids:
            return []

        # 有序存储查询结果（保证返回顺序与传入ID顺序一致）
        ordered_results = {}
        # 记录缓存未命中的ID，需要查数据库
        missing_ids = []

        # ===================== 第一步：查 Redis 缓存 =====================
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            # 从缓存获取数据
            cached = cache.get_json(self._cache_key(key))
            if cached:
                # 缓存命中，直接加入结果
                ordered_results[key] = cached
            else:
                # 缓存未命中，加入待查列表
                missing_ids.append(key)

        # ===================== 第二步：缓存未命中 → 查询数据库 =====================
        if missing_ids:
            db = SessionLocal()
            try:
                # 根据缺失ID批量查询数据库
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                # 遍历查询结果
                for row in rows:
                    # 转为字典格式
                    payload = self._to_dict(row)
                    # 加入结果集
                    ordered_results[row.chunk_id] = payload
                    # 回写 Redis 缓存，加速下次查询
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        # ===================== 第三步：按传入ID顺序返回结果 =====================
        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除该文件的所有父级分块
        功能：
            1. 删除 PostgreSQL 中对应数据
            2. 删除 Redis 中对应缓存
            3. 保持数据一致性
        用途：文件删除/更新时，清理旧的父块数据
        """
        if not filename:
            return 0

        db = SessionLocal()
        try:
            # 1. 查询该文件下所有父块记录（用于获取 chunk_id 删除缓存）
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)  # 要删除的条数

            # 2. 有数据才执行删除
            if deleted > 0:
                # 数据库批量删除
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()

                # 同步删除 Redis 中对应缓存
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))

            return deleted

        finally:
            db.close()