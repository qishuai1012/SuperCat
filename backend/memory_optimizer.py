"""
记忆优化系统
实现长期记忆管理、记忆压缩和智能记忆检索
"""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
import json
import logging
import sqlite3
import os
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import threading
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class MemoryImportance(Enum):
    """记忆重要性级别"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    type: MemoryType
    content: str
    importance: MemoryImportance
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_factor: float = 1.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.created_at, str):
            try:
                self.created_at = datetime.fromisoformat(self.created_at)
            except:
                self.created_at = datetime.now()
        if isinstance(self.last_accessed, str):
            try:
                self.last_accessed = datetime.fromisoformat(self.last_accessed)
            except:
                self.last_accessed = datetime.now()


@dataclass
class MemoryQuery:
    """记忆查询"""
    query: str
    memory_types: List[MemoryType]
    time_range: Optional[tuple] = None
    tags: Optional[List[str]] = None
    min_importance: MemoryImportance = MemoryImportance.LOW
    limit: int = 10


class MemoryDatabase:
    """记忆数据库 - 存储和管理长期记忆"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_database(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with self._lock, self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    content TEXT,
                    importance INTEGER,
                    tags TEXT,
                    metadata TEXT,
                    created_at DATETIME,
                    last_accessed DATETIME,
                    access_count INTEGER,
                    decay_factor REAL,
                    embedding BLOB
                )""")
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 TEXT,
                    memory_id_2 TEXT,
                    relation_type TEXT,
                    strength REAL,
                    FOREIGN KEY (memory_id_1) REFERENCES memories (id),
                    FOREIGN KEY (memory_id_2) REFERENCES memories (id)
                )""")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON memories(importance)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memories(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memories(last_accessed)")
                conn.commit()
        except Exception as e:
            logger.error(f"记忆库初始化失败: {e}")

    def save_memory(self, memory: MemoryEntry):
        try:
            with self._lock, self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO memories
                (id, type, content, importance, tags, metadata, created_at, last_accessed, access_count, decay_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    memory.id, memory.type.value, memory.content, memory.importance.value,
                    json.dumps(memory.tags, ensure_ascii=False),
                    json.dumps(memory.metadata, ensure_ascii=False),
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count, memory.decay_factor
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")

    def retrieve_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                sql = "SELECT * FROM memories WHERE 1=1"
                params = []
                if query.memory_types:
                    ph = ",".join(["?"] * len(query.memory_types))
                    sql += f" AND type IN ({ph})"
                    params.extend([t.value for t in query.memory_types])
                sql += " AND importance >= ?"
                params.append(query.min_importance.value)
                if query.time_range:
                    s, e = query.time_range
                    sql += " AND created_at BETWEEN ? AND ?"
                    params.extend([s.isoformat(), e.isoformat()])
                if query.tags:
                    conds = []
                    for t in query.tags:
                        conds.append("tags LIKE ?")
                        params.append(f'%"{t}"%')
                    sql += " AND (" + " OR ".join(conds) + ")"
                sql += " ORDER BY (importance * decay_factor * (1.0 + access_count * 0.1)) DESC LIMIT ?"
                params.append(query.limit)
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            res = []
            for r in rows:
                try:
                    tags = json.loads(r[4]) if r[4] and r[4].strip() else []
                    meta = json.loads(r[5]) if r[5] and r[5].strip() else {}
                    res.append(MemoryEntry(
                        id=r[0], type=MemoryType(r[1]), content=r[2],
                        importance=MemoryImportance(r[3]), tags=tags, metadata=meta,
                        created_at=r[6], last_accessed=r[7],
                        access_count=r[8], decay_factor=r[9]
                    ))
                except Exception as e:
                    logger.warning(f"解析记忆失败: {e}")
            return res
        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []

    def update_memory_access(self, memory_id: str):
        try:
            with self._lock, self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?""", (datetime.now().isoformat(), memory_id))
                conn.commit()
        except Exception as e:
            logger.error(f"更新记忆访问失败: {e}")

    def create_memory_relation(self, m1: str, m2: str, rt: str, s: float = 1.0):
        try:
            with self._lock, self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO memory_relations (memory_id_1, memory_id_2, relation_type, strength)
                VALUES (?, ?, ?, ?)""", (m1, m2, rt, s))
                conn.commit()
        except Exception as e:
            logger.error(f"创建记忆关联失败: {e}")

    def cleanup_old_memories(self, days=90):
        try:
            with self._lock, self._get_conn() as conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute("""
                DELETE FROM memories
                WHERE created_at < ? AND importance <= ? AND access_count < 3""",
                               (cutoff, MemoryImportance.MEDIUM.value))
                cnt = cursor.rowcount
                conn.commit()
                logger.info(f"清理了 {cnt} 条旧记忆")
                return cnt
        except Exception as e:
            logger.error(f"清理记忆失败: {e}")
            return 0


class MemoryCompressor:
    def __init__(self):
        self.model = None
        self._init()

    def _init(self):
        try:
            if API_KEY and MODEL:
                self.model = init_chat_model(
                    model=MODEL, model_provider="openai",
                    api_key=API_KEY, base_url=BASE_URL, temperature=0.1
                )
        except Exception as e:
            logger.warning(f"压缩模型初始化失败: {e}")
            self.model = None

    def compress_conversation(self, hist: List[Dict[str, str]]) -> str:
        if not hist:
            return ""
        if not self.model:
            return self._simple(hist)
        try:
            text = "\n".join([f"{m['role']}: {m['content']}" for m in hist])
            prompt = f"请总结以下对话，保留关键信息（≤200字）:\n{text}\n总结:"
            return self.model.invoke(prompt).content
        except:
            return self._simple(hist)

    def _simple(self, hist):
        if len(hist) <= 3:
            return "\n".join([f"{m['role']}: {m['content']}" for m in hist])
        selected = [hist[0], hist[len(hist)//2], hist[-1]]
        return "\n".join([f"{m['role']}: {m['content']}" for m in selected])

    def extract_key_knowledge(self, content: str) -> List[Dict]:
        if not content:
            return []
        if not self.model:
            return self._simple_knowledge(content)
        try:
            prompt = f"提取文本知识点，返回JSON格式：{content}"
            res = self.model.invoke(prompt).content
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                return json.loads(match.group()).get("knowledge_points", [])
        except:
            pass
        return self._simple_knowledge(content)

    def _simple_knowledge(self, content):
        res = []
        if any(p in content for p in ['是', '指', '定义']):
            res.append({"concept": "概念", "definition": content[:100], "importance": 3, "category": "def"})
        if any(p in content for p in ['步骤', '首先', '然后']):
            res.append({"concept": "流程", "definition": content[:100], "importance": 4, "category": "proc"})
        return res


class MemoryOptimizer:
    def __init__(self, db_path="data/memory.db"):
        self.db = MemoryDatabase(db_path)
        self.compressor = MemoryCompressor()
        self.max_working = 50
        self.threshold = 100
        self.decay = 0.95
        self.working_memory = []
        self._lock = threading.Lock()

    def add_to_working_memory(self, content: str, imp=MemoryImportance.MEDIUM, tags=None):
        with self._lock:
            mid = hashlib.md5(content.encode()).hexdigest()[:16]
            now = datetime.now()
            entry = MemoryEntry(
                id=mid, type=MemoryType.WORKING, content=content,
                importance=imp, tags=tags or [], metadata={"source": "working"},
                created_at=now, last_accessed=now, access_count=1
            )
            self.working_memory.append(entry)
            if len(self.working_memory) > self.max_working:
                self.working_memory.sort(key=lambda x: x.importance.value, reverse=True)
                self.working_memory = self.working_memory[:self.max_working]

    def promote_to_long_term_memory(self, entry: MemoryEntry, new_type=MemoryType.EPISODIC):
        try:
            long = MemoryEntry(
                id=entry.id, type=new_type, content=entry.content,
                importance=entry.importance, tags=entry.tags,
                metadata={**entry.metadata, "promoted": True},
                created_at=datetime.now(), last_accessed=datetime.now(),
                access_count=entry.access_count
            )
            self.db.save_memory(long)
            with self._lock:
                self.working_memory = [m for m in self.working_memory if m.id != entry.id]
        except Exception as e:
            logger.error(f"提升记忆失败: {e}")

    def retrieve_relevant_memories(self, query: str, context=None) -> List[MemoryEntry]:
        ctx = context or {}
        mq = MemoryQuery(
            query=query,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            min_importance=MemoryImportance.MEDIUM, limit=10
        )
        long_term = self.db.retrieve_memories(mq)
        relevant_work = []
        with self._lock:
            for m in self.working_memory:
                if self._rel(query, m.content) > 0.3:
                    relevant_work.append(m)
                    m.access_count += 1
                    m.last_accessed = datetime.now()
        all_mem = relevant_work + long_term
        all_mem.sort(key=lambda x: self._rel(query, x.content), reverse=True)
        return all_mem[:10]

    def _rel(self, q, c):
        q_words = set(q.lower().split())
        if not q_words:
            return 0.0
        c_words = set(c.lower().split())
        return len(q_words & c_words) / len(q_words)

    def update_memory_decay(self):
        with self._lock:
            now = datetime.now()
            for m in self.working_memory:
                days = (now - m.created_at).days
                m.decay_factor = (self.decay ** days)
            self.working_memory = [
                m for m in self.working_memory
                if m.decay_factor > 0.1 or m.importance.value >= MemoryImportance.HIGH.value
            ]

    def run_memory_maintenance(self):
        try:
            self.update_memory_decay()
            self.db.cleanup_old_memories()
            promoted = []
            with self._lock:
                for m in self.working_memory:
                    if m.importance.value >= MemoryImportance.HIGH.value and m.access_count >= 3:
                        promoted.append(m)
            for m in promoted:
                self.promote_to_long_term_memory(m)
            return {"status": "ok", "promoted": len(promoted)}
        except Exception as e:
            logger.error(f"记忆维护失败: {e}")
            return {"status": "fail"}

    def get_memory_statistics(self):
        return {"working": len(self.working_memory), "long_term": 0}


_memory_optimizer = None
_gbl_lock = threading.Lock()

def get_memory_optimizer():
    global _memory_optimizer
    with _gbl_lock:
        if _memory_optimizer is None:
            _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer

def add_memory(content: str, importance=2, tags=None):
    opt = get_memory_optimizer()
    imp = MemoryImportance(min(4, max(1, importance)))
    opt.add_to_working_memory(content, imp, tags or [])

def retrieve_memories(query: str, context=None) -> List[MemoryEntry]:
    opt = get_memory_optimizer()
    return opt.retrieve_relevant_memories(query, context)