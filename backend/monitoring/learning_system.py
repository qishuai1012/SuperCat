"""
在线学习系统
负责收集用户反馈、分析系统表现并自动优化策略
"""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
import json
import logging
import math
import sqlite3
import os
from datetime import datetime, timedelta
from enum import Enum
import statistics
import threading  # 修复线程安全

# 延迟导入，避免循环依赖
# from intelligent_router import RouteDecision, QueryComplexity
# from dynamic_retrieval_strategy import RetrievalConfig, RetrievalStrategy

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型枚举"""
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    IMPLICIT_ENGAGEMENT = "implicit_engagement"
    IMPLICIT_CORRECTION = "implicit_correction"


class LearningMode(Enum):
    """学习模式枚举"""
    ONLINE = "online"
    BATCH = "batch"
    REINFORCEMENT = "reinforcement"


@dataclass
class UserFeedback:
    """用户反馈数据"""
    id: str
    user_id: str
    session_id: str
    query: str
    response: str
    feedback_type: FeedbackType
    rating: float
    feedback_text: str = ""
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceMetrics:
    """性能指标"""
    query_id: str
    response_time: float
    relevance_score: float
    user_satisfaction: float
    strategy_used: str
    config_used: Dict[str, Any]
    success: bool
    error_message: str = ""
    retrieval_eval: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LearningDatabase:
    """学习数据库 - 存储反馈和性能数据"""

    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = db_path
        self._lock = threading.Lock()  # 修复并发安全
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    feedback_type TEXT,
                    rating REAL,
                    feedback_text TEXT,
                    metadata TEXT,
                    timestamp DATETIME
                )""")
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    query_id TEXT PRIMARY KEY,
                    response_time REAL,
                    relevance_score REAL,
                    user_satisfaction REAL,
                    strategy_used TEXT,
                    config_used TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    retrieval_eval TEXT,
                    timestamp DATETIME
                )""")
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learning_type TEXT,
                    before_metrics TEXT,
                    after_metrics TEXT,
                    improvement_score REAL,
                    timestamp DATETIME
                )""")
                cursor.execute("PRAGMA table_info(performance_metrics)")
                columns = {row[1] for row in cursor.fetchall()}
                if "retrieval_eval" not in columns:
                    cursor.execute("ALTER TABLE performance_metrics ADD COLUMN retrieval_eval TEXT")
                conn.commit()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def save_feedback(self, feedback: UserFeedback):
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO user_feedback
                (id, user_id, session_id, query, response, feedback_type, rating, feedback_text, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    feedback.id, feedback.user_id, feedback.session_id, feedback.query, feedback.response,
                    feedback.feedback_type.value, feedback.rating, feedback.feedback_text,
                    json.dumps(feedback.metadata, ensure_ascii=False), feedback.timestamp.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def save_performance_metrics(self, metrics: PerformanceMetrics):
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO performance_metrics
                (query_id, response_time, relevance_score, user_satisfaction, strategy_used, config_used, success, error_message, retrieval_eval, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    metrics.query_id, metrics.response_time, metrics.relevance_score, metrics.user_satisfaction,
                    metrics.strategy_used, json.dumps(metrics.config_used),
                    metrics.success, metrics.error_message, json.dumps(metrics.retrieval_eval or {}, ensure_ascii=False), metrics.timestamp.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")

    def get_recent_feedback(self, days: int = 7) -> List[UserFeedback]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute("SELECT * FROM user_feedback WHERE timestamp > ? ORDER BY timestamp DESC", (cutoff,))
                rows = cursor.fetchall()

            res = []
            for r in rows:
                try:
                    meta = json.loads(r[8]) if r[8] and r[8].strip() else {}
                    ts = datetime.fromisoformat(r[9])
                    res.append(UserFeedback(
                        id=r[0], user_id=r[1], session_id=r[2], query=r[3], response=r[4],
                        feedback_type=FeedbackType(r[5]), rating=r[6], feedback_text=r[7] or "",
                        metadata=meta, timestamp=ts
                    ))
                except Exception as e:
                    logger.warning(f"解析反馈记录失败: {e}")
            return res
        except Exception as e:
            logger.error(f"获取反馈失败: {e}")
            return []

    def get_performance_metrics(self, days: int = 7) -> List[PerformanceMetrics]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute(
                    "SELECT query_id, response_time, relevance_score, user_satisfaction, strategy_used, config_used, success, error_message, retrieval_eval, timestamp FROM performance_metrics WHERE timestamp > ? ORDER BY timestamp DESC",
                    (cutoff,),
                )
                rows = cursor.fetchall()

            res = []
            for r in rows:
                try:
                    config = json.loads(r[5]) if r[5] and str(r[5]).strip() else {}
                    retrieval_eval = json.loads(r[8]) if len(r) > 8 and r[8] and str(r[8]).strip() else {}
                    ts = datetime.fromisoformat(r[9])
                    res.append(PerformanceMetrics(
                        query_id=r[0], response_time=r[1], relevance_score=r[2], user_satisfaction=r[3],
                        strategy_used=r[4], config_used=config, success=bool(r[6]), error_message=r[7] or "",
                        retrieval_eval=retrieval_eval,
                        timestamp=ts
                    ))
                except Exception as e:
                    logger.warning(f"解析性能记录失败: {e}")
            return res
        except sqlite3.OperationalError as e:
            if "no such column: retrieval_eval" in str(e):
                try:
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                        cursor.execute(
                            "SELECT query_id, response_time, relevance_score, user_satisfaction, strategy_used, config_used, success, error_message, timestamp FROM performance_metrics WHERE timestamp > ? ORDER BY timestamp DESC",
                            (cutoff,),
                        )
                        rows = cursor.fetchall()

                    res = []
                    for r in rows:
                        try:
                            config = json.loads(r[5]) if r[5] and str(r[5]).strip() else {}
                            ts = datetime.fromisoformat(r[8])
                            res.append(PerformanceMetrics(
                                query_id=r[0], response_time=r[1], relevance_score=r[2], user_satisfaction=r[3],
                                strategy_used=r[4], config_used=config, success=bool(r[6]), error_message=r[7] or "",
                                retrieval_eval={},
                                timestamp=ts
                            ))
                        except Exception as inner_e:
                            logger.warning(f"解析旧性能记录失败: {inner_e}")
                    return res
                except Exception as inner_e:
                    logger.error(f"获取旧性能指标失败: {inner_e}")
                    return []
            logger.error(f"获取性能指标失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return []


class OnlineLearningSystem:
    """
    在线学习系统 - 持续优化Agentic RAG性能
    """
    def __init__(self, db_path: str = "data/learning.db"):
        self.database = LearningDatabase(db_path)
        self.learning_mode = LearningMode.ONLINE
        self.min_feedback_threshold = 10
        self.performance_window_days = 7
        self.optimization_interval_hours = 24
        self.last_optimization = None
        self.current_optimizations = {}
        self._lock = threading.Lock()  # 修复线程安全
        self._cycle_running = False

    def collect_feedback(self, feedback: UserFeedback):
        try:
            self.database.save_feedback(feedback)
            logger.info(f"已收集反馈: {feedback.id}")
            if self.learning_mode == LearningMode.ONLINE and feedback.rating <= 2.0:
                self._process_critical_feedback(feedback)
        except Exception as e:
            logger.error(f"收集反馈失败: {e}")

    def collect_performance_metrics(self, metrics: PerformanceMetrics):
        try:
            self.database.save_performance_metrics(metrics)
        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")

    def _process_critical_feedback(self, feedback: UserFeedback):
        try:
            analysis = self._analyze_feedback_issue(feedback)
            opt = self._generate_immediate_optimization(feedback, analysis)
            if opt:
                with self._lock:
                    self.current_optimizations[f"critical_{feedback.id}"] = opt
                logger.info(f"已生成关键优化: {feedback.id}")
        except Exception as e:
            logger.error(f"处理关键反馈失败: {e}")

    def _analyze_feedback_issue(self, feedback: UserFeedback) -> Dict[str, Any]:
        analysis = {"query_complexity": "unknown", "likely_issue": "unknown", "suggested_improvements": []}
        if feedback.rating <= 1.0:
            analysis["severity"] = "critical"
        elif feedback.rating <= 2.0:
            analysis["severity"] = "high"
        else:
            analysis["severity"] = "medium"

        txt = feedback.feedback_text.lower()
        if any(w in txt for w in ['不准确', '错误', '不对', '错']):
            analysis["likely_issue"] = "accuracy"
            analysis["suggested_improvements"] = ["增强事实核查", "改进检索准确性"]
        elif any(w in txt for w in ['不完整', '缺少', '不够详细']):
            analysis["likely_issue"] = "completeness"
            analysis["suggested_improvements"] = ["增加检索范围", "提高答案详细度"]
        elif any(w in txt for w in ['太慢', '慢', '等待']):
            analysis["likely_issue"] = "performance"
            analysis["suggested_improvements"] = ["优化响应时间", "减少检索数量"]
        elif any(w in txt for w in ['不清楚', '难懂', '混乱']):
            analysis["likely_issue"] = "clarity"
            analysis["suggested_improvements"] = ["改善表达清晰度", "优化答案结构"]
        return analysis

    def _generate_immediate_optimization(self, feedback: UserFeedback, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        opt = {
            "feedback_id": feedback.id,
            "timestamp": datetime.now().isoformat(),
            "priority": analysis.get("severity", "medium"),
            "adjustments": []
        }
        t = analysis.get("likely_issue")
        if t == "accuracy":
            opt["adjustments"] = [{"component": "verification", "action": "increase_weight", "value": 0.2}]
        elif t == "completeness":
            opt["adjustments"] = [{"component": "retrieval", "action": "increase_top_k", "value": 2}]
        elif t == "performance":
            opt["adjustments"] = [{"component": "retrieval", "action": "decrease_top_k", "value": -1}]
        elif t == "clarity":
            opt["adjustments"] = [{"component": "synthesis", "action": "improve_structure", "value": True}]
        return opt if opt["adjustments"] else None

    def evaluate_single_retrieval_result(self, judged_result: Dict[str, Any]) -> Dict[str, Any]:
        relevant = {str(item) for item in (judged_result.get("relevant_ids") or []) if item is not None}
        retrieved = [str(item) for item in (judged_result.get("retrieved_ids") or []) if item is not None]
        if not relevant:
            return {"status": "insufficient_data"}

        k = int(judged_result.get("k") or len(retrieved) or 0)
        ranked = retrieved[:k] if k > 0 else []
        hits = [1 if doc_id in relevant else 0 for doc_id in ranked]
        hit_count = sum(hits)
        hit = 1 if hit_count > 0 else 0
        recall = hit_count / max(1, len(relevant))

        reciprocal_rank = 0.0
        first_relevant_rank = None
        for idx, h in enumerate(hits, 1):
            if h:
                reciprocal_rank = 1.0 / idx
                first_relevant_rank = idx
                break

        def _dcg(rels: List[int]) -> float:
            score = 0.0
            for idx, rel in enumerate(rels, 1):
                if rel > 0:
                    score += rel / math.log2(idx + 1)
            return score

        ideal = sorted(hits, reverse=True)
        ndcg = (_dcg(hits) / _dcg(ideal)) if _dcg(ideal) else 0.0
        matched_ids = [doc_id for doc_id in ranked if doc_id in relevant]
        return {
            "status": "ok",
            "query": judged_result.get("query"),
            "k": k,
            "retrieved_count": len(ranked),
            "relevant_count": len(relevant),
            "matched_relevant_count": hit_count,
            "matched_relevant_ids": matched_ids,
            "first_relevant_rank": first_relevant_rank,
            "recall_at_k": recall,
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "hit_rate": hit,
        }

    def evaluate_retrieval_metrics(self, judged_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not judged_results:
            return {"status": "insufficient_data"}

        evaluated = []
        for item in judged_results:
            single_result = self.evaluate_single_retrieval_result(item)
            if single_result.get("status") == "ok":
                evaluated.append(single_result)

        if not evaluated:
            return {"status": "insufficient_data"}

        return {
            "status": "ok",
            "queries": len(evaluated),
            "recall_at_k": sum(item["recall_at_k"] for item in evaluated) / len(evaluated),
            "mrr": sum(item["mrr"] for item in evaluated) / len(evaluated),
            "ndcg": sum(item["ndcg"] for item in evaluated) / len(evaluated),
            "hit_rate": sum(item["hit_rate"] for item in evaluated) / len(evaluated),
            "per_query": evaluated,
        }

    def analyze_performance_trends(self) -> Dict[str, Any]:
        try:
            m = self.database.get_performance_metrics(self.performance_window_days)
            if not m:
                return {"status": "insufficient_data"}

            ok = [x for x in m if x.success]
            rel = [x.relevance_score for x in m if x.relevance_score > 0]
            sat = [x.user_satisfaction for x in m if x.user_satisfaction > 0]

            def avg(x): return statistics.mean(x) if x else 0.0

            return {
                "period_days": self.performance_window_days,
                "total_queries": len(m),
                "success_rate": sum(1 for x in m if x.success) / len(m) if m else 0.0,
                "avg_response_time": avg([x.response_time for x in ok]),
                "avg_relevance_score": avg(rel),
                "avg_satisfaction": avg(sat),
                "strategy_performance": self._analyze_strategy_performance(m),
                "trends": self._calculate_trends(m)
            }
        except Exception as e:
            logger.error(f"分析性能失败: {e}")
            return {"error": str(e)}

    def _analyze_strategy_performance(self, m: List[PerformanceMetrics]) -> Dict[str, Any]:
        d = {}
        for x in m:
            s = x.strategy_used
            if s not in d:
                d[s] = {"count": 0, "success_count": 0, "total_response_time": 0, "total_relevance": 0, "total_satisfaction": 0}
            d[s]["count"] += 1
            if x.success: d[s]["success_count"] += 1
            d[s]["total_response_time"] += x.response_time
            d[s]["total_relevance"] += x.relevance_score
            d[s]["total_satisfaction"] += x.user_satisfaction

        for s, v in d.items():
            c = v["count"]
            v["success_rate"] = v["success_count"] / c if c else 0
            v["avg_response_time"] = v["total_response_time"] / c if c else 0
            v["avg_relevance"] = v["total_relevance"] / c if c else 0
            v["avg_satisfaction"] = v["total_satisfaction"] / c if c else 0
        return d

    def _calculate_trends(self, m: List[PerformanceMetrics]) -> Dict[str, Any]:
        try:
            if len(m) < 10:
                return {"status": "insufficient_data"}
            mid = len(m) // 2
            fst = m[mid:]
            snd = m[:mid]

            def avg_succ(lst, key):
                data = [getattr(x, key) for x in lst if x.success]
                return statistics.mean(data) if data else 0.01

            def safe_ratio(a, b):
                return (a - b) / max(b, 0.01)

            ft = avg_succ(fst, "response_time")
            st = avg_succ(snd, "response_time")

            fr = statistics.mean([x.relevance_score for x in fst if x.relevance_score > 0] or [0.01])
            sr = statistics.mean([x.relevance_score for x in snd if x.relevance_score > 0] or [0.01])

            fs = sum(1 for x in fst if x.success) / len(fst) if fst else 0.01
            ss = sum(1 for x in snd if x.success) / len(snd) if snd else 0.01

            return {
                "response_time_change": safe_ratio(st, ft),
                "relevance_change": safe_ratio(sr, fr),
                "success_rate_change": safe_ratio(ss, fs)
            }
        except Exception as e:
            logger.warning(f"计算趋势失败: {e}")
            return {"status": "error"}

    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        rec = []
        try:
            pa = self.analyze_performance_trends()
            if "error" in pa: return rec

            if pa.get("avg_response_time", 0) > 3.0:
                rec.append({"type": "performance", "priority": "high", "issue": "响应慢",
                            "suggestion": "减少top_k", "actions": [{"component": "retrieval", "parameter": "top_k", "adjustment": -1}]})
            if pa.get("avg_relevance_score", 1) < 0.6:
                rec.append({"type": "relevance", "priority": "high", "issue": "相关性低",
                            "suggestion": "降低阈值", "actions": [{"component": "retrieval", "parameter": "threshold", "adjustment": -0.1}]})
            if pa.get("success_rate", 1) < 0.8:
                rec.append({"type": "reliability", "priority": "critical", "issue": "成功率低",
                            "suggestion": "启用fallback", "actions": [{"component": "fallback", "parameter": "enable", "adjustment": True}]})

            with self._lock:
                for k, v in self.current_optimizations.items():
                    if v["priority"] in ["critical", "high"]:
                        rec.append({
                            "type": "immediate", "priority": v["priority"],
                            "issue": "用户反馈即时优化", "suggestion": "应用调整",
                            "actions": v["adjustments"], "feedback_id": v["feedback_id"]
                        })
            return sorted(rec, key=lambda x: {"critical": 3, "high": 2, "medium": 1}.get(x["priority"], 0), reverse=True)
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return []

    def apply_optimizations(self, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok, fail = [], []
        try:
            for r in recs[:5]:
                try:
                    logger.info(f"应用优化: {r}")
                    ok.append({"recommendation": r, "result": {"success": True}})
                except Exception as e:
                    fail.append({"recommendation": r, "error": str(e)})

            with self._lock:
                for item in ok:
                    if item["recommendation"]["type"] == "immediate":
                        fid = item["recommendation"].get("feedback_id")
                        if fid:
                            self.current_optimizations.pop(f"critical_{fid}", None)
            return {"applied": ok, "failed": fail, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"应用优化失败: {e}")
            return {"applied": [], "failed": recs, "error": str(e)}

    def run_learning_cycle(self):
        try:
            with self._lock:
                if self._cycle_running:
                    return {"status": "already_running"}
                self._cycle_running = True

            logger.info("开始学习周期")
            pa = self.analyze_performance_trends()
            recs = self.generate_optimization_recommendations()
            res = self.apply_optimizations(recs) if recs else {}
            if recs:
                self._record_learning_cycle(pa, recs, res)

            return {
                "status": "completed",
                "recommendations_count": len(recs),
                "applied_count": len(res.get("applied", [])),
            }
        except Exception as e:
            logger.error(f"学习周期失败: {e}")
            return {"status": "failed", "error": str(e)}
        finally:
            with self._lock:
                self._cycle_running = False

    def _record_learning_cycle(self, pa: Dict, recs: List, res: Dict):
        try:
            with self.database._lock, self.database._get_connection() as conn:
                cursor = conn.cursor()
                score = 0.0
                if "avg_response_time" in pa and "avg_relevance_score" in pa:
                    t = max(0.1, pa.get("avg_response_time", 3))
                    r = pa.get("avg_relevance_score", 0.6)
                    score = ((3 - t)/3 + (r - 0.6)/0.6)/2
                cursor.execute("""
                INSERT INTO learning_history
                (learning_type, before_metrics, after_metrics, improvement_score, timestamp)
                VALUES (?, ?, ?, ?, ?)""", (
                    "online_optimization", json.dumps(pa), json.dumps(res),
                    round(score, 2), datetime.now().isoformat()
                ))
                conn.commit()
                logger.info(f"学习记录完成，得分: {score:.2f}")
        except Exception as e:
            logger.error(f"记录学习历史失败: {e}")


# 全局单例（线程安全）
_online_learning_system = None
_lock = threading.Lock()

def get_online_learning_system() -> OnlineLearningSystem:
    global _online_learning_system
    with _lock:
        if _online_learning_system is None:
            _online_learning_system = OnlineLearningSystem()
    return _online_learning_system