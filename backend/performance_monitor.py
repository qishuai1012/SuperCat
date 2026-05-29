"""
企业级性能监控模块
追踪每个步骤的耗时，识别瓶颈，支持导出分析报告
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """单步骤性能指标"""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error: Optional[str] = None):
        """结束计时"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


@dataclass
class QueryMetrics:
    """单次查询的完整性能指标"""
    query_id: str
    query_text: str
    user_id: str
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    steps: List[StepMetrics] = field(default_factory=list)
    complexity: Optional[str] = None
    strategy_used: Optional[str] = None
    api_calls: int = 0
    success: bool = True

    def add_step(self, step_name: str) -> StepMetrics:
        """添加新步骤"""
        step = StepMetrics(step_name=step_name, start_time=time.time())
        self.steps.append(step)
        return step

    def finish(self, success: bool = True):
        """结束查询计时"""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        self.success = success

    def get_bottleneck(self) -> Optional[StepMetrics]:
        """识别最慢的步骤"""
        if not self.steps:
            return None
        return max(self.steps, key=lambda s: s.duration or 0)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text[:100],  # 截断
            "user_id": self.user_id,
            "session_id": self.session_id,
            "total_duration": self.total_duration,
            "complexity": self.complexity,
            "strategy_used": self.strategy_used,
            "api_calls": self.api_calls,
            "success": self.success,
            "steps": [
                {
                    "name": s.step_name,
                    "duration": s.duration,
                    "success": s.success,
                    "error": s.error
                }
                for s in self.steps
            ],
            "bottleneck": self.get_bottleneck().step_name if self.get_bottleneck() else None
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.current_queries: Dict[str, QueryMetrics] = {}
        self.history: List[QueryMetrics] = []
        self.max_history = 1000

    def start_query(self, query_id: str, query_text: str, user_id: str, session_id: str) -> QueryMetrics:
        """开始监控查询"""
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            user_id=user_id,
            session_id=session_id,
            start_time=time.time()
        )
        self.current_queries[query_id] = metrics
        return metrics

    def finish_query(self, query_id: str, success: bool = True):
        """结束查询监控"""
        if query_id not in self.current_queries:
            return

        metrics = self.current_queries[query_id]
        metrics.finish(success)

        # 移到历史记录
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        del self.current_queries[query_id]

        # 记录日志
        bottleneck = metrics.get_bottleneck()
        logger.info(
            f"查询完成 [{query_id}] "
            f"耗时: {metrics.total_duration:.2f}s, "
            f"API调用: {metrics.api_calls}, "
            f"瓶颈: {bottleneck.step_name if bottleneck else 'N/A'}"
        )

    def get_metrics(self, query_id: str) -> Optional[QueryMetrics]:
        """获取查询指标"""
        return self.current_queries.get(query_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.history:
            return {"total_queries": 0}

        durations = [q.total_duration for q in self.history if q.total_duration]
        api_calls = [q.api_calls for q in self.history]

        step_stats = defaultdict(list)
        for query in self.history:
            for step in query.steps:
                if step.duration:
                    step_stats[step.step_name].append(step.duration)

        return {
            "total_queries": len(self.history),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "avg_api_calls": sum(api_calls) / len(api_calls) if api_calls else 0,
            "success_rate": sum(1 for q in self.history if q.success) / len(self.history),
            "step_statistics": {
                name: {
                    "avg": sum(times) / len(times),
                    "max": max(times),
                    "count": len(times)
                }
                for name, times in step_stats.items()
            }
        }

    def export_report(self, path: str = "logs/performance_report.json"):
        """导出性能报告"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "recent_queries": [q.to_dict() for q in self.history[-100:]]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"性能报告已导出: {path}")


# 全局监控器实例
_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
