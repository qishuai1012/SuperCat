"""
企业级性能配置管理
支持动态调整策略，无需重启服务
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from query_understanding.types import QueryComplexity


@dataclass
class StrategyConfig:
    """处理策略配置"""
    # 功能开关
    enable_routing: bool = True
    enable_decomposition: bool = True
    enable_multi_agent: bool = True
    enable_reflection: bool = True
    enable_query_expansion: bool = True
    enable_document_grading: bool = True
    enable_parallel_execution: bool = True

    # 超时控制（秒）
    router_timeout: float = 5.0
    retrieval_timeout: float = 10.0
    agent_timeout: float = 30.0
    reflection_timeout: float = 15.0

    # 阈值配置
    doc_quality_threshold: float = 0.5  # 文档质量阈值
    min_docs_for_skip_grading: int = 2  # 跳过评分的最小文档数

    # 并行配置
    max_parallel_tasks: int = 3

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存过期时间（秒）


@dataclass
class PerformanceConfig:
    """性能配置总控"""
    # 不同复杂度的策略
    simple_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        enable_multi_agent=False,
        enable_reflection=False,
        enable_query_expansion=False,
        enable_document_grading=False,
        router_timeout=3.0,
        agent_timeout=15.0
    ))

    medium_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        enable_multi_agent=False,
        enable_reflection=False,
        enable_query_expansion=True,
        enable_document_grading=True,
        router_timeout=5.0,
        agent_timeout=30.0
    ))

    complex_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        enable_multi_agent=True,
        enable_reflection=True,
        enable_query_expansion=True,
        enable_document_grading=True,
        router_timeout=10.0,
        agent_timeout=60.0
    ))

    complex_light_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        enable_routing=False,
        enable_decomposition=False,
        enable_multi_agent=True,
        enable_reflection=False,
        enable_query_expansion=True,
        enable_document_grading=True,
        router_timeout=10.0,
        agent_timeout=60.0
    ))

    complex_heavy_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        enable_routing=True,
        enable_decomposition=True,
        enable_multi_agent=True,
        enable_reflection=True,
        enable_query_expansion=True,
        enable_document_grading=True,
        router_timeout=10.0,
        agent_timeout=60.0
    ))

    # 全局配置
    enable_performance_monitoring: bool = True
    enable_auto_degradation: bool = True  # 自动降级
    max_response_time: float = 30.0  # 最大响应时间（秒）

    def get_strategy(self, complexity: QueryComplexity) -> StrategyConfig:
        """根据复杂度获取策略"""
        mapping = {
            QueryComplexity.SIMPLE: self.simple_strategy,
            QueryComplexity.MEDIUM: self.medium_strategy,
            QueryComplexity.COMPLEX: self.complex_strategy,
            QueryComplexity.COMPLEX_LIGHT: self.complex_light_strategy,
            QueryComplexity.COMPLEX_HEAVY: self.complex_heavy_strategy
        }
        return mapping.get(complexity, self.medium_strategy)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "simple_strategy": self.simple_strategy.__dict__,
            "medium_strategy": self.medium_strategy.__dict__,
            "complex_strategy": self.complex_strategy.__dict__,
            "complex_light_strategy": self.complex_light_strategy.__dict__,
            "complex_heavy_strategy": self.complex_heavy_strategy.__dict__,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_auto_degradation": self.enable_auto_degradation,
            "max_response_time": self.max_response_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceConfig':
        """从字典加载"""
        config = cls()
        if "simple_strategy" in data:
            config.simple_strategy = StrategyConfig(**data["simple_strategy"])
        if "medium_strategy" in data:
            config.medium_strategy = StrategyConfig(**data["medium_strategy"])
        if "complex_strategy" in data:
            config.complex_strategy = StrategyConfig(**data["complex_strategy"])
        if "complex_light_strategy" in data:
            config.complex_light_strategy = StrategyConfig(**data["complex_light_strategy"])
        if "complex_heavy_strategy" in data:
            config.complex_heavy_strategy = StrategyConfig(**data["complex_heavy_strategy"])
        config.enable_performance_monitoring = data.get("enable_performance_monitoring", True)
        config.enable_auto_degradation = data.get("enable_auto_degradation", True)
        config.max_response_time = data.get("max_response_time", 30.0)
        return config

    def save(self, path: str = "config/performance_config.json"):
        """保存配置到文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str = "config/performance_config.json") -> 'PerformanceConfig':
        """从文件加载配置"""
        if not os.path.exists(path):
            # 不存在则创建默认配置
            config = cls()
            config.save(path)
            return config

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# 全局配置实例
_performance_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """获取全局性能配置"""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig.load()
    return _performance_config


def reload_performance_config():
    """重新加载配置（支持热更新）"""
    global _performance_config
    _performance_config = PerformanceConfig.load()
