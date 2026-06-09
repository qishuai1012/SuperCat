"""
自适应调优系统
自动优化系统参数和策略以提升性能
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
import asyncio
import random
from datetime import datetime
from enum import Enum
import threading

# 延迟导入，避免循环依赖
# from learning_system import OnlineLearningSystem, PerformanceMetrics
# from dynamic_retrieval_strategy.md import RetrievalStrategy, RetrievalConfig


logger = logging.getLogger(__name__)


class TuningParameter(Enum):
    """调优参数枚举"""
    RETRIEVAL_TOP_K = "retrieval_top_k"
    RETRIEVAL_THRESHOLD = "retrieval_threshold"
    RERANK_ENABLED = "rerank_enabled"
    HYBRID_WEIGHT_DENSE = "hybrid_weight_dense"
    HYBRID_WEIGHT_SPARSE = "hybrid_weight_sparse"
    DECOMPOSITION_THRESHOLD = "decomposition_threshold"
    PARALLEL_PATHS = "parallel_paths"
    RESPONSE_TIMEOUT = "response_timeout"


class OptimizationObjective(Enum):
    """优化目标枚举"""
    RESPONSE_TIME = "response_time"      # 最小化响应时间
    RELEVANCE = "relevance"              # 最大化相关性
    SUCCESS_RATE = "success_rate"        # 最大化成功率
    USER_SATISFACTION = "user_satisfaction"  # 最大化用户满意度
    COMPREHENSIVENESS = "comprehensiveness"  # 最大化答案完整性


@dataclass
class ParameterRange:
    """参数范围定义"""
    min_value: float
    max_value: float
    step: float = 0.1
    is_integer: bool = False


@dataclass
class TuningConfig:
    """调优配置"""
    parameter: TuningParameter
    current_value: float
    range: ParameterRange
    sensitivity: float = 1.0  # 参数敏感度


@dataclass
class OptimizationResult:
    """优化结果"""
    parameter: TuningParameter
    old_value: float
    new_value: float
    improvement: float
    confidence: float
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]


class AdaptiveTuner:
    """
    自适应调优器 - 自动优化系统参数
    """

    def __init__(self, learning_system=None):
        # 线程安全锁
        self._lock = threading.Lock()
        # 延迟初始化，避免循环依赖
        self._learning_system = learning_system
        self._learning_system_loaded = False

        # ========== 修复1：把所有初始化从setter移到__init__，解决实例化报错 ==========
        # 参数配置
        self.parameter_configs = {
            TuningParameter.RETRIEVAL_TOP_K: TuningConfig(
                parameter=TuningParameter.RETRIEVAL_TOP_K,
                current_value=5,
                range=ParameterRange(1, 20, 1, True),
                sensitivity=0.8
            ),
            TuningParameter.RETRIEVAL_THRESHOLD: TuningConfig(
                parameter=TuningParameter.RETRIEVAL_THRESHOLD,
                current_value=0.3,
                range=ParameterRange(0.1, 0.8, 0.05),
                sensitivity=1.2
            ),
            TuningParameter.RERANK_ENABLED: TuningConfig(
                parameter=TuningParameter.RERANK_ENABLED,
                current_value=1.0,  # 1=True, 0=False
                range=ParameterRange(0, 1, 1, True),
                sensitivity=0.5
            ),
            TuningParameter.HYBRID_WEIGHT_DENSE: TuningConfig(
                parameter=TuningParameter.HYBRID_WEIGHT_DENSE,
                current_value=0.7,
                range=ParameterRange(0.1, 0.9, 0.1),
                sensitivity=1.0
            ),
            TuningParameter.HYBRID_WEIGHT_SPARSE: TuningConfig(
                parameter=TuningParameter.HYBRID_WEIGHT_SPARSE,
                current_value=0.3,
                range=ParameterRange(0.1, 0.9, 0.1),
                sensitivity=1.0
            ),
            TuningParameter.DECOMPOSITION_THRESHOLD: TuningConfig(
                parameter=TuningParameter.DECOMPOSITION_THRESHOLD,
                current_value=2,
                range=ParameterRange(1, 5, 1, True),
                sensitivity=0.7
            ),
            TuningParameter.PARALLEL_PATHS: TuningConfig(
                parameter=TuningParameter.PARALLEL_PATHS,
                current_value=2,
                range=ParameterRange(1, 5, 1, True),
                sensitivity=0.6
            ),
            TuningParameter.RESPONSE_TIMEOUT: TuningConfig(
                parameter=TuningParameter.RESPONSE_TIMEOUT,
                current_value=30.0,
                range=ParameterRange(10.0, 60.0, 5.0),
                sensitivity=0.4
            )
        }

        # 优化目标权重
        self.objective_weights = {
            OptimizationObjective.RESPONSE_TIME: 0.25,
            OptimizationObjective.RELEVANCE: 0.35,
            OptimizationObjective.SUCCESS_RATE: 0.25,
            OptimizationObjective.USER_SATISFACTION: 0.15
        }

        # 调优历史
        self.tuning_history = []
        self.max_history_size = 50

        # 当前优化状态
        self.is_optimizing = False
        self.current_experiment = None

        # 初始化实际调优方法
        self._setup_optimization_methods()

    @property
    def learning_system(self):
        # 加锁避免多线程重复导入
        with self._lock:
            if not self._learning_system_loaded and self._learning_system is None:
                from learning_system import OnlineLearningSystem
                self._learning_system = OnlineLearningSystem()
                self._learning_system_loaded = True
        return self._learning_system

    @learning_system.setter
    def learning_system(self, value):
        with self._lock:
            self._learning_system = value
            self._learning_system_loaded = True
        # 保留原写法，但不再放初始化代码

    def get_current_parameter_value(self, parameter: TuningParameter) -> float:
        """获取当前参数值"""
        with self._lock:
            return self.parameter_configs[parameter].current_value

    def set_parameter_value(self, parameter: TuningParameter, value: float):
        """设置参数值"""
        with self._lock:
            config = self.parameter_configs[parameter]

            # 确保值在范围内
            value = max(config.range.min_value, min(config.range.max_value, value))

            if config.range.is_integer:
                value = int(round(value))

            # ========== 修复2：混合权重强制约束和为1.0 ==========
            if parameter == TuningParameter.HYBRID_WEIGHT_DENSE:
                sparse_val = round(1.0 - value, 2)
                self.parameter_configs[TuningParameter.HYBRID_WEIGHT_SPARSE].current_value = sparse_val
            elif parameter == TuningParameter.HYBRID_WEIGHT_SPARSE:
                dense_val = round(1.0 - value, 2)
                self.parameter_configs[TuningParameter.HYBRID_WEIGHT_DENSE].current_value = dense_val

            config.current_value = value
            logger.info(f"参数 {parameter.value} 已更新为 {value}")

    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        try:
            # 获取性能分析
            performance_analysis = self.learning_system.analyze_performance_trends()

            if "error" in performance_analysis:
                return []

            bottlenecks = []

            # 分析响应时间
            avg_response_time = performance_analysis.get("avg_response_time", 0)
            if avg_response_time > 3.0:
                bottlenecks.append({
                    "type": "performance",
                    "metric": "response_time",
                    "current_value": avg_response_time,
                    "target_value": 2.0,
                    "severity": "high" if avg_response_time > 5.0 else "medium",
                    "affected_parameters": [
                        TuningParameter.RETRIEVAL_TOP_K,
                        TuningParameter.RERANK_ENABLED,
                        TuningParameter.RESPONSE_TIMEOUT
                    ]
                })

            # 分析相关性
            avg_relevance = performance_analysis.get("avg_relevance_score", 0)
            if avg_relevance < 0.6:
                bottlenecks.append({
                    "type": "quality",
                    "metric": "relevance",
                    "current_value": avg_relevance,
                    "target_value": 0.8,
                    "severity": "high" if avg_relevance < 0.4 else "medium",
                    "affected_parameters": [
                        TuningParameter.RETRIEVAL_THRESHOLD,
                        TuningParameter.HYBRID_WEIGHT_DENSE,
                        TuningParameter.HYBRID_WEIGHT_SPARSE
                    ]
                })

            # 分析成功率
            success_rate = performance_analysis.get("success_rate", 1.0)
            if success_rate < 0.8:
                bottlenecks.append({
                    "type": "reliability",
                    "metric": "success_rate",
                    "current_value": success_rate,
                    "target_value": 0.95,
                    "severity": "critical" if success_rate < 0.6 else "high",
                    "affected_parameters": [
                        TuningParameter.RESPONSE_TIMEOUT,
                        TuningParameter.DECOMPOSITION_THRESHOLD
                    ]
                })

            return bottlenecks

        except Exception as e:
            logger.error(f"分析性能瓶颈失败: {e}")
            return []

    def generate_optimization_strategy(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成优化策略"""
        strategy = {
            "approach": "incremental",  # incremental, aggressive, conservative
            "target_parameters": [],
            "expected_improvements": {},
            "risk_level": "low"
        }

        if not bottlenecks:
            return strategy

        # 根据瓶颈严重程度确定策略
        critical_bottlenecks = [b for b in bottlenecks if b["severity"] == "critical"]
        high_bottlenecks = [b for b in bottlenecks if b["severity"] == "high"]

        if critical_bottlenecks:
            strategy["approach"] = "aggressive"
            strategy["risk_level"] = "medium"
            strategy["target_parameters"] = critical_bottlenecks[0]["affected_parameters"]
        elif high_bottlenecks:
            strategy["approach"] = "incremental"
            strategy["risk_level"] = "low"
            strategy["target_parameters"] = high_bottlenecks[0]["affected_parameters"]
        else:
            strategy["approach"] = "conservative"
            strategy["risk_level"] = "low"
            # 选择所有瓶颈的参数
            all_params = set()
            for bottleneck in bottlenecks:
                all_params.update(bottleneck["affected_parameters"])
            strategy["target_parameters"] = list(all_params)

        # 生成预期改进
        for bottleneck in bottlenecks:
            metric = bottleneck["metric"]
            current = bottleneck["current_value"]
            target = bottleneck["target_value"]

            if current != 0:
                improvement = abs(target - current) / abs(current)
                strategy["expected_improvements"][metric] = min(improvement, 0.5)  # 最大50%改进

        return strategy

    async def run_parameter_optimization(self, target_parameters: List[TuningParameter] = None) -> List[OptimizationResult]:
        """运行参数优化"""
        with self._lock:
            if self.is_optimizing:
                logger.warning("已有优化任务在进行中")
                return []
            self.is_optimizing = True

        try:
            # 分析当前性能
            bottlenecks = self.analyze_performance_bottlenecks()

            # 生成优化策略
            strategy = self.generate_optimization_strategy(bottlenecks)

            # 如果没有指定目标参数，使用策略中的参数
            if target_parameters is None:
                target_parameters = strategy["target_parameters"]

            optimization_results = []

            # 对每个目标参数进行优化
            for parameter in target_parameters[:3]:  # 限制同时优化的参数数量
                result = await self._optimize_single_parameter(parameter, strategy["approach"])
                if result:
                    optimization_results.append(result)

                    # 应用优化（如果改进显著）
                    if result.improvement > 0.05 and result.confidence > 0.6:
                        self.set_parameter_value(parameter, result.new_value)

            # 记录优化历史
            with self._lock:
                self.tuning_history.extend(optimization_results)
                if len(self.tuning_history) > self.max_history_size:
                    self.tuning_history = self.tuning_history[-self.max_history_size:]

            return optimization_results

        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            return []

        finally:
            with self._lock:
                self.is_optimizing = False

    async def _optimize_single_parameter(self, parameter: TuningParameter, approach: str) -> Optional[OptimizationResult]:
        """优化单个参数"""
        try:
            config = self.parameter_configs[parameter]
            current_value = config.current_value

            # 获取当前性能指标
            baseline_metrics = await self._measure_performance()

            # 根据优化方法选择新值
            if approach == "aggressive":
                new_value = self._aggressive_tuning(parameter, current_value)
            elif approach == "incremental":
                new_value = self._incremental_tuning(parameter, current_value)
            else:  # conservative
                new_value = self._conservative_tuning(parameter, current_value)

            # 测试新值
            self.set_parameter_value(parameter, new_value)
            new_metrics = await self._measure_performance()

            # 计算改进
            improvement = self._calculate_improvement(baseline_metrics, new_metrics)
            confidence = self._calculate_confidence(baseline_metrics, new_metrics)

            # 恢复原值（等待决策）
            self.set_parameter_value(parameter, current_value)

            return OptimizationResult(
                parameter=parameter,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                confidence=confidence,
                metrics_before=baseline_metrics,
                metrics_after=new_metrics
            )

        except Exception as e:
            logger.error(f"优化参数 {parameter.value} 失败: {e}")
            return None

    def _aggressive_tuning(self, parameter: TuningParameter, current_value: float) -> float:
        """激进调优 - 大步长调整"""
        config = self.parameter_configs[parameter]
        range_size = config.range.max_value - config.range.min_value

        # ========== 修复3：修正调优方向，不再越调越差 ==========
        if parameter in [TuningParameter.RETRIEVAL_TOP_K, TuningParameter.PARALLEL_PATHS]:
            # 响应慢时减小数值，降低开销
            adjustment = -range_size * 0.3
            new_value = current_value + adjustment
        elif parameter == TuningParameter.RETRIEVAL_THRESHOLD:
            # 降低阈值提高召回
            adjustment = -range_size * 0.2
            new_value = current_value + adjustment
        elif parameter == TuningParameter.RERANK_ENABLED:
            # 切换重排状态
            new_value = 1.0 if current_value < 0.5 else 0.0
        else:
            # 默认小幅度调整
            adjustment = range_size * 0.1
            new_value = current_value + adjustment

        return max(config.range.min_value, min(config.range.max_value, new_value))

    def _incremental_tuning(self, parameter: TuningParameter, current_value: float) -> float:
        """增量调优 - 小步长调整"""
        config = self.parameter_configs[parameter]

        # 使用配置中的步长
        step = config.range.step

        # 随机选择方向（可以基于历史性能数据优化）
        direction = random.choice([-1, 1])
        new_value = current_value + (direction * step)

        return max(config.range.min_value, min(config.range.max_value, new_value))

    def _conservative_tuning(self, parameter: TuningParameter, current_value: float) -> float:
        """保守调优 - 微调"""
        config = self.parameter_configs[parameter]

        # 使用更小的步长
        step = config.range.step * 0.5

        # 基于历史性能选择方向
        direction = self._determine_tuning_direction(parameter)
        new_value = current_value + (direction * step)

        return max(config.range.min_value, min(config.range.max_value, new_value))

    def _determine_tuning_direction(self, parameter: TuningParameter) -> float:
        """基于历史数据确定调优方向"""
        # 简化的方向决策
        # 在实际应用中，这里应该基于详细的性能分析

        recent_history = self.tuning_history[-10:]  # 最近10次调优
        parameter_history = [h for h in recent_history if h.parameter == parameter]

        if len(parameter_history) < 2:
            return random.choice([-1, 1])

        # 分析最近的变化趋势
        recent_changes = [h.new_value - h.old_value for h in parameter_history[-3:]]
        recent_improvements = [h.improvement for h in parameter_history[-3:]]

        # 如果最近的变化带来了改进，继续同方向
        if len(recent_changes) > 0 and len(recent_improvements) > 0:
            avg_change = sum(recent_changes) / len(recent_changes)
            avg_improvement = sum(recent_improvements) / len(recent_improvements)

            if avg_improvement > 0 and avg_change != 0:
                return 1 if avg_change > 0 else -1

        return random.choice([-1, 1])

    async def _measure_performance(self) -> Dict[str, float]:
        """测量当前性能"""
        # 在实际应用中，这里应该运行一组标准测试查询
        # 这里使用学习系统中的实际性能数据作为简化
        # ========== 修复4：补充空异步等待，消除async警告 ==========
        await asyncio.sleep(0)

        try:
            performance_analysis = self.learning_system.analyze_performance_trends()

            return {
                "response_time": performance_analysis.get("avg_response_time", 2.0),
                "relevance": performance_analysis.get("avg_relevance_score", 0.7),
                "success_rate": performance_analysis.get("success_rate", 0.9),
                "satisfaction": performance_analysis.get("avg_satisfaction", 0.8)
            }

        except Exception as e:
            logger.warning(f"性能测量失败，使用默认值: {e}")
            return {
                "response_time": 2.0,
                "relevance": 0.7,
                "success_rate": 0.9,
                "satisfaction": 0.8
            }

    def _calculate_improvement(self, metrics_before: Dict[str, float], metrics_after: Dict[str, float]) -> float:
        """计算综合改进程度"""
        total_improvement = 0.0
        total_weight = 0.0

        for objective, weight in self.objective_weights.items():
            if objective == OptimizationObjective.RESPONSE_TIME:
                # 响应时间越小越好
                base = max(metrics_before["response_time"], 0.01)
                improvement = (metrics_before["response_time"] - metrics_after["response_time"]) / base
                total_improvement += improvement * weight

            elif objective == OptimizationObjective.RELEVANCE:
                # 相关性越大越好
                base = max(metrics_before["relevance"], 0.01)
                improvement = (metrics_after["relevance"] - metrics_before["relevance"]) / base
                total_improvement += improvement * weight

            elif objective == OptimizationObjective.SUCCESS_RATE:
                # 成功率越大越好
                base = max(metrics_before["success_rate"], 0.01)
                improvement = (metrics_after["success_rate"] - metrics_before["success_rate"]) / base
                total_improvement += improvement * weight

            elif objective == OptimizationObjective.USER_SATISFACTION:
                # 满意度越大越好
                base = max(metrics_before["satisfaction"], 0.01)
                improvement = (metrics_after["satisfaction"] - metrics_before["satisfaction"]) / base
                total_improvement += improvement * weight

            total_weight += weight

        return total_improvement / max(total_weight, 0.01)

    def _calculate_confidence(self, metrics_before: Dict[str, float], metrics_after: Dict[str, float]) -> float:
        """计算优化置信度"""
        # 基于指标变化的稳定性计算置信度
        confidence = 0.5  # 基础置信度

        # 检查是否有明显的改进趋势
        improvements = []

        base = max(metrics_before["response_time"], 0.01)
        time_improvement = (metrics_before["response_time"] - metrics_after["response_time"]) / base
        improvements.append(time_improvement)

        base = max(metrics_before["relevance"], 0.01)
        relevance_improvement = (metrics_after["relevance"] - metrics_before["relevance"]) / base
        improvements.append(relevance_improvement)

        base = max(metrics_before["success_rate"], 0.01)
        success_improvement = (metrics_after["success_rate"] - metrics_before["success_rate"]) / base
        improvements.append(success_improvement)

        # 基于改进的一致性调整置信度
        if improvements:
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            consistency = positive_improvements / len(improvements)
            confidence = 0.3 + (consistency * 0.6)  # 0.3-0.9范围

            # 如果改进幅度大，增加置信度
            avg_improvement = sum(max(0, imp) for imp in improvements) / len(improvements)
            if avg_improvement > 0.1:
                confidence = min(1.0, confidence + 0.1)

        return confidence

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []

        try:
            # 分析性能瓶颈
            bottlenecks = self.analyze_performance_bottlenecks()

            for bottleneck in bottlenecks:
                affected_params = bottleneck["affected_parameters"]

                for param in affected_params:
                    config = self.parameter_configs[param]
                    current_value = config.current_value

                    # 根据瓶颈类型推荐调整
                    if bottleneck["type"] == "performance":
                        if param == TuningParameter.RETRIEVAL_TOP_K:
                            new_value = max(1, current_value - 2)
                            reason = "减少检索数量以提高响应速度"
                        elif param == TuningParameter.RERANK_ENABLED:
                            new_value = 0.0  # 禁用重排
                            reason = "禁用重排以减少处理时间"
                        else:
                            continue

                    elif bottleneck["type"] == "quality":
                        if param == TuningParameter.RETRIEVAL_THRESHOLD:
                            new_value = max(0.1, current_value - 0.1)
                            reason = "降低阈值以提高召回率"
                        elif param == TuningParameter.HYBRID_WEIGHT_DENSE:
                            new_value = min(0.9, current_value + 0.1)
                            reason = "增加稠密向量权重以提高语义相关性"
                        else:
                            continue

                    else:
                        continue

                    recommendations.append({
                        "parameter": param.value,
                        "current_value": current_value,
                        "recommended_value": new_value,
                        "reason": reason,
                        "expected_improvement": bottleneck.get("severity", "medium"),
                        "confidence": 0.7
                    })

        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")

        return recommendations

    def _setup_optimization_methods(self):
        """设置实际调优方法的映射"""
        self._optimization_methods = {
            "retrieval.top_k": self._adjust_retrieval_top_k,
            "retrieval.threshold": self._adjust_retrieval_threshold,
            "rerank.enabled": self._adjust_rerank_enabled,
            "strategy.prefer_hybrid": self._adjust_prefer_hybrid,
        }

    def _adjust_retrieval_top_k(self, adjustment: int):
        """调整检索数量"""
        from rag_utils import LEAF_RETRIEVE_LEVEL
        # 这里应该调用实际的检索组件
        logger.info(f"调整检索数量: top_k += {adjustment}")

    def _adjust_retrieval_threshold(self, adjustment: float):
        """调整检索阈值"""
        from rag_utils import AUTO_MERGE_THRESHOLD
        # 这里应该调用实际的检索组件
        logger.info(f"调整检索阈值: threshold += {adjustment}")

    def _adjust_rerank_enabled(self, enabled: bool):
        """调整重排是否启用"""
        # 这里应该调用实际的重排组件
        logger.info(f"调整重排: enabled = {enabled}")

    def _adjust_prefer_hybrid(self, prefer: bool):
        """调整是否优先使用混合检索"""
        # 这里应该调用实际的路由组件
        logger.info(f"调整检索策略: prefer_hybrid = {prefer}")

    def apply_emergency_optimization(self, issue_type: str) -> Dict[str, Any]:
        """应用紧急优化"""
        emergency_actions = {
            "timeout": {
                TuningParameter.RETRIEVAL_TOP_K: 3,
                TuningParameter.RERANK_ENABLED: 0.0,
                TuningParameter.RESPONSE_TIMEOUT: 15.0
            },
            "low_relevance": {
                TuningParameter.RETRIEVAL_THRESHOLD: 0.2,
                TuningParameter.HYBRID_WEIGHT_DENSE: 0.8
            },
            "high_error_rate": {
                TuningParameter.DECOMPOSITION_THRESHOLD: 1,
                TuningParameter.PARALLEL_PATHS: 1
            }
        }

        if issue_type not in emergency_actions:
            return {"status": "unknown_issue", "message": f"未知的紧急问题类型: {issue_type}"}

        applied_changes = []

        try:
            for parameter, value in emergency_actions[issue_type].items():
                old_value = self.get_current_parameter_value(parameter)
                self.set_parameter_value(parameter, value)

                applied_changes.append({
                    "parameter": parameter.value,
                    "old_value": old_value,
                    "new_value": value
                })

            return {
                "status": "success",
                "applied_changes": applied_changes,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"应用紧急优化失败: {e}")
            return {"status": "error", "message": str(e)}

    async def optimize_if_needed(self):
        """检查性能瓶颈，必要时自动调优"""
        try:
            bottlenecks = self.analyze_performance_bottlenecks()
            if bottlenecks:
                self.generate_optimization_strategy(bottlenecks)
        except Exception as e:
            logger.warning(f"自适应调优跳过: {e}")


_adaptive_tuner = None

def get_adaptive_tuner() -> AdaptiveTuner:
    global _adaptive_tuner
    if _adaptive_tuner is None:
        _adaptive_tuner = AdaptiveTuner()
    return _adaptive_tuner