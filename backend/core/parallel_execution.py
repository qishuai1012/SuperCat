"""
并行执行框架
支持多路径任务并行执行和结果融合
"""

from typing import Dict, Any, List, Optional, Callable, Coroutine
from dataclasses import dataclass
import asyncio
import logging
import time
from enum import Enum
import threading
import os

from core.multi_agent_orchestrator import AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式枚举"""
    SEQUENTIAL = "sequential"     # 串行执行
    PARALLEL = "parallel"         # 完全并行
    PIPELINE = "pipeline"         # 流水线执行
    ADAPTIVE = "adaptive"         # 自适应模式


class FusionStrategy(Enum):
    """结果融合策略枚举"""
    CONCATENATE = "concatenate"   # 简单拼接
    RANK_FUSION = "rank_fusion"   # 排名融合
    WEIGHTED_VOTE = "weighted_vote" # 加权投票
    INTELLIGENT = "intelligent"   # 智能融合


@dataclass
class ExecutionPlan:
    """执行计划"""
    mode: ExecutionMode
    tasks: List[AgentTask]
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_task_ids]
    timeout: float = 30.0
    max_concurrency: int = 5
    fusion_strategy: FusionStrategy = FusionStrategy.INTELLIGENT


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    task_results: List[AgentResult]
    fused_result: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParallelExecutor:
    """
    并行执行器 - 管理复杂任务的并行执行和结果融合
    """

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self.execution_history = []

    def create_execution_plan(self,
                            tasks: List[AgentTask],
                            execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
                            fusion_strategy: FusionStrategy = FusionStrategy.INTELLIGENT) -> ExecutionPlan:
        """
        创建执行计划

        Args:
            tasks: 要执行的任务列表
            execution_mode: 执行模式
            fusion_strategy: 结果融合策略

        Returns:
            执行计划
        """
        # 分析任务依赖关系
        dependencies = self._analyze_dependencies(tasks)

        # 确定最优执行模式
        if execution_mode == ExecutionMode.ADAPTIVE:
            execution_mode = self._determine_optimal_mode(tasks, dependencies)

        # 优化任务执行顺序
        optimized_tasks = self._optimize_task_order(tasks, dependencies, execution_mode)

        return ExecutionPlan(
            mode=execution_mode,
            tasks=optimized_tasks,
            dependencies=dependencies,
            max_concurrency=self.max_concurrency,
            fusion_strategy=fusion_strategy
        )

    def _analyze_dependencies(self, tasks: List[AgentTask]) -> Dict[str, List[str]]:
        """分析任务依赖关系"""
        dependencies = {}

        for task in tasks:
            # 显式依赖
            explicit_deps = task.dependencies or []

            # 隐式依赖（基于任务类型）
            implicit_deps = self._find_implicit_dependencies(task, tasks)

            # 合并依赖
            all_deps = list(set(explicit_deps + implicit_deps))
            dependencies[task.id] = all_deps

        return dependencies

    def _find_implicit_dependencies(self, task: AgentTask, all_tasks: List[AgentTask]) -> List[str]:
        """查找隐式依赖"""
        implicit_deps = []

        # 基于任务类型的依赖规则
        dependency_rules = {
            AgentType.ANALYSIS: [AgentType.RETRIEVAL],  # 分析依赖检索
            AgentType.SYNTHESIS: [AgentType.ANALYSIS, AgentType.RETRIEVAL],  # 综合依赖分析和检索
            AgentType.VERIFICATION: [AgentType.SYNTHESIS, AgentType.ANALYSIS],  # 验证依赖综合和分析
        }

        required_types = dependency_rules.get(task.agent_type, [])
        task_map = {t.agent_type: t.id for t in all_tasks}
        for rt in required_types:
            if rt in task_map:
                implicit_deps.append(task_map[rt])

        return implicit_deps

    def _determine_optimal_mode(self, tasks: List[AgentTask], dependencies: Dict[str, List[str]]) -> ExecutionMode:
        """确定最优执行模式"""

        # 如果没有依赖，使用并行模式
        if all(len(deps) == 0 for deps in dependencies.values()):
            return ExecutionMode.PARALLEL

        # 如果依赖链很长，使用流水线模式
        max_dependency_depth = self._calculate_max_dependency_depth(dependencies)
        if max_dependency_depth > 3:
            return ExecutionMode.PIPELINE

        # 如果任务数量少，使用串行模式
        if len(tasks) <= 2:
            return ExecutionMode.SEQUENTIAL

        # 默认使用流水线模式
        return ExecutionMode.PIPELINE

    def _calculate_max_dependency_depth(self, dependencies: Dict[str, List[str]]) -> int:
        """计算最大依赖深度"""
        def get_depth(task_id: str, visited: set, max_recursion: int = 20) -> int:
            if task_id in visited or max_recursion <= 0:
                return 0  # 避免循环依赖

            visited.add(task_id)
            deps = dependencies.get(task_id, [])

            if not deps:
                return 1

            max_dep_depth = 0
            for dep_id in deps:
                dep_depth = get_depth(dep_id, visited.copy(), max_recursion - 1)
                max_dep_depth = max(max_dep_depth, dep_depth)

            return max_dep_depth + 1

        max_depth = 0
        for task_id in dependencies:
            depth = get_depth(task_id, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _optimize_task_order(self, tasks: List[AgentTask], dependencies: Dict[str, List[str]], mode: ExecutionMode) -> List[AgentTask]:
        """优化任务执行顺序"""
        if mode == ExecutionMode.PARALLEL:
            # 并行模式：按优先级排序
            return sorted(tasks, key=lambda t: t.priority, reverse=True)

        elif mode == ExecutionMode.SEQUENTIAL:
            # 串行模式：拓扑排序
            return self._topological_sort(tasks, dependencies)

        elif mode == ExecutionMode.PIPELINE:
            # 流水线模式：分层排序
            return self._pipeline_sort(tasks, dependencies)

        return tasks

    def _topological_sort(self, tasks: List[AgentTask], dependencies: Dict[str, List[str]]) -> List[AgentTask]:
        """拓扑排序"""
        task_map = {task.id: task for task in tasks}
        in_degree = {task.id: 0 for task in tasks}

        # 计算入度
        for task_id, deps in dependencies.items():
            for dep_id in deps:
                if dep_id in in_degree:
                    in_degree[task_id] += 1

        # 拓扑排序
        result = []
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]

        while queue:
            current_id = queue.pop(0)
            if current_id in task_map:
                result.append(task_map[current_id])

            # 更新依赖当前任务的其他任务
            for task_id, deps in dependencies.items():
                if current_id in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        return result

    def _pipeline_sort(self, tasks: List[AgentTask], dependencies: Dict[str, List[str]]) -> List[AgentTask]:
        """流水线排序 - 按依赖层级分组"""
        # 计算每个任务的层级
        levels = {}

        def calculate_level(task_id: str, visited: set) -> int:
            if task_id in visited:
                return 0

            visited.add(task_id)
            deps = dependencies.get(task_id, [])

            if not deps:
                levels[task_id] = 0
                return 0

            max_dep_level = 0
            for dep_id in deps:
                dep_level = calculate_level(dep_id, visited.copy())
                max_dep_level = max(max_dep_level, dep_level)

            levels[task_id] = max_dep_level + 1
            return levels[task_id]

        # 计算所有任务的层级
        for task in tasks:
            if task.id not in levels:
                calculate_level(task.id, set())

        # 按层级分组并排序
        level_groups = {}
        for task in tasks:
            level = levels.get(task.id, 0)
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(task)

        # 按层级排序，每层内按优先级排序
        result = []
        for level in sorted(level_groups.keys()):
            level_tasks = sorted(level_groups[level], key=lambda t: t.priority, reverse=True)
            result.extend(level_tasks)

        return result

    async def execute_plan(self, plan: ExecutionPlan, executor_func: Callable[[AgentTask], Coroutine]) -> ExecutionResult:
        """
        执行计划

        Args:
            plan: 执行计划
            executor_func: 任务执行函数

        Returns:
            执行结果
        """
        start_time = time.time()

        try:
            if plan.mode == ExecutionMode.PARALLEL:
                results = await self._execute_parallel(plan, executor_func)
            elif plan.mode == ExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential(plan, executor_func)
            elif plan.mode == ExecutionMode.PIPELINE:
                results = await self._execute_pipeline(plan, executor_func)
            else:
                results = await self._execute_parallel(plan, executor_func)

            # 融合结果
            fused_result = await self._fuse_results(results, plan.fusion_strategy)

            execution_time = time.time() - start_time

            result = ExecutionResult(
                success=True,
                task_results=results,
                fused_result=fused_result,
                execution_time=execution_time,
                metadata={
                    "execution_mode": plan.mode.value,
                    "total_tasks": len(plan.tasks),
                    "successful_tasks": sum(1 for r in results if r.success),
                    "failed_tasks": sum(1 for r in results if not r.success)
                }
            )

            # 记录执行历史
            self.execution_history.append(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"执行计划失败: {e}")

            return ExecutionResult(
                success=False,
                task_results=[],
                fused_result={"error": str(e)},
                execution_time=execution_time,
                metadata={"error": str(e)}
            )

    async def _execute_parallel(self, plan: ExecutionPlan, executor_func: Callable) -> List[AgentResult]:
        """并行执行"""
        # 检查是否有依赖，如果有则使用受限并行
        has_dependencies = any(len(deps) > 0 for deps in plan.dependencies.values())

        if has_dependencies:
            return await self._execute_parallel_with_dependencies(plan, executor_func)
        else:
            # 完全并行执行
            semaphore = asyncio.Semaphore(plan.max_concurrency)

            async def bounded_execute(task):
                async with semaphore:
                    try:
                        return await asyncio.wait_for(executor_func(task), timeout=task.timeout)
                    except asyncio.TimeoutError:
                        return AgentResult(
                            task_id=task.id,
                            agent_type=task.agent_type,
                            success=False,
                            output={"error": "任务执行超时"},
                            execution_time=task.timeout
                        )
                    except Exception as e:
                        return AgentResult(
                            task_id=task.id,
                            agent_type=task.agent_type,
                            success=False,
                            output={"error": str(e)},
                            execution_time=0.0
                        )

            tasks = [bounded_execute(task) for task in plan.tasks]
            return await asyncio.gather(*tasks)

    async def _execute_parallel_with_dependencies(self, plan: ExecutionPlan, executor_func: Callable) -> List[AgentResult]:
        """带依赖的并行执行"""
        results = {}
        completed = set()
        pending = set(task.id for task in plan.tasks)

        while pending:
            # 找到可以执行的任务（依赖已满足）
            ready_tasks = []
            for task in plan.tasks:
                if (task.id in pending and
                    all(dep_id in completed for dep_id in plan.dependencies.get(task.id, []))):
                    ready_tasks.append(task)

            if not ready_tasks:
                # 存在循环依赖，执行剩余任务
                if pending:
                    ready_tasks = [task for task in plan.tasks if task.id in pending]

            # 并行执行准备好的任务
            if ready_tasks:
                semaphore = asyncio.Semaphore(min(len(ready_tasks), plan.max_concurrency))

                async def bounded_execute(task):
                    async with semaphore:
                        try:
                            return await asyncio.wait_for(executor_func(task), timeout=task.timeout)
                        except asyncio.TimeoutError:
                            return AgentResult(
                                task_id=task.id,
                                agent_type=task.agent_type,
                                success=False,
                                output={"error": "任务执行超时"},
                                execution_time=task.timeout
                            )
                        except Exception as e:
                            return AgentResult(
                                task_id=task.id,
                                agent_type=task.agent_type,
                                success=False,
                                output={"error": str(e)},
                                execution_time=0.0
                            )

                batch_tasks = [bounded_execute(task) for task in ready_tasks]
                batch_results = await asyncio.gather(*batch_tasks)

                # 处理结果
                for i, result in enumerate(batch_results):
                    task_id = ready_tasks[i].id
                    results[task_id] = result
                    completed.add(task_id)
                    pending.discard(task_id)

        return [results.get(task.id, AgentResult(
            task_id=task.id,
            agent_type=task.agent_type,
            success=False,
            output={"error": "任务未执行"},
            execution_time=0.0
        )) for task in plan.tasks]

    async def _execute_sequential(self, plan: ExecutionPlan, executor_func: Callable) -> List[AgentResult]:
        """串行执行"""
        results = []

        for task in plan.tasks:
            try:
                result = await asyncio.wait_for(executor_func(task), timeout=task.timeout)
                results.append(result)
            except asyncio.TimeoutError:
                results.append(AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=False,
                    output={"error": "执行超时"},
                    execution_time=task.timeout
                ))
            except Exception as e:
                results.append(AgentResult(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    success=False,
                    output={"error": str(e)},
                    execution_time=0.0
                ))

        return results

    async def _execute_pipeline(self, plan: ExecutionPlan, executor_func: Callable) -> List[AgentResult]:
        """流水线执行"""
        # 按层级分组执行
        levels = {}
        for task in plan.tasks:
            levels[task.id] = 0

        def calc_level(task_id: str, visited: set):
            if task_id in visited:
                return 0
            visited.add(task_id)
            deps = plan.dependencies.get(task_id, [])
            if not deps:
                return 0
            max_lvl = 0
            for d in deps:
                max_lvl = max(max_lvl, calc_level(d, visited.copy()) + 1)
            levels[task_id] = max_lvl
            return max_lvl

        for t in plan.tasks:
            calc_level(t.id, set())

        # 按层级执行
        results = {}
        max_level = max(levels.values()) if levels else 0

        for current_level in range(max_level + 1):
            level_tasks = [task for task in plan.tasks if levels[task.id] == current_level]

            if level_tasks:
                # 当前层级并行执行
                semaphore = asyncio.Semaphore(plan.max_concurrency)

                async def bounded_execute(task):
                    async with semaphore:
                        try:
                            return await asyncio.wait_for(executor_func(task), timeout=task.timeout)
                        except Exception:
                            return AgentResult(
                                task_id=task.id,
                                agent_type=task.agent_type,
                                success=False,
                                output={"error": "执行失败"},
                                execution_time=0.0
                            )

                batch_tasks = [bounded_execute(task) for task in level_tasks]
                batch_results = await asyncio.gather(*batch_tasks)

                for i, result in enumerate(batch_results):
                    task_id = level_tasks[i].id
                    results[task_id] = result

        return [results.get(task.id, AgentResult(
            task_id=task.id,
            agent_type=task.agent_type,
            success=False,
            output={"error": "未执行"},
            execution_time=0.0
        )) for task in plan.tasks]

    async def _fuse_results(self, results: List[AgentResult], strategy: FusionStrategy) -> Dict[str, Any]:
        """融合结果"""
        if not results:
            return {"error": "没有可融合的结果"}

        if len(results) == 1:
            return results[0].output

        try:
            if strategy == FusionStrategy.CONCATENATE:
                return self._concatenate_results(results)
            elif strategy == FusionStrategy.RANK_FUSION:
                return await self._rank_fusion_results(results)
            elif strategy == FusionStrategy.WEIGHTED_VOTE:
                return await self._weighted_vote_fusion(results)
            elif strategy == FusionStrategy.INTELLIGENT:
                return await self._intelligent_fusion(results)
            else:
                return self._concatenate_results(results)

        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            return self._concatenate_results(results)

    def _concatenate_results(self, results: List[AgentResult]) -> Dict[str, Any]:
        """简单拼接结果"""
        responses = []
        for result in results:
            if result.success and "response" in result.output:
                responses.append(result.output["response"])

        return {
            "fused_response": "\n\n".join(responses),
            "source_count": len(responses),
            "fusion_strategy": "concatenate"
        }

    async def _rank_fusion_results(self, results: List[AgentResult]) -> Dict[str, Any]:
        """排名融合结果"""
        # 基于执行时间和成功率进行排名
        ranked_results = []

        for result in results:
            if result.success:
                # 简单的排名分数计算
                score = 1.0 / max(result.execution_time, 0.1)  # 时间越短分数越高
                if result.agent_type == AgentType.SYNTHESIS:
                    score *= 1.5  # 综合Agent的结果权重更高

                ranked_results.append((score, result))

        # 按分数排序
        ranked_results.sort(key=lambda x: x[0], reverse=True)

        # 取前几个结果融合
        top_results = ranked_results[:3]
        responses = [result.output.get("response", "") for _, result in top_results]

        return {
            "fused_response": "\n\n".join(responses),
            "ranked_scores": [score for score, _ in top_results],
            "source_count": len(top_results),
            "fusion_strategy": "rank_fusion"
        }

    async def _weighted_vote_fusion(self, results: List[AgentResult]) -> Dict[str, Any]:
        """加权投票融合"""
        # 基于Agent类型和成功率计算权重
        weights = {
            AgentType.SYNTHESIS: 3.0,
            AgentType.VERIFICATION: 2.5,
            AgentType.ANALYSIS: 2.0,
            AgentType.RETRIEVAL: 1.5,
            AgentType.PLANNING: 1.0
        }

        weighted_responses = []
        total_weight = 0

        for result in results:
            if result.success and "response" in result.output:
                weight = weights.get(result.agent_type, 1.0)
                if not result.success:
                    weight *= 0.5  # 失败结果权重降低

                weighted_responses.append({
                    "response": result.output["response"],
                    "weight": weight
                })
                total_weight += weight

        # 按权重排序
        weighted_responses.sort(key=lambda x: x["weight"], reverse=True)

        # 生成融合结果
        if weighted_responses:
            # 使用最高权重的结果作为基础
            main_response = weighted_responses[0]["response"]

            # 如果有其他高权重结果，进行补充
            if len(weighted_responses) > 1:
                supplementary = "\n\n补充信息:\n" + "\n".join(
                    f"- {item['response']}" for item in weighted_responses[1:3]
                )
                main_response += supplementary

            return {
                "fused_response": main_response,
                "weights": [item["weight"] for item in weighted_responses],
                "total_weight": total_weight,
                "source_count": len(weighted_responses),
                "fusion_strategy": "weighted_vote"
            }
        else:
            return {"error": "没有成功的结果可供融合"}

    async def _intelligent_fusion(self, results: List[AgentResult]) -> Dict[str, Any]:
        """智能融合 - 使用LLM进行结果融合"""
        try:
            from langchain.chat_models import init_chat_model

            # 初始化融合模型
            API_KEY = os.getenv("ARK_API_KEY")
            MODEL = os.getenv("MODEL")
            BASE_URL = os.getenv("BASE_URL")

            if not all([API_KEY, MODEL, BASE_URL]):
                # 降级到加权投票
                return await self._weighted_vote_fusion(results)

            fusion_model = init_chat_model(
                model=MODEL,
                model_provider="openai",
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0.1,
            )

            # 准备融合输入
            fusion_input = "请综合以下多个结果，生成一个完整、准确、连贯的答案：\n\n"

            for i, result in enumerate(results, 1):
                if result.success and "response" in result.output:
                    agent_type_str = result.agent_type.value
                    fusion_input += f"结果{i} (来自{agent_type_str}):\n"
                    fusion_input += f"{result.output['response']}\n\n"

            fusion_input += "请整合以上所有信息，消除重复和冲突，生成最终的完整答案："

            # 调用融合模型
            response = fusion_model.invoke(fusion_input)

            return {
                "fused_response": response.content,
                "source_count": len([r for r in results if r.success]),
                "fusion_strategy": "intelligent",
                "original_results": [
                    {
                        "agent_type": result.agent_type.value,
                        "success": result.success,
                        "execution_time": result.execution_time
                    }
                    for result in results
                ]
            }

        except Exception as e:
            logger.warning(f"智能融合失败，降级到加权投票: {e}")
            return await self._weighted_vote_fusion(results)


# 全局并行执行器
_parallel_executor = None
_executor_lock = threading.Lock()


def get_parallel_executor(max_concurrency: int = 5) -> ParallelExecutor:
    """获取全局并行执行器"""
    global _parallel_executor
    with _executor_lock:
        if _parallel_executor is None:
            _parallel_executor = ParallelExecutor(max_concurrency)
    return _parallel_executor


def create_and_execute_plan(tasks: List[AgentTask],
                          execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
                          fusion_strategy: FusionStrategy = FusionStrategy.INTELLIGENT,
                          executor_func: Callable = None) -> ExecutionResult:
    """便捷函数：创建并执行计划"""
    executor = get_parallel_executor()
    plan = executor.create_execution_plan(tasks, execution_mode, fusion_strategy)

    try:
        result = asyncio.run(executor.execute_plan(plan, executor_func))
        return result
    except Exception as e:
        return ExecutionResult(
            success=False,
            task_results=[],
            fused_result={"error": str(e)},
            execution_time=0.0
        )