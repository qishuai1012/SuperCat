"""
任务分解系统
负责将复杂查询分解为可管理的子任务
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
import json
import logging
import asyncio
from enum import Enum
import threading
import re

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class TaskType(Enum):
    """任务类型枚举"""
    RETRIEVAL = "retrieval"      # 检索任务
    ANALYSIS = "analysis"        # 分析任务
    SYNTHESIS = "synthesis"      # 综合任务
    VERIFICATION = "verification" # 验证任务
    COMPARISON = "comparison"    # 比较任务


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class SubTask:
    """子任务定义"""
    id: str
    type: TaskType
    description: str
    query: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = None
    estimated_complexity: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

    # 👇 👇 👇 加上这两段就修复了
    def __hash__(self):
        """让 SubTask 可以放入 set"""
        return hash(self.id)

    def __eq__(self, other):
        """判断两个任务是否相等"""
        if isinstance(other, SubTask):
            return self.id == other.id
        return False


class TaskDecomposer:
    """
    任务分解器 - 使用CoT方法将复杂查询分解为子任务
    """

    def __init__(self):
        self.decomposer_model = None
        self._init_decomposer_model()

        # 任务分解提示词
        self.decomposition_prompt = """
        你是一个专业的任务分解专家，负责将复杂查询分解为可执行的子任务。

        请分析以下复杂查询，并将其分解为逻辑清晰的子任务：

        原始查询: {original_query}
        查询背景: {context}

        分解要求：
        1. 每个子任务应该是独立的、可执行的
        2. 子任务之间可以有依赖关系
        3. 标明任务的优先级和执行顺序
        4. 使用合适的任务类型：
           - retrieval: 信息检索
           - analysis: 数据分析
           - synthesis: 信息综合
           - verification: 事实验证
           - comparison: 比较分析

        请返回JSON格式的分解结果：
        {{
            "can_decompose": true/false,
            "reasoning": "分解理由",
            "subtasks": [
                {{
                    "id": "唯一标识",
                    "type": "任务类型",
                    "description": "任务描述",
                    "query": "具体查询内容",
                    "priority": "HIGH/MEDIUM/LOW",
                    "dependencies": ["依赖的任务ID"],
                    "estimated_complexity": 复杂度评分(1-5)
                }}
            ],
            "execution_strategy": "sequential/parallel/mixed"
        }}
        """

    def _init_decomposer_model(self):
        """初始化分解模型"""
        try:
            if API_KEY and MODEL:
                self.decomposer_model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.2,
                    stream_usage=True,
                )
        except Exception as e:
            logger.warning(f"分解模型初始化失败: {e}")
            self.decomposer_model = None

    def _analyze_query_for_decomposition(self, query: str) -> Dict[str, Any]:
        """分析查询是否需要分解"""
        analysis = {
            'length': len(query),
            'question_count': query.count('？') + query.count('?'),
            'has_connectors': any(connector in query for connector in ['并且', '还有', '另外', '同时', '以及']),
            'has_multiple_topics': self._detect_multiple_topics(query),
            'is_procedural': self._is_procedural_query(query),
            'complexity_score': self._calculate_complexity_score(query)
        }

        # 判断是否需要分解
        needs_decomposition = (
            analysis['question_count'] > 1 or
            analysis['has_connectors'] or
            analysis['has_multiple_topics'] or
            analysis['complexity_score'] > 3.0
        )

        analysis['needs_decomposition'] = needs_decomposition
        return analysis

    def _detect_multiple_topics(self, query: str) -> bool:
        """检测多个主题"""
        topic_indicators = ['比较', '对比', '区别', '差异', '优点', '缺点', '特点']
        return any(indicator in query for indicator in topic_indicators)

    def _is_procedural_query(self, query: str) -> bool:
        """判断是否为过程性查询"""
        procedural_indicators = ['步骤', '流程', '方法', '怎么', '如何', '怎样', '过程', '顺序']
        return any(indicator in query for indicator in procedural_indicators)

    def _calculate_complexity_score(self, query: str) -> float:
        """计算查询复杂度分数"""
        score = 1.0

        # 基于长度的复杂度
        if len(query) > 100:
            score += 1.0
        elif len(query) > 50:
            score += 0.5

        # 基于问题数量的复杂度
        question_count = query.count('？') + query.count('?')
        score += question_count * 0.5

        # 基于连接词的复杂度
        connectors = ['并且', '还有', '另外', '同时', '以及', '或者', '但是']
        connector_count = sum(1 for connector in connectors if connector in query)
        score += connector_count * 0.3

        # 基于专业术语的复杂度
        technical_terms = ['原理', '机制', '算法', '模型', '架构', '框架']
        technical_count = sum(1 for term in technical_terms if term in query)
        score += technical_count * 0.2

        return min(score, 5.0)  # 上限为5

    def _rule_based_decomposition(self, query: str, context: Dict[str, Any] = None) -> List[SubTask]:
        """基于规则的分解fallback"""
        analysis = self._analyze_query_for_decomposition(query)

        if not analysis['needs_decomposition']:
            return [SubTask(
                id="single_task",
                type=TaskType.RETRIEVAL,
                description="单一检索任务",
                query=query,
                priority=TaskPriority.HIGH
            )]

        subtasks = []
        task_id = 1

        # 基于问题数量分解
        if analysis['question_count'] > 1:
            # 简单按问号分割
            questions = [q.strip() for q in query.replace('？', '?').split('?') if q.strip()]
            for i, q in enumerate(questions):
                if q:
                    subtasks.append(SubTask(
                        id=f"task_{task_id}",
                        type=TaskType.RETRIEVAL,
                        description=f"子问题{i+1}",
                        query=q + "?",
                        priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM
                    ))
                    task_id += 1

        # 基于连接词分解
        elif analysis['has_connectors']:
            connectors = ['并且', '还有', '另外', '同时', '以及']
            for connector in connectors:
                if connector in query:
                    parts = query.split(connector)
                    for i, part in enumerate(parts):
                        if part.strip():
                            subtasks.append(SubTask(
                                id=f"task_{task_id}",
                                type=TaskType.RETRIEVAL,
                                description=f"部分{i+1}",
                                query=part.strip(),
                                priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM
                            ))
                            task_id += 1
                    break

        # 如果是过程性查询，分解为步骤
        elif analysis['is_procedural']:
            subtasks = [
                SubTask(
                    id="understand_requirement",
                    type=TaskType.ANALYSIS,
                    description="理解需求",
                    query=f"理解查询需求: {query}",
                    priority=TaskPriority.HIGH
                ),
                SubTask(
                    id="find_procedure",
                    type=TaskType.RETRIEVAL,
                    description="查找相关流程",
                    query=f"查找{query}的具体步骤和流程",
                    priority=TaskPriority.HIGH,
                    dependencies=["understand_requirement"]
                ),
                SubTask(
                    id="organize_steps",
                    type=TaskType.SYNTHESIS,
                    description="整理步骤",
                    query="将找到的流程步骤按逻辑顺序整理",
                    priority=TaskPriority.MEDIUM,
                    dependencies=["find_procedure"]
                )
            ]

        # 默认返回原始查询
        if not subtasks:
            subtasks = [SubTask(
                id="single_task",
                type=TaskType.RETRIEVAL,
                description="单一检索任务",
                query=query,
                priority=TaskPriority.HIGH
            )]

        return subtasks

    def decompose(self, query: str, context: Dict[str, Any] = None) -> List[SubTask]:
        """
        分解复杂查询为子任务

        Args:
            query: 原始查询
            context: 上下文信息

        Returns:
            子任务列表
        """
        if context is None:
            context = {}

        # 如果分解模型不可用，使用规则基础分解
        if not self.decomposer_model:
            return self._rule_based_decomposition(query, context)

        try:
            # 准备分解输入
            decomposition_input = self.decomposition_prompt.format(
                original_query=query,
                context=json.dumps(context, ensure_ascii=False, indent=2)
            )

            # 调用分解模型
            response = self.decomposer_model.invoke(decomposition_input)

            # 解析分解结果
            decomposition_result = self._parse_decomposition_response(response.content)

            # 如果模型认为不需要分解，返回原始查询
            if not decomposition_result.get('can_decompose', False):
                return [SubTask(
                    id="single_task",
                    type=TaskType.RETRIEVAL,
                    description="单一检索任务",
                    query=query,
                    priority=TaskPriority.HIGH
                )]

            # 转换为SubTask对象
            return self._create_subtasks(decomposition_result['subtasks'])

        except Exception as e:
            logger.warning(f"智能任务分解失败，使用规则基础分解: {e}")
            return self._rule_based_decomposition(query, context)

    def _parse_decomposition_response(self, response_content: str) -> Dict[str, Any]:
        """解析分解模型响应"""
        try:
            if not response_content or not isinstance(response_content, str):
                return {"can_decompose": False}

            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("无法解析JSON响应")
        except Exception as e:
            logger.warning(f"解析分解响应失败: {e}")
            return {"can_decompose": False}

    def _create_subtasks(self, subtask_data: List[Dict[str, Any]]) -> List[SubTask]:
        """创建子任务对象"""
        subtasks = []

        if not isinstance(subtask_data, list):
            return subtasks

        for task_data in subtask_data:
            try:
                # 映射任务类型
                type_map = {
                    "retrieval": TaskType.RETRIEVAL,
                    "analysis": TaskType.ANALYSIS,
                    "synthesis": TaskType.SYNTHESIS,
                    "verification": TaskType.VERIFICATION,
                    "comparison": TaskType.COMPARISON
                }

                priority_map = {
                    "HIGH": TaskPriority.HIGH,
                    "MEDIUM": TaskPriority.MEDIUM,
                    "LOW": TaskPriority.LOW
                }

                task_type = type_map.get(task_data.get("type", "retrieval"), TaskType.RETRIEVAL)
                priority = priority_map.get(task_data.get("priority", "MEDIUM"), TaskPriority.MEDIUM)

                subtask = SubTask(
                    id=task_data.get("id", f"task_{len(subtasks)+1}"),
                    type=task_type,
                    description=task_data.get("description", ""),
                    query=task_data.get("query", ""),
                    priority=priority,
                    dependencies=task_data.get("dependencies", []),
                    estimated_complexity=float(task_data.get("estimated_complexity", 1.0)),
                    metadata={"original_data": task_data}
                )

                subtasks.append(subtask)

            except Exception as e:
                logger.warning(f"创建子任务失败: {e}")
                continue

        return subtasks

    def optimize_task_order(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        优化任务执行顺序

        Args:
            subtasks: 子任务列表

        Returns:
            优化排序后的子任务列表
        """
        if not subtasks:
            return []

        # 构建依赖图
        dependency_graph = {}
        for task in subtasks:
            dependency_graph[task.id] = task.dependencies

        # 拓扑排序
        ordered_tasks = []
        remaining_tasks = {task.id: task for task in subtasks}
        safe_counter = len(subtasks) + 10

        while remaining_tasks and safe_counter > 0:
            safe_counter -= 1
            # 找到没有依赖或依赖已满足的任务
            ready_tasks = []
            for task_id, task in remaining_tasks.items():
                if all(dep in [t.id for t in ordered_tasks] for dep in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # 存在循环依赖，按优先级排序
                ready_tasks = list(remaining_tasks.values())

            # 按优先级排序
            ready_tasks.sort(key=lambda x: x.priority.value, reverse=True)

            # 执行最高优先级的任务
            selected_task = ready_tasks[0]
            ordered_tasks.append(selected_task)
            del remaining_tasks[selected_task.id]

        return ordered_tasks

    def estimate_execution_plan(self, subtasks: List[SubTask]) -> Dict[str, Any]:
        """
        估计执行计划

        Args:
            subtasks: 子任务列表

        Returns:
            执行计划信息
        """
        if not subtasks:
            return {
                "total_tasks": 0,
                "total_complexity": 0,
                "estimated_time": 0,
                "parallel_groups": 0,
                "max_parallelism": 1,
                "dependency_depth": 0
            }

        total_complexity = sum(task.estimated_complexity for task in subtasks)

        # 识别可以并行执行的任务组
        parallel_groups = self._group_parallel_tasks(subtasks)

        return {
            "total_tasks": len(subtasks),
            "total_complexity": total_complexity,
            "estimated_time": total_complexity * 0.5,  # 假设每个复杂度单位需要0.5秒
            "parallel_groups": len(parallel_groups),
            "max_parallelism": max(len(group) for group in parallel_groups) if parallel_groups else 1,
            "dependency_depth": self._calculate_dependency_depth(subtasks)
        }

    def _group_parallel_tasks(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        """将任务分组为可以并行执行的组"""
        if not subtasks:
            return []

        groups = []
        remaining = set(subtasks)
        safe_counter = len(subtasks) + 10

        while remaining and safe_counter > 0:
            safe_counter -= 1
            # 找到当前可以并行执行的任务
            current_group = []
            completed_ids = set()

            # 添加已完成的任务ID
            for group in groups:
                completed_ids.update(task.id for task in group)

            # 找到没有依赖或依赖已满足的任务
            for task in list(remaining):
                if all(dep in completed_ids for dep in task.dependencies):
                    current_group.append(task)
                    remaining.remove(task)

            if current_group:
                groups.append(current_group)
            else:
                # 防止无限循环，添加剩余任务
                if remaining:
                    groups.append(list(remaining))
                break

        return groups

    def _calculate_dependency_depth(self, subtasks: List[SubTask]) -> int:
        """计算依赖深度"""
        if not subtasks:
            return 0

        def get_depth(task_id: str, visited: set) -> int:
            if task_id in visited:
                return 0

            visited.add(task_id)
            task = next((t for t in subtasks if t.id == task_id), None)
            if not task or not task.dependencies:
                return 1

            max_dep_depth = 0
            for dep_id in task.dependencies:
                dep_depth = get_depth(dep_id, visited.copy())
                max_dep_depth = max(max_dep_depth, dep_depth)

            return max_dep_depth + 1

        max_depth = 0
        task_ids = {t.id for t in subtasks}

        for task in subtasks:
            if task.id in task_ids:
                depth = get_depth(task.id, set())
                max_depth = max(max_depth, depth)

        return max_depth


# 全局分解器实例
_task_decomposer = None
_decomposer_lock = threading.Lock()


def get_task_decomposer() -> TaskDecomposer:
    """获取全局任务分解器实例"""
    global _task_decomposer
    with _decomposer_lock:
        if _task_decomposer is None:
            _task_decomposer = TaskDecomposer()
    return _task_decomposer


def decompose_query(query: str, context: Dict[str, Any] = None) -> List[SubTask]:
    """便捷函数：分解查询"""
    decomposer = get_task_decomposer()
    return decomposer.decompose(query, context)