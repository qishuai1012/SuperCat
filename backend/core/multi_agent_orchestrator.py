"""
多Agent协作系统
负责协调多个专业Agent完成复杂任务
"""

from typing import Dict, Any, List, Optional, TypedDict, Callable
from dataclasses import asdict, dataclass, is_dataclass
import json
import logging
import asyncio
import re
import time
from enum import Enum
import threading
from contextlib import asynccontextmanager

from core.memory_optimizer import MemoryCompressor
from chat.result_builder import ResultBuilder
from schemas import CompactTrace, CompressionMetadata, ContextBundle, KnowledgePoint, MessageDigest, PlanningSummary, RetrievedChunk, SubtaskResultSummary, VerificationSummary

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

from core.task_decomposer import TaskType, get_task_decomposer

logger = logging.getLogger(__name__)

# 容错导入工具，避免启动报错
try:
    from tools import get_current_weather, search_knowledge_base
except ImportError:
    def search_knowledge_base(*args, **kwargs):
        return {}
    def get_current_weather(*args, **kwargs):
        return {}

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


def _serialize_for_json(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return _serialize_for_json(value.model_dump())
    if is_dataclass(value):
        return _serialize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_json(item) for item in value]
    return str(value)


class AgentType(Enum):
    """Agent类型枚举"""
    RETRIEVAL = "retrieval"      # 检索Agent
    ANALYSIS = "analysis"        # 分析Agent
    SYNTHESIS = "synthesis"      # 综合Agent
    VERIFICATION = "verification" # 验证Agent
    PLANNING = "planning"        # 规划Agent
    COORDINATOR = "coordinator"  # 协调Agent


@dataclass
class AgentTask:
    """Agent任务定义"""
    id: str
    agent_type: AgentType
    input_data: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentResult:
    """Agent执行结果"""
    task_id: str
    agent_type: AgentType
    success: bool
    output: Dict[str, Any]
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SpecializedAgent:
    """专业Agent基类"""

    def __init__(self, agent_type: AgentType, system_prompt: str = None):
        self.agent_type = agent_type
        self.model = None
        self.agent = None
        self._memory_compressor = MemoryCompressor()
        self._result_builder = ResultBuilder()
        self._init_agent(system_prompt)

    def _init_agent(self, system_prompt: str = None):
        """初始化Agent"""
        try:
            logger.info(f"开始初始化 Agent {self.agent_type.value}, API_KEY存在: {bool(API_KEY)}, MODEL存在: {bool(MODEL)}")
            
            if API_KEY and MODEL:
                logger.info(f"正在创建模型: {MODEL}, base_url: {BASE_URL}")
                self.model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.2,
                    stream_usage=True,
                )
                logger.info(f"模型创建成功: {self.model}")

                # 根据Agent类型设置系统提示
                if not system_prompt:
                    system_prompt = self._get_default_system_prompt()

                # 根据Agent类型选择工具
                tools = self._get_specialized_tools()
                logger.info(f"Agent {self.agent_type.value} 使用工具: {[t.name for t in tools] if tools else '无'}")

                self.agent = create_agent(
                    model=self.model,
                    tools=tools,
                    system_prompt=system_prompt,
                )
                logger.info(f"Agent {self.agent_type.value} 创建成功")
            else:
                logger.warning(f"Agent {self.agent_type.value} 初始化跳过: API_KEY或MODEL为空")
        except Exception as e:
            logger.error(f"Agent {self.agent_type.value} 初始化失败: {e}")
            self.agent = None

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        prompts = {
            AgentType.RETRIEVAL: """你是专业的信息检索专家。你的任务是：
1. 准确理解用户查询意图
2. 选择合适的检索策略
3. 获取最相关的信息
4. 评估检索结果的质量
""",
            AgentType.ANALYSIS: """你是专业的信息分析专家。你的任务是：
1. 深入分析信息内容
2. 识别关键信息和模式
3. 进行逻辑推理和判断
4. 发现信息间的关联性
""",
            AgentType.SYNTHESIS: """你是专业的信息综合专家。你的任务是：
1. 整合多个信息源
2. 消除信息冲突
3. 生成连贯的结论
4. 确保答案的完整性和准确性
""",
            AgentType.VERIFICATION: """你是专业的事实核查专家。你的任务是：
1. 验证信息的准确性
2. 识别潜在的错误或偏见
3. 评估信息来源的可靠性
4. 提供验证结果和建议
""",
            AgentType.PLANNING: """你是专业的任务规划专家。你的任务是：
1. 分析复杂任务的组成部分
2. 制定执行计划和策略
3. 优化任务执行顺序
4. 评估计划的可行性和风险
""",
            AgentType.COORDINATOR: """你是专业的任务协调专家。你的任务是：
1. 协调多个专业Agent的工作
2. 分配任务和资源
3. 监控执行进度
4. 整合和优化最终结果
"""
        }
        return prompts.get(self.agent_type, "你是专业的人工智能助手。")

    def _get_specialized_tools(self) -> List[Callable]:
        """获取专业工具"""
        tool_mapping = {
            AgentType.RETRIEVAL: [search_knowledge_base],
            AgentType.ANALYSIS: [],
            AgentType.SYNTHESIS: [],
            AgentType.VERIFICATION: [search_knowledge_base],
            AgentType.COORDINATOR: [search_knowledge_base]
        }
        return tool_mapping.get(self.agent_type, [])

    def _compact_text(self, value: Any, limit: int = 1200) -> str:
        text = str(value) if value is not None else ""
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _compact_list(self, items: Any, limit: int = 8) -> list:
        if not isinstance(items, (list, tuple, set)):
            return []
        return [self._compact_value(item) for item in list(items)[:limit]]

    def _compact_value(self, value: Any, *, depth: int = 0) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            if depth >= 2:
                return {str(k): self._compact_text(v) for k, v in list(value.items())[:12]}
            compacted = {}
            for key, item in list(value.items())[:20]:
                if key in {"content", "response", "summary", "final_answer", "error", "reason", "strategy", "verdict"}:
                    compacted[str(key)] = self._compact_text(item)
                elif key in {"messages", "subtask_results", "all_results", "history", "recent_messages", "key_points", "issues", "checks", "risks"}:
                    compacted[str(key)] = self._compact_list(item)
                else:
                    compacted[str(key)] = self._compact_value(item, depth=depth + 1)
            return compacted
        if isinstance(value, (list, tuple, set)):
            return self._compact_list(value)
        return self._compact_text(value)

    def _get_context_value(self, context: Any, key: str, default: Any = None) -> Any:
        if context is None:
            return default
        if isinstance(context, dict):
            return context.get(key, default)
        return getattr(context, key, default)

    def _to_knowledge_point(self, item: Any) -> KnowledgePoint | None:
        if isinstance(item, KnowledgePoint):
            return item
        if not isinstance(item, dict):
            return None
        concept = item.get("concept") or item.get("name") or item.get("title")
        definition = item.get("definition") or item.get("content") or item.get("summary")
        if not concept and not definition:
            return None
        return KnowledgePoint(
            concept=str(concept or "")[:120],
            definition=str(definition or "")[:300],
            importance=item.get("importance"),
            category=item.get("category"),
        )

    def _to_retrieved_chunk(self, item: Any) -> RetrievedChunk | None:
        if isinstance(item, RetrievedChunk):
            return item
        if not isinstance(item, dict):
            return None
        filename = item.get("filename") or item.get("file_name") or item.get("source")
        text = item.get("text") or item.get("content")
        if not filename and not text:
            return None
        return RetrievedChunk(
            filename=str(filename or "unknown")[:160],
            page_number=item.get("page_number"),
            text=str(text or "")[:400],
            score=item.get("score"),
            rrf_rank=item.get("rrf_rank"),
            rerank_score=item.get("rerank_score"),
        )

    def _build_context_bundle(self, context: Any) -> ContextBundle | None:
        if context is None:
            return None
        history = self._get_context_value(context, "history", None) or []
        history_text = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("type") or item.get("role") or "message"
                content = str(item.get("content", ""))
            else:
                role = getattr(item, "type", None) or getattr(item, "role", None) or "message"
                content = str(getattr(item, "content", item))
            history_text.append({"role": str(role), "content": content})
        compressed_history = self._memory_compressor.compress_conversation(history_text)
        key_knowledge = [kp for item in self._memory_compressor.extract_key_knowledge(compressed_history) if (kp := self._to_knowledge_point(item))]
        recent_messages = []
        for item in history[-6:]:
            if isinstance(item, dict):
                recent_messages.append(MessageDigest(type=str(item.get("type", "unknown")), content_preview=str(item.get("content", ""))[:180]))
            else:
                recent_messages.append(MessageDigest(type=str(getattr(item, "type", "unknown")), content_preview=str(getattr(item, "content", item))[:180]))
        raw_chars = sum(len(m["content"]) for m in history_text)
        compression_meta = CompressionMetadata(
            strategy="memory_compressor",
            raw_chars=raw_chars,
            compact_chars=len(compressed_history),
            compression_ratio=(len(compressed_history) / max(1, raw_chars)) if history_text else 1.0,
            dropped_fields=["full_history"],
            history_messages_total=len(history_text),
            history_messages_retained=len(recent_messages),
        )
        return ContextBundle(
            query=str(self._get_context_value(context, "user_text", "") or self._get_context_value(context, "query", "") or ""),
            user_id=self._get_context_value(context, "user_id", None),
            session_id=self._get_context_value(context, "session_id", None),
            history_summary=compressed_history,
            recent_messages=recent_messages,
            key_knowledge=key_knowledge,
            compression_meta=compression_meta,
        )

    def _build_subtask_summary(self, result: AgentResult) -> SubtaskResultSummary:
        output = result.output if isinstance(result.output, dict) else {}
        response_text = None
        rag_trace = None
        if isinstance(output, dict):
            response_text = output.get("response") or output.get("summary") or output.get("final_answer")
            rag_trace = output.get("rag_trace")
        if response_text is None:
            response_text = result.output
        output_text = str(response_text)[:600]
        key_points = [kp for item in (output.get("key_points", []) if isinstance(output, dict) else []) if (kp := self._to_knowledge_point(item))]
        evidence = [chunk for item in (output.get("evidence", []) if isinstance(output, dict) else []) if (chunk := self._to_retrieved_chunk(item))]
        compact_trace = CompactTrace(
            tool_used=bool(rag_trace or result.metadata.get("tool_used")),
            tool_name=str(result.metadata.get("tool_name")) if result.metadata.get("tool_name") else None,
            query=str(result.metadata.get("query")) if result.metadata.get("query") else None,
            retrieval_stage=str(result.metadata.get("retrieval_stage")) if result.metadata.get("retrieval_stage") else None,
            reason=str(result.metadata.get("error") or result.metadata.get("reason") or "")[:260] or None,
            filter_summary=result.metadata.get("filter_summary") if result.metadata.get("filter_summary") else None,
        )
        issues = [str(item)[:200] for item in (output.get("issues", []) if isinstance(output, dict) else [])]
        return SubtaskResultSummary(
            task_id=result.task_id,
            agent_type=result.agent_type.value,
            success=result.success,
            summary=output_text,
            key_points=key_points,
            evidence=evidence,
            issues=issues,
            confidence=float(result.metadata.get("confidence")) if result.metadata.get("confidence") is not None else None,
            compact_trace=compact_trace,
            execution_time=result.execution_time,
        )

    def _build_planning_summary(self, planning_output: Dict[str, Any], main_query: str) -> PlanningSummary:
        output = planning_output if isinstance(planning_output, dict) else {"response": str(planning_output)}
        return PlanningSummary(
            goal=str(main_query)[:240],
            strategy=str(output.get("strategy") or output.get("response") or "")[:200] or None,
            subtasks=[str(item)[:200] for item in output.get("subtasks", [])][:10],
            risks=[str(item)[:200] for item in output.get("risks", [])][:8],
            checks=[str(item)[:200] for item in output.get("checks", [])][:8],
        )

    def _build_verification_summary(self, verification_output: Dict[str, Any]) -> VerificationSummary:
        output = verification_output if isinstance(verification_output, dict) else {"response": str(verification_output)}
        verdict = output.get("verdict") or output.get("response") or ""
        return VerificationSummary(
            verdict=str(verdict)[:240],
            issues_found=[str(item)[:200] for item in output.get("issues_found", [])][:8],
            supported_claims=[str(item)[:200] for item in output.get("supported_claims", [])][:8],
            unsupported_claims=[str(item)[:200] for item in output.get("unsupported_claims", [])][:8],
            recommended_changes=[str(item)[:200] for item in output.get("recommended_changes", [])][:8],
        )

    def _prepare_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        compacted: Dict[str, Any] = {}
        context_bundle = payload.get("context_bundle")
        if context_bundle is None and payload.get("context") is not None:
            context_bundle = self._build_context_bundle(payload.get("context"))

        for key, value in payload.items():
            if key == "context":
                compacted[key] = _serialize_for_json(value)
            elif key == "context_bundle":
                compacted[key] = _serialize_for_json(context_bundle) if context_bundle is not None else None
            elif key in {"subtask_results", "all_results"}:
                compacted[key] = [_serialize_for_json(item) for item in list(value)[:6]] if isinstance(value, (list, tuple, set)) else []
            elif key in {"analysis_result", "synthesis_prep", "planning_output", "verification_input", "final_answer", "final_answer_text", "original_query", "query", "focus", "requirements", "strategy", "summary"}:
                compacted[key] = str(value)[:2000]
            else:
                compacted[key] = _serialize_for_json(value)

        if context_bundle is not None and "context_bundle" not in compacted:
            compacted["context_bundle"] = _serialize_for_json(context_bundle)

        dumped = json.dumps(_serialize_for_json(compacted), ensure_ascii=False)
        if len(dumped) > 12000:
            for key in ("context", "subtask_results", "all_results", "analysis_result", "synthesis_prep", "planning_output", "final_answer", "final_answer_text"):
                if key in compacted and isinstance(compacted[key], list):
                    compacted[key] = compacted[key][:4]
                elif key in compacted:
                    compacted[key] = str(compacted[key])[:900]
        return compacted

    async def execute(self, task: AgentTask) -> AgentResult:
        """执行任务"""
        # 修复：Agent未初始化直接返回，不往下执行
        if not self.agent:
            logger.error(f"Agent {self.agent_type.value} 未初始化")
            return AgentResult(
                task_id=task.id,
                agent_type=self.agent_type,
                success=False,
                output={"error": "Agent未初始化"},
                execution_time=0.0
            )

        start_time = asyncio.get_event_loop().time()
        logger.info(f"Agent {self.agent_type.value} 开始执行任务: {task.id}, 超时时间: {task.timeout}秒")

        try:
            # 修复：增加任务超时控制
            result = await asyncio.wait_for(
                self._run_agent(task),
                timeout=task.timeout
            )

            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Agent {self.agent_type.value} 任务完成: {task.id}, 耗时: {execution_time:.2f}秒")

            return AgentResult(
                task_id=task.id,
                agent_type=self.agent_type,
                success=True,
                output=result,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Agent {self.agent_type.value} 任务超时: {task.id}, 耗时: {execution_time:.2f}秒")
            return AgentResult(
                task_id=task.id,
                agent_type=self.agent_type,
                success=False,
                output={"error": "任务执行超时"},
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Agent {self.agent_type.value} 执行失败: {e}")

            return AgentResult(
                task_id=task.id,
                agent_type=self.agent_type,
                success=False,
                output={"error": str(e)},
                execution_time=execution_time
            )

    async def _run_agent(self, task: AgentTask) -> Dict[str, Any]:
        """内部执行Agent逻辑，原样保留原有调用格式"""
        try:
            from tools import reset_tool_call_guards
            reset_tool_call_guards()
        except Exception:
            pass
        input_message = self._prepare_input(task.input_data)
        logger.info(f"Agent {self.agent_type.value} 开始调用模型，输入长度: {len(input_message)} 字符")
        model_start = asyncio.get_event_loop().time()
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]},
            config={"recursion_limit": 5},
        )
        model_end = asyncio.get_event_loop().time()
        logger.info(f"Agent {self.agent_type.value} 模型调用完成，耗时: {model_end - model_start:.2f}秒")
        return self._process_output(result)

    def _prepare_input(self, input_data: Dict[str, Any]) -> str:
        """准备输入消息"""
        prepared = self._prepare_payload(input_data)
        return json.dumps(_serialize_for_json(prepared), ensure_ascii=False, indent=2)

    def _process_output(self, result: Any) -> Dict[str, Any]:
        """处理输出结果"""
        try:
            if isinstance(result, dict):
                if "output" in result:
                    return {"response": result["output"]}
                elif "messages" in result and result["messages"]:
                    msg = result["messages"][-1]
                    return {"response": getattr(msg, "content", str(msg))}
            elif hasattr(result, "content"):
                return {"response": result.content}

            return {"response": str(result)}
        except Exception as e:
            return {"error": f"输出处理失败: {e}", "raw_result": str(result)}


class RetrievalAgent(SpecializedAgent):
    """专门负责信息检索的Agent"""

    def __init__(self):
        super().__init__(AgentType.RETRIEVAL)

    def _get_default_system_prompt(self) -> str:
        return """你是专业的信息检索专家。你的任务是：
1. 准确理解用户查询意图
2. 选择合适的检索策略（直接检索、退步查询、假设文档等）
3. 获取最相关的信息
4. 评估检索结果的质量和相关性
5. 返回结构化的检索结果
"""


class AnalysisAgent(SpecializedAgent):
    """专门负责信息分析的Agent"""

    def __init__(self):
        super().__init__(AgentType.ANALYSIS)

    def _get_default_system_prompt(self) -> str:
        return """你是专业的信息分析专家。你的任务是：
1. 深入分析信息内容，识别关键信息和模式
2. 进行逻辑推理和判断，发现信息间的关联性
3. 评估信息的可靠性、时效性和完整性
4. 识别潜在的偏见、错误或不一致之处
5. 提供详细的分析结论和建议
"""


class SynthesisAgent(SpecializedAgent):
    """专门负责信息综合的Agent"""

    def __init__(self):
        super().__init__(AgentType.SYNTHESIS)

    def _get_default_system_prompt(self) -> str:
        return """你是专业的信息综合专家。你的任务是：
1. 整合多个信息源，消除信息冲突和重复
2. 生成连贯、完整的结论和答案
3. 确保答案的逻辑性、准确性和完整性
4. 优化表达方式，使其清晰易懂
5. 提供综合性的最终回答
"""


class VerificationAgent(SpecializedAgent):
    """专门负责事实核查的Agent"""

    def __init__(self):
        super().__init__(AgentType.VERIFICATION)

    def _get_default_system_prompt(self) -> str:
        return """你是专业的事实核查专家。你的任务是：
1. 验证信息的准确性和真实性
2. 识别潜在的错误、偏见或不实信息
3. 评估信息来源的可靠性和权威性
4. 进行交叉验证，确保信息一致性
5. 提供验证结果和改进建议
"""


class PlanningAgent(SpecializedAgent):
    """专门负责任务规划的Agent"""

    def __init__(self):
        super().__init__(AgentType.PLANNING)

    def _get_default_system_prompt(self) -> str:
        return """你是专业的任务规划专家。你的任务是：
1. 分析复杂任务的组成部分和需求
2. 制定详细的执行计划和策略
3. 优化任务执行顺序，考虑依赖关系
4. 评估计划的可行性和潜在风险
5. 提供可执行的任务分解方案
"""


class CoordinatorAgent(SpecializedAgent):
    """协调Agent 补全缺失类型，不改动原有逻辑"""
    def __init__(self):
        super().__init__(AgentType.COORDINATOR)


class RequestLevelOrchestrationError(Exception):
    pass


class MultiAgentOrchestrator:
    """
    多Agent协调器 - 负责管理和协调整个多Agent系统
    """

    def __init__(self):
        self.agents = {}
        self._agent_lock = threading.Lock()
        self.task_decomposer = get_task_decomposer()
        self.parallel_executor = None
        self._memory_compressor = MemoryCompressor()
        self._result_builder = ResultBuilder()
        self._init_specialized_agents()

    def _init_specialized_agents(self):
        """初始化专业Agent"""
        # 修复：补齐 COORDINATOR 避免KeyError
        self.agents = {
            AgentType.RETRIEVAL: RetrievalAgent(),
            AgentType.ANALYSIS: AnalysisAgent(),
            AgentType.SYNTHESIS: SynthesisAgent(),
            AgentType.VERIFICATION: VerificationAgent(),
            AgentType.PLANNING: PlanningAgent(),
            AgentType.COORDINATOR: CoordinatorAgent()
        }

    def _compact_text(self, value: Any, limit: int = 1200) -> str:
        text = str(value) if value is not None else ""
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _compact_list(self, items: Any, limit: int = 8) -> list:
        if not isinstance(items, (list, tuple, set)):
            return []
        return [self._compact_value(item) for item in list(items)[:limit]]

    def _compact_value(self, value: Any, *, depth: int = 0) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            if depth >= 2:
                return {str(k): self._compact_text(v) for k, v in list(value.items())[:12]}
            compacted = {}
            for key, item in list(value.items())[:20]:
                if key in {"content", "response", "summary", "final_answer", "error", "reason", "strategy", "verdict"}:
                    compacted[str(key)] = self._compact_text(item)
                elif key in {"messages", "subtask_results", "all_results", "history", "recent_messages", "key_points", "issues", "checks", "risks"}:
                    compacted[str(key)] = self._compact_list(item)
                else:
                    compacted[str(key)] = self._compact_value(item, depth=depth + 1)
            return compacted
        if isinstance(value, (list, tuple, set)):
            return self._compact_list(value)
        return self._compact_text(value)

    def _get_context_value(self, context: Any, key: str, default: Any = None) -> Any:
        if context is None:
            return default
        if isinstance(context, dict):
            return context.get(key, default)
        return getattr(context, key, default)

    def _to_knowledge_point(self, item: Any) -> KnowledgePoint | None:
        if isinstance(item, KnowledgePoint):
            return item
        if not isinstance(item, dict):
            return None
        concept = item.get("concept") or item.get("name") or item.get("title")
        definition = item.get("definition") or item.get("content") or item.get("summary")
        if not concept and not definition:
            return None
        return KnowledgePoint(
            concept=str(concept or "")[:120],
            definition=str(definition or "")[:300],
            importance=item.get("importance"),
            category=item.get("category"),
        )

    def _to_retrieved_chunk(self, item: Any) -> RetrievedChunk | None:
        if isinstance(item, RetrievedChunk):
            return item
        if not isinstance(item, dict):
            return None
        filename = item.get("filename") or item.get("file_name") or item.get("source")
        text = item.get("text") or item.get("content")
        if not filename and not text:
            return None
        return RetrievedChunk(
            filename=str(filename or "unknown")[:160],
            page_number=item.get("page_number"),
            text=str(text or "")[:400],
            score=item.get("score"),
            rrf_rank=item.get("rrf_rank"),
            rerank_score=item.get("rerank_score"),
        )

    def _build_context_bundle(self, context: Any) -> ContextBundle | None:
        if context is None:
            return None
        history = self._get_context_value(context, "history", None) or []
        history_text = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("type") or item.get("role") or "message"
                content = str(item.get("content", ""))
            else:
                role = getattr(item, "type", None) or getattr(item, "role", None) or "message"
                content = str(getattr(item, "content", item))
            history_text.append({"role": str(role), "content": content})
        compressed_history = self._memory_compressor.compress_conversation(history_text)
        key_knowledge = [kp for item in self._memory_compressor.extract_key_knowledge(compressed_history) if (kp := self._to_knowledge_point(item))]
        recent_messages = []
        for item in history[-6:]:
            if isinstance(item, dict):
                recent_messages.append(MessageDigest(type=str(item.get("type", "unknown")), content_preview=str(item.get("content", ""))[:180]))
            else:
                recent_messages.append(MessageDigest(type=str(getattr(item, "type", "unknown")), content_preview=str(getattr(item, "content", item))[:180]))
        raw_chars = sum(len(m["content"]) for m in history_text)
        compression_meta = CompressionMetadata(
            strategy="memory_compressor",
            raw_chars=raw_chars,
            compact_chars=len(compressed_history),
            compression_ratio=(len(compressed_history) / max(1, raw_chars)) if history_text else 1.0,
            dropped_fields=["full_history"],
            history_messages_total=len(history_text),
            history_messages_retained=len(recent_messages),
        )
        return ContextBundle(
            query=str(self._get_context_value(context, "user_text", "") or self._get_context_value(context, "query", "") or ""),
            user_id=self._get_context_value(context, "user_id", None),
            session_id=self._get_context_value(context, "session_id", None),
            history_summary=compressed_history,
            recent_messages=recent_messages,
            key_knowledge=key_knowledge,
            compression_meta=compression_meta,
        )

    def _build_subtask_summary(self, result: AgentResult) -> SubtaskResultSummary:
        output = result.output if isinstance(result.output, dict) else {}
        response_text = None
        rag_trace = None
        if isinstance(output, dict):
            response_text = output.get("response") or output.get("summary") or output.get("final_answer")
            rag_trace = output.get("rag_trace")
        if response_text is None:
            response_text = result.output
        output_text = str(response_text)[:600]
        key_points = [kp for item in (output.get("key_points", []) if isinstance(output, dict) else []) if (kp := self._to_knowledge_point(item))]
        evidence = [chunk for item in (output.get("evidence", []) if isinstance(output, dict) else []) if (chunk := self._to_retrieved_chunk(item))]
        compact_trace = CompactTrace(
            tool_used=bool(rag_trace or result.metadata.get("tool_used")),
            tool_name=str(result.metadata.get("tool_name")) if result.metadata.get("tool_name") else None,
            query=str(result.metadata.get("query")) if result.metadata.get("query") else None,
            retrieval_stage=str(result.metadata.get("retrieval_stage")) if result.metadata.get("retrieval_stage") else None,
            reason=str(result.metadata.get("error") or result.metadata.get("reason") or "")[:260] or None,
            filter_summary=result.metadata.get("filter_summary") if result.metadata.get("filter_summary") else None,
        )
        issues = [str(item)[:200] for item in (output.get("issues", []) if isinstance(output, dict) else [])]
        return SubtaskResultSummary(
            task_id=result.task_id,
            agent_type=result.agent_type.value,
            success=result.success,
            summary=output_text,
            key_points=key_points,
            evidence=evidence,
            issues=issues,
            confidence=float(result.metadata.get("confidence")) if result.metadata.get("confidence") is not None else None,
            compact_trace=compact_trace,
            execution_time=result.execution_time,
        )

    def _build_planning_summary(self, planning_output: Dict[str, Any], main_query: str) -> PlanningSummary:
        output = planning_output if isinstance(planning_output, dict) else {"response": str(planning_output)}
        return PlanningSummary(
            goal=str(main_query)[:240],
            strategy=str(output.get("strategy") or output.get("response") or "")[:200] or None,
            subtasks=[str(item)[:200] for item in output.get("subtasks", [])][:10],
            risks=[str(item)[:200] for item in output.get("risks", [])][:8],
            checks=[str(item)[:200] for item in output.get("checks", [])][:8],
        )

    def _build_verification_summary(self, verification_output: Dict[str, Any]) -> VerificationSummary:
        output = verification_output if isinstance(verification_output, dict) else {"response": str(verification_output)}
        verdict = output.get("verdict") or output.get("response") or ""
        return VerificationSummary(
            verdict=str(verdict)[:240],
            issues_found=[str(item)[:200] for item in output.get("issues_found", [])][:8],
            supported_claims=[str(item)[:200] for item in output.get("supported_claims", [])][:8],
            unsupported_claims=[str(item)[:200] for item in output.get("unsupported_claims", [])][:8],
            recommended_changes=[str(item)[:200] for item in output.get("recommended_changes", [])][:8],
        )

    async def coordinate_task(self, main_query: str, context: Dict[str, Any] = None, query_complexity: str = None) -> Dict[str, Any]:
        """
        协调整个多Agent任务执行

        Args:
            main_query: 主任务查询
            context: 上下文信息
            query_complexity: 查询复杂度 (simple/medium/complex)

        Returns:
            综合执行结果
        """
        if context is None:
            context = {}

        # 检查关键Agent是否已初始化
        critical_agents = [AgentType.PLANNING, AgentType.SYNTHESIS]
        for agent_type in critical_agents:
            agent = self.agents.get(agent_type)
            if not agent or not agent.agent:
                logger.warning(f"关键Agent {agent_type.value} 未初始化，降级到默认策略")
                return await self._execute_default_strategy(main_query, context, fallback_reason=f"agent_unavailable:{agent_type.value}")

        # ====================== 🚀 【条件执行：根据复杂度选择流程】 ======================
        # 使用外部传入的复杂度判断，避免重复判断
        is_simple = query_complexity == "simple" if query_complexity else False

        logger.info(f"查询复杂度: {query_complexity}, 使用精简流程: {is_simple}")

        # 简单问题：跳过Planning和Verification
        if is_simple:
            logger.info("✅ 简单查询，使用精简流程（跳过Planning/Verification）")
            return await self._execute_simple_strategy(main_query, context)

        # 轻度复杂：跳过Planning，直接并行执行
        if query_complexity == "complex_light":
            logger.info("✅ 轻度复杂查询，跳过Planning，直接并行执行")
            return await self._execute_complex_light_strategy(main_query, context)

        decomposition_plan = None
        if query_complexity == "complex_heavy":
            decomposition_plan = self._build_decomposition_plan(main_query, context)
        # ================================================================================

        try:
            # 1. 任务规划
            planning_task = AgentTask(
                id="planning_main",
                agent_type=AgentType.PLANNING,
                input_data={
                    "query": main_query,
                    "context": context,
                    "context_bundle": self._build_context_bundle(context),
                    "requirements": "分析任务复杂度，制定执行计划"
                },
                timeout=120.0  # 增加到120秒
            )

            planning_result = await self.agents[AgentType.PLANNING].execute(planning_task)

            if not planning_result.success:
                logger.warning("任务规划失败，使用默认执行策略")
                return await self._execute_default_strategy(main_query, context, fallback_reason="planning_failed")

            planning_summary = self._build_planning_summary(planning_result.output, main_query)

            # 2. 解析规划结果并生成子任务
            subtasks = self._parse_planning_result(planning_result.output, main_query, context)
            if decomposition_plan and decomposition_plan.get("subtasks"):
                subtasks = self._merge_decomposition_subtasks(subtasks, decomposition_plan["subtasks"], main_query, context)

            # 3. 执行子任务分解与分派
            logger.info("🚀 执行复杂问题子任务")
            subtask_results = await self._execute_subtasks(subtasks)
            successful_subtasks = [result for result in subtask_results if result.success]
            compact_subtask_results = [self._build_subtask_summary(result) for result in subtask_results]

            if not successful_subtasks:
                logger.warning("子任务执行全部失败，降级到默认策略")
                return await self._execute_default_strategy(main_query, context, fallback_reason="subtasks_failed")

            # 4. 汇总子任务结果生成最终答案
            final_synthesis_task = AgentTask(
                id="synthesis_final",
                agent_type=AgentType.SYNTHESIS,
                input_data={
                    "subtask_results": compact_subtask_results,
                    "planning_output": planning_summary,
                    "context_bundle": self._build_context_bundle(context),
                    "original_query": main_query,
                    "requirements": "整合子任务结果，生成最终答案"
                },
                timeout=120.0
            )
            final_result = await self.agents[AgentType.SYNTHESIS].execute(final_synthesis_task)
            final_summary = self._build_subtask_summary(final_result)

            # 5. 验证结果
            all_task_results = [*subtask_results, final_result]
            verification_task = AgentTask(
                id="verification_final",
                agent_type=AgentType.VERIFICATION,
                input_data={
                    "final_answer_text": final_summary.summary,
                    "planning_output": planning_summary,
                    "subtask_results": compact_subtask_results,
                    "original_query": main_query,
                    "context_bundle": self._build_context_bundle(context),
                },
                timeout=120.0
            )
            verification_result = await self.agents[AgentType.VERIFICATION].execute(verification_task)
            verification_summary = self._build_verification_summary(verification_result.output)

            if final_result.success and verification_result.success:
                return self._normalized_success_result(
                    final_summary.summary,
                    subtask_results=compact_subtask_results + [final_summary],
                    verification=verification_summary,
                    planning=planning_summary,
                    execution_metadata={
                        "total_tasks": len(all_task_results),
                        "successful_tasks": sum(1 for r in all_task_results if r.success),
                        "total_execution_time": sum(r.execution_time for r in all_task_results),
                        "strategy": "complex_heavy",
                        "decomposition": decomposition_plan,
                        "raw_subtask_results": [self._result_to_dict(result) for result in all_task_results],
                    },
                )
            return self._normalized_failure_result(
                final_summary.summary or "抱歉，处理您的请求时遇到困难。",
                error=final_result.output.get("error") if isinstance(final_result.output, dict) else None,
                subtask_results=compact_subtask_results + [final_summary],
                verification=verification_summary,
                planning=planning_summary,
                execution_metadata={
                    "total_tasks": len(all_task_results),
                    "successful_tasks": sum(1 for r in all_task_results if r.success),
                    "total_execution_time": sum(r.execution_time for r in all_task_results),
                    "strategy": "complex_heavy",
                    "decomposition": decomposition_plan,
                    "raw_subtask_results": [self._result_to_dict(result) for result in all_task_results],
                },
            )

        except RequestLevelOrchestrationError:
            raise
        except Exception as e:
            logger.error(f"多Agent协调失败: {e}")
            self._raise_if_request_level_error(e)
            return await self._execute_default_strategy(main_query, context, fallback_reason="coordinate_task_exception")

    async def _execute_default_strategy(self, query: str, context: Dict[str, Any], fallback_reason: str | None = None) -> Dict[str, Any]:
        """执行默认策略（简化版）"""
        try:
            # 直接检索
            retrieval_task = AgentTask(
                id="retrieval_default",
                agent_type=AgentType.RETRIEVAL,
                input_data={"query": query, "context": context},
                timeout=120.0  # 增加到120秒
            )

            retrieval_result = await self.agents[AgentType.RETRIEVAL].execute(retrieval_task)
            retrieval_summary = self._build_subtask_summary(retrieval_result)

            # 简单综合
            synthesis_task = AgentTask(
                id="synthesis_default",
                agent_type=AgentType.SYNTHESIS,
                input_data={
                    "subtask_results": [retrieval_summary],
                    "context_bundle": self._build_context_bundle(context),
                    "original_query": query
                },
                timeout=120.0  # 增加到120秒
            )

            synthesis_result = await self.agents[AgentType.SYNTHESIS].execute(synthesis_task)
            synthesis_summary = self._build_subtask_summary(synthesis_result)

            if synthesis_result.success:
                return self._normalized_success_result(
                    synthesis_summary.summary,
                    subtask_results=[retrieval_summary, synthesis_summary],
                    execution_metadata={
                        "total_tasks": 2,
                        "successful_tasks": int(retrieval_result.success) + int(synthesis_result.success),
                        "total_execution_time": retrieval_result.execution_time + synthesis_result.execution_time,
                        "strategy": "default",
                        "raw_subtask_results": [self._result_to_dict(retrieval_result), self._result_to_dict(synthesis_result)],
                    },
                    fallback_reason=fallback_reason,
                )
            return self._normalized_failure_result(
                synthesis_summary.summary or "抱歉，处理您的请求时遇到困难。",
                error=synthesis_result.output.get("error") if isinstance(synthesis_result.output, dict) else None,
                fallback_reason=fallback_reason,
                subtask_results=[retrieval_summary, synthesis_summary],
                execution_metadata={
                    "total_tasks": 2,
                    "successful_tasks": 1 if retrieval_result.success else 0,
                    "total_execution_time": retrieval_result.execution_time + synthesis_result.execution_time,
                    "strategy": "default",
                    "raw_subtask_results": [self._result_to_dict(retrieval_result), self._result_to_dict(synthesis_result)],
                },
            )

        except Exception as e:
            logger.error(f"默认策略执行失败: {e}")
            self._raise_if_request_level_error(e)
            return self._normalized_failure_result(
                "抱歉，处理您的请求时遇到困难。",
                error=str(e),
                fallback_reason=fallback_reason,
                execution_metadata={"strategy": "default"},
            )

    async def _execute_simple_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行简单策略（跳过Planning和Verification）"""
        try:
            # 直接检索
            retrieval_task = AgentTask(
                id="retrieval_simple",
                agent_type=AgentType.RETRIEVAL,
                input_data={"query": query, "context": context},
                timeout=120.0
            )

            retrieval_result = await self.agents[AgentType.RETRIEVAL].execute(retrieval_task)

            # 简单综合
            synthesis_task = AgentTask(
                id="synthesis_simple",
                agent_type=AgentType.SYNTHESIS,
                input_data={
                    "subtask_results": [self._result_to_dict(retrieval_result)],
                    "original_query": query
                },
                timeout=120.0
            )

            synthesis_result = await self.agents[AgentType.SYNTHESIS].execute(synthesis_task)

            if synthesis_result.success:
                return self._normalized_success_result(
                    synthesis_result.output.get("response", ""),
                    subtask_results=[retrieval_result],
                    execution_metadata={
                        "total_tasks": 1,
                        "successful_tasks": 1 if retrieval_result.success else 0,
                        "total_execution_time": retrieval_result.execution_time,
                        "strategy": "simple",
                    },
                )
            return self._normalized_failure_result(
                synthesis_result.output.get("response", "") or "抱歉，处理您的请求时遇到困难。",
                error=synthesis_result.output.get("error") if isinstance(synthesis_result.output, dict) else None,
                subtask_results=[retrieval_result],
                execution_metadata={
                    "total_tasks": 1,
                    "successful_tasks": 1 if retrieval_result.success else 0,
                    "total_execution_time": retrieval_result.execution_time,
                    "strategy": "simple",
                },
            )

        except Exception as e:
            logger.error(f"简单策略执行失败: {e}")
            self._raise_if_request_level_error(e)
            return self._normalized_failure_result(
                "抱歉，处理您的请求时遇到困难。",
                error=str(e),
                execution_metadata={"strategy": "simple"},
            )

    async def _execute_complex_light_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行轻度复杂策略（跳过Planning，先检索再并行Analysis/Synthesis）"""
        try:
            # 先检索知识库
            retrieval_task = AgentTask(
                id="retrieval_complex_light",
                agent_type=AgentType.RETRIEVAL,
                input_data={"query": query, "context": context},
                timeout=120.0
            )
            retrieval_result = await self.agents[AgentType.RETRIEVAL].execute(retrieval_task)
            retrieval_dict = self._result_to_dict(retrieval_result)
            logger.info("🚀 检索完成，并行执行 Analysis 和 Synthesis")

            # 创建分析任务
            analysis_task = AgentTask(
                id="analysis_complex_light",
                agent_type=AgentType.ANALYSIS,
                input_data={
                    "query": query,
                    "context": context,
                    "retrieval_result": retrieval_dict,
                    "requirements": "分析查询内容和需求"
                },
                timeout=120.0
            )

            # 创建综合任务
            synthesis_task = AgentTask(
                id="synthesis_complex_light",
                agent_type=AgentType.SYNTHESIS,
                input_data={
                    "original_query": query,
                    "context": context,
                    "retrieval_result": retrieval_dict,
                    "requirements": "准备综合框架"
                },
                timeout=120.0
            )

            # 并行执行
            analysis_result, synthesis_prep = await asyncio.gather(
                self.agents[AgentType.ANALYSIS].execute(analysis_task),
                self.agents[AgentType.SYNTHESIS].execute(synthesis_task),
                return_exceptions=True
            )

            # 处理异常
            if isinstance(analysis_result, Exception):
                logger.error(f"Analysis 执行失败: {analysis_result}")
                self._raise_if_request_level_error(analysis_result)
                return await self._execute_default_strategy(query, context, fallback_reason="complex_light_analysis_exception")
            if isinstance(synthesis_prep, Exception):
                logger.error(f"Synthesis 准备失败: {synthesis_prep}")
                synthesis_prep = None

            # 最终综合
            final_synthesis_task = AgentTask(
                id="synthesis_final_light",
                agent_type=AgentType.SYNTHESIS,
                input_data={
                    "analysis_result": self._result_to_dict(analysis_result),
                    "synthesis_prep": self._result_to_dict(synthesis_prep) if synthesis_prep and not isinstance(synthesis_prep, Exception) else None,
                    "original_query": query,
                    "requirements": "整合分析结果，生成最终答案"
                },
                timeout=120.0
            )

            final_result = await self.agents[AgentType.SYNTHESIS].execute(final_synthesis_task)

            all_results = [analysis_result]
            if synthesis_prep and not isinstance(synthesis_prep, Exception):
                all_results.append(synthesis_prep)
            all_results.append(final_result)

            if final_result.success:
                return self._normalized_success_result(
                    final_result.output.get("response", ""),
                    subtask_results=all_results,
                    execution_metadata={
                        "total_tasks": len(all_results),
                        "successful_tasks": sum(1 for r in all_results if r.success),
                        "total_execution_time": sum(r.execution_time for r in all_results),
                        "strategy": "complex_light",
                        "optimization": "parallel_without_planning",
                    },
                )
            return self._normalized_failure_result(
                final_result.output.get("response", "") or "抱歉，处理您的请求时遇到困难。",
                error=final_result.output.get("error") if isinstance(final_result.output, dict) else None,
                subtask_results=all_results,
                execution_metadata={
                    "total_tasks": len(all_results),
                    "successful_tasks": sum(1 for r in all_results if r.success),
                    "total_execution_time": sum(r.execution_time for r in all_results),
                    "strategy": "complex_light",
                    "optimization": "parallel_without_planning",
                },
            )

        except Exception as e:
            logger.error(f"简单策略执行失败: {e}")
            self._raise_if_request_level_error(e)
            return self._normalized_failure_result(
                "抱歉，处理您的请求时遇到困难。",
                error=str(e),
                execution_metadata={"strategy": "complex_light"},
            )

    def _parse_planning_result(self, planning_output: Dict[str, Any], main_query: str, context: Dict[str, Any]) -> List[AgentTask]:
        """解析规划结果并生成子任务"""
        subtasks = []

        try:
            response = planning_output.get("response", "")

            # 简单的任务分解逻辑
            if "检索" in response or "搜索" in response:
                subtasks.append(AgentTask(
                    id="retrieval_task",
                    agent_type=AgentType.RETRIEVAL,
                    input_data={"query": main_query, "context": context},
                    timeout=120.0
                ))

            if "分析" in response:
                subtasks.append(AgentTask(
                    id="analysis_task",
                    agent_type=AgentType.ANALYSIS,
                    input_data={"query": main_query, "context": context, "requirements": "深入分析查询内容"},
                    dependencies=["retrieval_task"],
                    timeout=120.0
                ))

            if not subtasks:
                subtasks.append(AgentTask(
                    id="default_retrieval",
                    agent_type=AgentType.RETRIEVAL,
                    input_data={"query": main_query, "context": context},
                    timeout=120.0
                ))

        except Exception as e:
            logger.warning(f"解析规划结果失败: {e}")
            subtasks = [AgentTask(
                id="fallback_retrieval",
                agent_type=AgentType.RETRIEVAL,
                input_data={"query": main_query, "context": context},
                timeout=60.0
            )]

        return subtasks

    def _build_decomposition_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            subtasks = self.task_decomposer.decompose(query, context)
            ordered_subtasks = self.task_decomposer.optimize_task_order(subtasks)
            execution_plan = self.task_decomposer.estimate_execution_plan(ordered_subtasks)
            return {
                "enabled": True,
                "subtasks": ordered_subtasks,
                "execution_plan": execution_plan,
            }
        except Exception as e:
            logger.warning(f"任务分解失败，继续使用原复杂路径: {e}")
            return {
                "enabled": False,
                "error": str(e),
                "subtasks": [],
                "execution_plan": None,
            }

    def _merge_decomposition_subtasks(
        self,
        planned_tasks: List[AgentTask],
        decomposition_subtasks: List[Any],
        main_query: str,
        context: Dict[str, Any],
    ) -> List[AgentTask]:
        merged_tasks = list(planned_tasks)
        existing_ids = {task.id for task in merged_tasks}

        for subtask in decomposition_subtasks:
            agent_task = self._subtask_to_agent_task(subtask, main_query, context)
            if not agent_task:
                continue
            if agent_task.id in existing_ids:
                continue
            merged_tasks.append(agent_task)
            existing_ids.add(agent_task.id)

        return merged_tasks or [
            AgentTask(
                id="fallback_retrieval",
                agent_type=AgentType.RETRIEVAL,
                input_data={"query": main_query, "context": context},
                timeout=60.0,
            )
        ]

    def _subtask_to_agent_task(self, subtask: Any, main_query: str, context: Dict[str, Any]) -> AgentTask | None:
        task_type_map = {
            TaskType.RETRIEVAL: AgentType.RETRIEVAL,
            TaskType.ANALYSIS: AgentType.ANALYSIS,
            TaskType.SYNTHESIS: AgentType.SYNTHESIS,
            TaskType.VERIFICATION: AgentType.VERIFICATION,
            TaskType.COMPARISON: AgentType.ANALYSIS,
        }
        agent_type = task_type_map.get(getattr(subtask, "type", None))
        if agent_type is None:
            return None

        dependencies = list(getattr(subtask, "dependencies", []) or [])
        subtask_query = getattr(subtask, "query", "") or main_query
        description = getattr(subtask, "description", "") or subtask_query
        estimated_complexity = float(getattr(subtask, "estimated_complexity", 1.0) or 1.0)
        timeout = min(max(estimated_complexity * 30.0, 60.0), 180.0)
        metadata = {
            "decomposition_task": True,
            "task_type": getattr(getattr(subtask, "type", None), "value", str(getattr(subtask, "type", ""))),
            "description": description,
            "original_query": main_query,
        }

        if agent_type == AgentType.RETRIEVAL:
            input_data = {"query": subtask_query, "context": context, "requirements": description}
        elif agent_type == AgentType.ANALYSIS:
            input_data = {"query": main_query, "context": context, "focus": subtask_query, "requirements": description}
        elif agent_type == AgentType.SYNTHESIS:
            input_data = {"original_query": main_query, "context": context, "requirements": description, "focus": subtask_query}
        else:
            input_data = {"query": main_query, "context": context, "requirements": description, "focus": subtask_query}

        return AgentTask(
            id=getattr(subtask, "id", description) or description,
            agent_type=agent_type,
            input_data=input_data,
            timeout=timeout,
            dependencies=dependencies,
        )

    def _build_analysis_batch_prompt(self, tasks: List[AgentTask]) -> str:
        shared_context = None
        task_payload = []
        for task in tasks:
            context_value = task.input_data.get("context_bundle")
            if context_value is None:
                context_value = task.input_data.get("context")
            if shared_context is None and context_value is not None:
                shared_context = context_value

            task_payload.append({
                "task_id": task.id,
                "query": task.input_data.get("query"),
                "focus": task.input_data.get("focus"),
                "requirements": task.input_data.get("requirements"),
                "description": task.input_data.get("description") or task.input_data.get("requirements") or task.input_data.get("query"),
                "dependencies": list(task.dependencies or []),
            })

        payload = _serialize_for_json({"shared_context": shared_context, "tasks": task_payload})
        return (
            "你是专业的信息分析专家。现在需要一次性完成多个同层分析子任务。\n\n"
            "请严格遵守：\n"
            "1. 只输出 JSON 数组，不要输出解释、Markdown 或代码块。\n"
            "2. 数组中的每一项必须对应一个 task_id。\n"
            "3. 不要漏项，不要新增 task_id。\n"
            "4. 如果某个任务无法完成，success 设为 false，并在 issues 中说明原因。\n\n"
            "输入如下：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "输出格式示例：\n"
            "[\n"
            "  {\n"
            "    \"task_id\": \"analysis_1\",\n"
            "    \"success\": true,\n"
            "    \"response\": \"...\",\n"
            "    \"key_points\": [{\"concept\": \"...\", \"definition\": \"...\"}],\n"
            "    \"issues\": [],\n"
            "    \"confidence\": 0.82\n"
            "  }\n"
            "]"
        )

    def _extract_json_array(self, text: str) -> Any:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("results", "items", "tasks", "data"):
                    if isinstance(parsed.get(key), list):
                        return parsed.get(key)
        except Exception:
            pass

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        return None

    def _build_analysis_batch_result(self, task: AgentTask, item: dict, batch_duration: float, batch_size: int, index: int, batch_task_ids: List[str]) -> AgentResult:
        response_text = str(item.get("response") or item.get("summary") or item.get("final_answer") or "").strip()
        if not response_text:
            response_text = "未生成分析结果"

        key_points = item.get("key_points") or []
        normalized_key_points = []
        if isinstance(key_points, list):
            for kp in key_points:
                if isinstance(kp, dict):
                    normalized_key_points.append(kp)
                elif isinstance(kp, str) and kp.strip():
                    text = kp.strip()
                    normalized_key_points.append({"concept": text[:120], "definition": text[:300]})

        issues = item.get("issues") or []
        normalized_issues = [str(issue)[:200] for issue in issues if str(issue).strip()] if isinstance(issues, list) else []
        evidence = item.get("evidence") or []

        output = {
            "task_id": task.id,
            "response": response_text,
            "summary": response_text,
            "final_answer": response_text,
            "key_points": normalized_key_points,
            "issues": normalized_issues,
        }
        if isinstance(evidence, list) and evidence:
            output["evidence"] = evidence
        if "confidence" in item:
            output["confidence"] = item.get("confidence")
        if "success" in item:
            output["success"] = bool(item.get("success"))

        return AgentResult(
            task_id=task.id,
            agent_type=AgentType.ANALYSIS,
            success=bool(item.get("success", True)),
            output=output,
            metadata={
                "batch_analysis": True,
                "batch_task_ids": batch_task_ids,
                "batch_size": batch_size,
                "batch_task_id": task.id,
                "batch_index": index,
                "batch_duration_ms": round(batch_duration * 1000, 2),
            },
            execution_time=batch_duration / max(1, batch_size),
        )

    async def _execute_analysis_batch(self, tasks: List[AgentTask]) -> List[AgentResult]:
        analysis_agent = self.agents.get(AgentType.ANALYSIS)
        if not analysis_agent or not analysis_agent.agent:
            logger.warning("Analysis Agent 未初始化，降级到逐个执行")
            return await asyncio.gather(*(self._execute_agent_task(task) for task in tasks))

        batch_prompt = self._build_analysis_batch_prompt(tasks)
        started_at = time.perf_counter()

        try:
            raw_response = await analysis_agent.agent.ainvoke(
                {"messages": [HumanMessage(content=batch_prompt)]},
                config={"recursion_limit": 3},
            )
            response_text = analysis_agent._process_output(raw_response).get("response", "")
            parsed_items = self._extract_json_array(response_text)
            if not isinstance(parsed_items, list):
                raise ValueError("batch analysis response is not a JSON array")

            item_map: Dict[str, dict] = {}
            for idx, item in enumerate(parsed_items):
                if not isinstance(item, dict):
                    continue
                task_id = str(item.get("task_id") or (tasks[idx].id if idx < len(tasks) else "")).strip()
                if task_id:
                    item_map[task_id] = item

            missing_task_ids = [task.id for task in tasks if task.id not in item_map]
            if missing_task_ids:
                raise ValueError(f"batch analysis missing task_ids: {missing_task_ids}")

            batch_duration = time.perf_counter() - started_at
            batch_task_ids = [task.id for task in tasks]
            results: List[AgentResult] = []
            for index, task in enumerate(tasks):
                results.append(self._build_analysis_batch_result(task, item_map[task.id], batch_duration, len(tasks), index, batch_task_ids))
            return results

        except Exception as e:
            logger.warning(f"Analysis 批量执行失败，降级到逐个执行: {e}")
            return await asyncio.gather(*(self._execute_agent_task(task) for task in tasks))

    async def _execute_subtasks(self, subtasks: List[AgentTask]) -> List[AgentResult]:
        """执行子任务，优先批量合并 Analysis 任务"""
        if not subtasks:
            return []

        results_by_task_id: Dict[str, AgentResult] = {}
        completed_tasks = set()
        max_loop = len(subtasks) * 3
        loop_count = 0

        while len(completed_tasks) < len(subtasks) and loop_count < max_loop:
            loop_count += 1
            ready_tasks = []
            for task in subtasks:
                if task.id not in completed_tasks and all(dep in completed_tasks for dep in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                remaining = [task for task in subtasks if task.id not in completed_tasks]
                if remaining:
                    ready_tasks = [remaining[0]]
                else:
                    break

            analysis_tasks = [task for task in ready_tasks if task.agent_type == AgentType.ANALYSIS]
            other_tasks = [task for task in ready_tasks if task.agent_type != AgentType.ANALYSIS]
            batch_results: List[AgentResult] = []
            other_results: List[AgentResult] = []

            if len(analysis_tasks) > 1:
                coroutines = [self._execute_analysis_batch(analysis_tasks)]
                coroutines.extend(self._execute_agent_task(task) for task in other_tasks)
                gathered = await asyncio.gather(*coroutines, return_exceptions=True)

                batch_value = gathered[0]
                if isinstance(batch_value, Exception):
                    logger.warning(f"Analysis 批量执行异常，降级到逐个执行: {batch_value}")
                    batch_results = await asyncio.gather(*(self._execute_agent_task(task) for task in analysis_tasks), return_exceptions=True)
                else:
                    batch_results = batch_value
                    if not isinstance(batch_results, list):
                        logger.warning("Analysis 批量执行返回格式异常，降级到逐个执行")
                        batch_results = await asyncio.gather(*(self._execute_agent_task(task) for task in analysis_tasks), return_exceptions=True)

                other_raw_results = gathered[1:]
                for task, result in zip(other_tasks, other_raw_results):
                    if isinstance(result, Exception):
                        other_results.append(AgentResult(
                            task_id=task.id,
                            agent_type=task.agent_type,
                            success=False,
                            output={"error": str(result)},
                            execution_time=0.0,
                        ))
                    else:
                        other_results.append(result)
            else:
                single_results = await asyncio.gather(*(self._execute_agent_task(task) for task in ready_tasks), return_exceptions=True)
                for task, result in zip(ready_tasks, single_results):
                    if isinstance(result, Exception):
                        other_results.append(AgentResult(
                            task_id=task.id,
                            agent_type=task.agent_type,
                            success=False,
                            output={"error": str(result)},
                            execution_time=0.0,
                        ))
                    else:
                        other_results.append(result)

            if len(analysis_tasks) > 1:
                normalized_analysis_results: List[AgentResult] = []
                for task, result in zip(analysis_tasks, batch_results):
                    if isinstance(result, Exception):
                        normalized_analysis_results.append(AgentResult(
                            task_id=task.id,
                            agent_type=task.agent_type,
                            success=False,
                            output={"error": str(result)},
                            execution_time=0.0,
                        ))
                    else:
                        normalized_analysis_results.append(result)
                normalized_results = [*normalized_analysis_results, *other_results]
            else:
                normalized_results = other_results

            for result in normalized_results:
                results_by_task_id[result.task_id] = result
                completed_tasks.add(result.task_id)

        return [results_by_task_id.get(task.id, AgentResult(
            task_id=task.id,
            agent_type=task.agent_type,
            success=False,
            output={"error": "任务未执行"},
            execution_time=0.0,
        )) for task in subtasks]

    async def _execute_agent_task(self, task: AgentTask) -> AgentResult:
        agent = self.agents.get(task.agent_type)
        if not agent:
            return AgentResult(
                task_id=task.id,
                agent_type=task.agent_type,
                success=False,
                output={"error": f"agent_not_found:{task.agent_type.value}"},
                execution_time=0.0,
            )
        return await agent.execute(task)

    def _is_request_level_error(self, error: Exception) -> bool:
        message = str(error)
        if re.search(r"Error code:\s*(401|403|429|5\d\d)", message):
            return True
        lowered = message.lower()
        return any(token in lowered for token in ("timeout", "connection", "embedding", "milvus"))

    def _normalized_failure_result(
        self,
        final_answer: str,
        *,
        error: str | None = None,
        fallback_reason: str | None = None,
        request_error: str | None = None,
        subtask_results: list | None = None,
        planning: dict | None = None,
        verification: dict | None = None,
        execution_metadata: dict | None = None,
        rag_trace: dict | None = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "final_answer": final_answer,
            "rag_trace": rag_trace,
            "subtask_results": subtask_results or [],
            "planning": planning,
            "verification": verification,
            "execution_metadata": execution_metadata or {},
            "fallback_reason": fallback_reason,
            "error": error,
            "request_error": request_error,
        }

    def _normalized_success_result(
        self,
        final_answer: str,
        *,
        subtask_results: list | None = None,
        planning: dict | None = None,
        verification: dict | None = None,
        execution_metadata: dict | None = None,
        rag_trace: dict | None = None,
        fallback_reason: str | None = None,
        error: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "final_answer": final_answer,
            "rag_trace": rag_trace,
            "subtask_results": subtask_results or [],
            "planning": planning,
            "verification": verification,
            "execution_metadata": execution_metadata or {},
            "fallback_reason": fallback_reason,
            "error": error,
            "request_error": None,
        }

    def _raise_if_request_level_error(self, error: Exception):
        if self._is_request_level_error(error):
            raise RequestLevelOrchestrationError(str(error)) from error

    def _result_to_dict(self, result: AgentResult) -> Dict[str, Any]:
        """将AgentResult转换为字典"""
        return _serialize_for_json({
            "task_id": result.task_id,
            "agent_type": result.agent_type,
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
            "execution_time": result.execution_time
        })


# 全局多Agent协调器
_multi_agent_orchestrator = None
# 修复：全局单例加线程锁
_global_agent_lock = threading.Lock()


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """获取全局多Agent协调器"""
    global _multi_agent_orchestrator
    with _global_agent_lock:
        if _multi_agent_orchestrator is None:
            _multi_agent_orchestrator = MultiAgentOrchestrator()
    return _multi_agent_orchestrator

# 修复：兼容异步事件循环，不破坏原有同步调用接口
def coordinate_multi_agent_task(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """便捷函数：协调多Agent任务"""
    orchestrator = get_multi_agent_orchestrator()
    # 保留原有接口形态，改用安全的异步运行方式
    try:
        return asyncio.run(orchestrator.coordinate_task(query, context))
    except Exception as e:
        logger.error(f"同步调用多Agent任务异常: {e}")
        return {
            "success": False,
            "final_answer": "任务执行异常",
            "error": str(e)
        }