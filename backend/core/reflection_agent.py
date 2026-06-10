"""
反思机制系统
负责评估答案质量并实现自我修正
"""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
import json
import logging
import asyncio
from enum import Enum
import threading

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class QualityDimension(Enum):
    """质量评估维度"""
    RELEVANCE = "relevance"      # 相关性
    COMPLETENESS = "completeness" # 完整性
    ACCURACY = "accuracy"        # 准确性
    CLARITY = "clarity"          # 清晰度
    DEPTH = "depth"              # 深度


class ReflectionAction(Enum):
    """反思后采取的行动"""
    ACCEPT = "accept"           # 接受当前答案
    REVISE = "revise"           # 修订答案
    RESEARCH = "research"       # 需要进一步研究
    CLARIFY = "clarify"         # 需要澄清问题


@dataclass
class QualityAssessment:
    """质量评估结果"""
    dimension: QualityDimension
    score: float  # 0-1分
    feedback: str
    suggestions: List[str]


@dataclass
class ReflectionResult:
    """反思结果"""
    overall_score: float
    quality_assessments: List[QualityAssessment]
    action: ReflectionAction
    reasoning: str
    revision_suggestions: List[str]
    confidence: float


class ReflectionAgent:
    """
    反思Agent - 评估答案质量并建议改进
    """

    def __init__(self):
        self.reflection_model = None
        self._init_reflection_model()

        # 质量评估提示词
        self.assessment_prompt = """
        你是一个专业的答案质量评估专家，负责评估AI生成的答案质量。

        请评估以下答案的质量：

        原始问题: {original_question}
        生成答案: {generated_answer}
        检索上下文: {retrieved_context}
        用户背景: {user_context}

        请从以下维度进行评估（每个维度给出0-1的分数和具体反馈）：

        1. 相关性 (Relevance): 答案是否直接回应了用户问题
        2. 完整性 (Completeness): 答案是否涵盖了问题的所有方面
        3. 准确性 (Accuracy): 答案中的信息是否准确无误
        4. 清晰度 (Clarity): 答案是否表达清晰、易于理解
        5. 深度 (Depth): 答案是否提供了足够的深度和细节

        基于评估结果，请决定下一步行动：
        - accept: 答案质量良好，可以直接返回给用户
        - revise: 答案有改进空间，需要修订
        - research: 需要进一步研究获取更多信息
        - clarify: 需要澄清用户的原始问题

        请返回JSON格式评估结果：
        {{
            "overall_score": 总体分数(0-1),
            "assessments": [
                {{
                    "dimension": "评估维度",
                    "score": 分数(0-1),
                    "feedback": "具体反馈",
                    "suggestions": ["改进建议"]
                }}
            ],
            "action": "建议行动",
            "reasoning": "决策理由",
            "revision_suggestions": ["具体修订建议"],
            "confidence": 评估置信度(0-1)
        }}
        """

        # 修订提示词
        self.revision_prompt = """
        你是一个专业的答案修订专家，负责改进已有的答案。

        原始问题: {original_question}
        当前答案: {current_answer}
        质量评估: {quality_assessment}
        修订建议: {revision_suggestions}
        可用上下文: {context}

        请基于评估反馈修订答案，改进以下方面：
        1. 提高相关性和准确性
        2. 填补信息空白
        3. 改善表达清晰度
        4. 增加必要的深度和细节

        要求：
        - 保持答案的核心正确信息
        - 修正发现的问题
        - 确保修订后的答案更全面、准确
        - 避免引入新的错误

        请提供修订后的完整答案。
        """

    def _init_reflection_model(self):
        """初始化反思模型"""
        try:
            if API_KEY and MODEL:
                self.reflection_model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.1,  # 低温度确保评估一致性
                    stream_usage=True,
                )
        except Exception as e:
            logger.warning(f"反思模型初始化失败: {e}")
            self.reflection_model = None

    def _rule_based_assessment(self, question: str, answer: str, context: str = None) -> ReflectionResult:
        """基于规则的质量评估fallback"""
        assessments = []

        # 相关性评估
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        relevance_score = len(question_keywords.intersection(answer_keywords)) / max(len(question_keywords), 1)

        assessments.append(QualityAssessment(
            dimension=QualityDimension.RELEVANCE,
            score=min(relevance_score, 1.0),
            feedback=f"相关性评估: 关键词匹配度 {relevance_score:.2f}",
            suggestions=["增加更多与问题直接相关的信息"] if relevance_score < 0.6 else []
        ))

        # 完整性评估
        answer_length = len(answer)
        completeness_score = min(answer_length / 200, 1.0)  # 假设200字为完整答案

        assessments.append(QualityAssessment(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            feedback=f"完整性评估: 答案长度 {answer_length} 字符",
            suggestions=["提供更详细的解释和例子"] if completeness_score < 0.6 else []
        ))

        # 清晰度评估
        clarity_indicators = ['首先', '其次', '最后', '因此', '所以', '例如', '具体来说']
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in answer) / len(clarity_indicators)

        assessments.append(QualityAssessment(
            dimension=QualityDimension.CLARITY,
            score=clarity_score,
            feedback=f"清晰度评估: 逻辑连接词使用 {clarity_score:.2f}",
            suggestions=["增加逻辑连接词，改善表达结构"] if clarity_score < 0.5 else []
        ))

        # 准确性评估（简单启发式）
        accuracy_score = 0.8  # 默认较高，实际应用中需要更复杂的评估
        if len(answer) < 50:
            accuracy_score = 0.5  # 过短的答案可能不准确

        assessments.append(QualityAssessment(
            dimension=QualityDimension.ACCURACY,
            score=accuracy_score,
            feedback=f"准确性评估: 基于长度的启发式评估 {accuracy_score:.2f}",
            suggestions=["提供更具体、可验证的信息"] if accuracy_score < 0.7 else []
        ))

        # 深度评估
        depth_indicators = ['深入分析', '根本原因', '机制', '原理', '本质']
        depth_score = sum(1 for indicator in depth_indicators if indicator in answer) / len(depth_indicators)

        assessments.append(QualityAssessment(
            dimension=QualityDimension.DEPTH,
            score=depth_score,
            feedback=f"深度评估: 深度分析指标 {depth_score:.2f}",
            suggestions=["增加深入分析和原理解释"] if depth_score < 0.4 else []
        ))

        # 总体评估
        overall_score = sum(assessment.score for assessment in assessments) / len(assessments)

        # 决定行动
        if overall_score >= 0.8:
            action = ReflectionAction.ACCEPT
        elif overall_score >= 0.6:
            action = ReflectionAction.REVISE
        elif overall_score >= 0.4:
            action = ReflectionAction.RESEARCH
        else:
            action = ReflectionAction.CLARIFY

        revision_suggestions = []
        for assessment in assessments:
            revision_suggestions.extend(assessment.suggestions)

        return ReflectionResult(
            overall_score=overall_score,
            quality_assessments=assessments,
            action=action,
            reasoning=f"基于规则评估，总体分数: {overall_score:.2f}",
            revision_suggestions=revision_suggestions,
            confidence=0.7
        )

    def assess_answer_quality(self,
                            question: str,
                            answer: str,
                            context: str = None,
                            user_context: Dict[str, Any] = None) -> ReflectionResult:
        """
        评估答案质量

        Args:
            question: 原始问题
            answer: 生成的答案
            context: 检索到的上下文
            user_context: 用户背景信息

        Returns:
            质量评估结果
        """
        if user_context is None:
            user_context = {}

        # 如果反思模型不可用，使用规则基础评估
        if not self.reflection_model:
            return self._rule_based_assessment(question, answer, context)

        try:
            # 准备评估输入
            assessment_input = self.assessment_prompt.format(
                original_question=question,
                generated_answer=answer,
                retrieved_context=context or "无额外上下文",
                user_context=json.dumps(user_context, ensure_ascii=False, indent=2)
            )

            # 调用反思模型
            response = self.reflection_model.invoke(assessment_input)

            # 解析评估结果
            assessment_result = self._parse_assessment_response(response.content)

            # 转换为ReflectionResult对象
            return self._create_reflection_result(assessment_result)

        except Exception as e:
            logger.warning(f"智能质量评估失败，使用规则基础评估: {e}")
            return self._rule_based_assessment(question, answer, context)

    def _parse_assessment_response(self, response_content: str) -> Dict[str, Any]:
        """解析评估模型响应"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("无法解析JSON响应")
        except Exception as e:
            logger.warning(f"解析评估响应失败: {e}")
            return self._get_default_assessment()

    def _get_default_assessment(self) -> Dict[str, Any]:
        """获取默认评估结果"""
        return {
            "overall_score": 0.7,
            "assessments": [],
            "action": "accept",
            "reasoning": "默认评估",
            "revision_suggestions": [],
            "confidence": 0.5
        }

    def _create_reflection_result(self, assessment_data: Dict[str, Any]) -> ReflectionResult:
        """创建反思结果对象"""
        try:
            # 转换质量评估
            assessments = []
            for assessment in assessment_data.get("assessments", []):
                dimension_map = {
                    "relevance": QualityDimension.RELEVANCE,
                    "completeness": QualityDimension.COMPLETENESS,
                    "accuracy": QualityDimension.ACCURACY,
                    "clarity": QualityDimension.CLARITY,
                    "depth": QualityDimension.DEPTH
                }

                dimension = dimension_map.get(assessment.get("dimension", "relevance").lower(), QualityDimension.RELEVANCE)

                quality_assessment = QualityAssessment(
                    dimension=dimension,
                    score=float(assessment.get("score", 0.5)),
                    feedback=assessment.get("feedback", ""),
                    suggestions=assessment.get("suggestions", [])
                )
                assessments.append(quality_assessment)

            # 转换行动
            action_map = {
                "accept": ReflectionAction.ACCEPT,
                "revise": ReflectionAction.REVISE,
                "research": ReflectionAction.RESEARCH,
                "clarify": ReflectionAction.CLARIFY
            }

            action = action_map.get(assessment_data.get("action", "accept").lower(), ReflectionAction.ACCEPT)

            return ReflectionResult(
                overall_score=float(assessment_data.get("overall_score", 0.5)),
                quality_assessments=assessments,
                action=action,
                reasoning=assessment_data.get("reasoning", ""),
                revision_suggestions=assessment_data.get("revision_suggestions", []),
                confidence=float(assessment_data.get("confidence", 0.5))
            )

        except Exception as e:
            logger.warning(f"创建反思结果失败: {e}")
            return ReflectionResult(
                overall_score=0.5,
                quality_assessments=[],
                action=ReflectionAction.ACCEPT,
                reasoning="创建结果失败",
                revision_suggestions=[],
                confidence=0.3
            )

    def revise_answer(self,
                     question: str,
                     current_answer: str,
                     reflection_result: ReflectionResult,
                     context: str = None) -> str:
        """
        基于反思结果修订答案

        Args:
            question: 原始问题
            current_answer: 当前答案
            reflection_result: 反思结果
            context: 可用上下文

        Returns:
            修订后的答案
        """
        if reflection_result.action != ReflectionAction.REVISE:
            return current_answer

        if not self.reflection_model:
            return current_answer

        try:
            # 准备修订输入
            revision_input = self.revision_prompt.format(
                original_question=question,
                current_answer=current_answer,
                quality_assessment=json.dumps({
                    "overall_score": reflection_result.overall_score,
                    "feedback": [assessment.feedback for assessment in reflection_result.quality_assessments]
                }, ensure_ascii=False, indent=2),
                revision_suggestions=json.dumps(reflection_result.revision_suggestions, ensure_ascii=False, indent=2),
                context=context or "无额外上下文"
            )

            # 调用修订
            response = self.reflection_model.invoke(revision_input)
            return response.content

        except Exception as e:
            logger.warning(f"答案修订失败: {e}")
            return current_answer

    def create_improvement_plan(self, reflection_result: ReflectionResult) -> Dict[str, Any]:
        """
        创建改进计划

        Args:
            reflection_result: 反思结果

        Returns:
            改进计划
        """
        plan = {
            "action": reflection_result.action.value,
            "priority": "high" if reflection_result.overall_score < 0.5 else "medium",
            "steps": [],
            "estimated_effort": "low"
        }

        if reflection_result.action == ReflectionAction.REVISE:
            plan["steps"] = [
                "分析质量评估反馈",
                "识别主要改进点",
                "收集额外信息（如果需要）",
                "修订答案内容",
                "验证修订结果"
            ]
            plan["estimated_effort"] = "medium"

        elif reflection_result.action == ReflectionAction.RESEARCH:
            plan["steps"] = [
                "确定信息缺口",
                "执行额外检索",
                "验证信息准确性",
                "整合新信息",
                "生成改进答案"
            ]
            plan["estimated_effort"] = "high"

        elif reflection_result.action == ReflectionAction.CLARIFY:
            plan["steps"] = [
                "分析问题歧义",
                "生成澄清问题",
                "等待用户反馈",
                "基于澄清重新回答"
            ]
            plan["estimated_effort"] = "medium"

        return plan


# 全局反思Agent实例
_reflection_agent = None
_reflection_lock = threading.Lock()


def get_reflection_agent() -> ReflectionAgent:
    """获取全局反思Agent实例"""
    global _reflection_agent
    with _reflection_lock:
        if _reflection_agent is None:
            _reflection_agent = ReflectionAgent()
    return _reflection_agent


def assess_answer(question: str, answer: str, context: str = None) -> ReflectionResult:
    """便捷函数：评估答案质量"""
    agent = get_reflection_agent()
    return agent.assess_answer_quality(question, answer, context)