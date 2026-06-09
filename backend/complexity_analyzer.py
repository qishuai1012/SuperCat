"""
最终企业版 复杂度分析器
✅ 规则兜底
✅ m3e-base 语义判断（真正懂中文）
✅ 无硬编码、不按长度/标点乱判
✅ 接口 100% 兼容旧代码
"""
import re
import numpy as np
from typing import Dict, Any, Optional
from query_understanding.types import QueryComplexity

try:
    # sentence_transformers 是一个专门做文本语义理解的 Python 库
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


class ComplexityAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        # 问候
        self.greetings = {'hi', 'hello', '你好', '您好', '嗨', '哈喽'}

        # 简单问题
        self.simple_patterns = [
            re.compile(r'^(什么是|谁是|哪里|何时|解释|定义|介绍)\s*.+[？?]?$', re.I),
        ]

        # 中等问题
        self.medium_patterns = [
            re.compile(r'^(查询|查看|获取|告诉我|有没有|是否)\s*.+', re.I),
        ]

        # 轻度复杂关键词
        self.light_words = {
            '为什么', '如何', '分析', '比较', '区别', '原理', '优缺点'
        }

        # 重度复杂关键词
        self.heavy_words = {
            '详细分析', '架构设计', '技术选型', '实现方案', '权衡', '深入探讨'
        }

        # 加载 m3e-base 模型
        self.model = None
        if MODEL_AVAILABLE and model_path:
            self.model = SentenceTransformer(model_path)

        # 4 类标准意图（语义判断基准）
        self.standard_questions = {
            "SIMPLE": ["什么是RAG", "介绍一下Python", "这篇文章讲的是什么", "这个文档说了什么", "是什么内容", "知识库里有这个吗"],
            "MEDIUM": ["查询用户列表", "查看系统状态", "获取最新数据", "怎么安装", "如何配置"],
            "COMPLEX_LIGHT": ["对比两种方案区别", "解释原理", "分析优缺点", "为什么会这样"],
            "COMPLEX_HEAVY": ["设计系统架构", "技术选型与权衡", "详细分析实现方案", "从多个角度深入比较"]
        }

    def _get_embedding(self, text: str):
        if not self.model:
            return None
        return self.model.encode(text, normalize_embeddings=True)

    def _semantic_predict(self, query: str) -> Optional[QueryComplexity]:
        if not self.model:
            return None

        query_emb = self._get_embedding(query)
        max_score = -1
        best_level = None

        for level, questions in self.standard_questions.items():
            embs = self.model.encode(questions, normalize_embeddings=True)
            score = np.mean(np.dot(embs, query_emb.T))
            if score > max_score:
                max_score = score
                best_level = level

        if best_level == "SIMPLE":
            return QueryComplexity.SIMPLE
        elif best_level == "MEDIUM":
            return QueryComplexity.MEDIUM
        elif best_level == "COMPLEX_LIGHT":
            return QueryComplexity.COMPLEX_LIGHT
        elif best_level == "COMPLEX_HEAVY":
            return QueryComplexity.COMPLEX_HEAVY
        return None

    def analyze(self, query: str, context: Dict[str, Any] = None) -> QueryComplexity:
        query = query.strip()
        query_lower = query.lower()

        # 1. 问候 → 简单
        if query_lower in self.greetings:
            return QueryComplexity.SIMPLE

        # 2. 规则简单问题
        for pat in self.simple_patterns:
            if pat.match(query):
                return QueryComplexity.SIMPLE

        # 3. 模型语义判断（核心！）
        sem_level = self._semantic_predict(query)
        if sem_level:
            return sem_level

        # 4. 重度关键词
        for w in self.heavy_words:
            if w in query_lower:
                return QueryComplexity.COMPLEX_HEAVY

        # 5. 轻度关键词
        if any(w in query_lower for w in self.light_words):
            return QueryComplexity.COMPLEX_LIGHT

        # 6. 中等规则
        for pat in self.medium_patterns:
            if pat.match(query):
                return QueryComplexity.MEDIUM

        return QueryComplexity.MEDIUM


# 全局单例
_analyzer: Optional[ComplexityAnalyzer] = None

def get_complexity_analyzer(model_path: str = None):
    global _analyzer
    if _analyzer is None:
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "m3e-base")
        _analyzer = ComplexityAnalyzer(model_path)
    return _analyzer