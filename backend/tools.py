from typing import Optional
from contextvars import ContextVar
import asyncio
import threading
import os
import logging
import requests
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

KB_NO_RESULT_SENTINEL = "KB_NO_RESULT"
KB_NO_RESULT_MESSAGE = "未在知识库中找到足够依据来回答这个问题。请换个问法、补充关键词，或先上传相关资料。"


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _build_kb_no_result_suggestions(query: str, rag_trace: dict | None = None) -> list[str]:
    rag_trace = rag_trace or {}
    query = " ".join(str(query or "").split())
    suggestions = []

    if query:
        if len(query) > 24:
            suggestions.append("把问题改短一点，只保留最关键的名词、接口名、报错词或配置项。")
        if len(query) <= 12:
            suggestions.append("给问题补充业务对象、模块名或上下文，避免问题过短。")
        if not any(token in query for token in ("报错", "错误", "异常", "接口", "函数", "类", "字段", "配置", "日志", "文件", "表", "索引")):
            suggestions.append("补充更具体的定位词，例如文档名、接口名、函数名、类名、字段名或报错信息。")
        if any(token in query for token in ("怎么", "如何", "为什么", "原因", "排查")):
            suggestions.append("先改问相关资料里提到了哪些模块、接口、配置或错误现象，再继续追问原因或步骤。")

    suggestions.append("可以换用同义词、英文术语、缩写或全称后再试一次。")

    if rag_trace.get("step_back_generated") or rag_trace.get("hyde_generated_count"):
        suggestions.append("如果这是代码或文档问题，直接贴模块名、目录名、报错原文或关键配置会更容易命中。")

    return _dedupe_preserve_order(suggestions)[:3]


def _build_kb_no_result_explanation(rag_trace: dict | None = None) -> str:
    rag_trace = rag_trace or {}
    reason = rag_trace.get("reason")
    if reason == "all_docs_below_similarity_threshold":
        return "检索到了一些候选片段，但相关度都不够高，无法作为可靠依据。"
    if rag_trace.get("retrieval_mode") == "failed":
        return "这次检索流程没有成功返回可用片段。"
    if rag_trace.get("filtered_count") and not rag_trace.get("retained_count"):
        return "候选片段在相似度过滤后全部被淘汰了。"
    if rag_trace.get("expanded_retrieval_count", 1) > 1:
        return "已经尝试了原问题和扩展查询，但仍未找到足够相关的依据。"
    return "知识库里没有检索到足够相关的片段。"


def _build_kb_no_result_context(query: str, rag_trace: dict | None = None) -> dict:
    rag_trace = dict(rag_trace or {})
    return {
        "rag_trace": rag_trace,
        "kb_no_result": True,
        "kb_no_result_reason": rag_trace.get("reason") or rag_trace.get("hit_reason") or "no_relevant_docs",
        "kb_no_result_explanation": _build_kb_no_result_explanation(rag_trace),
        "kb_no_result_suggestions": _build_kb_no_result_suggestions(query, rag_trace),
    }


def should_skip_grading(query: str) -> bool:
    """返回是否命中“可考虑跳过 grader”的轻量启发式信号。"""
    query = (query or "").strip()
    if not query:
        return True

    from complexity_analyzer import get_complexity_analyzer

    complexity = get_complexity_analyzer().analyze(query).value
    if complexity == "simple":
        return True

    factual_prefixes = ("什么是", "谁是", "哪里", "哪个", "何时", "多少", "介绍", "解释", "定义")
    if len(query) <= 20 and query.startswith(factual_prefixes):
        return True

    if len(query) <= 12 and all(token not in query for token in ("为什么", "如何", "分析", "比较", "评估", "综合")):
        return True

    return False

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# --- 请求级隔离状态 ---
#
# _RAG_STEP_QUEUE / _RAG_STEP_LOOP：在协程中写入，在线程中读取。
#   asyncio.to_thread 会复制当前 context，因此 ContextVar 天然隔离多请求，线程可安全读取。
#
# rag_context / call_count：在线程中写入，需要对协程可见（tool 返回后读取）。
#   ContextVar 在线程中写入后不会同步回协程，改用 asyncio Task 对象属性：
#   asyncio.to_thread 复制 context 时也复制了 current_task() 的引用，
#   对 Task 属性的修改是对同一对象的 mutation，协程可直接看到。

_RAG_STEP_QUEUE: ContextVar = ContextVar('rag_step_queue', default=None)
_RAG_STEP_LOOP: ContextVar = ContextVar('rag_step_loop', default=None)

# 非 async 上下文（单元测试、直接调用）的兜底存储
_thread_local = threading.local()


def _task_store() -> dict:
    """返回当前请求的可变状态字典。协程和 asyncio.to_thread 线程均可读写。"""
    try:
        task = asyncio.current_task()
        if task is not None:
            if not hasattr(task, '_tools_state'):
                task._tools_state = {'rag_context': None, 'call_count': 0, 'rag_options': {}}
            return task._tools_state
    except RuntimeError:
        # 线程池中没有事件循环，降级到线程本地存储
        pass
    # 兜底：同步测试或非 async 调用场景
    if not hasattr(_thread_local, '_tools_state'):
        _thread_local._tools_state = {'rag_context': None, 'call_count': 0, 'rag_options': {}}
    return _thread_local._tools_state


def _set_last_rag_context(context: dict):
    _task_store()['rag_context'] = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    store = _task_store()
    context = store.get('rag_context')
    if clear:
        store['rag_context'] = None
    return context



def set_rag_options(**options):
    store = _task_store()
    current = dict(store.get('rag_options') or {})
    current.update({k: v for k, v in options.items() if v is not None})
    store['rag_options'] = current



def get_rag_options(clear: bool = False) -> dict:
    store = _task_store()
    options = dict(store.get('rag_options') or {})
    if clear:
        store['rag_options'] = {}
    return options


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    store = _task_store()
    store['call_count'] = 0
    store['rag_options'] = {}


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    _RAG_STEP_QUEUE.set(queue)
    if queue:
        try:
            _RAG_STEP_LOOP.set(asyncio.get_running_loop())
        except RuntimeError:
            _RAG_STEP_LOOP.set(asyncio.get_event_loop())
    else:
        _RAG_STEP_LOOP.set(None)


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    queue = _RAG_STEP_QUEUE.get()
    loop = _RAG_STEP_LOOP.get()
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    store = _task_store()
    if store['call_count'] >= 1:
        logger.warning(f"🔒 已拦截重复KB调用: search_knowledge_base, 查询='{query[:50]}...'")
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    store['call_count'] += 1
    logger.info(f"🔧 Agent首次调用工具: search_knowledge_base, 查询='{query[:50]}...'")

    from rag_pipeline import run_rag_graph

    rag_options = get_rag_options(clear=True)
    expansion_hint = rag_options.get("expansion_hint")
    skip_grading = should_skip_grading(query)
    logger.info(
        f"🔧 KB检索策略: skip_grading={skip_grading}, expansion_hint={expansion_hint}, "
        f"call_count={store['call_count']}, query='{query[:50]}...'"
    )

    try:
        rag_result = run_rag_graph(query, skip_grading=skip_grading, expansion_hint=expansion_hint)
    except Exception as e:
        import traceback
        logger.error(f"search_knowledge_base 异常: {e}")
        logger.error(f"异常堆栈:\n{traceback.format_exc()}")
        return f"RAG_ERROR: {e}"

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}

    if not docs:
        rag_trace = dict(rag_trace or {})
        rag_trace["kb_no_result"] = True
        _set_last_rag_context(_build_kb_no_result_context(query, rag_trace))
        return f"{KB_NO_RESULT_SENTINEL}: {KB_NO_RESULT_MESSAGE}"

    rag_trace = dict(rag_trace or {})
    rag_trace["kb_no_result"] = False
    _set_last_rag_context({"rag_trace": rag_trace, "kb_no_result": False})

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("calculator")
def calculator(expression: str) -> str:
    """
    计算器工具：执行数学表达式计算。

    支持的运算符：
    - 加减乘除: + - * /
    - 幂运算: **
    - 括号: ()
    - 数学函数: abs, round, sqrt, pow, sin, cos, tan 等

    示例：
    - "2 + 3 * 4" → 14
    - "(10 + 5) / 3" → 5
    - "pow(2, 10)" → 1024
    - "sqrt(16)" → 4.0
    """
    import math

    if not expression:
        return "错误：表达式不能为空"

    allowed_chars = set("0123456789+-*/().eE \t\n\rabsqrtpow sincostanloglnPIpi")
    for char in expression:
        if char not in allowed_chars:
            return f"错误：表达式包含非法字符 '{char}'"

    try:
        expr = expression.replace('^', '**')
        result = eval(expr, {"__builtins__": None}, {
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "ln": math.log,
            "PI": math.pi,
            "pi": math.pi,
            "e": math.e,
        })
        return f"计算结果：{result}"

    except SyntaxError:
        return f"错误：表达式语法错误: {expression}"
    except ZeroDivisionError:
        return "错误：不能除以零"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool("web_search")
def web_search(query: str) -> str:
    """
    网页搜索工具：使用搜索引擎搜索最新信息。

    参数：
    - query: 搜索关键词

    示例：
    - "2024年世界杯冠军"
    - "最新科技新闻"
    - "Python 3.12 新特性"

    返回：
    - 搜索结果摘要列表
    """
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    if SERP_API_KEY:
        try:
            params = {
                "q": query,
                "api_key": SERP_API_KEY,
                "engine": "google",
                "num": 5
            }
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            results = []
            if "organic_results" in data:
                for i, result in enumerate(data["organic_results"], 1):
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    link = result.get("link", "")
                    if title or snippet:
                        results.append(f"[{i}] {title}\n{snippet}\n链接: {link}")

            if results:
                return "搜索结果：\n\n" + "\n\n---\n\n".join(results)
            else:
                return "未找到相关搜索结果"

        except Exception as e:
            logger.error(f"网页搜索失败: {e}")
            return f"搜索失败: {str(e)}"
    else:
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []
            if "Abstract" in data and data["Abstract"]:
                results.append(f"摘要: {data['Abstract']}")
                if "AbstractURL" in data:
                    results.append(f"来源: {data['AbstractURL']}")

            if "RelatedTopics" in data:
                for i, topic in enumerate(data["RelatedTopics"], 1):
                    text = topic.get("Text", "")
                    link = topic.get("FirstURL", "")
                    if text:
                        results.append(f"[{i}] {text}")
                        if link:
                            results.append(f"    链接: {link}")

            if results:
                return "搜索结果：\n\n" + "\n\n".join(results)
            else:
                return "未找到相关搜索结果"

        except Exception as e:
            logger.error(f"网页搜索失败: {e}")
            return f"搜索失败: {str(e)}"
