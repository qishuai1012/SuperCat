"""
性能测试脚本
测试不同复杂度查询的响应时间和API调用次数
"""

import asyncio
import time
from backend.agent import get_agentic_processor
from backend.performance_monitor import get_performance_monitor

# 测试用例
TEST_CASES = [
    {
        "name": "简单问题",
        "queries": [
            "什么是Python",
            "谁是马云",
            "北京在哪里"
        ]
    },
    {
        "name": "中等问题",
        "queries": [
            "Python和Java有什么区别",
            "如何学习机器学习",
            "什么是RAG技术"
        ]
    },
    {
        "name": "复杂问题",
        "queries": [
            "请详细分析Python在人工智能领域的应用，并比较其与其他语言的优缺点",
            "如何设计一个高性能的RAG系统，需要考虑哪些关键因素"
        ]
    }
]


async def test_query(query: str, user_id: str = "test_user", session_id: str = "test_session"):
    """测试单个查询"""
    processor = get_agentic_processor()

    start = time.time()
    result = await processor.process_query(query, user_id, session_id)
    duration = time.time() - start

    return {
        "query": query[:50],
        "duration": duration,
        "response_length": len(result.get("response", "")),
        "success": bool(result.get("response"))
    }


async def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("性能测试开始")
    print("=" * 60)

    for category in TEST_CASES:
        print(f"\n【{category['name']}】")
        print("-" * 60)

        for query in category['queries']:
            print(f"\n查询: {query[:50]}...")
            result = await test_query(query)

            print(f"  耗时: {result['duration']:.2f}秒")
            print(f"  响应长度: {result['response_length']} 字符")
            print(f"  状态: {'成功' if result['success'] else '失败'}")

    # 输出统计信息
    print("\n" + "=" * 60)
    print("性能统计")
    print("=" * 60)

    monitor = get_performance_monitor()
    stats = monitor.get_statistics()

    print(f"总查询数: {stats['total_queries']}")
    print(f"平均耗时: {stats['avg_duration']:.2f}秒")
    print(f"最大耗时: {stats['max_duration']:.2f}秒")
    print(f"最小耗时: {stats['min_duration']:.2f}秒")
    print(f"平均API调用: {stats['avg_api_calls']:.1f}次")
    print(f"成功率: {stats['success_rate']*100:.1f}%")

    # 导出详细报告
    monitor.export_report("logs/performance_report.json")
    print("\n详细报告已导出: logs/performance_report.json")


if __name__ == "__main__":
    asyncio.run(run_tests())
