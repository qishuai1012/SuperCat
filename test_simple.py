"""
简单性能测试 - 只测试一个查询
"""

import asyncio
import time
from backend.agent import get_agentic_processor

async def test_simple_query():
    """测试简单查询"""
    processor = get_agentic_processor()

    query = "什么是Python"
    print(f"查询: {query}")
    print("=" * 60)

    start = time.time()
    result = await processor.process_query(query, "test_user", "test_session")
    duration = time.time() - start

    print(f"\n耗时: {duration:.2f}秒")
    print(f"响应长度: {len(result.get('response', ''))} 字符")

    # 安全打印响应内容（避免emoji编码错误）
    response_text = result.get('response', '')
    try:
        print(f"响应内容: {response_text[:200]}...")
    except UnicodeEncodeError:
        # 移除emoji等特殊字符
        safe_text = response_text.encode('gbk', errors='ignore').decode('gbk')
        print(f"响应内容: {safe_text[:200]}...")

    print(f"成功: {bool(result.get('response'))}")

    return result

if __name__ == "__main__":
    asyncio.run(test_simple_query())
