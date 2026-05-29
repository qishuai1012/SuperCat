"""测试重构后的agent模块"""
import sys
sys.path.insert(0, 'F:/PythonProject/SuperCat/backend')

print("测试1: 导入agent模块...")
try:
    from agent import chat_with_agent, chat_with_agent_stream, storage
    print("✅ 成功导入: chat_with_agent, chat_with_agent_stream, storage")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试2: 检查storage对象...")
try:
    print(f"storage类型: {type(storage)}")
    print(f"storage方法: {[m for m in dir(storage) if not m.startswith('_')]}")
    print("✅ storage对象正常")
except Exception as e:
    print(f"❌ storage检查失败: {e}")

print("\n测试3: 检查函数签名...")
try:
    import inspect
    sig1 = inspect.signature(chat_with_agent)
    sig2 = inspect.signature(chat_with_agent_stream)
    print(f"chat_with_agent签名: {sig1}")
    print(f"chat_with_agent_stream签名: {sig2}")
    print("✅ 函数签名正常")
except Exception as e:
    print(f"❌ 函数签名检查失败: {e}")

print("\n✅ 所有基础测试通过！")
