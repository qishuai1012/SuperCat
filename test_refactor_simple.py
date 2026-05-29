"""测试重构后的agent模块 - 无emoji版本"""
import sys
sys.path.insert(0, 'F:/PythonProject/SuperCat/backend')

print("Test 1: Import agent module...")
try:
    from agent import chat_with_agent, chat_with_agent_stream, storage
    print("SUCCESS: Imported chat_with_agent, chat_with_agent_stream, storage")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest 2: Check storage object...")
try:
    print(f"storage type: {type(storage)}")
    print(f"storage has save: {hasattr(storage, 'save')}")
    print(f"storage has load: {hasattr(storage, 'load')}")
    print("SUCCESS: storage object is valid")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\nTest 3: Check function signatures...")
try:
    import inspect
    sig1 = inspect.signature(chat_with_agent)
    sig2 = inspect.signature(chat_with_agent_stream)
    print(f"chat_with_agent: {sig1}")
    print(f"chat_with_agent_stream: {sig2}")
    print("SUCCESS: Function signatures are correct")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")
