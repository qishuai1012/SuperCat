"""
Agentic RAG Usage Example
Demonstrates how to use the upgraded system
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def simple_usage_demo():
    """Simple usage demonstration"""

    print("Agentic RAG Usage Example")
    print("=" * 50)

    print("\n1. Basic Usage (Automatic Agentic Features)")
    print("-" * 40)
    print("# Your existing code works automatically with Agentic features")
    print("from backend.agent import chat_with_agent, chat_with_agent_stream")
    print("")
    print("# Non-streaming chat (now with Agentic capabilities)")
    print("result = chat_with_agent('What is artificial intelligence?', 'user123', 'session456')")
    print("print(result['response'])")
    print("")
    print("# Streaming chat (enhanced with real-time Agentic processing)")
    print("async for chunk in chat_with_agent_stream('Complex query...', 'user123', 'session456'):")
    print("    print(chunk)")

    print("\n2. Advanced Features Usage")
    print("-" * 40)

    try:
        from backend.intelligent_router import get_intelligent_router
        print("# Intelligent Routing")
        router = get_intelligent_router()
        decision = router.route_query("Complex query about machine learning")
        print(f"Strategy: {decision.strategy.value}")
        print(f"Complexity: {decision.query_complexity.value}")
        print(f"Will decompose: {decision.needs_decomposition}")
    except Exception as e:
        print(f"Router demo skipped: {e}")

    try:
        from backend.task_decomposer import get_task_decomposer
        print("\n# Task Decomposition")
        decomposer = get_task_decomposer()
        subtasks = decomposer.decompose("Compare AI and machine learning")
        print(f"Decomposed into {len(subtasks)} subtasks")
        for task in subtasks:
            print(f"  - {task.type.value}: {task.description}")
    except Exception as e:
        print(f"Decomposition demo skipped: {e}")

    try:
        from backend.reflection_agent import get_reflection_agent
        print("\n# Quality Assessment")
        reflection = get_reflection_agent()
        assessment = reflection.assess_answer_quality(
            "What is AI?",
            "AI is artificial intelligence...",
            "Context information"
        )
        print(f"Quality score: {assessment.overall_score:.2f}")
        print(f"Recommended action: {assessment.action.value}")
    except Exception as e:
        print(f"Reflection demo skipped: {e}")

    print("\n3. System Information")
    print("-" * 40)

    try:
        from backend.memory_optimizer import get_memory_optimizer
        print("# Memory System")
        memory = get_memory_optimizer()
        stats = memory.get_memory_statistics()
        print(f"Working memory: {stats['working_memory']['total_count']} items")
        print(f"Long-term memory: {stats['long_term_memory']['total_count']} items")
    except Exception as e:
        print(f"Memory demo skipped: {e}")

    print("\n4. Learning System")
    print("-" * 40)
    print("# The system automatically:")
    print("# - Collects user feedback")
    print("# - Analyzes performance trends")
    print("# - Generates optimization recommendations")
    print("# - Applies adaptive tuning")

    print("\n" + "=" * 50)
    print("Key Points:")
    print("• Existing code works automatically with enhanced capabilities")
    print("• No breaking changes to existing APIs")
    print("• Agentic features activate automatically for complex queries")
    print("• System continuously learns and improves")
    print("• All enhancements are transparent to end users")

if __name__ == "__main__":
    simple_usage_demo()