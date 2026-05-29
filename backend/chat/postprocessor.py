import logging

from performance_config import get_performance_config
from query_understanding.types import QueryComplexity

logger = logging.getLogger(__name__)


class PostProcessor:
    def __init__(self, reflection, config):
        self.reflection = reflection
        self.config = config

    def get_reflection_control(self, context, response: str) -> dict:
        complexity_value = (context.complexity or "medium").upper()
        complexity = QueryComplexity.__members__.get(complexity_value, QueryComplexity.MEDIUM)
        strategy_config = get_performance_config().get_strategy(complexity)

        if not self.config.enable_reflection:
            return {
                "reflection_enabled_by_config": False,
                "reflection_attempted": False,
                "reflection_revised": False,
                "reflection_skipped_reason": "global_disabled",
            }

        if not strategy_config.enable_reflection:
            return {
                "reflection_enabled_by_config": False,
                "reflection_attempted": False,
                "reflection_revised": False,
                "reflection_skipped_reason": f"strategy_disabled:{complexity.value}",
            }

        if not self.reflection:
            return {
                "reflection_enabled_by_config": True,
                "reflection_attempted": False,
                "reflection_revised": False,
                "reflection_skipped_reason": "reflection_unavailable",
            }

        if not response:
            return {
                "reflection_enabled_by_config": True,
                "reflection_attempted": False,
                "reflection_revised": False,
                "reflection_skipped_reason": "empty_response",
            }

        return {
            "reflection_enabled_by_config": True,
            "reflection_attempted": True,
            "reflection_revised": False,
            "reflection_skipped_reason": None,
        }

    async def process(self, result, context, include_memory: bool = True):
        reflection_meta = self.get_reflection_control(context, result.response)
        result.metadata.update(reflection_meta)
        result.metadata.setdefault("was_revised", False)

        if reflection_meta["reflection_attempted"]:
            try:
                reflection_result = self.reflection.assess_answer_quality(context.user_text, result.response)
                if reflection_result and reflection_result.action.value == "revise":
                    revised = self.reflection.revise_answer(context.user_text, result.response, reflection_result)
                    if revised:
                        result.response = revised
                        result.metadata["was_revised"] = True
                        result.metadata["reflection_revised"] = True
            except Exception as e:
                result.metadata["reflection_attempted"] = False
                result.metadata["reflection_skipped_reason"] = f"reflection_error:{e}"
                logger.warning(f"反思失败: {e}")

        return result
