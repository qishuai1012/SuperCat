from dataclasses import asdict, is_dataclass
from enum import Enum


class ResultBuilder:
    async def finalize_result(self, result, context, postprocessor):
        return await postprocessor.process(result, context, include_memory=False)

    def build_response_payload(self, result) -> dict:
        serialized_metadata = {
            key: self.serialize_for_json(value)
            for key, value in result.metadata.items()
        }
        route_decision = serialized_metadata.get("route_decision")

        return {
            "response": result.response,
            "rag_trace": result.rag_trace,
            "agentic_info": {
                "route_decision": route_decision,
                "was_revised": bool(serialized_metadata.get("was_revised", False)),
            },
            **serialized_metadata,
        }

    def serialize_for_json(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return self.serialize_for_json(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): self.serialize_for_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self.serialize_for_json(item) for item in value]
        return str(value)
