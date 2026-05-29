class StrategySelector:
    def __init__(self, strategies):
        self.strategies = sorted(strategies, key=lambda s: s.get_priority())

    async def select(self, context):
        execution_class = getattr(context, "execution_class", None)
        if execution_class:
            for strategy in self.strategies:
                if getattr(strategy, "execution_class", None) == execution_class:
                    return strategy

        for strategy in self.strategies:
            if await strategy.can_handle(context):
                return strategy
        return self.strategies[0]
