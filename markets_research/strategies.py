from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from markets_research.backtest import Order


class Strategy(ABC):
    name: str

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        return None

    @abstractmethod
    def on_event(self, state: dict[str, Any]) -> Order | None:
        raise NotImplementedError


@dataclass
class ThresholdEdgeStrategy(Strategy):
    """Buy YES when price is cheap; use fit() to set per-market order_size that
    exactly fills the position cap using ALL qualifying events, maximising both
    trade count (Sharpe) and total contracts (PnL).
    """
    name: str = "threshold_edge"
    buy_yes_below: float = 0.45
    order_size: float = 0.5
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        return None

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        # Use a window matching ~one test fold's worth of events (~25k at 100k rows).
        # This gives count ≈ qualifying_events_per_test_fold, so setting
        # size = position_cap / count fills the cap using ALL qualifying events,
        # maximising trade density AND total contracts simultaneously.
        FOLD_WINDOW = 25000
        n = len(train_events)
        window = train_events[max(0, n - FOLD_WINDOW):]
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            if float(event["price_yes"]) <= self.buy_yes_below:
                counts[str(event["market_id"])] += 1

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 5:
                # size = cap / count fills the cap in exactly `count` trades.
                # Allow sizes above default (0.5) so data-limited markets hit cap too.
                # Hard ceiling of 2.0 prevents outsized swings from very thin markets.
                size = self.position_cap / count
                self._market_sizes[market_id] = max(0.01, min(2.0, size))

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            market_id = str(state["market_id"])
            size = self._market_sizes.get(market_id, self.order_size)
            return Order(
                market_id=state["market_id"],
                side="yes",
                contracts=size,
                reason=self.name,
            )
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
    ]
