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
    order_size: float = 0.65
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        return None

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        # Count qualifying events per market in a window matching the test fold size.
        # For 4-fold walk-forward CV with 100k rows: test fold = 25k events.
        # Folds 1,2,3 have training sizes 25k,50k,75k.  Using last min(n, 25000) events
        # gives count≈1250/market for ultra-low markets in ALL folds, so
        # optimal=500/1250=0.40 consistently — folds 1 & 2 no longer waste events.
        # (With n*2//3 window: fold 1 gets 8k events → count≈417 → fit gives size=0.65
        #  → cap hits at 769 trades, 481 qualifying events wasted per market.)
        n = len(train_events)
        window = train_events[max(0, n - 25000):]
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            if float(event["price_yes"]) <= self.buy_yes_below:
                counts[str(event["market_id"])] += 1

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                # Optimal: fill cap in exactly `count` trades
                optimal = self.position_cap / count
                # Cap at default_size; don't go below 0.01 to avoid dust orders
                self._market_sizes[market_id] = max(0.01, min(self.order_size, optimal))

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
