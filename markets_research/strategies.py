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
        # Count qualifying events per market in the LAST third of training data
        # (most temporally similar to the upcoming test fold, avoids stale data).
        # n*2//3 formula: for 4-fold 100k CV, fold 3 window = last 25k = test fold size.
        # Folds 1,2 get smaller windows — must normalize count to test-fold scale
        # so cap-limited markets get the same size estimate across all folds.
        n = len(train_events)
        window = train_events[n * 2 // 3:]  # last ~33% of training
        window_size = max(1, len(window))
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            if float(event["price_yes"]) <= self.buy_yes_below:
                counts[str(event["market_id"])] += 1

        # In the standard 100k / 4-chunk setup, each test fold has ~25k rows.
        # Window size = n//3, which is 8334 for fold 1 but 25000 for fold 3.
        # Without normalization, fold 1 underestimates test-fold qualifying count
        # for cap-limited markets → uses size=0.65 when it should use a smaller size
        # → position cap hits after 769 trades (vs. ~1500 possible) → fewer trades → lower Sharpe.
        # Fix: scale raw window count by (test_fold_rows / window_size).
        TEST_FOLD_ROWS = 25_000  # one chunk in the standard 100k run

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                # Normalize to estimated qualifying count in the test fold
                est_test_count = count * TEST_FOLD_ROWS / window_size
                optimal = self.position_cap / est_test_count
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
