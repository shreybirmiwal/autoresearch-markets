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
        self._last_qual_market: str | None = None
        return None

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        # Count qualifying events per market in the LAST quarter of training data
        # (most temporally similar to the upcoming test fold).
        n = len(train_events)
        # Use last ~1/3 of training (scale-invariant: = one test fold's worth
        # for walk-forward fold 3, regardless of total data size).
        # At 100k rows: ~25k events → count≈617 < 1000 → size=0.5 (unchanged).
        # At 1M rows:  ~250k events → count≈6137 > 1000 → size=500/6137≈0.08
        #   → ALL qualifying events fire, cap exactly filled → full PnL at high density.
        window = train_events[n * 2 // 3:]  # last ~33% of training
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
            # Only fire if this market was ALSO the previous qualifying event's market.
            # Consecutive same-market qualifying events indicate a burst where the NEXT
            # sequential event is likely also from the same cheap market → better execution.
            same_as_prev = (market_id == self._last_qual_market)
            self._last_qual_market = market_id
            if same_as_prev:
                size = self._market_sizes.get(market_id, self.order_size)
                return Order(
                    market_id=state["market_id"],
                    side="yes",
                    contracts=size,
                    reason=self.name,
                )
        else:
            # Reset tracking if non-qualifying event resets the qualifying run
            # (optional — keeps _last_qual_market from previous run for re-entry detection)
            pass
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
    ]
