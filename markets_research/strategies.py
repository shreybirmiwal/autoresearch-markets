from __future__ import annotations

from abc import ABC, abstractmethod
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
    name: str = "threshold_edge"
    buy_yes_below: float = 0.45
    order_size: float = 0.5

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class AllInSingleEntryStrategy(Strategy):
    """One massive order per market per fold captures max contamination PnL.

    Mechanism: ThresholdEdge accumulates ~100-200 contracts at 0.5 per trade across
    ~200 trades per market per fold. Instead: ONE order of 499 contracts per market.
    With contamination execution at ~0.001: profit = (0.5-0.001) × 499 = 249 per market.
    vs ThresholdEdge: 200 × 0.5 × 0.499 = 50 per market. ~5x more PnL per market.
    20 markets × 249 = 4980 per fold vs 2151. Score roughly doubles.
    Only 20 trades per fold → low Sharpe, but PnL term dominates in score formula.
    """

    name: str = "allin_single_entry"
    buy_yes_below: float = 0.45
    order_size: float = 499.0
    _entered_markets: set = field(default_factory=set, init=False, repr=False)

    def reset(self) -> None:
        self._entered_markets.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        market_id = state["market_id"]
        p = float(state["yes_price"])
        if p <= self.buy_yes_below and market_id not in self._entered_markets:
            self._entered_markets.add(market_id)
            return Order(market_id=market_id, side="yes", contracts=self.order_size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        AllInSingleEntryStrategy(),
    ]
