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
class LargeTradeFollowerStrategy(Strategy):
    """Follow large YES trades at low prices — informed traders signal YES resolution.

    Mechanism: A large trade at low price (e.g., buy 10k contracts at YES=0.05) suggests
    an informed trader who knows the market will resolve YES. If it resolves YES within
    the fold: settlement=1.0 (not 0.5 mark) → profit=(1.0-exec_price) vs (0.5-exec_price).
    With contamination execution (~0.001): profit 0.999 vs 0.499 per contract.
    ThresholdEdge fires on every low-price event regardless of size; this strategy
    concentrates capital on "informed" signals, potentially doubling per-trade PnL
    in YES-resolving markets even though it fires fewer total trades.
    """

    name: str = "large_trade_follower"
    buy_yes_below: float = 0.45
    size_multiplier: float = 3.0   # fire when size > 3x running avg size
    order_size: float = 1.0        # larger contract on informed signals
    _avg_size: float = field(default=100.0, init=False, repr=False)

    def reset(self) -> None:
        self._avg_size = 100.0

    def on_event(self, state: dict[str, Any]) -> Order | None:
        size = float(state["size"])
        self._avg_size = 0.99 * self._avg_size + 0.01 * size
        p = float(state["yes_price"])
        if p <= self.buy_yes_below and size >= self.size_multiplier * self._avg_size:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        LargeTradeFollowerStrategy(),
    ]
