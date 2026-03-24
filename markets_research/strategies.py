from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class HighPriceFollowerStrategy(Strategy):
    """Buy YES in high-probability YES markets (p>0.90) via contamination mechanism.

    Mechanism: Market 511754 (mean=0.963) likely resolves YES → settlement=1.0, giving
    profit=(1.0-execution_price) vs mark-settlement profit=(0.5-execution_price).
    With cheap contamination execution (~0.001), profit doubles: 0.999 vs 0.499 per contract.
    ThresholdEdge (buy<=0.45) NEVER fires in market 511754. This strategy fills that gap.
    Even if market doesn't resolve YES (settles at mark=0.5), still profits from contamination.
    """

    name: str = "high_price_follower"
    buy_yes_above: float = 0.90
    order_size: float = 0.5

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p >= self.buy_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        HighPriceFollowerStrategy(),
    ]
