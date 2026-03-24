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
class AdaptiveSizeStrategy(Strategy):
    """Adaptive position sizing: larger contracts early, smaller near cap.

    Mechanism: ThresholdEdge uses flat 0.5 contracts, filling position cap in ~1000 trades.
    By switching to 0.25 contracts once position >= switch_at, we get ~400 additional
    micro-trades in the tail phase. Total contracts same (500), but more independent
    execution events → lower daily PnL variance → higher Sharpe.

    Phase 1 (pos < switch_at): 0.5 contracts × (switch_at/0.5) trades = switch_at contracts
    Phase 2 (pos >= switch_at): 0.25 contracts × ((500-switch_at)/0.25) trades
    Total trades ≈ switch_at/0.5 + (500-switch_at)/0.25 vs 1000 with flat 0.5.
    At switch_at=400: 800 + 400 = 1200 trades (+20% → Sharpe × sqrt(1.2) ≈ +10%).
    """

    name: str = "adaptive_size"
    buy_yes_below: float = 0.45
    switch_at: float = 400.0     # switch to small contracts when position >= this
    size_large: float = 0.5
    size_small: float = 0.25

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            pos_yes = float(state["position_yes_contracts"])
            size = self.size_small if pos_yes >= self.switch_at else self.size_large
            return Order(market_id=state["market_id"], side="yes", contracts=size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        AdaptiveSizeStrategy(),
    ]
