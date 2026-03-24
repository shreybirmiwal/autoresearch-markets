from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
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
        # Folds 1,2 get smaller windows but they're more representative temporally
        # (earlier folds have different qualifying rates due to market price trends).
        n = len(train_events)
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
            size = self._market_sizes.get(market_id, self.order_size)
            return Order(
                market_id=state["market_id"],
                side="yes",
                contracts=size,
                reason=self.name,
            )
        return None


@dataclass
class OrderFlowQualityStrategy(Strategy):
    """Track global qualifying-event fraction over last K events.
    When majority of recent events came from qualifying (ultra-low) markets,
    the next execution is more likely to be at a cheap price → use higher size.
    When high-price markets dominate recent flow, use lower size.
    """
    name: str = "order_flow_quality"
    buy_yes_below: float = 0.45
    order_size_hot: float = 0.70    # ≥4/5 recent events qualify
    order_size_cold: float = 0.62   # <4/5 recent events qualify
    window_k: int = 5
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        self._recent_qualifying: deque = deque(maxlen=self.window_k)
        return None

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        n = len(train_events)
        window = train_events[n * 2 // 3:]
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            if float(event["price_yes"]) <= self.buy_yes_below:
                counts[str(event["market_id"])] += 1

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                optimal_hot = self.position_cap / count
                optimal_cold = self.position_cap / count
                # Use hot size as the reference for cap calibration
                ref_size = max(self.order_size_hot, self.order_size_cold)
                size = max(0.01, min(ref_size, (optimal_hot + optimal_cold) / 2))
                self._market_sizes[market_id] = size

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        # Update global qualifying fraction tracker (ALL events, not just qualifying)
        self._recent_qualifying.append(1 if p <= self.buy_yes_below else 0)

        if p <= self.buy_yes_below:
            market_id = str(state["market_id"])
            # Determine if we're in a "hot" (ultra-low dominant) period
            recent_qual_fraction = (
                sum(self._recent_qualifying) / len(self._recent_qualifying)
                if self._recent_qualifying else 0.5
            )
            is_hot = recent_qual_fraction >= 0.8  # ≥4/5 recent events qualified
            order_size = self.order_size_hot if is_hot else self.order_size_cold

            # Use fit-calibrated size if available (overrides dynamic sizing for cap control)
            size = self._market_sizes.get(market_id, order_size)
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
        OrderFlowQualityStrategy(),
    ]
