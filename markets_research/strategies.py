from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class MeanReversionStrategy(Strategy):
    name: str = "mean_reversion"
    window: int = 50
    z_entry: float = 1.2
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._history: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        self._history.append(p)
        if len(self._history) < self.window:
            return None
        arr = np.array(self._history, dtype=np.float64)
        std = arr.std()
        if std <= 1e-9:
            return None
        z = (p - arr.mean()) / std
        if z <= -self.z_entry:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if z >= self.z_entry:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class OnlineLogisticLikeStrategy(Strategy):
    name: str = "online_logistic_like"
    lr: float = 0.05
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._w = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._w[:] = 0.0

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        for event in train_events:
            px = float(event.get("yes_price", event.get("price_yes", 0.5)))
            x = np.array([1.0, px, np.log1p(float(event["size"]))], dtype=np.float64)
            y = float(event.get("label", 0.5))
            pred = 1.0 / (1.0 + np.exp(-float(np.dot(self._w, x))))
            grad = (pred - y) * x
            self._w -= self.lr * grad

    def on_event(self, state: dict[str, Any]) -> Order | None:
        x = np.array([1.0, float(state["yes_price"]), np.log1p(float(state["size"]))], dtype=np.float64)
        pred_yes = 1.0 / (1.0 + np.exp(-float(np.dot(self._w, x))))
        if pred_yes - float(state["yes_price"]) > 0.05:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if float(state["yes_price"]) - pred_yes > 0.05:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class TrendFilteredThresholdStrategy(Strategy):
    """Buy YES only when the per-market recent price trend is non-negative.

    Mechanism: Consistent price drops in a market signal informed sellers
    who know the market will resolve NO. Buying YES against a falling trend
    is wrong-sided. We only buy YES when price is flat or rising over the
    last `lookback` events for that market, indicating no strong informed
    selling. This selectively captures cheap YES positions in markets that
    are holding value, not drifting to 0.
    """
    name: str = "trend_filtered_threshold"
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    lookback: int = 3
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._market_prices: dict[str, deque] = {}

    def reset(self) -> None:
        self._market_prices.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])

        if mid not in self._market_prices:
            self._market_prices[mid] = deque(maxlen=self.lookback + 1)
        self._market_prices[mid].append(p)

        hist = self._market_prices[mid]

        if p <= self.buy_yes_below:
            # Only buy YES if not all recent moves are downward
            if len(hist) >= 2:
                # Check if every consecutive pair is strictly decreasing
                all_falling = all(hist[i] > hist[i + 1] for i in range(len(hist) - 1))
                if all_falling:
                    return None
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        if p >= self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class HighWaterMarkStrategy(Strategy):
    """Buy YES only in markets that have demonstrated price viability.

    Mechanism: Markets that have ONLY traded at very low prices (< min_viable_price)
    have achieved informational consensus at a near-zero probability — the market
    participants collectively see no meaningful YES potential. Markets that have
    ever traded above the threshold retain genuine uncertainty. This filters out
    the large cluster of near-zero markets that systematically resolve NO, while
    preserving full exposure to the rare markets that eventually resolve YES
    (which always show prices above the threshold early on).
    """
    name: str = "high_water_mark"
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    min_viable_price: float = 0.10
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._max_price: dict[str, float] = {}

    def reset(self) -> None:
        self._max_price.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])

        self._max_price[mid] = max(self._max_price.get(mid, 0.0), p)

        if p <= self.buy_yes_below:
            if self._max_price[mid] < self.min_viable_price:
                return None
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        if p >= self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        TrendFilteredThresholdStrategy(),
        HighWaterMarkStrategy(),
    ]

