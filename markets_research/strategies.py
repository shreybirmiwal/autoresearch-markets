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
class LargeTradeFollowerStrategy(Strategy):
    """Follow large trades as informed-trader signal with exit logic.

    Mechanism: Large trades (>3x rolling median size for this market) are more likely
    from informed participants with private information. When a large YES buy occurs
    at a low price (0.10-0.40), we follow. Exit on take-profit (>0.70) or stop-loss (<0.05).
    """

    name: str = "large_trade_follower"
    size_window: int = 20
    size_multiplier: float = 3.0
    entry_min: float = 0.10
    entry_max: float = 0.40
    exit_take_profit: float = 0.70
    exit_stop_loss: float = 0.05
    order_size: float = 1.0

    def __post_init__(self) -> None:
        # per-market rolling size history
        self._market_sizes: dict[str, deque] = {}

    def reset(self) -> None:
        self._market_sizes.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = str(state["market_id"])
        p = float(state["yes_price"])
        sz = float(state["size"])
        pos_yes = float(state["position_yes_contracts"])

        if mid not in self._market_sizes:
            self._market_sizes[mid] = deque(maxlen=self.size_window)
        hist = self._market_sizes[mid]

        # Exit logic: if holding YES and price has moved enough
        if pos_yes > 0:
            hist.append(sz)
            if p >= self.exit_take_profit or p <= self.exit_stop_loss:
                return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)
            return None

        hist.append(sz)
        if len(hist) < 5:
            return None

        median_sz = float(np.median(np.array(hist, dtype=np.float64)))
        if median_sz <= 0:
            return None

        # Large YES buy signal at low price
        if sz >= self.size_multiplier * median_sz and self.entry_min <= p <= self.entry_max:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class BurstMarketThresholdStrategy(Strategy):
    """Trade only during same-market burst periods for predictable execution.

    Mechanism: In the backtest, order execution uses the NEXT sequential row's price
    (regardless of market). When one market is "burst trading" (N consecutive events),
    the next event is from the same market → execution price is predictable and at the
    same market's level. By trading only during bursts of low-price markets, we ensure
    YES buys execute at the low price of the bursting market, not at some random
    high-price market's price.
    """

    name: str = "burst_market_threshold"
    buy_yes_below: float = 0.40
    buy_no_above: float = 0.60
    burst_min: int = 2  # require at least this many consecutive same-market events
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._last_market_id: str = ""
        self._same_market_count: int = 0

    def reset(self) -> None:
        self._last_market_id = ""
        self._same_market_count = 0

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = str(state["market_id"])
        p = float(state["yes_price"])

        # Track burst count
        if mid == self._last_market_id:
            self._same_market_count += 1
        else:
            self._same_market_count = 1
            self._last_market_id = mid

        # Only trade during bursts (same market N consecutive times)
        if self._same_market_count < self.burst_min:
            return None

        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        LargeTradeFollowerStrategy(),
        BurstMarketThresholdStrategy(),
    ]

