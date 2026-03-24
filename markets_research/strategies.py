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
class InformedFlowFilterStrategy(Strategy):
    """Threshold entry with cooldown after large informed NO trades.

    Mechanism: In thin prediction markets, a large trade with significant
    downward price impact signals an informed NO buyer (someone who knows
    the market will likely resolve NO). Buying YES immediately after such
    a signal is wrong-sided. We add a per-market cooldown: after detecting
    a large downward trade (size > threshold * recent_median AND price drop
    > min_impact), skip YES buys for `cooldown` events in that market.
    """
    name: str = "informed_flow_filter"
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    size_multiplier: float = 3.0
    min_price_drop: float = 0.01
    cooldown: int = 5
    window: int = 20
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._prev_price: dict[str, float] = {}
        self._sizes: dict[str, deque] = {}
        self._cooldown_left: dict[str, int] = {}

    def reset(self) -> None:
        self._prev_price.clear()
        self._sizes.clear()
        self._cooldown_left.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])
        sz = float(state["size"])

        if mid not in self._sizes:
            self._sizes[mid] = deque(maxlen=self.window)
            self._cooldown_left[mid] = 0

        self._sizes[mid].append(sz)

        # Detect large informed downward trade
        if mid in self._prev_price and len(self._sizes[mid]) >= 5:
            price_drop = self._prev_price[mid] - p
            median_sz = float(np.median(list(self._sizes[mid])))
            if (price_drop >= self.min_price_drop and median_sz > 0
                    and sz >= self.size_multiplier * median_sz):
                self._cooldown_left[mid] = self.cooldown

        self._prev_price[mid] = p

        # Decrement cooldown
        if self._cooldown_left.get(mid, 0) > 0:
            self._cooldown_left[mid] -= 1

        if p <= self.buy_yes_below:
            if self._cooldown_left.get(mid, 0) > 0:
                return None  # Skip: large informed seller active
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
        InformedFlowFilterStrategy(),
    ]

