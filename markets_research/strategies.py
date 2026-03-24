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
class StableCheapBoostStrategy(Strategy):
    """Larger position when market has been cheap for a long time (stable, likely mispriced).

    Mechanism: A market freshly dropped to cheap prices may reflect informed
    selling (smart money knows it's NO). A market stuck at cheap prices for
    50+ events has no active selling pressure — it's a stable longshot that
    the market may be anchoring incorrectly. These stable cheap markets are
    better candidates for YES buys. Conversely, fresh cheap entries (<10
    consecutive events) are more likely informed selling — use smaller size.
    """
    name: str = "stable_cheap_boost"
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    cheap_threshold: float = 0.20
    stable_min_events: int = 50
    fresh_max_events: int = 10
    stable_size: float = 1.5
    fresh_size: float = 0.4
    base_size: float = 1.0

    def __post_init__(self) -> None:
        self._cheap_count: dict[str, int] = {}

    def reset(self) -> None:
        self._cheap_count.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])

        # Track consecutive events in cheap zone
        if mid not in self._cheap_count:
            self._cheap_count[mid] = 0

        if p < self.cheap_threshold:
            self._cheap_count[mid] += 1
        else:
            self._cheap_count[mid] = 0  # reset on exit from cheap zone

        if p <= self.buy_yes_below:
            count = self._cheap_count[mid]
            if count >= self.stable_min_events:
                size = self.stable_size
            elif count <= self.fresh_max_events and count > 0:
                size = self.fresh_size
            else:
                size = self.base_size
            return Order(market_id=mid, side="yes", contracts=size, reason=self.name)

        if p >= self.buy_no_above:
            return Order(market_id=mid, side="no", contracts=1.0, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        TrendFilteredThresholdStrategy(),
        StableCheapBoostStrategy(),
    ]

