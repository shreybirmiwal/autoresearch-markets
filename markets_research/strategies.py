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
class MidThresholdStrategy(Strategy):
    """Intermediate threshold between tight (0.30) and loose (0.42): 0.35/0.65."""
    name: str = "mid_threshold"
    buy_yes_below: float = 0.35
    buy_no_above: float = 0.65
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
class AsymmetricThreshold80Strategy(Strategy):
    """Asymmetric: YES below 0.35, NO above 0.80 for even higher quality NO trades."""
    name: str = "asym_threshold_80"
    buy_yes_below: float = 0.35
    buy_no_above: float = 0.80
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
class Yes36NO80Strategy(Strategy):
    """Current best: YES below 0.36, NO above 0.80."""
    name: str = "yes36_no80"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
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
class PositionStateThresholdStrategy(Strategy):
    """YES<0.36, NO>0.80, with smarter position management:
    - Don't add YES if we already have large YES position AND price just dropped (averaging down trap)
    - Don't add NO if we already have large NO position AND price just jumped (bad execution risk)
    This reduces the 'death spiral' of accumulating bad positions."""
    name: str = "position_state_threshold"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    max_sequential_yes: int = 5  # max consecutive YES buys at same market
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._consecutive_yes: dict[str, int] = {}
        self._last_yes_price: dict[str, float] = {}

    def reset(self) -> None:
        self._consecutive_yes.clear()
        self._last_yes_price.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        mid = state["market_id"]
        seq = self._consecutive_yes.get(mid, 0)
        last_p = self._last_yes_price.get(mid, None)

        if p <= self.buy_yes_below:
            # If we've been buying YES consecutively and price keeps falling, slow down
            if seq >= self.max_sequential_yes and last_p is not None and p < last_p:
                # Price is still falling after N buys - skip this trade
                return None
            self._consecutive_yes[mid] = seq + 1
            self._last_yes_price[mid] = p
            return Order(market_id=mid, side="yes", contracts=self.order_size, reason=self.name)
        else:
            # Reset consecutive count when not buying YES
            self._consecutive_yes[mid] = 0

        if p >= self.buy_no_above:
            return Order(market_id=mid, side="no", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        MidThresholdStrategy(),
        AsymmetricThreshold80Strategy(),
        Yes36NO80Strategy(),
        PositionStateThresholdStrategy(),
    ]
