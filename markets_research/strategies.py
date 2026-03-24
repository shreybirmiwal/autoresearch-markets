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
class ThresholdEdgeWithExitStrategy(Strategy):
    """Like ThresholdEdgeStrategy but with exit logic to recycle capital."""
    name: str = "threshold_edge_exit"
    buy_yes_below: float = 0.42
    buy_no_above: float = 0.58
    exit_yes_above: float = 0.55  # exit yes positions when price recovers
    exit_no_below: float = 0.45  # exit no positions when price falls
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._yes_pos: dict[str, float] = {}
        self._no_pos: dict[str, float] = {}

    def reset(self) -> None:
        self._yes_pos.clear()
        self._no_pos.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        mid = state["market_id"]

        yes_pos = self._yes_pos.get(mid, 0.0)
        no_pos = self._no_pos.get(mid, 0.0)

        # Exit YES position when price has recovered
        if yes_pos > 0 and p >= self.exit_yes_above:
            self._yes_pos[mid] = 0.0
            return Order(market_id=mid, side="yes", contracts=-yes_pos, reason=self.name)

        # Exit NO position when price has fallen
        if no_pos > 0 and p <= self.exit_no_below:
            self._no_pos[mid] = 0.0
            return Order(market_id=mid, side="no", contracts=-no_pos, reason=self.name)

        # Entry signals
        if p <= self.buy_yes_below:
            self._yes_pos[mid] = yes_pos + self.order_size
            return Order(market_id=mid, side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.buy_no_above:
            self._no_pos[mid] = no_pos + self.order_size
            return Order(market_id=mid, side="no", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        ThresholdEdgeWithExitStrategy(),
    ]
