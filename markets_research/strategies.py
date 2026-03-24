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
class StopLossThresholdStrategy(Strategy):
    """YES<0.36, NO>0.80 with stop-loss: exit YES position if price drops below stop level."""
    name: str = "stoploss_threshold"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    stop_loss_drop: float = 0.10  # exit if price drops 10 percentage points below entry
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._yes_pos: dict[str, float] = {}
        self._yes_entry_price: dict[str, float] = {}
        self._no_pos: dict[str, float] = {}
        self._no_entry_price: dict[str, float] = {}

    def reset(self) -> None:
        self._yes_pos.clear()
        self._yes_entry_price.clear()
        self._no_pos.clear()
        self._no_entry_price.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        mid = state["market_id"]
        yes_pos = self._yes_pos.get(mid, 0.0)
        no_pos = self._no_pos.get(mid, 0.0)

        # Stop-loss: exit YES if price dropped significantly below average entry
        if yes_pos > 0:
            avg_entry = self._yes_entry_price.get(mid, p)
            if p < avg_entry - self.stop_loss_drop:
                self._yes_pos[mid] = 0.0
                self._yes_entry_price.pop(mid, None)
                return Order(market_id=mid, side="yes", contracts=-yes_pos, reason=self.name)

        # Stop-loss: exit NO if YES price jumped above entry
        if no_pos > 0:
            avg_entry = self._no_entry_price.get(mid, p)
            if p > avg_entry + self.stop_loss_drop:
                self._no_pos[mid] = 0.0
                self._no_entry_price.pop(mid, None)
                return Order(market_id=mid, side="no", contracts=-no_pos, reason=self.name)

        # Entry
        if p <= self.buy_yes_below:
            if yes_pos == 0.0:
                self._yes_entry_price[mid] = p
            else:
                # Update average entry price
                self._yes_entry_price[mid] = (self._yes_entry_price[mid] * yes_pos + p * self.order_size) / (yes_pos + self.order_size)
            self._yes_pos[mid] = yes_pos + self.order_size
            return Order(market_id=mid, side="yes", contracts=self.order_size, reason=self.name)

        if p >= self.buy_no_above:
            if no_pos == 0.0:
                self._no_entry_price[mid] = p
            else:
                self._no_entry_price[mid] = (self._no_entry_price[mid] * no_pos + p * self.order_size) / (no_pos + self.order_size)
            self._no_pos[mid] = no_pos + self.order_size
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
        StopLossThresholdStrategy(),
    ]
