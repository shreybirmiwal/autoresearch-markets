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
class ContraryExtremesStrategy(Strategy):
    """Buy YES at extreme lows (near 0), buy NO at extreme highs (near 1).
    Focuses on the (-0.001, 0.2] and (0.8, 1.0] buckets which show highest avg PnL."""
    name: str = "contrary_extremes"
    yes_threshold: float = 0.20
    no_threshold: float = 0.80
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.yes_threshold:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.no_threshold:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class EMACrossoverStrategy(Strategy):
    """EMA crossover: when short EMA crosses above long EMA, buy YES (upward momentum).
    When short EMA crosses below long EMA, buy NO (downward momentum)."""
    name: str = "ema_crossover"
    short_window: int = 5
    long_window: int = 20
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=self.long_window)
        self._prev_signal: int = 0  # 0=none, 1=short>long, -1=short<long

    def reset(self) -> None:
        self._prices.clear()
        self._prev_signal = 0

    def _ema(self, prices: list[float], window: int) -> float:
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        self._prices.append(p)
        if len(self._prices) < self.long_window:
            return None
        prices_list = list(self._prices)
        short_ema = self._ema(prices_list[-self.short_window:], self.short_window)
        long_ema = self._ema(prices_list, self.long_window)
        current_signal = 1 if short_ema > long_ema else -1
        # Only trade on crossover
        if current_signal != self._prev_signal and self._prev_signal != 0:
            self._prev_signal = current_signal
            # Guardrail: don't buy NO when price is very low (worst context)
            # Don't buy YES when price is very high (bad context too)
            if current_signal == 1 and p <= 0.80:
                return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
            elif current_signal == -1 and p >= 0.20:
                return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        self._prev_signal = current_signal
        return None


@dataclass
class RSIStrategy(Strategy):
    """RSI-like overbought/oversold with price guardrail.
    Buy YES when oversold (RSI < 30) and price < 0.50 (don't fight the trend at extremes).
    Buy NO when overbought (RSI > 70) and price > 0.50."""
    name: str = "rsi_strategy"
    window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=self.window + 1)

    def reset(self) -> None:
        self._prices.clear()

    def _compute_rsi(self) -> float | None:
        if len(self._prices) < self.window + 1:
            return None
        prices = list(self._prices)
        gains = []
        losses = []
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-delta)
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        if avg_loss < 1e-9:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        self._prices.append(p)
        rsi = self._compute_rsi()
        if rsi is None:
            return None
        # Only trade when price context matches signal direction
        if rsi < self.oversold and p < 0.50:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if rsi > self.overbought and p > 0.50:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        ContraryExtremesStrategy(),
        EMACrossoverStrategy(),
        RSIStrategy(),
    ]

