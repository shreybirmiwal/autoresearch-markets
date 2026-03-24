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
class MomentumReversalStrategy(Strategy):
    """Momentum reversal: buy YES when price has been falling consistently,
    buy NO when price has been rising consistently."""
    name: str = "momentum_reversal"
    window: int = 8
    min_consecutive_moves: int = 5  # require 5 out of 7 moves in same direction
    buy_yes_below: float = 0.50  # only reverse falling markets below 0.50
    buy_no_above: float = 0.50  # only reverse rising markets above 0.50
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

        arr = np.array(self._history)
        moves = np.diff(arr)

        falling_count = int((moves < 0).sum())
        rising_count = int((moves > 0).sum())

        # Strong falling trend at low price - buy YES (reversal)
        if p <= self.buy_yes_below and falling_count >= self.min_consecutive_moves:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        # Strong rising trend at high price - buy NO (reversal)
        if p >= self.buy_no_above and rising_count >= self.min_consecutive_moves:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class Yes36NO80Size2Strategy(Strategy):
    """YES below 0.36, NO above 0.80 with doubled order size (2 contracts).
    Sharpe should remain similar but PnL should roughly double."""
    name: str = "yes36_no80_size2"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    order_size: float = 2.0

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
class LogisticGatedThresholdStrategy(Strategy):
    """Gate the yes36_no80 signal with an online logistic model.
    Only buy YES when BOTH: price < 0.36 AND logistic predicts YES resolution.
    Only buy NO when BOTH: price > 0.80 AND logistic predicts NO resolution."""
    name: str = "logistic_gated_threshold"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    lr: float = 0.05
    min_logistic_edge: float = 0.03  # logistic must predict at least 3pp above price
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
        p = float(state["yes_price"])
        x = np.array([1.0, p, np.log1p(float(state["size"]))], dtype=np.float64)
        pred_yes = 1.0 / (1.0 + np.exp(-float(np.dot(self._w, x))))

        if p <= self.buy_yes_below and pred_yes - p >= self.min_logistic_edge:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.buy_no_above and p - pred_yes >= self.min_logistic_edge:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class DeepValueOnlyStrategy(Strategy):
    """Only buy YES at VERY low prices (<0.20) with larger order size.
    The <0.20 bucket has avg_pnl=0.458 vs 0.182 for 0.20-0.40 range."""
    name: str = "deep_value_only"
    buy_yes_below: float = 0.20
    buy_no_above: float = 0.80
    order_size: float = 3.0  # 3x size since fewer trades but higher quality

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
class BandedNoStrategy(Strategy):
    """Buy NO only when YES price is in the 0.62-0.78 band (profitable NO range).
    Avoids the >0.80 range (where yes36_no80 already trades) and the <0.62 range."""
    name: str = "banded_no"
    no_buy_low: float = 0.62   # YES price lower bound for NO buys
    no_buy_high: float = 0.78  # YES price upper bound for NO buys
    buy_yes_below: float = 0.36  # Also buy YES at low prices
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if self.no_buy_low <= p <= self.no_buy_high:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class HybridThresholdLogisticStrategy(Strategy):
    """Hybrid: threshold for YES<0.36/NO>0.80, logistic for the middle zone 0.36-0.80.
    Best of both worlds: high-confidence threshold signals + learned signals in middle."""
    name: str = "hybrid_threshold_logistic"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    lr: float = 0.05
    logistic_edge: float = 0.08  # logistic must predict 8pp edge in middle zone
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
        p = float(state["yes_price"])

        # Threshold zone - high confidence
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        # Middle zone - use logistic with strict edge requirement
        x = np.array([1.0, p, np.log1p(float(state["size"]))], dtype=np.float64)
        pred_yes = 1.0 / (1.0 + np.exp(-float(np.dot(self._w, x))))

        if p < 0.50 and pred_yes - p > self.logistic_edge:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p > 0.50 and p - pred_yes > self.logistic_edge:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class MarketAdaptiveThresholdStrategy(Strategy):
    """Per-market adaptive threshold: learn each market's price distribution
    during training, then trade more aggressively in markets where YES<0.36 is
    historically correlated with high future prices (mean reversion type)."""
    name: str = "market_adaptive_threshold"
    buy_yes_below: float = 0.36
    buy_no_above: float = 0.80
    order_size: float = 1.0

    def __post_init__(self) -> None:
        # market_id -> average price (learned from training)
        self._market_avg_price: dict[str, float] = {}
        self._market_price_std: dict[str, float] = {}

    def reset(self) -> None:
        self._market_avg_price.clear()
        self._market_price_std.clear()

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        """Learn per-market price statistics from training data."""
        from collections import defaultdict
        market_prices: dict = defaultdict(list)
        for event in train_events:
            mid = str(event.get("market_id", ""))
            px = float(event.get("yes_price", event.get("price_yes", 0.5)))
            market_prices[mid].append(px)

        for mid, prices in market_prices.items():
            arr = np.array(prices)
            self._market_avg_price[mid] = float(arr.mean())
            self._market_price_std[mid] = float(arr.std())

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        mid = str(state["market_id"])

        # Get market statistics
        avg_price = self._market_avg_price.get(mid, 0.5)
        price_std = self._market_price_std.get(mid, 0.15)

        # Only trade markets with meaningful price variation (not stuck at one level)
        if price_std < 0.05:
            return None

        # Standard threshold trades
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
        MidThresholdStrategy(),
        AsymmetricThreshold80Strategy(),
        Yes36NO80Strategy(),
        MarketAdaptiveThresholdStrategy(),
    ]
