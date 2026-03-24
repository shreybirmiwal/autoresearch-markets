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
class BandedThresholdStrategy(Strategy):
    """Buy YES only in profitable mid-range (0.20-0.42), skip near-zero prices.
    Buy NO only above 0.60, skip the uncertain 0.40-0.60 range.
    Mechanism: YES < 0.20 markets are often near-resolved-NO (skip them).
    NO in 0.40-0.60 is buying uncertainty with negative edge (skip them).
    """
    name: str = "banded_threshold"
    yes_floor: float = 0.20
    yes_ceil: float = 0.42
    no_floor: float = 0.60
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if self.yes_floor <= p <= self.yes_ceil:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        if p >= self.no_floor:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class ExitAwareBandedStrategy(Strategy):
    """Buy YES in 0.20-0.42, exit YES when price rises above 0.55; buy NO above 0.65.
    Mechanism: capture the mispricing edge in the 0.20-0.42 band, then exit
    when price converges toward fair value (0.55+), recycling capital for new
    opportunities. Avoids position cap and reduces reversal risk.
    """
    name: str = "exit_aware_banded"
    yes_floor: float = 0.20
    yes_ceil: float = 0.42
    no_floor: float = 0.65
    exit_yes_above: float = 0.55
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state.get("position_yes_contracts", 0))

        # Exit YES position when price has risen to profitable territory
        if pos_yes > 0 and p >= self.exit_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        # Buy YES in the profitable mid-range
        if self.yes_floor <= p <= self.yes_ceil:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        # Buy NO only in high-confidence range (avoid 0.60-0.65 gray zone)
        if p >= self.no_floor:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class ConfidenceScaledBandedStrategy(Strategy):
    """Buy YES in 0.20-0.42 with size scaled by distance from 0.5 (fair value).
    Cheaper YES = higher expected gain + larger implied mispricing → buy more.
    Exit YES when price recovers above 0.55 (recycle capital).
    Formula: contracts = max(1, (0.50 - price) / 0.10) capped at 4.
    """
    name: str = "confidence_scaled_banded"
    yes_floor: float = 0.20
    yes_ceil: float = 0.42
    no_floor: float = 0.65
    exit_yes_above: float = 0.55
    max_contracts: float = 4.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state.get("position_yes_contracts", 0))

        # Exit YES position when price recovers
        if pos_yes > 0 and p >= self.exit_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        # Buy YES with confidence-scaled size: more contracts when price is cheaper
        if self.yes_floor <= p <= self.yes_ceil:
            size = max(1.0, min(self.max_contracts, (0.50 - p) / 0.10))
            return Order(market_id=state["market_id"], side="yes", contracts=size, reason=self.name)

        # Buy NO in high-confidence range
        if p >= self.no_floor:
            return Order(market_id=state["market_id"], side="no", contracts=1.0, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        BandedThresholdStrategy(),
        ExitAwareBandedStrategy(),
        ConfidenceScaledBandedStrategy(),
    ]

