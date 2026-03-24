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
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
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
        return None




@dataclass
class TightExitRecyclerStrategy(Strategy):
    """AboveMarkRecycler variant with tighter exit threshold for more recycling cycles.

    Mechanism: Market 253597 (2M trades, mean=0.389, max=0.543) oscillates between
    low prices and 0.54. The original recycler at exit=0.55 never exits in this market.
    Lowering exit to 0.52 enables recycling when market 253597 peaks at 0.53-0.54,
    freeing capital to re-enter at low prices. More cycles per market → more total PnL.
    Each exit at 0.52 still captures (0.52-buy_price) > (0.5-buy_price) above mark.
    """

    name: str = "tight_exit_recycler"
    buy_yes_below: float = 0.38
    sell_yes_above: float = 0.52  # just above 0.5 mark, triggers in more markets
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state["position_yes_contracts"])

        if pos_yes > 0 and p >= self.sell_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        return None



@dataclass
class RecyclingThresholdStrategy(Strategy):
    """Wider buy range + above-mark exit: buy YES when p<=0.42, sell YES when p>=0.52.

    Mechanism: ThresholdEdge (buy<=0.42, hold to mark=0.5) vs TightExitRecycler (buy<=0.38,
    sell>=0.52). ThresholdEdge wins slightly due to wider buy range capturing more trades.
    TightExitRecycler's exit at 0.52 (above mark) extracts more profit per exit.
    Combining both: wider entry (0.42) + above-mark exit (0.52) should strictly dominate.
    Buy at p<=0.42, exit at p>=0.52 = up to 0.10/contract locked profit vs 0.08 at mark.
    Recycled capital re-enters at low prices for more cycles.
    """

    name: str = "recycling_threshold"
    buy_yes_below: float = 0.42
    sell_yes_above: float = 0.52
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state["position_yes_contracts"])

        if pos_yes > 0 and p >= self.sell_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        TightExitRecyclerStrategy(),
        RecyclingThresholdStrategy(),
    ]

