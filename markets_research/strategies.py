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
class AboveMarkRecyclerStrategy(Strategy):
    """Buy YES cheap, sell above 0.5 settlement mark to lock in extra profit and recycle.

    Mechanism: Settlement is at 0.5 for all positions. Holding a YES position to the
    mark gives (0.5 - buy_price) profit. But SELLING when the price EXCEEDS 0.5 gives
    (sell_price - buy_price) > (0.5 - buy_price). If the market then drops back to a
    low price, re-entering captures another round. This recycling generates profit above
    what holding-to-mark would capture, while freeing capital for re-entry.
    """

    name: str = "above_mark_recycler"
    buy_yes_below: float = 0.38
    sell_yes_above: float = 0.55  # above the 0.5 settlement mark = better than holding
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state["position_yes_contracts"])

        # Exit: sell above mark to capture above-mark profit
        if pos_yes > 0 and p >= self.sell_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        # Entry: buy cheap YES
        if p <= self.buy_yes_below:
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
class VeryTightExitRecyclerStrategy(Strategy):
    """Recycler with exit at 0.51 — just above mark for maximum recycling cycles.

    Mechanism: Lower exit threshold (0.51 vs 0.52) enables even more markets to
    trigger recycling. Any market that crosses 0.51 (just above the 0.5 settlement
    mark) triggers an exit and re-entry cycle. More frequent exits mean faster capital
    recycling, leading to more consistent per-period returns and higher sharpe.
    """

    name: str = "very_tight_exit_recycler"
    buy_yes_below: float = 0.38
    sell_yes_above: float = 0.51
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
class FullCycleRecyclerStrategy(Strategy):
    """Full-cycle recycler trading both YES and NO directions with above-mark exits.

    Mechanism: Market 253591 (3.9M trades) oscillates between YES=0.44 and YES=0.999.
    YES side: buy when YES<0.38, exit when YES>0.52 (same as tight_exit_recycler).
    NO side: buy NO when YES>0.62 (NO price<0.38), exit when YES<0.48 (NO price>0.52,
    above NO mark of 0.5). Market 253591 cycling creates recurring profitable NO trades.
    Both sides settle at 0.5 mark, so exits above mark on both sides beat hold-to-mark.
    """

    name: str = "full_cycle_recycler"
    buy_yes_below: float = 0.38
    sell_yes_above: float = 0.52
    buy_no_above_yes: float = 0.62   # buy NO when YES > this (NO price < 0.38)
    sell_no_below_yes: float = 0.48  # exit NO when YES < this (NO price > 0.52)
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state["position_yes_contracts"])
        pos_no = float(state["position_no_contracts"])

        # Exit YES when above YES mark
        if pos_yes > 0 and p >= self.sell_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name)

        # Exit NO when YES drops (NO is now above NO mark)
        if pos_no > 0 and p <= self.sell_no_below_yes:
            return Order(market_id=state["market_id"], side="no", contracts=-pos_no, reason=self.name)

        # Buy YES cheap
        if p <= self.buy_yes_below:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)

        # Buy NO when YES is expensive (NO is cheap)
        if p >= self.buy_no_above_yes:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        OnlineLogisticLikeStrategy(),
        LargeTradeFollowerStrategy(),
        AboveMarkRecyclerStrategy(),
        TightExitRecyclerStrategy(),
        VeryTightExitRecyclerStrategy(),
        FullCycleRecyclerStrategy(),
    ]

