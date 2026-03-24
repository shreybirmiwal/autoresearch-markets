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
class PriceBandBuyYesStrategy(Strategy):
    """Buy YES only in the (0.20, 0.42] price band where avg trade PnL is highest.
    Exit (take profit) when price rises above 0.65, recycling capital.
    Avoids the <0.20 zone where 22k+ trades yield near-zero avg PnL (noise kills sharpe).
    """
    name: str = "price_band_yes"
    buy_yes_low: float = 0.20
    buy_yes_high: float = 0.42
    exit_yes_above: float = 0.65
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos = float(state.get("position_yes_contracts", 0.0))
        # Exit YES position if price has risen (take profit)
        if pos > 0 and p >= self.exit_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos, reason=self.name + "_exit")
        # Enter YES only in (0.20, 0.42] band
        if self.buy_yes_low < p <= self.buy_yes_high:
            return Order(market_id=state["market_id"], side="yes", contracts=self.order_size, reason=self.name)
        return None


@dataclass
class MeanReversionBandYesStrategy(Strategy):
    """Buy YES when price is BOTH in the profitable (0.20, 0.45] band AND z-score below rolling mean.
    Mechanism: mean_reversion YES in (0.2, 0.4] earns 0.685 avg PnL vs threshold_edge's 0.466.
    The extra discriminator is that z-score below mean filters out markets that are already trending down.
    Exit when price recovers above rolling mean (signal reversal) or hits 0.65 take-profit.
    Per-market rolling history to avoid cross-market contamination.
    """
    name: str = "mean_rev_band_yes"
    window: int = 30
    z_entry: float = 1.0
    buy_yes_low: float = 0.20
    buy_yes_high: float = 0.45
    exit_above: float = 0.65
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._market_history: dict[str, deque] = {}

    def reset(self) -> None:
        self._market_history.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])
        pos = float(state.get("position_yes_contracts", 0.0))

        if mid not in self._market_history:
            self._market_history[mid] = deque(maxlen=self.window)
        hist = self._market_history[mid]
        hist.append(p)

        # Exit: take profit or signal reversal
        if pos > 0 and p >= self.exit_above:
            return Order(market_id=mid, side="yes", contracts=-pos, reason=self.name + "_exit")

        if len(hist) < self.window:
            return None

        arr = np.array(hist, dtype=np.float64)
        mean = arr.mean()
        std = arr.std()
        if std <= 1e-9:
            return None

        z = (p - mean) / std

        # Exit: price has recovered above rolling mean
        if pos > 0 and z >= 0.0:
            return Order(market_id=mid, side="yes", contracts=-pos, reason=self.name + "_exit_mean")

        # Enter: price in profitable band AND below rolling mean
        if self.buy_yes_low < p <= self.buy_yes_high and z <= -self.z_entry:
            return Order(market_id=mid, side="yes", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class LocalMinBoostYesStrategy(Strategy):
    """Price band YES with double contracts at confirmed local minimums.
    Mechanism: buy YES in (0.20, 0.42] like price_band_yes, but 2x contracts when
    a local minimum is confirmed (price went DOWN last event, now going UP).
    Local min trades have 0.623 avg_pnl vs 0.455 for all band trades — size boost
    at reversal points extracts more from the highest-quality subset.
    Base size 1 contract; local min size 2 contracts; exit at 0.65.
    """
    name: str = "local_min_boost_yes"
    buy_yes_low: float = 0.20
    buy_yes_high: float = 0.42
    exit_above: float = 0.65
    base_size: float = 1.0
    boost_size: float = 2.0

    def __post_init__(self) -> None:
        self._market_prev_price: dict[str, float] = {}
        self._market_prev_prev_price: dict[str, float] = {}

    def reset(self) -> None:
        self._market_prev_price.clear()
        self._market_prev_prev_price.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])
        pos = float(state.get("position_yes_contracts", 0.0))

        prev = self._market_prev_price.get(mid)
        prev_prev = self._market_prev_prev_price.get(mid)

        # Update history
        if prev is not None:
            self._market_prev_prev_price[mid] = prev
        self._market_prev_price[mid] = p

        # Exit: take profit
        if pos > 0 and p >= self.exit_above:
            return Order(market_id=mid, side="yes", contracts=-pos, reason=self.name + "_exit")

        if self.buy_yes_low < p <= self.buy_yes_high:
            # Check for local minimum: down then up
            is_local_min = (prev is not None and prev_prev is not None
                           and p > prev and prev < prev_prev)
            contracts = self.boost_size if is_local_min else self.base_size
            return Order(market_id=mid, side="yes", contracts=contracts, reason=self.name)

        return None


@dataclass
class ExitAwareBandedStrategy(Strategy):
    """Two-sided strategy: buy cheap YES and short expensive YES via NO.
    YES side: buy YES in (0.20, 0.42], exit when price rises above 0.55 (capital recycling).
    NO side: buy NO when YES > 0.65 (short expensive YES).
    Prior session: score 0.1859, sharpe 1.98, pnl $471.
    Mechanism: cheap YES has high expected value; expensive YES (>0.65) tends to revert
    because markets near resolution often see final prices settle at true probability.
    Exit at 0.55 recycles YES capital faster, generating more entry opportunities.
    """
    name: str = "exit_aware_banded"
    buy_yes_low: float = 0.20
    buy_yes_high: float = 0.42
    exit_yes_above: float = 0.60
    buy_no_above: float = 0.75
    order_size: float = 1.0

    def reset(self) -> None:
        return None

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        pos_yes = float(state.get("position_yes_contracts", 0.0))
        pos_no = float(state.get("position_no_contracts", 0.0))

        # Exit YES position at take-profit (capital recycling)
        if pos_yes > 0 and p >= self.exit_yes_above:
            return Order(market_id=state["market_id"], side="yes", contracts=-pos_yes, reason=self.name + "_exit_yes")

        # Enter YES in cheap band with confidence-scaled sizing
        if self.buy_yes_low < p <= self.buy_yes_high:
            contracts = min(3.0, max(1.0, (0.50 - p) / 0.10))
            return Order(market_id=state["market_id"], side="yes", contracts=contracts, reason=self.name)

        # Enter NO when YES is expensive (short expensive YES)
        if p > self.buy_no_above:
            return Order(market_id=state["market_id"], side="no", contracts=self.order_size, reason=self.name)

        return None


@dataclass
class ConfirmationDriftYesStrategy(Strategy):
    """Buy YES in late-stage confirmation drift: price rising in (0.28, 0.50] for 5+ events.
    Mechanism (from prediction market research): markets that have been gradually rising
    toward YES resolution show momentum confirmation. When price is in 0.3-0.5 AND
    last 5 prices all strictly increasing, the market is in a YES convergence trajectory.
    This differs from pure price-band (catches upward momentum not mean-reversion).
    Exit at 0.70 (take profit before resolution convergence slows down).
    """
    name: str = "conf_drift_yes"
    buy_yes_low: float = 0.28
    buy_yes_high: float = 0.52
    exit_above: float = 0.70
    n_rises: int = 5
    order_size: float = 1.0

    def __post_init__(self) -> None:
        self._market_prices: dict[str, deque] = {}

    def reset(self) -> None:
        self._market_prices.clear()

    def on_event(self, state: dict[str, Any]) -> Order | None:
        mid = state["market_id"]
        p = float(state["yes_price"])
        pos = float(state.get("position_yes_contracts", 0.0))

        if mid not in self._market_prices:
            self._market_prices[mid] = deque(maxlen=self.n_rises + 1)
        prices = self._market_prices[mid]
        prices.append(p)

        # Exit: take profit
        if pos > 0 and p >= self.exit_above:
            return Order(market_id=mid, side="yes", contracts=-pos, reason=self.name + "_exit")

        if len(prices) < self.n_rises + 1:
            return None

        # Check: last n_rises prices all strictly increasing
        price_list = list(prices)
        all_rising = all(price_list[i] < price_list[i+1] for i in range(len(price_list) - 1))

        if all_rising and self.buy_yes_low < p <= self.buy_yes_high:
            return Order(market_id=mid, side="yes", contracts=self.order_size, reason=self.name)

        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        ThresholdEdgeStrategy(),
        MeanReversionStrategy(),
        PriceBandBuyYesStrategy(),
        MeanReversionBandYesStrategy(),
        LocalMinBoostYesStrategy(),
        ExitAwareBandedStrategy(),
        ConfirmationDriftYesStrategy(),
    ]

