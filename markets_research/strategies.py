from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import pandas as pd

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
class HybridEdgeStrategy(Strategy):
    """Buy YES extended range when EITHER in good UTC hours OR in a cheap price cluster.
    Mechanism: combines two independent signals that both identify when the global sequence
    is in a cheap execution state:
      1. Time-of-day: UTC 3-11 (US nighttime) has fewer expensive political market events
         contaminating the sequence → better execution for 0.45-0.58 range buys.
      2. Cheap cluster: when ≥40% of last 20 global events had price ≤ 0.45, the sequence
         is currently in a cheap cluster → next execution likely cheap → extend to 0.58.
    Both signals find when extended buying (up to 0.58) has positive expected value.
    fit() uses n*2//3 window to estimate adaptive order sizes per market.
    """
    name: str = "hybrid_edge"
    base_threshold: float = 0.45
    extended_threshold: float = 0.58
    good_hour_start: int = 3
    good_hour_end: int = 11
    rolling_window: int = 20
    cheap_fraction_min: float = 0.40
    order_size: float = 0.65
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        self._recent_prices: deque = deque(maxlen=self.rolling_window)
        return None

    def _hour_of(self, ts: Any) -> int | None:
        try:
            return int(pd.Timestamp(ts).hour)
        except Exception:
            return None

    def _qualifies(self, price: float, ts: Any) -> bool:
        if price <= self.base_threshold:
            return True
        if price <= self.extended_threshold:
            h = self._hour_of(ts)
            if h is not None and self.good_hour_start <= h <= self.good_hour_end:
                return True
            if len(self._recent_prices) >= self.rolling_window:
                cheap_frac = sum(1 for p in self._recent_prices if p <= self.base_threshold) / self.rolling_window
                if cheap_frac >= self.cheap_fraction_min:
                    return True
        return False

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        n = len(train_events)
        window = train_events[n * 2 // 3:]
        recent: deque = deque(maxlen=self.rolling_window)
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            p = float(event["price_yes"])
            ts = event.get("event_ts")
            qualifies = False
            if p <= self.base_threshold:
                qualifies = True
            elif p <= self.extended_threshold:
                h = self._hour_of(ts)
                if h is not None and self.good_hour_start <= h <= self.good_hour_end:
                    qualifies = True
                elif len(recent) >= self.rolling_window:
                    cheap_frac = sum(1 for rp in recent if rp <= self.base_threshold) / self.rolling_window
                    if cheap_frac >= self.cheap_fraction_min:
                        qualifies = True
            if qualifies:
                counts[str(event["market_id"])] += 1
            recent.append(p)

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                optimal = self.position_cap / count
                self._market_sizes[market_id] = max(0.01, min(self.order_size, optimal))

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        qualifies = self._qualifies(p, state.get("event_ts"))
        self._recent_prices.append(p)
        if qualifies:
            market_id = str(state["market_id"])
            size = self._market_sizes.get(market_id, self.order_size)
            return Order(
                market_id=state["market_id"],
                side="yes",
                contracts=size,
                reason=self.name,
            )
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        HybridEdgeStrategy(),
    ]
