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
    """Buy YES extended range when EITHER in good UTC hours OR in a cheap cluster,
    with a market-switch gate and optional VWAP signal.
    Mechanism:
      1. Base: always buy YES ≤ 0.45.
      2. Extended (0.45-0.60): buy when continuation (not market switch) AND
         (good hours UTC 3-11 OR cheap fraction ≥ 40% of last 20 events OR
          VWAP of last 20 events ≤ 0.30).
      3. Market-switch gate: skip extended at first event from any market (run_pos=1).
         Empirical: market-switch extended events have 11% cheap execution → unprofitable.
    fit() uses n*2//3 window to estimate adaptive order sizes per market.
    """
    name: str = "hybrid_edge"
    base_threshold: float = 0.45
    extended_threshold: float = 0.60
    good_hour_start: int = 3
    good_hour_end: int = 11
    rolling_window: int = 5
    cheap_fraction_min: float = 0.40
    vwap_threshold: float = 0.30  # VWAP signal: if recent VWAP ≤ this, extend
    order_size: float = 0.65
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        self._recent_prices: deque = deque(maxlen=self.rolling_window)
        self._recent_sizes: deque = deque(maxlen=self.rolling_window)
        self._prev_market_id: Any = None
        return None

    def _hour_of(self, ts: Any) -> int | None:
        try:
            return int(pd.Timestamp(ts).hour)
        except Exception:
            return None

    def _cheap_conditions(self, ts: Any) -> bool:
        """Return True if time-of-day or cluster signal warrants extended buying."""
        h = self._hour_of(ts)
        if h is not None and self.good_hour_start <= h <= self.good_hour_end:
            return True
        if len(self._recent_prices) >= self.rolling_window:
            cheap_frac = sum(1 for p in self._recent_prices if p <= self.base_threshold) / self.rolling_window
            if cheap_frac >= self.cheap_fraction_min:
                return True
            total_size = sum(self._recent_sizes)
            if total_size > 0:
                vwap = sum(p * s for p, s in zip(self._recent_prices, self._recent_sizes)) / total_size
                if vwap <= self.vwap_threshold:
                    return True
        return False

    def _qualifies(self, price: float, ts: Any, market_id: Any) -> bool:
        if price <= self.base_threshold:
            return True
        if price <= self.extended_threshold:
            if market_id != self._prev_market_id:
                return False  # market-switch gate
            return self._cheap_conditions(ts)
        return False

    def fit(self, train_events: list[dict[str, Any]]) -> None:
        n = len(train_events)
        window = train_events[n * 2 // 3:]
        recent_p: deque = deque(maxlen=self.rolling_window)
        recent_s: deque = deque(maxlen=self.rolling_window)
        prev_mid: Any = None
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            p = float(event["price_yes"])
            s = float(event.get("size", 1.0))
            ts = event.get("event_ts")
            mid = str(event["market_id"])
            qualifies = False
            if p <= self.base_threshold:
                qualifies = True
            elif p <= self.extended_threshold and mid == prev_mid:
                h = self._hour_of(ts)
                if h is not None and self.good_hour_start <= h <= self.good_hour_end:
                    qualifies = True
                elif len(recent_p) >= self.rolling_window:
                    cheap_frac = sum(1 for rp in recent_p if rp <= self.base_threshold) / self.rolling_window
                    if cheap_frac >= self.cheap_fraction_min:
                        qualifies = True
                    else:
                        total_s = sum(recent_s)
                        if total_s > 0:
                            vwap = sum(rp * rs for rp, rs in zip(recent_p, recent_s)) / total_s
                            if vwap <= self.vwap_threshold:
                                qualifies = True
            if qualifies:
                counts[mid] += 1
            recent_p.append(p)
            recent_s.append(s)
            prev_mid = mid

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                optimal = self.position_cap / count
                self._market_sizes[market_id] = max(0.01, min(self.order_size, optimal))

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        s = float(state.get("size", 1.0))
        market_id = state["market_id"]
        qualifies = self._qualifies(p, state.get("event_ts"), market_id)
        self._recent_prices.append(p)
        self._recent_sizes.append(s)
        self._prev_market_id = market_id
        if qualifies:
            mid = str(market_id)
            size = self._market_sizes.get(mid, self.order_size)
            return Order(
                market_id=market_id,
                side="yes",
                contracts=size,
                reason=self.name,
            )
        return None


def default_strategy_registry() -> list[Strategy]:
    return [
        HybridEdgeStrategy(),
    ]
