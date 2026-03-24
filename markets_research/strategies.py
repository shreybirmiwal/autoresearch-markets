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
    """Buy YES extended range when EITHER in good UTC hours OR in a cheap cluster
    OR in a diverse-market sequence (≥3 distinct markets in last 5 events),
    with a market-switch gate for extended range buys.
    Mechanism:
      1. Base: always buy YES ≤ 0.45.
      2. Extended (0.45-0.60): buy when continuation (not market switch) AND
         (good hours UTC 3-11 OR ≥40% of last 5 events were cheap
          OR ≥3 distinct markets appeared in last 5 events).
      3. Market-switch gate: skip extended at first event from any market (run_pos=1).
      4. Diverse-market signal: when ≥3 distinct markets appear in last 5 events,
         cheap markets are actively interleaving. Extended range buy executes at one of
         those cheap markets → excellent execution quality (avg PnL 0.107 vs 0.006 for
         single-market runs).
    fit() uses n*2//3 window to estimate adaptive order sizes per market.
    """
    name: str = "hybrid_edge"
    base_threshold: float = 0.45
    extended_threshold: float = 0.60
    good_hour_start: int = 3
    good_hour_end: int = 11
    rolling_window: int = 5
    cheap_fraction_min: float = 0.40
    diverse_market_min: int = 3
    order_size: float = 0.65
    position_cap: float = 500.0

    def reset(self) -> None:
        self._market_sizes: dict[str, float] = {}
        self._recent_prices: deque = deque(maxlen=self.rolling_window)
        self._recent_market_ids: deque = deque(maxlen=self.rolling_window)
        self._prev_market_id: Any = None
        return None

    def _hour_of(self, ts: Any) -> int | None:
        try:
            return int(pd.Timestamp(ts).hour)
        except Exception:
            return None

    def _cheap_conditions(self, ts: Any) -> bool:
        h = self._hour_of(ts)
        if h is not None and self.good_hour_start <= h <= self.good_hour_end:
            return True
        if len(self._recent_prices) >= self.rolling_window:
            cheap_frac = sum(1 for p in self._recent_prices if p <= self.base_threshold) / self.rolling_window
            if cheap_frac >= self.cheap_fraction_min:
                return True
        # Diverse-market signal: many active markets = cheap interleaving = good execution
        if len(self._recent_market_ids) >= self.rolling_window:
            if len(set(self._recent_market_ids)) >= self.diverse_market_min:
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
        recent: deque = deque(maxlen=self.rolling_window)
        recent_mids: deque = deque(maxlen=self.rolling_window)
        prev_mid: Any = None
        counts: dict[str, int] = defaultdict(int)
        for event in window:
            p = float(event["price_yes"])
            ts = event.get("event_ts")
            mid = str(event["market_id"])
            qualifies = False
            if p <= self.base_threshold:
                qualifies = True
            elif p <= self.extended_threshold and mid == prev_mid:
                h = self._hour_of(ts)
                if h is not None and self.good_hour_start <= h <= self.good_hour_end:
                    qualifies = True
                elif len(recent) >= self.rolling_window:
                    cheap_frac = sum(1 for rp in recent if rp <= self.base_threshold) / self.rolling_window
                    if cheap_frac >= self.cheap_fraction_min:
                        qualifies = True
                elif len(recent_mids) >= self.rolling_window and len(set(recent_mids)) >= self.diverse_market_min:
                    qualifies = True
            if qualifies:
                counts[mid] += 1
            recent.append(p)
            recent_mids.append(mid)
            prev_mid = mid

        self._market_sizes = {}
        for market_id, count in counts.items():
            if count >= 10:
                optimal = self.position_cap / count
                self._market_sizes[market_id] = max(0.01, min(self.order_size, optimal))

    def on_event(self, state: dict[str, Any]) -> Order | None:
        p = float(state["yes_price"])
        market_id = state["market_id"]
        qualifies = self._qualifies(p, state.get("event_ts"), market_id)
        self._recent_prices.append(p)
        self._recent_market_ids.append(str(market_id))
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
