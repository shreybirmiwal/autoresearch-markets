"""
Calibration chart over ALL data lake trades (not just strategy trades).
For every trade: buy_price = what you paid per contract (YES: price_yes, NO: 1-price_yes)
                 outcome   = settlement (1 or 0)
                 pnl       = outcome - buy_price
In a perfectly calibrated market every 5c bucket should average $0.
"""

import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DARK_BG = "#0f0f0f"
ACCENT  = "#00ff88"
RED     = "#ff4444"
YELLOW  = "#ffcc00"
GRAY    = "#888888"
WHITE   = "#e8e8e8"

print("Loading data lake...")
trades  = pd.read_parquet(glob.glob("data_lake/trades/*.parquet")[0])
markets = pd.read_parquet(glob.glob("data_lake/markets/*.parquet")[0])

# Keep only resolved markets
markets = markets[markets["is_resolved"]][["market_id", "settlement_price_yes", "fee_bps"]]
trades  = trades.merge(markets, on="market_id", how="inner")
print(f"Trades joined with resolved markets: {len(trades):,}")

# Buy price: what you paid per contract regardless of side
trades["buy_price"] = np.where(trades["side"] == "yes",
                               trades["price_yes"],
                               1.0 - trades["price_yes"])

# Outcome: did your contract pay out?
trades["outcome"] = np.where(trades["side"] == "yes",
                              trades["settlement_price_yes"],
                              1.0 - trades["settlement_price_yes"])

# Raw PnL per contract (no fee adjustment — pure calibration signal)
trades["pnl"] = trades["outcome"] - trades["buy_price"]

# Bin into 5c buckets
bins   = np.arange(0.0, 1.01, 0.05)
labels = [f"{b:.2f}" for b in bins[:-1]]
trades["price_bin"] = pd.cut(trades["buy_price"], bins=bins, labels=labels, include_lowest=True)

stats = (trades.groupby("price_bin", observed=True)["pnl"]
               .agg(mean="mean",
                    sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
                    count="count")
               .reset_index())

print("\nBucket  | Mean PnL/contract | ±95% CI   | N trades")
print("--------|-------------------|-----------|----------")
for _, row in stats.iterrows():
    ci  = 1.96 * row["sem"]
    sig = "**" if abs(row["mean"]) > ci else "  "
    print(f"{row['price_bin']:7s} | {row['mean']:+.4f}           | ±{ci:.4f}  | {int(row['count']):10,}  {sig}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

x      = np.arange(len(stats))
means  = stats["mean"].values
sems   = stats["sem"].values
colors = [ACCENT if m >= 0 else RED for m in means]

bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
ax.errorbar(x, means, yerr=1.96 * sems, fmt="none",
            color=WHITE, linewidth=1.2, capsize=4, zorder=4)
ax.axhline(0, color=WHITE, linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)

for bar, m in zip(bars, means):
    color = WHITE if abs(m) < 0.001 else (ACCENT if m > 0 else RED)
    ax.text(bar.get_x() + bar.get_width() / 2,
            m + (0.001 if m >= 0 else -0.001),
            f"{m:+.3f}", ha="center", va="bottom" if m >= 0 else "top",
            color=color, fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(stats["price_bin"], rotation=45, ha="right", color=WHITE, fontsize=9)
ax.tick_params(colors=WHITE)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    f"Market Calibration: Expected PnL vs. Buy Price  ({len(trades):,.0f} trades, all data lake)\n"
    "Efficient market → every bar ≈ $0  |  Raw PnL = outcome − price (no fee adj.)",
    color=WHITE, fontsize=13, fontweight="bold", pad=14)
ax.set_ylabel("Mean PnL per contract ($)", color=GRAY, fontsize=11)
ax.set_xlabel("Buy price bucket (5¢ bins)", color=GRAY, fontsize=11)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=ACCENT, label="Positive edge"),
                   Patch(facecolor=RED,    label="Negative edge")],
          facecolor="#1a1a1a", edgecolor=GRAY, labelcolor=WHITE, fontsize=10, loc="lower left")


plt.tight_layout(pad=2.0)
out = Path("charts/calibration_all_trades.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"\nSaved → {out}")
