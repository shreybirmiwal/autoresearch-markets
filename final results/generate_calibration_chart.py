"""
Chart: Expected PnL per contract vs. buy price.

In a perfectly calibrated market (no edge, no fees), a contract bought at price X
should return exactly $X on average, giving $0 PnL. This chart checks whether
that holds in the actual trade data.

For YES buys: buy_price = exec_yes_price
For NO  buys: buy_price = 1 - exec_yes_price  (what you actually paid per NO contract)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DARK_BG  = "#0f0f0f"
ACCENT   = "#00ff88"
RED      = "#ff4444"
YELLOW   = "#ffcc00"
GRAY     = "#888888"
WHITE    = "#e8e8e8"

df = pd.read_csv("results/trade_attribution.csv")

# Derive the actual "buy price" – what you paid per contract regardless of side
df["buy_price"] = np.where(df["side"] == "yes",
                           df["exec_yes_price"],
                           1.0 - df["exec_yes_price"])

# Bin into 5-cent buckets: [0.00, 0.05), [0.05, 0.10), …, [0.95, 1.00]
bins   = np.arange(0.0, 1.01, 0.05)
labels = [f"{b:.2f}" for b in bins[:-1]]
df["price_bin"] = pd.cut(df["buy_price"], bins=bins, labels=labels, include_lowest=True)

stats = (df.groupby("price_bin", observed=True)["pnl_per_contract"]
           .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), count="count")
           .reset_index())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 11),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor(DARK_BG)

ax = axes[0]
ax.set_facecolor(DARK_BG)

x     = np.arange(len(stats))
means = stats["mean"].values
sems  = stats["sem"].values

colors = [ACCENT if m >= 0 else RED for m in means]
bars   = ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
ax.errorbar(x, means, yerr=1.96 * sems, fmt="none",
            color=WHITE, linewidth=1.2, capsize=4, zorder=4)

ax.axhline(0, color=WHITE, linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)

# Annotate each bar with the mean value
for bar, m in zip(bars, means):
    color = WHITE if abs(m) < 0.005 else (ACCENT if m > 0 else RED)
    ax.text(bar.get_x() + bar.get_width() / 2, m + (0.003 if m >= 0 else -0.003),
            f"{m:+.3f}", ha="center", va="bottom" if m >= 0 else "top",
            color=color, fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(stats["price_bin"], rotation=45, ha="right",
                   color=WHITE, fontsize=9)
ax.tick_params(colors=WHITE)
ax.spines["bottom"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("Expected PnL per Contract vs. Buy Price\n"
             "(efficient market → every bar ≈ $0)",
             color=WHITE, fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Mean PnL per contract ($)", color=GRAY, fontsize=11)
ax.set_xlabel("Buy price bucket (5¢ bins)", color=GRAY, fontsize=11)

from matplotlib.patches import Patch
legend = [Patch(facecolor=ACCENT, label="Positive edge"),
          Patch(facecolor=RED,    label="Negative edge")]
ax.legend(handles=legend, facecolor="#1a1a1a", edgecolor=GRAY,
          labelcolor=WHITE, fontsize=10, loc="upper right")

# ── Lower panel: trade count per bin ─────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
ax2.bar(x, stats["count"], color=GRAY, alpha=0.7, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(stats["price_bin"], rotation=45, ha="right",
                    color=WHITE, fontsize=9)
ax2.tick_params(colors=WHITE)
ax2.spines["bottom"].set_color(GRAY)
ax2.spines["left"].set_color(GRAY)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylabel("# trades", color=GRAY, fontsize=10)
ax2.set_xlabel("Buy price bucket", color=GRAY, fontsize=10)
ax2.set_title("Trade count per bucket", color=WHITE, fontsize=11, pad=8)

plt.tight_layout(pad=2.0)
out = Path("charts/calibration_pnl_by_price.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"Saved → {out}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\nBucket  | Mean PnL/contract | ±95% CI   | N trades")
print("--------|-------------------|-----------|----------")
for _, row in stats.iterrows():
    ci = 1.96 * row["sem"]
    sig = "**" if abs(row["mean"]) > ci else "  "
    print(f"{row['price_bin']:7s} | {row['mean']:+.4f}           | ±{ci:.4f}  | {int(row['count']):6d}  {sig}")
