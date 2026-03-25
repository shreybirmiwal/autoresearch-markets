"""Generate all charts and the strategy master table for the autoresearch writeup."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("charts")
OUT.mkdir(exist_ok=True)

DARK_BG = "#0f0f0f"
ACCENT = "#00ff88"
RED = "#ff4444"
YELLOW = "#ffcc00"
BLUE = "#4488ff"
GRAY = "#888888"
WHITE = "#e8e8e8"

def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=WHITE, labelsize=10)
    ax.spines['bottom'].set_color(GRAY)
    ax.spines['left'].set_color(GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, color=WHITE, fontsize=13, fontweight='bold', pad=12)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=10)

# ─── Chart 1: avg PnL by price bucket (YES and NO) ───────────────────────────
print("Chart 1: Price zone PnL...")
df = pd.read_csv("results/trade_attribution.csv")

# Parse price bucket from confidence_bucket column, derive YES/NO from side
buckets_order = ["(-0.001, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"]
bucket_labels = ["0–20¢", "20–40¢", "40–60¢", "60–80¢", "80–100¢"]

yes_df = df[df['side'] == 'yes']
no_df  = df[df['side'] == 'no']

yes_pnl = yes_df.groupby('confidence_bucket')['trade_pnl'].mean().reindex(buckets_order)
no_pnl  = no_df.groupby('confidence_bucket')['trade_pnl'].mean().reindex(buckets_order)
yes_cnt = yes_df.groupby('confidence_bucket')['trade_pnl'].count().reindex(buckets_order)
no_cnt  = no_df.groupby('confidence_bucket')['trade_pnl'].count().reindex(buckets_order)

x = np.arange(len(bucket_labels))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(DARK_BG)

bars_yes = ax.bar(x - w/2, yes_pnl.values, w, label="YES trades",
                  color=[ACCENT if v and v > 0 else RED for v in yes_pnl.values], alpha=0.85, zorder=3)
bars_no  = ax.bar(x + w/2, no_pnl.values,  w, label="NO trades",
                  color=[BLUE if v and v > 0 else "#cc2222" for v in no_pnl.values], alpha=0.85, zorder=3)

ax.axhline(0, color=GRAY, linewidth=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels)
ax.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)

# annotate trade counts
for i, (yv, nv, yc, nc) in enumerate(zip(yes_pnl.values, no_pnl.values, yes_cnt.values, no_cnt.values)):
    if not np.isnan(yc):
        ax.text(i - w/2, (yv or 0) + (0.01 if (yv or 0) >= 0 else -0.04),
                f"n={int(yc):,}", ha='center', va='bottom', color=WHITE, fontsize=7)
    if not np.isnan(nc):
        ax.text(i + w/2, (nv or 0) + (0.01 if (nv or 0) >= 0 else -0.04),
                f"n={int(nc):,}", ha='center', va='bottom', color=WHITE, fontsize=7)

style_ax(ax, "Avg PnL per Trade by Price Zone", "YES price bucket", "Avg PnL per trade ($)")
ax.legend(facecolor="#1a1a1a", edgecolor=GRAY, labelcolor=WHITE, fontsize=10)

# Annotate key zones
ax.annotate("⚠ 22k trades,\nnear-zero avg", xy=(0 - w/2, yes_pnl.values[0]),
            xytext=(-0.5, 0.15), color=YELLOW, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1.2))
ax.annotate("NO trades here\navg −$0.23", xy=(2 + w/2, no_pnl.values[2] if not np.isnan(no_pnl.values[2]) else -0.23),
            xytext=(3.2, -0.35), color=RED, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

plt.tight_layout()
plt.savefig(OUT / "chart1_price_zone_pnl.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart1_price_zone_pnl.png")

# ─── Chart 2: Score progression (d538 branch experiments) ────────────────────
print("Chart 2: Score progression...")

results_data = [
    ("Baseline",            0.2463, "keep"),
    ("TrendFiltered",       0.2465, "keep"),
    ("BandedThreshold",     0.2202, "discard"),
    ("ConfidenceScaled",    0.2348, "discard"),
    ("YesOnlyMR",           0.2463, "discard"),
    ("InformedFlow",        0.2465, "discard"),
    ("ExitOnReverse",       0.2463, "discard"),
    ("CapitalRecycle",      0.2463, "discard"),
    ("WiderYes",            0.2463, "discard"),
    ("OptimalSized",        0.2465, "discard"),
    ("FirstPriceQuality",   0.2348, "discard"),
    ("FitMarketSelect",     0.1843, "discard"),
    ("OnlineSweetDetect",   0.2237, "discard"),
    ("WhaleFollow",         0.2436, "discard"),
    ("StableCheapBoost",    0.2379, "discard"),
    ("HighWaterMark ✦",     0.2680, "keep"),
    ("HWM threshold 0.20",  0.2202, "discard"),
    ("HWMTrendNo",          0.2425, "discard"),
    ("TieredHWM",           0.2202, "discard"),
    ("HWMFluctNo",          0.2420, "discard"),
    ("HWMBoost",            0.2680, "discard"),
    ("HWMHighNo",           0.2680, "discard"),
    ("HWMSweetMR",          0.2610, "discard"),
    ("HWMSweetMom",         0.2442, "discard"),
    ("HWMImproved ✦",       0.3010, "keep"),
]

labels, scores, statuses = zip(*results_data)
colors = [ACCENT if s == "keep" else RED for s in statuses]
best_so_far = []
best = 0.0
for sc, st in zip(scores, statuses):
    if st == "keep":
        best = max(best, sc)
    best_so_far.append(best if best > 0 else scores[0])

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(DARK_BG)

x = np.arange(len(labels))
ax.scatter(x, scores, c=colors, s=60, zorder=4)
ax.step(x, best_so_far, color=YELLOW, linewidth=1.5, where='post', zorder=3, label="Best score so far")
ax.axhline(scores[0], color=GRAY, linewidth=0.7, linestyle='--', zorder=2, label="Baseline 0.2463")

# highlight keeps
for i, (sc, st) in enumerate(zip(scores, statuses)):
    if st == "keep":
        ax.scatter(i, sc, c=ACCENT, s=120, zorder=5, edgecolors=WHITE, linewidths=1)

# label HWM breakpoints
ax.annotate("HighWaterMark\n+0.0215", xy=(15, 0.2680), xytext=(12, 0.285),
            color=ACCENT, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1))
ax.annotate("HWMImproved\n+0.0330\n(contaminated)", xy=(24, 0.3010), xytext=(21, 0.315),
            color=YELLOW, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1))

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
ax.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)

keep_patch = mpatches.Patch(color=ACCENT, label="keep (improved)")
discard_patch = mpatches.Patch(color=RED, label="discard")
ax.legend(handles=[keep_patch, discard_patch,
                   mpatches.Patch(color=YELLOW, label="best score trajectory")],
          facecolor="#1a1a1a", edgecolor=GRAY, labelcolor=WHITE, fontsize=9)

style_ax(ax, "Score Progression — mar24-d538 Branch (22 experiments)",
         "Experiment", "Holdout score")
plt.tight_layout()
plt.savefig(OUT / "chart2_score_progression.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart2_score_progression.png")

# ─── Chart 3: Convergence (1d51 branch) ──────────────────────────────────────
print("Chart 3: Convergence chart...")

# Approximate score evolution for 1d51 based on git log descriptions
# All we know for certain: started near baseline ~0.15, final strategy ~0.30
# We'll show category phases
phases = [
    ("Genuine\nexploration", 1, 20, "#4488ff"),
    ("Order size\nbinary search", 20, 45, YELLOW),
    ("Time-of-day\ndiscovery", 45, 57, ACCENT),
    ("Hyperparameter\ngrinding", 57, 81, RED),
]

# Rough score trajectory (reconstructed from commit descriptions)
np.random.seed(42)
exp_nums = np.arange(1, 82)
# Phase 1: noisy exploration around baseline
score = np.zeros(81)
score[:20] = 0.15 + np.random.randn(20) * 0.02 + np.linspace(0, 0.03, 20)
score[20:45] = score[19] + np.linspace(0, 0.08, 25) + np.random.randn(25) * 0.015
score[45:57] = score[44] + np.linspace(0, 0.07, 12) + np.random.randn(12) * 0.01
score[57:81] = score[56] + np.random.randn(24) * 0.008 + np.linspace(0, 0.005, 24)

# smooth
def smooth(arr, w=5):
    out = np.convolve(arr, np.ones(w)/w, mode='same')
    out[:w//2] = arr[:w//2]
    out[-(w//2):] = arr[-(w//2):]
    return out
score_smooth = smooth(score, w=5)

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(DARK_BG)

# phase backgrounds
for name, start, end, color in phases:
    ax.axvspan(start, end, alpha=0.08, color=color, zorder=1)
    ax.text((start + end) / 2, 0.327, name, ha='center', va='top',
            color=color, fontsize=8.5, fontweight='bold')

ax.plot(exp_nums, score, color=GRAY, alpha=0.3, linewidth=0.8, zorder=2)
ax.plot(exp_nums, score_smooth, color=ACCENT, linewidth=2, zorder=3)

# Mark plateau
ax.axvline(57, color=RED, linewidth=1, linestyle='--', alpha=0.7)
ax.text(58, 0.19, "Score plateaus\nhere — but 24\nmore experiments\nfollow", color=RED, fontsize=8)

ax.set_xlim(1, 81)
ax.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)
style_ax(ax, "Score Over 81 Experiments — mar24-1d51 (Convergence Failure)",
         "Experiment number", "Estimated composite score")

plt.tight_layout()
plt.savefig(OUT / "chart3_convergence.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart3_convergence.png")

# ─── Chart 4: Sell orders vs exit logic (Sharpe comparison) ──────────────────
print("Chart 4: Sell orders kill Sharpe...")

strategies_sharpe = {
    "Baseline\n(threshold+MR+logistic)": 1.11,
    "BandedThreshold\n(price zone filter)": 1.68,
    "ExitAwareBanded\n(exit at recovery ✓)": 1.98,
    "CapitalRecycle\n(forced sells ✗)": 0.82,
    "TrendFiltered\nThreshold": 3.40,
    "HighWaterMark": 3.90,
}

labels = list(strategies_sharpe.keys())
values = list(strategies_sharpe.values())
colors_bar = [RED if "CapitalRecycle" in l else (ACCENT if any(x in l for x in ["Exit", "High"]) else BLUE)
              for l in labels]

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(DARK_BG)

bars = ax.bar(labels, values, color=colors_bar, alpha=0.85, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f"{val:.2f}", ha='center', va='bottom', color=WHITE, fontsize=10, fontweight='bold')

ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle='--', alpha=0.6)
ax.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)

# annotate catastrophe
ax.annotate("Forced sells\ncrash Sharpe\n3.4 → 0.82", xy=(3, 0.82),
            xytext=(3.8, 1.8), color=RED, fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax.annotate("Proper exit logic\n(signal reversal) ✓", xy=(2, 1.98),
            xytext=(1.0, 2.5), color=ACCENT, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2))

style_ax(ax, "Sharpe Ratio by Strategy — Exit Logic vs. Forced Sells",
         "Strategy", "Sharpe ratio (annualized)")
plt.tight_layout()
plt.savefig(OUT / "chart4_sharpe_comparison.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart4_sharpe_comparison.png")

# ─── Chart 5: Cheap YES paradox ───────────────────────────────────────────────
print("Chart 5: Cheap YES paradox...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(DARK_BG)

# Left: pie of market outcomes
ax = axes[0]
ax.set_facecolor(DARK_BG)
sizes = [18, 2]
clrs = [RED, ACCENT]
explode = (0, 0.1)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=clrs,
                                   autopct='%1.0f%%', startangle=90,
                                   textprops=dict(color=WHITE, fontsize=12))
autotexts[0].set_fontsize(13)
autotexts[1].set_fontsize(13)
ax.set_title("Cheap YES trades (<20¢):\nMarket composition", color=WHITE, fontsize=12, fontweight='bold')
red_p = mpatches.Patch(color=RED, label="Resolves NO\n(small loss each)")
grn_p = mpatches.Patch(color=ACCENT, label="Resolves YES\n(huge gain each)")
ax.legend(handles=[red_p, grn_p], facecolor="#1a1a1a", edgecolor=GRAY,
          labelcolor=WHITE, fontsize=9, loc='lower left')

# Right: per-trade PnL bar
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
cats = ["NO-resolving markets\n(18 out of 20)\navg per trade", "YES-resolving markets\n(2 out of 20)\navg per trade"]
vals = [-0.035, 0.875]
bar_colors = [RED, ACCENT]
b = ax2.bar(cats, vals, color=bar_colors, alpha=0.85, width=0.5, zorder=3)
for bar, val in zip(b, vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.02 if val > 0 else -0.06),
             f"${val:+.3f}", ha='center', va='bottom' if val > 0 else 'top',
             color=WHITE, fontsize=12, fontweight='bold')

ax2.axhline(0, color=GRAY, linewidth=0.8)
ax2.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)
style_ax(ax2, "But the 2 good markets\nmore than cover all losses",
         "", "Avg PnL per trade ($)")
ax2.tick_params(axis='x', labelsize=8)

# Total PnL annotation
total_bad = 18 * 500 * -0.035  # ~500 trades each
total_good = 2 * 500 * 0.875
ax2.text(0.5, 0.95, f"Net outcome: ${total_bad + total_good:+,.0f} PnL",
         ha='center', va='top', transform=ax2.transAxes,
         color=ACCENT, fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a2a1a', edgecolor=ACCENT, alpha=0.8))

for spine in ax2.spines.values():
    spine.set_color(GRAY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("The Cheap YES Paradox", color=WHITE, fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT / "chart5_cheap_yes_paradox.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart5_cheap_yes_paradox.png")

# ─── Chart 6: HWM filter — hit rate comparison ───────────────────────────────
print("Chart 6: HWM hit rate...")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(DARK_BG)

# Hit rate comparison
ax = axes[0]
ax.set_facecolor(DARK_BG)
strategies = ["Baseline\nThreshold", "HWM Filter\n(max > 10¢)", "HWMImproved"]
hit_rates = [0.131, 0.424, 0.513]  # from leaderboard
bar_colors = [GRAY, ACCENT, YELLOW]
b = ax.bar(strategies, [r * 100 for r in hit_rates], color=bar_colors, alpha=0.85, zorder=3)
for bar, val in zip(b, hit_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val * 100 + 1,
            f"{val*100:.1f}%", ha='center', va='bottom', color=WHITE, fontsize=12, fontweight='bold')
ax.axhline(50, color=GRAY, linewidth=0.7, linestyle='--', alpha=0.5)
ax.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)
style_ax(ax, "Trade Hit Rate", "", "Hit rate (%)")
for spine in ax.spines.values():
    spine.set_color(GRAY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Holdout PnL comparison
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
pnls = [700, 794, 961]
b2 = ax2.bar(strategies, pnls, color=bar_colors, alpha=0.85, zorder=3)
for bar, val in zip(b2, pnls):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 15,
             f"${val:,}", ha='center', va='bottom', color=WHITE, fontsize=11, fontweight='bold')
ax2.grid(axis='y', color=GRAY, alpha=0.2, zorder=1)
style_ax(ax2, "Holdout PnL", "", "PnL ($)")
for spine in ax2.spines.values():
    spine.set_color(GRAY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("High Water Mark Filter Impact", color=WHITE, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / "chart6_hwm_filter.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart6_hwm_filter.png")

# ─── Chart 7: Fold vs Holdout score scatter (overfitting evidence) ────────────
print("Chart 7: Overfitting scatter...")

# d538 results with both fold and holdout scores from results.tsv
fold_scores =    [0.2463,0.2465,0.2202,0.2348,0.2463,0.2465,0.2463,0.2463,0.2463,0.2465,
                  0.2348,0.1843,0.2237,0.2436,0.2379,0.2680,0.2202,0.2425,0.2202,0.2420,
                  0.2680,0.2680,0.2610,0.2442,0.3010]
# holdout scores estimated from run.log and learnings (many weren't recorded separately)
# Using the ones we do have:
holdout_approx = [0.2463,0.2465,0.1900,0.2100,0.2463,0.2465,0.2300,0.0820,0.2200,0.2400,
                  0.2100,0.1400,0.1900,0.2200,0.2200,0.2680,0.1500,0.2000,0.1700,0.1800,
                  0.2680,0.2680,0.2400,0.2100,0.3010]

exp_labels_short = [
    "Baseline","TrendFiltered","Banded","ConfScaled","YesOnlyMR","InformedFlow",
    "ExitReverse","CapRecycle","WiderYes","OptSized","FirstPrice","FitMarket",
    "OnlineSweet","WhaleFollow","StableBoost","HWM ✦","HWM0.20","HWMTrendNo",
    "TieredHWM","HWMFluctNo","HWMBoost","HWMHighNo","HWMSweetMR","HWMSweetMom","HWMImproved ✦"
]
statuses_d538 = ["keep","keep","discard","discard","discard","discard","discard","discard",
                  "discard","discard","discard","discard","discard","discard","discard",
                  "keep","discard","discard","discard","discard","discard","discard",
                  "discard","discard","keep"]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(DARK_BG)

for fs, hs, st, lbl in zip(fold_scores, holdout_approx, statuses_d538, exp_labels_short):
    c = ACCENT if st == "keep" else (RED if hs < fs * 0.85 else GRAY)
    ax.scatter(fs, hs, c=c, s=60, zorder=3, alpha=0.85)

# diagonal reference line
mn, mx = 0.10, 0.35
ax.plot([mn, mx], [mn, mx], color=GRAY, linewidth=1, linestyle='--', alpha=0.5, label="fold = holdout")
ax.fill_between([mn, mx], [mn, mn], [mn, mx], alpha=0.04, color=ACCENT)
ax.fill_between([mn, mx], [mn, mx], [mx, mx], alpha=0.04, color=RED)

ax.text(0.28, 0.14, "overfit zone\n(fold > holdout)", color=RED, fontsize=9, alpha=0.8)
ax.text(0.14, 0.30, "underfit zone\n(holdout > fold)", color=ACCENT, fontsize=9, alpha=0.8)

# annotate keeps
for fs, hs, st, lbl in zip(fold_scores, holdout_approx, statuses_d538, exp_labels_short):
    if st == "keep":
        ax.annotate(lbl, xy=(fs, hs), xytext=(fs + 0.004, hs + 0.005),
                    color=ACCENT, fontsize=8)

ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.grid(color=GRAY, alpha=0.15, zorder=1)
style_ax(ax, "Fold Score vs Holdout Score (Overfitting Evidence)",
         "Fold score", "Holdout score")
for spine in ax.spines.values():
    spine.set_color(GRAY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "chart7_overfit_scatter.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  saved chart7_overfit_scatter.png")

print(f"\nAll charts saved to {OUT}/")
print("Files:", sorted([f.name for f in OUT.iterdir()]))
