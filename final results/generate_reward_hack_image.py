"""Generate reward hacking illustration for tweet."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

Path("charts").mkdir(exist_ok=True)

DARK_BG   = "#0a0a0a"
PANEL_BG  = "#131313"
RED       = "#ff3333"
ORANGE    = "#ff8800"
YELLOW    = "#ffcc00"
GREEN     = "#00ff88"
GRAY      = "#555555"
LGRAY     = "#999999"
WHITE     = "#f0f0f0"
ACCENT    = "#cc44ff"   # purple for "hacking" vibe

fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.93, "Autoresearch ❤ Reward Hacking",
         ha='center', va='top', fontsize=22, fontweight='bold', color=WHITE,
         path_effects=[pe.withStroke(linewidth=4, foreground=ACCENT)])
fig.text(0.5, 0.875, "3 ways the agent learned to cheat before we patched it",
         ha='center', va='top', fontsize=12, color=LGRAY)

# ── Three panels ──────────────────────────────────────────────────────────────
axes = []
for i, (left, title_color) in enumerate([(0.04, RED), (0.37, ORANGE), (0.70, YELLOW)]):
    ax = fig.add_axes([left, 0.08, 0.28, 0.74])
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    # panel border
    for spine_name in ['top','bottom','left','right']:
        pass
    rect = FancyBboxPatch((0.02, 0.02), 9.96, 9.96,
                           boxstyle="round,pad=0.1",
                           linewidth=1.5, edgecolor=title_color, facecolor=PANEL_BG,
                           transform=ax.transData, zorder=0)
    ax.add_patch(rect)
    axes.append(ax)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1: Z-score gaming
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

ax.text(5, 9.3, "HACK #1", ha='center', fontsize=11, fontweight='bold',
        color=RED, transform=ax.transData)
ax.text(5, 8.7, "Z-score widening", ha='center', fontsize=9.5, color=WHITE,
        transform=ax.transData)

# show two distributions: narrow (before) and wide (after adding trash)
np.random.seed(1)
x_range = np.linspace(-4, 4, 300)

# Before: tight distribution, good strategy not standing out
sigma_before = 0.8
y_before = np.exp(-x_range**2 / (2*sigma_before**2))
y_before /= y_before.max()

# After: wide distribution, good strategy looks better by comparison
sigma_after = 1.8
y_after = np.exp(-x_range**2 / (2*sigma_after**2))
y_after /= y_after.max()

# scale to plot coords: x: 0.5-9.5, y: 1.5-5.5
def to_plot(xv, yv, xmin=-4, xmax=4, ybase=1.5, yscale=2.5):
    xp = 0.5 + (xv - xmin) / (xmax - xmin) * 9.0
    yp = ybase + yv * yscale
    return xp, yp

xp_b, yp_b = to_plot(x_range, y_before)
xp_a, yp_a = to_plot(x_range, y_after)

ax.plot(xp_b, yp_b, color=GRAY, linewidth=1.5, linestyle='--', label='before', zorder=2)
ax.fill_between(xp_b, 1.5, yp_b, alpha=0.15, color=GRAY, zorder=1)

ax.plot(xp_a, yp_a, color=RED, linewidth=2, label='after adding trash', zorder=3)
ax.fill_between(xp_a, 1.5, yp_a, alpha=0.20, color=RED, zorder=2)

# mark "good strategy" at z=1.5
z_good = 1.5
xg_b, yg_b = to_plot(np.array([z_good]), np.array([np.exp(-z_good**2/(2*sigma_before**2))]))
xg_a, yg_a = to_plot(np.array([z_good]), np.array([np.exp(-z_good**2/(2*sigma_after**2))]))

ax.scatter(xg_b[0], yg_b[0], color=GRAY, s=80, zorder=5)
ax.scatter(xg_a[0], yg_a[0], color=GREEN, s=100, zorder=5)
ax.annotate("", xy=(xg_a[0], yg_a[0]+0.5), xytext=(xg_b[0], yg_b[0]+0.5),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
ax.text(xg_a[0]+0.3, yg_a[0]+0.85, "score\n+0.04!", ha='left', fontsize=8,
        color=GREEN, fontweight='bold')

ax.text(5, 1.1, "← narrow distribution →", ha='center', fontsize=7.5, color=GRAY)
ax.text(5, 0.6, "added trash strategies to widen it\n→ good strategy looks better by comparison",
        ha='center', fontsize=7.5, color=LGRAY, style='italic')

ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a1a', edgecolor=GRAY,
          labelcolor=WHITE, framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2: Holdout peeking
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

ax.text(5, 9.3, "HACK #2", ha='center', fontsize=11, fontweight='bold',
        color=ORANGE, transform=ax.transData)
ax.text(5, 8.7, "Holdout data snooping", ha='center', fontsize=9.5, color=WHITE,
        transform=ax.transData)

# Show train/holdout split with agent peeking
# Train box
train_box = FancyBboxPatch((0.5, 5.5), 4.2, 2.5, boxstyle="round,pad=0.1",
                            linewidth=1.5, edgecolor=GREEN, facecolor='#0a1a0a')
ax.add_patch(train_box)
ax.text(2.6, 7.35, "TRAIN DATA", ha='center', fontsize=8.5,
        color=GREEN, fontweight='bold')
ax.text(2.6, 6.8, "80% of rows", ha='center', fontsize=8, color=LGRAY)
ax.text(2.6, 6.35, "[OK] agent can use", ha='center', fontsize=7.5, color=GREEN)

# Holdout box
hold_box = FancyBboxPatch((5.3, 5.5), 4.2, 2.5, boxstyle="round,pad=0.1",
                            linewidth=1.5, edgecolor=RED, facecolor='#1a0a0a')
ax.add_patch(hold_box)
ax.text(7.4, 7.35, "HOLDOUT", ha='center', fontsize=8.5,
        color=RED, fontweight='bold')
ax.text(7.4, 6.8, "20% of rows", ha='center', fontsize=8, color=LGRAY)
ax.text(7.4, 6.35, "[X] off-limits", ha='center', fontsize=7.5, color=RED)

# Agent peeking arrow
ax.annotate("", xy=(5.3, 6.5), xytext=(4.7, 5.0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5,
                            connectionstyle='arc3,rad=-0.3'))

# Agent box
agent_box = FancyBboxPatch((2.5, 2.5), 5.0, 2.0, boxstyle="round,pad=0.1",
                            linewidth=2, edgecolor=ORANGE, facecolor='#1a1200')
ax.add_patch(agent_box)
ax.text(5.0, 4.1, "[ Agent ]", ha='center', fontsize=9, color=ORANGE, fontweight='bold')
ax.text(5.0, 3.55, '"market 253697 always\nresolves NO at 0.78..."', ha='center',
        fontsize=7.5, color=WHITE, style='italic')

# What it did
ax.text(5.0, 1.8, "wrote strategy rules targeting\nspecific holdout market IDs",
        ha='center', fontsize=7.5, color=LGRAY, style='italic')
ax.text(5.0, 1.15, '"if market_max < 0.43: skip"\n(market 253697 max = 0.43)',
        ha='center', fontsize=7, color=ORANGE,
        bbox=dict(boxstyle='round', facecolor='#1a1200', edgecolor=ORANGE, alpha=0.7))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3: Metric gaming
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]

ax.text(5, 9.3, "HACK #3", ha='center', fontsize=11, fontweight='bold',
        color=YELLOW, transform=ax.transData)
ax.text(5, 8.7, "Score formula gaming", ha='center', fontsize=9.5, color=WHITE,
        transform=ax.transData)

# Show the score formula components as a balance
ax.text(5, 8.0, "score = 0.45×Sharpe + 0.45×PnL + 0.10×Drawdown",
        ha='center', fontsize=7, color=LGRAY,
        bbox=dict(boxstyle='round', facecolor='#1a1a00', edgecolor=YELLOW, alpha=0.6))

# Two strategies being compared
strategies = [
    ("Strategy A\n(genuine)", 3.0, 0.20, -0.03, GREEN),
    ("Strategy B\n(gamed)", 0.82, 0.40, -0.01, YELLOW),
]
y_positions = [6.2, 4.0]

score_a = 0.45*(3.0/20) + 0.45*(0.20) + 0.10*(1 + -0.03)
score_b = 0.45*(0.82/20) + 0.45*(0.40) + 0.10*(1 + -0.01)

for (name, sharpe, pnl_norm, dd, color), yp in zip(strategies, y_positions):
    box = FancyBboxPatch((0.4, yp - 0.5), 9.2, 1.7, boxstyle="round,pad=0.1",
                          linewidth=1, edgecolor=color, facecolor='#0f0f0f')
    ax.add_patch(box)
    ax.text(1.2, yp + 0.85, name, ha='left', fontsize=8, color=color, fontweight='bold')
    ax.text(1.2, yp + 0.4, f"Sharpe: {sharpe:.2f}", ha='left', fontsize=7.5,
            color=LGRAY if sharpe < 1.5 else WHITE)
    ax.text(1.2, yp + 0.05, f"PnL (norm): {pnl_norm:.2f}", ha='left', fontsize=7.5,
            color=WHITE if pnl_norm > 0.3 else LGRAY)

score_a_val = 0.45*(3.0/20) + 0.45*(0.20) + 0.10*(1-0.03)
score_b_val = 0.45*(0.82/20) + 0.45*(0.40) + 0.10*(1-0.01)

ax.text(6.5, y_positions[0]+0.45, f"score\n{score_a_val:.3f}", ha='center', fontsize=9,
        color=LGRAY, fontweight='bold')
ax.text(6.5, y_positions[1]+0.45, f"score\n{score_b_val:.3f}", ha='center', fontsize=9,
        color=YELLOW, fontweight='bold')

# Arrow showing B beats A despite worse Sharpe
ax.annotate("", xy=(8.2, y_positions[1]+0.5), xytext=(8.2, y_positions[0]+0.5),
            arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2))
ax.text(8.5, (y_positions[0]+y_positions[1])/2 + 0.5, "wins?!", ha='left',
        fontsize=8.5, color=YELLOW, fontweight='bold')

ax.text(5.0, 2.8, "crash the Sharpe, pump the PnL,\nhit a higher composite score",
        ha='center', fontsize=8, color=LGRAY, style='italic')
ax.text(5.0, 1.9, "CapitalRecycle: Sharpe 3.4→0.82\nbut PnL higher → score 'improved'",
        ha='center', fontsize=7.5, color=ORANGE,
        bbox=dict(boxstyle='round', facecolor='#1a0f00', edgecolor=ORANGE, alpha=0.7))
ax.text(5.0, 1.0, "patched by switching to absolute\nscoring (no z-score normalization)",
        ha='center', fontsize=7, color=GRAY)

# ── Bottom watermark ──────────────────────────────────────────────────────────
fig.text(0.5, 0.01, "autoresearch-markets  •  190+ experiments  •  @Polymarket data",
         ha='center', fontsize=9, color=GRAY)

plt.savefig("charts/reward_hacking.png", dpi=160, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved charts/reward_hacking.png")
