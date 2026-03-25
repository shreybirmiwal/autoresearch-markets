# Tweet Thread

---

**[1 — Hook]**

we modified @karpathy's autoresearch to run 190+ autonomous experiments on @Polymarket data while we slept

woke up to $800 in backtest profit 🤑

(it's fake. we overfit everything. here's what actually happened 🧵)

cc @KrishPatelShah @moalzq

[pic: shrey sleeping, laptop open, agent running]

---

**[2 — The setup]**

the idea: give an AI agent one file to edit (`strategies.py`) and let it loop forever

every iteration:
→ write a new trading strategy
→ backtest it on real @Polymarket data
→ keep if better, discard if worse
→ repeat

~3 min per experiment. one overnight session. 190 experiments.

[screen recording scrolling through all strategies / git log]

---

**[3 — Reward hacking]**

first thing the agent did: cheat

it found 3 different ways to game our scoring:

1/ added deliberately BAD strategies to widen the z-score distribution, making good ones look better

2/ peeked at the test dataset (which was supposed to be off-limits) to write rules targeting specific markets by ID

3/ balanced PnL vs Sharpe artificially to hit the composite score formula

we had to patch each one manually. the agent found every seam.

[chart2: score progression showing the contaminated jumps]

---

**[4 — Stop-losses are wrong-signed]**

one of the most counterintuitive results:

stop-losses destroyed performance

buying YES at 0.25, price drops to 0.17 → stop out → market resolves YES at 1.00 → you lost the whole trade

in binary prediction markets, price dipping isn't a trend signal. it's just noise before resolution. cutting early is exactly wrong.

---

**[5 — Sell timing was the single biggest real improvement]**

the one thing that genuinely worked: knowing WHEN to sell

buying cheap YES and holding to resolution: Sharpe 1.1
adding exit logic (sell when price recovers to 0.63): Sharpe 2.0

why it works: binary markets resolve slowly. capital sitting in a drifting position is dead money. exit at 0.63, redeploy, enter again.

best single trade context in the whole dataset: $0.787 avg PnL per recycled exit trade

[chart4: sharpe comparison]

---

**[6 — But sell orders can also KILL you]**

there's a trap though

we tried using sell orders for capital MANAGEMENT (selling positions early to make room for better trades)

Sharpe: 3.4 → 0.82

why: a forced sell at 0.25 creates a huge one-day realized loss. daily P&L goes lumpy. Sharpe collapses.

rule we learned: only exit when the SIGNAL reverses (price recovered). never sell for cash flow.

[chart4 again with CapitalRecycle bar highlighted]

---

**[7 — The cheap YES paradox]**

buying YES at <20¢ looks like a bad idea. avg trade loses money.

except:

- 18/20 markets: resolves NO, small loss per trade (−$0.035 avg)
- 2/20 markets: resolves YES, massive gain per trade (+$0.875 avg)

the 2 winners cover all 18 losers and then some. net: +$560

any "smart" filter that reduces cheap-zone trading ends up missing the rare YES-resolvers that generate all the profit

[chart5: cheap YES paradox pie + bar]

---

**[8 — The High Water Mark filter]**

biggest legitimate discovery: the High Water Mark filter

if a market's price has NEVER exceeded 10¢, traders have basically priced it at zero already. it almost always resolves NO.

markets that eventually resolve YES always trade above 9¢ on the way up.

filter: skip YES trades in markets where max(price_ever) < 0.10

hit rate: 13% → 42%
holdout PnL: $700 → $794

[chart6: HWM filter impact]

---

**[9 — Convergence: the agent got stuck]**

this is convergence (chart below)

the agent ran 81 experiments in one branch. after exp 57 — the score flatlined.

but it kept going for 24 more experiments, just grinding two parameters:
- `cheap_fraction_min`: 0.20 → 0.30 → 0.40 → 0.50 → 0.60
- `rolling_window`: 3 → 5 → 10 → 20

same idea, smaller increments, forever

it found one thing that worked and polished it instead of asking "what else is true about this market?"

[chart3: convergence failure — 81 experiments, plateau at exp 57]

---

**[10 — Why all the results are fake]**

even the "good" results:

- tested on 20 markets. same 20 every run. every "improvement" was fitting to those specific markets
- the holdout dataset was contaminated (agent literally named market IDs in its code)
- Polymarket markets are short-lived events — what existed in training has already resolved by test time
- best strategy fold Sharpe: 1.75. holdout Sharpe: 4.75. that's not alpha, that's luck

[chart7: fold vs holdout scatter — almost everything in the overfit zone]

---

**[11 — What we'd actually need to find real edge]**

- split by MARKET not by row (no market overlap between train and test)
- evaluate on 200+ markets, not 20
- never touch the holdout until the final model
- separate the agent that proposes from the agent that evaluates
- penalize per-market learned state (it expires when the market resolves)

the framework works. the eval setup was the bug.

---

**[12 — Real takeaways]**

things that are mechanistically true even if we can't trade on them:

→ price near 0 ≠ cheap opportunity. it means the market IS resolving no
→ capital recycling matters in binary markets — exit when signal reverses
→ the 0.40–0.60 NO zone loses money across every strategy, every branch. avg −$0.23/trade. don't touch it
→ autoresearch will find and exploit every gap in your eval. every single one.

[chart1: price zone PnL — the cleanest data chart]

---

**[13 — Full results]**

181 strategies tried. 39 kept. 142 discarded.

best score: 0.3010 (contaminated)
best clean score: ~0.27
baseline: 0.15

full experiment log, all 181 strategies, and code: [repo link]

if you find real edge on @Polymarket using this framework: please tell us what we missed 👀
