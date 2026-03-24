#!/usr/bin/env bash
# reset_and_run.sh
# Usage: ./reset_and_run.sh
#   Resets all experiment state to a clean master baseline,
#   then launches an autonomous Claude Code autoresearch session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ── 1. Safety check ────────────────────────────────────────────────────────────
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "master" ]]; then
  echo "ERROR: You must be on master before resetting. Currently on: $CURRENT_BRANCH"
  echo "Run: git checkout master"
  exit 1
fi

# ── 2. Sync with remote ────────────────────────────────────────────────────────
echo "==> Pulling latest master from remote..."
git fetch origin
git reset --hard origin/master
echo "    Done."

# ── 3. Wipe experiment artefacts ──────────────────────────────────────────────
echo "==> Clearing experiment state..."

# Run logs
rm -f run.log run_1m.log

# Learning journal and results log (untracked — safe to delete)
rm -f learnings.md

# Reset results.tsv to header-only
echo -e "commit\tscore\tfinal_pnl\tsharpe\tstatus\tdescription" > results.tsv

# Clear results/ output directory
rm -f results/leaderboard.csv results/trade_attribution.csv results/report.json

echo "    Done."

# ── 4. Confirm data lake is present ───────────────────────────────────────────
if [[ ! -d "data_lake/trades" ]] || [[ -z "$(ls data_lake/trades/*.parquet 2>/dev/null)" ]]; then
  echo "WARNING: data_lake/trades/ is missing or empty — Claude will need to ingest data first."
else
  echo "==> data_lake looks good."
fi

# ── 5. Launch Claude Code autoresearch session ─────────────────────────────────
echo ""
echo "==> Launching autonomous Claude Code session (dangerously-skip-permissions)..."
echo "    Claude will follow program.md instructions and begin experimenting."
echo ""

PROMPT="$(cat <<'CLAUDE_PROMPT'
You are starting a fresh autoresearch session for the autoresearch-markets project.

Follow ALL instructions in program.md exactly. Here is a summary of what you must do:

1. Read program.md, README.md, prepare.py, train.py, and markets_research/strategies.py for full context.
2. Generate a fresh run tag (today's date + 4-char hex suffix, e.g. mar24-a3f1).
   Use: python3 -c "import secrets; print(secrets.token_hex(2))"
3. Create and checkout a new branch: git checkout -b autoresearch/<tag>
4. Verify data_lake/ exists and has parquet files. If missing, stop and tell the user.
5. Initialize results.tsv with just the header:
   commit	score	final_pnl	sharpe	status	description
6. Begin the experiment loop:
   a. Read results/report.json for win/loss context and next hypotheses (skip on first run).
   b. Hypothesize a strategy change in markets_research/strategies.py.
   c. Run: python train.py 2>&1 | tee run.log
   d. If IMPROVED: git commit the change, log "keep" in results.tsv, record WHY in learnings.md.
   e. If NO_IMPROVEMENT or FIRST_RUN with bad score: git checkout markets_research/strategies.py, log "discard" in results.tsv, record WHY in learnings.md.
   f. Repeat from step (a) indefinitely.

IMPORTANT constraints from program.md:
- ONLY modify markets_research/strategies.py — never touch train.py, prepare.py, or other markets_research/ files.
- Do not install new packages.
- Do not modify the scoring logic.
- Record mechanistic WHY (not just what changed) in learnings.md after every experiment.

Start now.
CLAUDE_PROMPT
)"

claude --dangerously-skip-permissions -p "$PROMPT"
