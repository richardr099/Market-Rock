# Market-Rock — System Report

Status as of v2 (self-developing portfolio). Each claim below carries its evidence. Where
something has not been verified, the report says so.

---

## 0. What was actually run vs. only reasoned about (v2)

| Item | Status | Evidence |
|---|---|---|
| Python pipeline: genomes, simulator, evolution, gate, lifecycle, sizing, ledger, CLI, report | **Run** | `pytest tests` → 29 passed |
| Unattended nightly loop on noise never puts a strategy live | **Run** | `tests/test_evolve.py::test_noise_never_goes_live` (4 nights, >300 strategies evaluated) |
| Loop discovers a planted edge, paper-trades it on forward data, promotes it within the user's ceiling, with no clones | **Run** | `tests/test_evolve.py::test_planted_edge_is_discovered_paper_traded_and_promoted` |
| Kill switch, resume, rollback/pin, trade-log tamper detection | **Run** | `tests/test_evolve.py::test_halt_resume_rollback_and_log_rotation` |
| Failing live strategy retired; failed tune auto-rolls back to its parent | **Run** | `tests/test_portfolio.py` |
| C# genome interpreter compiles | **Run, against stubs only** | `dotnet build tools/nt8-compile-check` (C# language version 5), 0 warnings |
| C# ↔ Python parity: bars, tick-rule volume, 8 features, and every entry decision of a 5-genome portfolio covering all 7 condition types | **Run** through the real `MarketRockStrategy.cs` | `tests/test_parity.py::test_csharp_replay_matches_python` |
| Tests detect injected bugs | **Run** | Mutations caught: paper trial always passes, clone filter off, C# `CROSS_UP` boundary, C# `TIME_IN` boundary (v2); plus the v1 set |
| Compiles inside real NinjaTrader 8 / behaves correctly on Rithmic | **NOT verified** | No NT8 here. Press F5 in the NinjaScript Editor; run on sim first. |
| NT8 threading model (finding A-15) | **NOT verified** | Still open. |
| **The system makes money on real futures data** | **No evidence either way** | Synthetic data only. The planted-edge test proves the machinery can find and exploit an edge *when one exists*. It says nothing about whether ES has one at this frequency. |

## 1. GitHub gap analysis

Reviewed: FinRL / FinRL-X (AI4Finance), QuantConnect LEAN's Algorithm
Framework, GitHub `ninjatrader8` topic repositories, GA/walk-forward engines
listed in *awesome-ai-in-finance* (e.g. finclaw).

**What they do better, bluntly:**

* **FinRL** runs several RL agents (A2C/PPO/DDPG) and, each window, picks the one
  with the best validation Sharpe. It is multi-asset and has a data pipeline we
  don't have. Its "weight-centric" design makes a single object the only
  interface between modules.
* **LEAN** has a clean Alpha → Portfolio Construction → Risk → Execution
  pipeline in which the risk model can veto targets. It also has mature fill and
  slippage models and supports many asset classes.
* **GA engines** (finclaw et al.) search much larger strategy spaces (hundreds
  of factors) with walk-forward evaluation.
* **NT8 repos** offer polished UI panels (e.g. "why I'm not trading" overlays)
  and ATM-strategy integration.

**What we took, and how:**

| Idea | Source | Integration |
|---|---|---|
| A single contract between learner and executor | FinRL-X weight-centric interface | `params.json` + SHA-256 is the only thing Python can hand to NT8. Hard limits are not in it, so the learner cannot touch them. |
| Risk model as a separate layer that only vetoes or reduces | LEAN Risk Management model | `RiskGuard` in C#: `AllowedQty` can only shrink size; `CanEnter` can only block. The autonomy policy (`autonomy.py`) can only de-risk without a human. |
| Window-wise selection on validation data | FinRL ensemble | Evolution strategy candidates are scored on purged session folds (mean − 0.5·std), then gated on a hold-out the search never saw. |
| Walk-forward evaluation | GA engines | Purged K-fold over sessions, with an embargo, plus a time-ordered hold-out. |
| Reconcile positions after a broker hiccup | Rithmic/NT8 support threads (positions closed at the broker still showing open in NT8) | `Reconcile()` compares `Position` with `PositionAccount` after every reconnect and flattens on divergence. At startup a mismatch locks trading instead of touching a position that isn't ours. |

**What those projects lack, and we added:** a multiple-testing correction.
GA and RL frameworks report walk-forward Sharpe without deflating for how many
configurations were tried. Every proposal here must pass the **Deflated Sharpe
Ratio** computed with the *cumulative* trial count, which persists across
nights. Without that, a nightly optimizer is a machine for finding luck.

**Rejected, with reasons:**

* *Deep RL policy (FinRL-style) for this market.* A single-instrument intraday
  AMT signal fires about 0.5 times per session. An RL policy needs orders of
  magnitude more independent decisions than that, so it would fit noise. It also
  breaks run-to-run determinism.
* *LEAN serialization for the bridge.* LEAN's format is Newtonsoft JSON, built
  for its own object graph. The bridge here carries 14 scalars, so plain
  canonical JSON + SHA-256 is smaller, auditable by eye, and hash-verifiable.
  Bars and trades travel as CSV, which both sides already read.
* *Hundreds of factors.* More search dimensions raise the deflation penalty
  faster than they raise the true signal at this trade frequency.

**v2 adds:** strategy invention by genetic search over a genome vocabulary
(the GA-engine idea, kept small so the deflation penalty stays meaningful),
a forward paper trial before any capital, and Kelly sizing under a ruin
constraint.

**Not integrated, still a gap:** multi-instrument and portfolio construction;
L2 order-book features (we classify volume with the tick rule, typically about
75–85% agreement with quote-based classification in the literature); queue
position and market-replay fill realism; an on-chart status panel.

---

## 2. Adversarial findings (red-team log)

Found and fixed in v1. All still apply to v2 except where noted: `params.json`
is now `portfolio.txt`; A-12's `promote` command no longer exists (v2
promotes automatically through the paper trial; halt/pin replace it, A-22);
A-17's trade-log check is now proven by `test_halt_resume_rollback_and_log_rotation`.
v1 test names in this table refer to the v1 commit (`b29d433`).

| # | Severity | Finding | Fix | Proof |
|---|---|---|---|---|
| A-1 | **Critical (desync)** | NT8 processes the primary series before the tick series at equal timestamps. Ticks stamped exactly at a bar's end would be processed after that bar closed, so live order flow would differ from what was trained on. | A bar is finalized on the first tick *after* its end. All ticks with time ≤ end are counted. Entry goes out at that tick, which matches Python's next-bar-open fill. | `test_parity` (fixture deliberately includes ticks exactly at bar end) |
| A-2 | **Critical (silent no-trade)** | `AllowedQty` cast `floor(room/loss)` straight to `int`. With a large room this overflowed to a negative number, sizing every trade to zero. **Found by the parity harness, not by inspection.** | Clamp in double space, then cast. | `test_parity` (C# produced 0 of 15 entries before the fix) |
| A-3 | High (desync) | Python `round()` and C# `Math.Round` both use banker's rounding (round half to even) but are easy to replace inconsistently. Tick levels and stop ticks must agree exactly. | `floor(x/tick + 0.5)` on both sides. | `test_level_rounding_is_half_up_not_bankers` |
| A-4 | **Critical (risk)** | Historical playback on enable produces fictional P&L that fed the same risk guard used live, which could cause false locks or a false sense of headroom. | Separate guard instance during historical playback. On `State.Realtime` the guard is replaced with the persisted live state (`riskstate.txt`). | code review (needs NT8 to test) |
| A-5 | High (risk) | Going live mid-session left the daily-loss anchor on yesterday. | `guard.NewDay(currentSessionId)` at the realtime transition. | code review |
| A-6 | **Critical (risk)** | If the account had prior P&L, strategy-computed equity and broker equity differed. The guard's trailing floor could then be *looser* than the broker's. | Use the worse of the two equity numbers and the higher of the two for the peak. On the first live run, adopt the account's actual P&L. Optional `KnownFloorUsd` property taken from the firm's dashboard; the floor is never looser than that. | code review |
| A-7 | High | Risk flatten was re-sent on every tick while an exit was working (possible overfill or rejected-order storm). | `exitPending` latch, cleared on fill, cancel, or reject. | code review |
| A-8 | High | Startup reconcile would flatten a manual position belonging to someone else. | At startup a mismatch only locks the day and alerts. After a reconnect, a divergence of our own position is flattened. | code review |
| A-9 | Medium | `entryPending` could stick if an order event was lost. | Reset at each session roll when flat. | code review |
| A-10 | Medium | Default `RealtimeErrorHandling` stops the strategy on a reject. That disables it with a position possibly left unprotected. | `IgnoreAllErrors` + an explicit handler. A rejected protective or exit order triggers `Account.Flatten` and a day lock. | code review |
| A-11 | High | Torn read: NT8 could read a half-written `params.json`. | Atomic `os.replace`, JSON before hash. C# verifies SHA-256, bounds and key set; on failure it keeps the last good parameters (or makes no entries if there are none). | `test_live_params_hash_and_ledger_tamper` |
| A-12 | High | Promoting a candidate built before an auto-derisk would silently undo the derisk. | `promote` refuses unless `base_params_sha256` equals the current live hash. | `test_promote_refuses_stale_base_and_blank_approver` |
| A-13 | **Critical (overfitting)** | A per-run trial count lets the nightly loop "forget" how many configurations it has tried, inflating significance. | Cumulative `n_trials` in `state/search.json`, fed into the DSR. | `test_noise_is_rejected...` asserts the count grows across runs |
| A-14 | High (leakage) | Hold-out bars visible to the search. | Search runs on a slice ending before the hold-out. The hold-out is scored on the full history (for warm-up), counting only hold-out sessions. | code structure + positive and negative gate tests |
| A-15 | **Open** | If NT8 delivers `OnOrderUpdate`/`OnExecutionUpdate`/`OnConnectionStatusUpdate` on a different thread from `OnBarUpdate`, the guard's `double` fields could race (bool/int flags are atomic, doubles are not guaranteed). | Not fixed. I could not verify NT8's threading model here. If it is multi-threaded, wrap `RiskGuard` mutations in a `lock`. | **UNVERIFIED** |
| A-16 | Medium | A duplicate history export appended a second copy, which would corrupt training data. | History export truncates. The Python loader rejects non-increasing times (refuses rather than guessing). | `Bars.validate` |
| A-17 | Info | Trade log rewritten or rotated → Bayesian posteriors double-count or skip trades. | Hash of the first row plus a processed-row count. `propose` refuses on a mismatch. | `test_auto_derisk_from_losing_live_fills` |

**Memory review (C#):** no allocation on the tick path; one `double[]` per
session for the value-area computation; the `Dictionary` is reused and cleared
each session; ring buffers are fixed-size; no event subscriptions (only
overrides), so there are no handler leaks; both `StreamWriter`s are disposed
in `State.Terminated`; `Print` is throttled.

---

## 3. The self-improvement mechanism (v2: self-developing)

The objective is to **maximise expected long-run growth of the account
(fractional Kelly), subject to P(hitting the trailing-drawdown floor) ≤
`ruin_prob` (5%)**. Raw profit is never the objective.

### 3.1 Strategies are data (genomes)

A genome is a direction, a conjunction of 1–4 conditions, and an exit:

    dir=-1;stop=1.25;rr=1.5;hold=30;c=LE delta_z -1.0|LE er 0.4

Conditions: `CROSS_UP/CROSS_DOWN/ABOVE/BELOW {VAH,VAL,POC}`,
`GE/LE {delta_z, er, vol_pct} x`, `TIME_IN a b`. NT8 interprets the same text,
so a strategy invented tonight trades tomorrow without code changes. It runs
only on interpreter code the parity test has verified. The system can invent
any strategy expressible in that vocabulary. It cannot invent new code, which
is deliberate (see §5).

### 3.2 Nightly loop (`marketrock evolve`, unattended)

1. **Learn from live fills.** Each live trade is converted to an R-multiple
   (P&L ÷ planned risk) and attributed to the genome that opened it.
2. **Retire failures.** For a LIVE genome with n ≥ 10 live trades, let μ_f
   and σ_f be the mean and SD of its forward-trial R-multiples. Then
   z = (mean_live − μ_f)/(σ_f/√n). Retire if z < −2.5, or if n ≥ 30 and the
   live t-stat is < −1, or if cumulative live R < −10. If the retired genome
   was a tune that had replaced its parent, the parent goes back to LIVE
   (**auto-rollback**).
3. **Paper trials.** For each PAPER genome, simulate it on bars recorded
   *after its birth*, which no search has scored it on, using the same
   pessimistic fill model. With forward stats (n, mean, t) and backtest mean
   μ_b:
     pass ⇔ n ≥ 20 ∧ t ≥ 1.0 ∧ (mean − 0.5·μ_b)/(σ_b/√n) ≥ −2
   The 0.5 is the Harvey–Liu haircut: the winner of a search is always
   optimistic in-sample (finding A-18). A tuned child must also beat its
   parent's forward mean over the same window. Winners go LIVE at
   `start_stage` = 25%. Losers are REJECTED. A trial that hasn't reached 20
   trades within 60 sessions expires.
4. **Invent and tune.** A genetic search (population 32, 4 generations,
   tournament selection, crossover, mutation) seeded with the AMT seeds plus
   every LIVE/PAPER genome, and small parameter-only mutations of each LIVE
   genome. Fitness on the search slice only:
     F = mean_k S_k − 0.5·std_k S_k − 0.01·n_conditions  (purged 5-fold, per-session Sharpe)
5. **Gate.** Same as v1: Deflated Sharpe ≥ 0.95 using the *cumulative* trial
   count, ≥ 4/5 positive folds, and a hold-out the search never saw (not blown,
   ≥ 15 trades, profitable, drawdown ≤ 50% of the limit). Candidates whose
   entry bars overlap an existing LIVE/PAPER genome by Jaccard > 0.5 are
   dropped as clones (finding A-19). Survivors become PAPER.
6. **Size** (below), **publish** `portfolio.txt` + hash, write the ledger and
   the daily report.

### 3.3 Sizing: fractional Kelly under a ruin constraint

For each LIVE genome, pool forward and live R-multiples, then shrink toward
zero with n₀ = 20 pseudo-trades:

    mean = ΣR/(n+n₀),  σ² = max(sample var, 0.5²),  μ_lo = mean − σ/√(n+n₀)
    D = (trailing_dd − buffer) / n_live                (drawdown room per strategy)
    r_kelly = kelly_fraction · μ_lo/σ² · D             (half-Kelly by default)
    r_ruin  = 2·μ_lo·D / (σ²·ln(1/ruin_prob))          (drifted random walk: P(ruin) = exp(−2μD/σ²r))
    stage   = min(1, 0.25 + 0.75·n_live_trades/50)
    risk    = min(stage·min(r_kelly, r_ruin), max_risk_per_trade_usd), floored at probe_risk_usd

Sizing moves up only on **lower-bound** evidence. It grows with live
confirmation and can never exceed **your** `max_risk_per_trade_usd`. NT8
additionally clamps to its own `MaxRiskPerTradeUsd` property and to the live
drawdown room (`RiskGuard.AllowedQty`).

### 3.4 What you control

`state/config.json`: `max_risk_per_trade_usd` (sizing ceiling),
`ruin_prob`, `kelly_fraction`, slot counts, trial lengths, instrument costs,
`halted`. Commands: `halt`, `resume`, `rollback --to <sha>` (pins the live
portfolio until `resume`). NT8 properties: every hard account limit, plus
`MaxRiskPerTradeUsd`. None of these are writable by the learning loop.

### 3.5 v2 adversarial findings

| # | Severity | Finding | Fix | Proof |
|---|---|---|---|---|
| A-18 | High (lost edge) | The paper trial compared forward results with the raw backtest. A real edge (+0.105R over 232 forward trades) was rejected because the search winner's in-sample mean is always inflated. | Compare with a 50%-haircut backtest; the forward t-stat ≥ 1 must still hold on its own. | smoke run: rejected before, promoted after; `test_paper_trial_rejects_strategy_without_forward_edge` keeps the no-edge case rejected |
| A-19 | High (false diversification) | Live slots filled with near-identical genomes (e.g. max-hold 36 vs 38). They fire on the same bars, add no diversification, and split the drawdown budget. | Jaccard clone filter on entry bars at admission. | planted-edge test asserts live overlap ≤ 0.6; mutant (filter off) caught |
| A-20 | Medium | The stage multiplied *after* the ceiling, so a 25%-stage genome could never reach the ceiling even when its optimal size was far above it. | risk = min(stage·optimal, ceiling). | `test_sizing_is_capped_staged_and_monotone_in_edge` |
| A-21 | **Important (practical)** | With a $2,500 trailing drawdown and several strategies, the Kelly/ruin-optimal risk per trade is often **below one ES contract**. The probe floor then overrides the staging. | Not a code fix: **trade MES (micro, $1.25/tick)** so sizes can follow the maths. Set `tick_value: 1.25` in config.json and run NT8 on MES. | smoke run sizes: $16 optimal vs $100 floor |
| A-22 | Medium | Halt/rollback could be silently undone by the next nightly run. | `halted` and `pinned` live in config.json and are respected by `evolve`; only `resume` clears them. | `test_halt_resume_rollback_and_log_rotation` |

## 4. Known limits (read before trading real money)

1. **No evidence of edge.** Synthetic data only. Run `evolve` nightly on at
   least 250 sessions of real exported bars. The system won't trade until
   something passes both the gate and a forward paper trial, which is the
   correct default.
2. **Trade frequency.** About 0.5 trades per session means about one year of
   data per meaningful gate decision. A gate that rarely passes is the system
   working, not failing.
3. **Training-data source.** `bars.csv` (realtime log) has gaps whenever NT8
   is off, and gaps change the rolling features. Train on `bars_history.csv`
   (ExportHistory mode with tick history loaded). That requires tick history
   from your data provider; Rithmic's historical tick depth is limited.
4. **Fill model divergence.** Python is deliberately pessimistic (stop-first,
   1 tick of slippage per side). Compare `trades.csv` with backtests monthly.
5. **Commission.** NT8's `ProfitCurrency` includes commission only if a
   commission template is configured on the account. Configure one, or live
   monitoring and sizing learn from gross P&L.
7. **One position at a time.** Live genomes share one position (the first
   signal in priority order wins), mirroring the backtest. D is divided by
   the number of live genomes, which is conservative.
8. **Search space.** The vocabulary is small on purpose (finding A-21 and the
   DSR penalty). New condition types need a code change in both Python and
   C#, followed by the parity test.
6. Finding A-15 (threading) is open.

---

## 5. What was deliberately not built

* **Self-modifying code deployed live.** The system invents strategies as
  data inside a verified interpreter. It never writes or deploys new code. An
  optimiser that can change its own executor can also break it, and a backtest
  can't catch every way that happens.
* **Learnable account limits.** Hard limits and the sizing ceiling are
  outside the learning loop's write path, in both Python and C#.
* **Deep RL.** See §1: sample-starved at this trade frequency, and it breaks
  determinism.
