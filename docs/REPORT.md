# Market-Rock — System Report

Status as of the first commit. Each claim below carries its evidence. Where
something has not been verified, the report says so.

---

## 0. What was actually run vs. only reasoned about

| Item | Status | Evidence |
|---|---|---|
| Python pipeline (features, backtest, validation, optimizer, autonomy, ledger, CLI) | **Run** | `pytest tests` → 23 passed |
| Gate rejects pure noise | **Run** | `tests/test_gate.py::test_noise_is_rejected_and_promote_refuses` |
| Gate *can* approve a real (planted) edge | **Run** | `tests/test_gate.py::test_planted_edge_passes_gate_then_human_promotes` |
| C# strategy compiles | **Run, against stubs only** | `dotnet build tools/nt8-compile-check` (C# language version 5). The stubs are my reading of the NT8 API, not the real assemblies. |
| C# ↔ Python parity (bars, tick-rule volume, all 7 features, every entry decision) | **Run** through the real `MarketRockStrategy.cs` code | `tests/test_parity.py::test_csharp_replay_matches_python` (needs the .NET SDK) |
| Tests detect injected bugs | **Run** | Mutations caught: profile lookahead, autonomy bypass, sample-vs-population std in C#, value-area tie direction in C#. One mutation not caught: C# percentile `<` → `<=`. It only differs on exact float ties, which this data never produces. |
| Compiles inside real NinjaTrader 8 | **NOT verified** | No NT8 in this environment. Press F5 in the NinjaScript Editor before anything else. |
| Rithmic disconnect / reconnect behaviour | **NOT verified** | The code path exists and compiles. It has not been exercised against a live Rithmic feed. |
| NT8 historical fills of managed stop/target vs the Python fill model | **NOT verified** | Python resolves stop and target hit in the same bar as a stop. NT8 uses its own intrabar fill logic. |
| Whether NT8 calls `OnOrderUpdate` / `OnExecutionUpdate` on the same thread as `OnBarUpdate` | **NOT verified** | See finding A-15. |
| **The strategy has an edge on real futures data** | **No evidence either way** | No real tick data was available in this environment. Every performance number in this repository is from synthetic data. |

---

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

**Not integrated, still a gap:** multi-instrument and portfolio construction;
L2 order-book features (we classify volume with the tick rule, typically about
75–85% agreement with quote-based classification in the literature); queue
position and market-replay fill realism; an on-chart status panel.

---

## 2. Adversarial findings (red-team log)

Every item below was found while building this commit and fixed in it.
"Proof" names the test that fails if the fix is removed, where one exists.

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

## 3. The self-improvement mechanism

Two learners. They run on different evidence, with different authority.

### 3.1 Alpha parameters: trust-region evolution strategy (proposes, never applies)

Search space: θ = (va_pct, delta_z_entry, stop_atr, target_rr, er_trend,
max_hold_bars, regime_vol_lo, regime_vol_hi), mapped to unit space
u ∈ [0,1]^8 via uᵢ = (θᵢ − loᵢ)/(hiᵢ − loᵢ).

**Proposal (one night).** Incumbent u₀; step size σ (persisted); λ = 24.

  uₖ = clip( u₀ + clip(σ·zₖ, −Δ, +Δ), 0, 1 ),  zₖ ~ N(0, I)

Δᵢ = `max_step` (default 0.10). No parameter can move more than 10% of its
range per night, however good a candidate looks. The RNG seed is
SHA-256(bar bytes ‖ incumbent hash ‖ n_trials), so a night is reproducible.

**Fitness (search slice only).** Daily P&L series d, split into K = 5
contiguous session folds with a 1-session embargo; Sₖ = mean(d_k)/std(d_k).

  F(θ) = mean_k Sₖ − 0.5 · std_k Sₖ,  F = −∞ if the account was blown or trades < 60

The penalty rewards a parameter set that works in every fold over one that
works brilliantly in one.

**Step-size adaptation (Rechenberg's 1/5 rule).**

  σ ← clip( σ · exp( 1[F(best) > F(incumbent)] − 0.2 ), 0.01, 0.10 )

**Gate (all must pass).** With N = cumulative trials ever, T sessions,
γ₃ skew, γ₄ kurtosis of d, and V = cross-candidate variance of SR:

  SR₀ = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ],  γ = 0.5772…  
  DSR = Φ( (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4 · SR²) ) ≥ 0.95

plus: at least 4 of 5 folds positive; on the hold-out (default the last 20
sessions, never seen by the search): not blown, ≥ 15 trades, net P&L > 0,
Sharpe ≥ incumbent's hold-out Sharpe, max drawdown ≤ 50% of the trailing
limit.

**Then a human.** `marketrock promote --approver "<name>"` is the only path
to live for these parameters. Every promotion is a hash-chained ledger entry.

### 3.2 Regime sizing: Bayesian lower-confidence-bound Kelly (may only de-risk on its own)

For each regime r ∈ {rotational, trend}, from **live fills** only:

* Win probability: pᵣ ~ Beta(αᵣ, βᵣ), prior Beta(2, 2). Each new session first
  decays the evidence toward the prior,
   α ← 2 + 0.97·(α − 2),  β ← 2 + 0.97·(β − 2),
  then adds that session's wins to α and losses to β. The effective memory is
  about 1/(1−0.97) ≈ 33 sessions, so the posterior tracks regime drift.
* Payoff ratio: b = (decayed mean win)/(decayed mean loss); defaults to
  `target_rr` until both exist.
* Kelly fraction at posterior quantiles q:
   f(q) = p_q − (1 − p_q)/b
* Decision:
   if f(0.95) ≤ 0 → disable the regime (even the optimistic case has no edge);  
   otherwise multiplier m = clip( f(0.05) / 0.10, 0.25, 1 ).

  A regime reaches full size only when the *pessimistic* Kelly reaches 10%.
  With little evidence it trades at 25% ("probation"). It is switched off only
  when the data rule out an edge with 95% confidence, not merely when the data
  are thin.

**Authority (semi-autonomy).** `autonomy.split_changes` classifies each change
by the parameter's `risk_dir`. Lowering a multiplier, disabling a regime,
cutting `max_contracts`/`risk_per_trade_usd`, or narrowing the volatility band
is applied to live automatically (ledger approver `auto:derisk`). Raising any
of them, re-enabling a regime, or touching any alpha parameter goes to the
human candidate. `apply_derisk` re-checks its own output and raises if
anything non-derisk slipped through.

### 3.3 What cannot be learned

Trailing drawdown, daily loss limit, consistency cap, safety buffer and the
broker floor are NinjaScript properties set by the user. They are not in
`params.json`, so no output of the learning loop can loosen them.

---

## 4. Known limits (read before trading real money)

1. **No evidence of edge.** Synthetic data only. Run `propose` on at least
   250 sessions of real exported bars; until the gate passes there, the
   honest default is to not trade.
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
   commission template is configured on the account. Configure one, or the
   Bayesian layer learns from gross P&L.
6. Finding A-15 (threading) is open.
