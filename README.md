# Market-Rock

A semi-autonomous intraday futures strategy:

* **Python** (NumPy/Numba) learns: order-flow and Auction Market Theory features, a
  conservative backtester, overfitting controls (purged folds, Deflated Sharpe),
  a trust-region evolution strategy, and Bayesian regime sizing.
* **NinjaTrader 8** (C#, Rithmic-friendly) executes, behind a hard risk guard for
  prop-firm rules: intraday trailing drawdown, daily loss, consistency cap.

**Semi-autonomous means:** the system may make itself *safer* on its own (shrink
size, switch off a regime that is losing). It cannot make itself riskier or change
its strategy logic without a named human running `marketrock promote`. The account's
hard limits are never writable by the learning loop.

> No claim of profitability is made. Every number produced so far comes from
> synthetic data. Read [docs/REPORT.md](docs/REPORT.md), including §0 (what has and
> hasn't been verified) and §4 (known limits), before connecting a funded account.

## Layout

```
marketrock/            Python package (research + learning + CLI)
  features.py          causal features (mirrored exactly in C#)
  strategy.py          regime filter, AMT signals, pessimistic backtester
  validation.py        PSR, Deflated Sharpe, purged folds
  optimizer.py         evolution strategy + Bayesian regime sizing + gate
  autonomy.py          what may change without a human (de-risk only)
  store.py             params.json/sha256 bridge, hash-chained ledger
  cli.py               init | backtest | propose | promote | rollback | verify
ninjatrader/MarketRockStrategy.cs   the NT8 strategy
tools/nt8-compile-check/            stubbed NT8 API for CI compile checks
tools/parity/                       replays ticks through the real C# strategy
tests/                              pytest (incl. Python<->C# parity)
docs/REPORT.md                      system report: gaps, red-team log, maths
```

## Daily operation

```
           NT8 (live)                                Python (nightly, e.g. Task Scheduler)
  ┌─────────────────────────┐   bars_history.csv  ┌────────────────────────────────────┐
  │ MarketRockStrategy.cs   │ ──────────────────▶ │ marketrock propose                 │
  │  hard risk guard        │   trades.csv        │  1. Bayesian regime update         │
  │  reads params.json      │ ──────────────────▶ │     └─ de-risk → applied (auto)    │
  │  (hash-verified, loaded │                     │  2. ES search on non-hold-out data │
  │   at session start)     │ ◀────────────────── │  3. gate: DSR, folds, hold-out     │
  └─────────────────────────┘   params.json       │     └─ candidate.json + report     │
                                                  └────────────────────────────────────┘
                                                        human: marketrock promote
```

### Setup

1. **NT8:** copy `ninjatrader/MarketRockStrategy.cs` to
   `Documents\NinjaTrader 8\bin\Custom\Strategies\`, then compile it (NinjaScript Editor, F5).
2. **Python 3.12+:** `pip install -e .[fast]` (`fast` adds Numba; without it the code
   runs the same, just slower).
3. Initialise the live parameters in the folder NT8 reads (the strategy's `Data directory`
   property; default `Documents\NinjaTrader 8\marketrock`):
   ```
   marketrock --live-dir "%USERPROFILE%\Documents\NinjaTrader 8\marketrock" --state-dir state init --approver "Your Name"
   ```
4. On a 1-minute chart of the instrument, with tick history loaded, enable the strategy
   once with **Export historical bars** on. This writes `bars_history.csv`, the
   training file. Then disable it, turn the option off, and set the **Hard limits**
   group to your firm's rules. Fill in **Broker liquidation threshold** from your
   prop-firm dashboard if it shows one.
5. Enable the strategy on the account (sim first).

### Each night

```
marketrock --live-dir <DataDir> --state-dir state --out-dir proposals \
           --daily-loss 1000 --trailing-dd 2500 \
           propose --bars <DataDir>/bars_history.csv --trades <DataDir>/trades.csv
```

* De-risk changes are applied automatically. NT8 picks them up at the next session start.
* If `proposals/report.json` says `AWAITING_APPROVAL`, read its `gates` and `changes`,
  then `marketrock ... promote --approver "Your Name"`.
* `marketrock ... rollback --to <sha256> --approver "Your Name"` restores any earlier version.
* `marketrock ... verify` checks the ledger chain and that live params are the last ledgered version.

Pass the same `--tick/--tick-value/--commission/--slippage` values as your instrument
(the defaults are ES).

## Tests

```
pip install -e .[fast,test]
pytest -q                     # parity test runs only if the .NET 8 SDK is on PATH (or $DOTNET)
dotnet build tools/nt8-compile-check
```
