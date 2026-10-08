# Market-Rock

A self-developing intraday futures system for **MES** (Micro E-mini S&P 500):

* **Python** (NumPy/Numba) invents, tests and manages strategies. Strategies are
  *genomes* (rules built from order-flow and Auction-Market-Theory conditions). A
  genetic search invents and tunes them. Each one must pass a statistical gate
  (Deflated Sharpe with a cumulative trial count, purged folds, an unseen hold-out)
  and then a **forward paper trial** on data recorded after it was invented.
* **NinjaTrader 8** (C#, Rithmic-friendly) interprets genomes as data and trades the
  live portfolio behind a hard prop-firm risk guard: intraday trailing drawdown,
  daily loss, consistency cap, and a per-trade risk ceiling.

**The goal it optimises:** maximum long-run account growth (fractional Kelly) while
keeping the probability of hitting your trailing-drawdown floor at or below 5%.

**Autonomy (as configured):**

| Action | Who |
|---|---|
| Invent, tune, paper-trade strategies | automatic |
| Promote a strategy that wins its paper trial (starts at 25% size) | automatic |
| Grow a strategy's size as live results confirm it | automatic, **up to your ceiling** |
| Retire a failing strategy; roll back a failed tune | automatic |
| Hard account limits, the sizing ceiling | **you only** (config + NT8 properties) |
| Kill switch, rollback/pin | **you** (`halt`, `resume`, `rollback`) |

> No profit claim is made. Every result so far comes from synthetic data. The
> tests show the machinery finds and exploits an edge *when one exists* and stays
> flat when none does. Read [docs/REPORT.md](docs/REPORT.md) §0 and §4 before
> connecting a funded account.

## Layout

```
marketrock/
  features.py    causal features (mirrored exactly in C#)
  genome.py      strategy-as-data: vocabulary, parse/validate, signals, mutation
  strategy.py    portfolio backtester (one position at a time, pessimistic fills)
  evolve.py      genetic search, fitness, statistical gate
  portfolio.py   lifecycle (PAPER -> LIVE -> RETIRED), Kelly/ruin sizing, monitoring
  validation.py  PSR, Deflated Sharpe, purged folds
  store.py       portfolio.txt/sha256 bridge, hash-chained ledger
  report.py      daily plain-language report
  cli.py         init | evolve | backtest | halt | resume | rollback | verify
ninjatrader/MarketRockStrategy.cs   NT8 genome interpreter + risk guard
tools/nt8-compile-check/            stubbed NT8 API for CI compile checks
tools/parity/                       replays ticks through the real C# strategy
tests/                              pytest, incl. Python<->C# parity
docs/REPORT.md                      gaps, red-team log, the maths
```

## Setup

1. **NT8:** copy `ninjatrader/MarketRockStrategy.cs` to
   `Documents\NinjaTrader 8\bin\Custom\Strategies\` and compile it (NinjaScript Editor, F5).
2. **Python 3.11+:** `pip install -e .[fast]`
3. **Initialise** in the folder NT8 reads (the strategy's *Data directory*, default
   `Documents\NinjaTrader 8\marketrock`):
   ```
   marketrock --live-dir "<DataDir>" --state-dir state --report-dir reports init --approver "Your Name"
   ```
   Then edit `state/config.json`:
   * `max_risk_per_trade_usd`: **your ceiling on automatic sizing**. To let the system
     size fully on its own, set it to your hard per-trade limit. To approve every
     increase yourself, keep it at your current size and raise it when you choose.
   * Instrument: defaults are **MES** (Micro E-mini S&P 500: tick 0.25, $1.25/tick).
     Set `commission_rt` to your broker's round-trip cost per contract (default $1.50
     including fees). MES gives the sizing maths 1/10-ES granularity (REPORT finding A-21).
   * `trailing_dd_usd`, `daily_loss_usd`, `buffer_usd`: copy your firm's rules.
4. On a **1-minute MES chart** (front month) with tick history loaded, enable the strategy once with
   **Export historical bars** on. That writes `bars_history.csv` (training data). Then
   set the **Hard limits** group, including **Max risk per trade**, and enable it on the
   account (**sim first**).

## Every night (schedule it, e.g. Windows Task Scheduler)

```
marketrock --live-dir "<DataDir>" --state-dir state --report-dir reports \
           evolve --bars "<DataDir>/bars_history.csv" --trades "<DataDir>/trades.csv"
```

Read `reports/<date>.md`. NT8 picks up portfolio changes at the next session start.

* `marketrock ... halt --approver "Your Name"`: empty portfolio until `resume`.
* `marketrock ... rollback --to <sha256> --approver "Your Name"`: restore and pin an
  earlier portfolio (the hashes are in `state/ledger.jsonl` and `<DataDir>/history/`).
* `marketrock ... verify`: ledger chain intact and live file equals the last ledgered version.

## Tests

```
pip install -e .[fast,test]
pytest -q                     # parity test runs when the .NET 8 SDK is on PATH (or $DOTNET)
dotnet build tools/nt8-compile-check
```
