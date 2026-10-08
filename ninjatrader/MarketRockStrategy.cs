// Market-Rock NT8 execution layer.
//
// Mirrors marketrock/features.py + strategy.py. Any change to a feature,
// signal or rounding rule must be made in BOTH places (tests/test_parity.py
// checks the shared constants and the parameter bounds table).
//
// Design summary (see docs/ARCHITECTURE.md):
//  * Primary series: 1-minute bars (apply the strategy to a 1-minute chart).
//    Secondary series: 1-tick, used to classify volume with the TICK RULE.
//    The tick rule is used both historically and live, so the bar log the
//    Python learner trains on is produced by the same classifier that trades.
//  * A completed bar is FINALIZED on the first tick of the next bar, not in
//    the primary OnBarUpdate: NT8 processes the primary series before the
//    secondary at equal timestamps, so ticks stamped exactly at the bar end
//    would otherwise be lost. Entry orders therefore go out at the next bar's
//    first tick, which is exactly the Python fill model (next bar's open).
//  * Hard account limits are user-set properties here and nowhere else. The
//    learning loop can only write params.json, which cannot touch them.
//  * params.json is accepted only if its SHA-256 matches params.sha256 and
//    every key is present and in bounds. Otherwise: no new entries.

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class MarketRockStrategy : Strategy
    {
        // ---- constants shared with features.py (checked by test_parity.py) ----
        private const int AtrN = 14;
        private const int DeltaZN = 30;
        private const int VolPctL = 1000;
        private const int ErN = 20;
        private const int RegimeBlock = 0, RegimeRotational = 1, RegimeTrend = 2;
        private const string LongName = "MR_L", ShortName = "MR_S";

        // name, lo, hi -- must equal marketrock/params.py SPECS
        private static readonly object[][] ParamBounds = new object[][]
        {
            new object[] { "va_pct", 0.60, 0.80 },
            new object[] { "delta_z_entry", 0.00, 3.00 },
            new object[] { "stop_atr", 0.50, 3.00 },
            new object[] { "target_rr", 0.80, 3.00 },
            new object[] { "er_trend", 0.20, 0.70 },
            new object[] { "max_hold_bars", 5.0, 120.0 },
            new object[] { "regime_vol_lo", 0.00, 0.40 },
            new object[] { "regime_vol_hi", 0.60, 1.00 },
            new object[] { "risk_per_trade_usd", 50.0, 500.0 },
            new object[] { "max_contracts", 1.0, 10.0 },
            new object[] { "enable_rotational", 0.0, 1.0 },
            new object[] { "enable_trend", 0.0, 1.0 },
            new object[] { "size_mult_rotational", 0.0, 1.0 },
            new object[] { "size_mult_trend", 0.0, 1.0 },
        };

        // ---- parameters (from params.json) ----
        private Dictionary<string, double> prm;
        private string prmHash = "";
        private bool prmValid;

        // ---- pending (closed but not yet finalized) primary bar ----
        private bool pendingBar;
        private DateTime pendingEnd;
        private double pOpen, pHigh, pLow, pClose, pVolume;
        private bool pendingFirstOfSession;
        private int pendingSessionId;

        // ---- tick-rule accumulation for the bar being built ----
        private double buyVol, sellVol;
        private double lastTickPx = double.NaN;
        private int lastTickDir;
        private DateTime lastTickTime = DateTime.MinValue;
        private bool staleSinceLastBar;

        // ---- feature state ----
        private int barCount;
        private double trSeedSum;
        private double atr = double.NaN;
        private double prevBarClose = double.NaN;
        private Ring deltaRing, volRing, closeRing;
        private readonly Dictionary<long, double> hist = new Dictionary<long, double>(4096);
        private long histLo = long.MaxValue, histHi = long.MinValue;
        private double sessPoc = double.NaN, sessVah = double.NaN, sessVal = double.NaN;
        private SessionIterator sessionIterator;
        // last finalized bar's features (read by tools/parity to prove Python parity)
        private double fDz = double.NaN, fVolPct = double.NaN, fEr = double.NaN;
        private int currentSessionId;

        // ---- trade state ----
        private int barsHeld;
        private int entryRegime;
        private int openTradeRegime;
        private int lastTradeCount;
        private bool entryPending;
        private bool exitPending;
        private bool startupReconcile;

        // ---- risk / health ----
        private RiskGuard guard;
        private bool connectionOk = true;
        private DateTime reconnectUtc = DateTime.MinValue;
        private bool needReconcile;
        private DateTime lastAccountPollUtc = DateTime.MinValue;
        private double accountEquity = double.NaN;
        private DateTime lastPrintUtc = DateTime.MinValue;

        // ---- io ----
        private StreamWriter tradeLog, barLog;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "MarketRockStrategy";
                Description = "AMT value-area / order-flow strategy with hard prop-firm risk guard.";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                StartBehavior = StartBehavior.WaitUntilFlat;
                // We handle rejects ourselves (OnOrderUpdate) so a single reject
                // does not disable the strategy and orphan a live position.
                RealtimeErrorHandling = RealtimeErrorHandling.IgnoreAllErrors;
                ConnectionLossHandling = ConnectionLossHandling.KeepRunning;
                DisconnectDelaySeconds = 10;
                BarsRequiredToTrade = 0;
                IsUnmanaged = false;
                TraceOrders = false;

                AccountStartBalance = 50000;
                TrailingDrawdownUsd = 2500;
                UseTrailCap = true;
                TrailCapUsd = 100;
                DailyLossLimitUsd = 1000;
                RiskBufferUsd = 150;
                ConsistencyMaxPct = 0.30;
                ConsistencyBaseUsd = 3000;
                KnownFloorUsd = 0;
                CommissionPerContractRt = 4.5;
                ReconnectCooldownSec = 30;
                StaleDataSec = 20;
                DataDir = Path.Combine(NinjaTrader.Core.Globals.UserDataDir, "marketrock");
                ExportHistory = false;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Tick, 1);
            }
            else if (State == State.DataLoaded)
            {
                deltaRing = new Ring(DeltaZN);
                volRing = new Ring(VolPctL);
                closeRing = new Ring(ErN + 1);
                sessionIterator = new SessionIterator(BarsArray[0]);
                guard = NewGuard();
                Directory.CreateDirectory(DataDir);
                TryLoadParams(true);
                if (ExportHistory)
                    barLog = OpenCsv("bars_history.csv", "time,session,open,high,low,close,volume,buy_volume,sell_volume", false);
            }
            else if (State == State.Realtime)
            {
                // Historical playback must never contaminate the real risk state:
                // swap in the persisted realtime guard.
                guard = NewGuard();
                guard.Load(Path.Combine(DataDir, "riskstate.txt"));
                lastTradeCount = SystemPerformance.AllTrades.Count;
                if (barLog != null) { barLog.Flush(); barLog.Dispose(); }
                barLog = OpenCsv("bars.csv", "time,session,open,high,low,close,volume,buy_volume,sell_volume", true);
                tradeLog = OpenCsv("trades.csv", "exit_time,session,regime,direction,qty,pnl_usd,params_sha256", true);
                guard.NewDay(currentSessionId); // re-anchor the day if we go live mid-session
                needReconcile = true;
                startupReconcile = true;
            }
            else if (State == State.Terminated)
            {
                if (guard != null && guard.Persist && DataDir != null)
                    guard.Save(Path.Combine(DataDir, "riskstate.txt"));
                if (tradeLog != null) { tradeLog.Flush(); tradeLog.Dispose(); tradeLog = null; }
                if (barLog != null) { barLog.Flush(); barLog.Dispose(); barLog = null; }
            }
        }

        private RiskGuard NewGuard()
        {
            return new RiskGuard
            {
                StartEquity = AccountStartBalance,
                TrailDd = TrailingDrawdownUsd,
                UseTrailCap = UseTrailCap,
                TrailCap = TrailCapUsd,
                DailyLoss = DailyLossLimitUsd,
                Buffer = RiskBufferUsd,
                ConsistencyPct = ConsistencyMaxPct,
                ConsistencyBase = ConsistencyBaseUsd,
                Hwm = AccountStartBalance,
                KnownFloor = KnownFloorUsd,
                Persist = State == State.Realtime,
            };
        }

        private StreamWriter OpenCsv(string file, string header, bool append)
        {
            string path = Path.Combine(DataDir, file);
            bool exists = append && File.Exists(path) && new FileInfo(path).Length > 0;
            var w = new StreamWriter(new FileStream(path, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read), new UTF8Encoding(false));
            if (!exists) w.WriteLine(header);
            w.Flush();
            return w;
        }

        // ------------------------------------------------------------------
        // Bar flow
        // ------------------------------------------------------------------
        protected override void OnBarUpdate()
        {
            if (BarsInProgress == 0)
            {
                // Record the closed primary bar; finalize on the next tick.
                pendingBar = true;
                pendingEnd = Times[0][0];
                pOpen = Opens[0][0]; pHigh = Highs[0][0]; pLow = Lows[0][0]; pClose = Closes[0][0]; pVolume = Volumes[0][0];
                pendingFirstOfSession = BarsArray[0].IsFirstBarOfSession;
                if (pendingFirstOfSession || currentSessionId == 0)
                {
                    sessionIterator.GetNextSession(Times[0][0], true);
                    DateTime td = sessionIterator.ActualTradingDayExchange;
                    currentSessionId = td.Year * 10000 + td.Month * 100 + td.Day;
                }
                pendingSessionId = currentSessionId;
                return;
            }

            if (BarsInProgress != 1)
                return;

            DateTime t = Times[1][0];
            double px = Closes[1][0];
            double v = Volumes[1][0];

            if (State == State.Realtime)
            {
                if (lastTickTime != DateTime.MinValue && (t - lastTickTime).TotalSeconds > StaleDataSec && !BarsArray[1].IsFirstBarOfSession)
                    staleSinceLastBar = true;
                if (needReconcile && connectionOk)
                    Reconcile();
                RiskTick(px);
            }
            lastTickTime = t;

            if (pendingBar && t > pendingEnd)
                FinalizeBar(BarsArray[1].IsFirstBarOfSession);

            // Tick rule: uptick = buy, downtick = sell, unchanged = previous side.
            int dir = double.IsNaN(lastTickPx) ? 0 : (px > lastTickPx ? 1 : (px < lastTickPx ? -1 : lastTickDir));
            if (dir > 0) buyVol += v; else if (dir < 0) sellVol += v; else { buyVol += v / 2; sellVol += v / 2; }
            lastTickDir = dir;
            lastTickPx = px;
        }

        private static long Level(double p, double tick)
        {
            return (long)Math.Floor(p / tick + 0.5); // half-up; matches features._level
        }

        private void FinalizeBar(bool nextTickIsNewSession)
        {
            pendingBar = false;
            double tick = TickSize;
            double delta = buyVol - sellVol;
            double bVol = buyVol, sVol = sellVol;
            buyVol = 0; sellVol = 0;

            if (pendingFirstOfSession)
            {
                RollProfile(tick);
                guard.NewDay(pendingSessionId);
                if (Position.MarketPosition == MarketPosition.Flat) { entryPending = false; exitPending = false; }
                TryLoadParams(false); // promotions take effect at session boundaries only
            }
            else if (!prmValid)
                TryLoadParams(false);

            // --- volume profile: uniform distribution over the bar's ticks ---
            long a = Level(pLow, tick), b = Level(pHigh, tick);
            double per = pVolume / (b - a + 1);
            for (long k = a; k <= b; k++)
            {
                double cur;
                hist.TryGetValue(k, out cur);
                hist[k] = cur + per;
            }
            if (a < histLo) histLo = a;
            if (b > histHi) histHi = b;

            // --- ATR (Wilder, SMA seed) ---
            double tr = double.IsNaN(prevBarClose) ? pHigh - pLow
                : Math.Max(pHigh - pLow, Math.Max(Math.Abs(pHigh - prevBarClose), Math.Abs(pLow - prevBarClose)));
            if (barCount < AtrN)
            {
                trSeedSum += tr;
                if (barCount == AtrN - 1) atr = trSeedSum / AtrN;
            }
            else
                atr = (atr * (AtrN - 1) + tr) / AtrN;

            // --- delta z (population std over window incl. current) ---
            deltaRing.Push(delta);
            double dz = double.NaN;
            if (deltaRing.Full)
            {
                double s = 0, ss = 0;
                for (int i = 0; i < DeltaZN; i++) { double x = deltaRing.Get(i); s += x; ss += x * x; }
                double mu = s / DeltaZN, var = ss / DeltaZN - mu * mu;
                dz = var > 1e-12 ? (delta - mu) / Math.Sqrt(var) : 0.0;
            }

            // --- volatility percentile rank vs previous L values ---
            double xv = atr / pClose;
            double volPct = double.NaN;
            if (volRing.Full && !double.IsNaN(xv))
            {
                int cnt = 0; bool ok = true;
                for (int i = 0; i < VolPctL; i++)
                {
                    double y = volRing.Get(i);
                    if (double.IsNaN(y)) { ok = false; break; }
                    if (y < xv) cnt++;
                }
                if (ok) volPct = (double)cnt / VolPctL;
            }
            volRing.Push(xv);

            // --- efficiency ratio ---
            closeRing.Push(pClose);
            double er = double.NaN;
            if (closeRing.Full)
            {
                double path = 0;
                for (int i = 1; i <= ErN; i++) path += Math.Abs(closeRing.Get(i) - closeRing.Get(i - 1));
                er = path > 0 ? Math.Abs(closeRing.Get(ErN) - closeRing.Get(0)) / path : 0.0;
            }

            fDz = dz; fVolPct = volPct; fEr = er;
            double prevClose = prevBarClose;
            prevBarClose = pClose;
            barCount++;

            if (barLog != null)
            {
                long epoch = (long)(TimeZoneInfo.ConvertTimeToUtc(pendingEnd, NinjaTrader.Core.Globals.GeneralOptions.TimeZoneInfo)
                    - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
                barLog.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                    epoch, pendingSessionId, pOpen, pHigh, pLow, pClose, pVolume, bVol, sVol));
                if (barCount % 30 == 0) barLog.Flush();
            }

            // --- time exit ---
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                barsHeld++;
                if (prmValid && barsHeld >= (int)prm["max_hold_bars"])
                    FlattenStrategy("time");
            }

            // --- regime + signal ---
            if (!prmValid || double.IsNaN(prevClose) || double.IsNaN(dz) || double.IsNaN(volPct)
                || double.IsNaN(er) || double.IsNaN(atr) || double.IsNaN(sessVah))
                return;
            int regime = RegimeBlock;
            if (volPct >= prm["regime_vol_lo"] && volPct <= prm["regime_vol_hi"])
                regime = er >= prm["er_trend"] ? RegimeTrend : RegimeRotational;
            if (regime == RegimeBlock) return;

            double thr = prm["delta_z_entry"];
            int sig = 0;
            if (regime == RegimeRotational && prm["enable_rotational"] > 0.5)
            {
                if (prevClose < sessVal && pClose >= sessVal && dz >= thr) sig = 1;
                else if (prevClose > sessVah && pClose <= sessVah && dz <= -thr) sig = -1;
            }
            else if (regime == RegimeTrend && prm["enable_trend"] > 0.5)
            {
                if (prevClose <= sessVah && pClose > sessVah && dz >= thr) sig = 1;
                else if (prevClose >= sessVal && pClose < sessVal && dz <= -thr) sig = -1;
            }
            bool stale = staleSinceLastBar;
            staleSinceLastBar = false;
            if (sig == 0 || nextTickIsNewSession || stale || entryPending) return;
            if (Position.MarketPosition != MarketPosition.Flat) return;
            if (!guard.CanEnter() || !HealthyForEntry()) return;

            int stopTicks = Math.Max(1, (int)Math.Floor(prm["stop_atr"] * atr / tick + 0.5));
            int tgtTicks = Math.Max(1, (int)Math.Floor(stopTicks * prm["target_rr"] + 0.5));
            double tickValue = Instrument.MasterInstrument.PointValue * tick;
            double mult = regime == RegimeRotational ? prm["size_mult_rotational"] : prm["size_mult_trend"];
            int qty = (int)Math.Floor(prm["risk_per_trade_usd"] * mult / (stopTicks * tickValue));
            qty = Math.Min(qty, (int)prm["max_contracts"]);
            // Pre-trade: the worst case of this trade must fit inside BOTH limits.
            double lossPerContract = stopTicks * tickValue + CommissionPerContractRt + 2 * tickValue;
            qty = guard.AllowedQty(qty, lossPerContract);
            if (qty < 1) return;

            entryRegime = regime;
            string name = sig > 0 ? LongName : ShortName;
            SetStopLoss(name, CalculationMode.Ticks, stopTicks, false);
            SetProfitTarget(name, CalculationMode.Ticks, tgtTicks);
            entryPending = true;
            if (sig > 0) EnterLong(0, qty, name); else EnterShort(0, qty, name);
        }

        private void RollProfile(double tick)
        {
            if (hist.Count > 0)
            {
                int n = (int)(histHi - histLo + 1);
                double[] h = new double[n]; // once per session
                foreach (var kv in hist) h[kv.Key - histLo] = kv.Value;
                double total = 0; int poc = 0;
                for (int k = 0; k < n; k++) { total += h[k]; if (h[k] > h[poc]) poc = k; }
                int lo = poc, hi = poc; double acc = h[poc], target = prm != null && prmValid ? prm["va_pct"] * total : 0.7 * total;
                while (acc < target && (lo > 0 || hi < n - 1))
                {
                    double up = hi < n - 1 ? h[hi + 1] : -1.0;
                    double dn = lo > 0 ? h[lo - 1] : -1.0;
                    if (up >= dn) { hi++; acc += up; } else { lo--; acc += dn; }
                }
                sessPoc = (poc + histLo) * tick;
                sessVah = (hi + histLo) * tick;
                sessVal = (lo + histLo) * tick;
            }
            hist.Clear();
            histLo = long.MaxValue; histHi = long.MinValue;
        }

        // ------------------------------------------------------------------
        // Risk and health
        // ------------------------------------------------------------------
        private void RiskTick(double px)
        {
            if (DateTime.UtcNow - lastAccountPollUtc > TimeSpan.FromSeconds(1))
            {
                lastAccountPollUtc = DateTime.UtcNow;
                double nl = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
                accountEquity = nl > 0 ? nl : double.NaN;
            }
            double unreal = Position.MarketPosition == MarketPosition.Flat ? 0.0
                : Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, px);
            RiskAction act = guard.Check(unreal, accountEquity);
            if (act != RiskAction.Ok && Position.MarketPosition != MarketPosition.Flat)
                FlattenStrategy(act == RiskAction.FlattenPermanent ? "trailing-drawdown" : "daily-loss");
        }

        private bool HealthyForEntry()
        {
            if (State != State.Realtime) return true;
            if (!connectionOk || needReconcile) return false;
            return (DateTime.UtcNow - reconnectUtc).TotalSeconds >= ReconnectCooldownSec;
        }

        private void Reconcile()
        {
            needReconcile = false;
            MarketPosition sp = Position.MarketPosition, ap = PositionAccount.MarketPosition;
            int sq = Position.Quantity, aq = PositionAccount.Quantity;
            bool atStartup = startupReconcile;
            startupReconcile = false;
            if (sp != ap || sq != aq)
            {
                if (atStartup)
                {
                    // Not our position (manual / other strategy): do not touch it, do not trade.
                    Log(string.Format("MarketRock: account holds {0}x{1} at start, strategy {2}x{3}; trading locked for the day", ap, aq, sp, sq), LogLevel.Alert);
                }
                else
                {
                    Log(string.Format("MarketRock: position diverged after reconnect strategy={0}x{1} account={2}x{3}; flattening", sp, sq, ap, aq), LogLevel.Alert);
                    Account.Flatten(new[] { Instrument });
                }
                guard.LockDay();
            }
        }

        private void FlattenStrategy(string why)
        {
            if (exitPending || Position.MarketPosition == MarketPosition.Flat) return;
            exitPending = true;
            if (Position.MarketPosition == MarketPosition.Long) ExitLong(0, Position.Quantity, "MR_X_" + why, LongName);
            else if (Position.MarketPosition == MarketPosition.Short) ExitShort(0, Position.Quantity, "MR_X_" + why, ShortName);
            Throttled("MarketRock: flatten (" + why + ")");
        }

        protected override void OnConnectionStatusUpdate(ConnectionStatusEventArgs connectionStatusUpdate)
        {
            bool ok = connectionStatusUpdate.Status == ConnectionStatus.Connected
                && connectionStatusUpdate.PriceStatus == ConnectionStatus.Connected;
            if (!ok)
                connectionOk = false;
            else if (!connectionOk)
            {
                connectionOk = true;
                reconnectUtc = DateTime.UtcNow;
                needReconcile = true; // positions may have changed while we were blind
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled,
            double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (order.Name == LongName || order.Name == ShortName)
            {
                if (orderState == OrderState.Filled || orderState == OrderState.Rejected || orderState == OrderState.Cancelled)
                    entryPending = false;
                if (orderState == OrderState.Filled) { barsHeld = 0; openTradeRegime = entryRegime; }
            }
            if (order.Name.StartsWith("MR_X_") && (orderState == OrderState.Cancelled || orderState == OrderState.Rejected))
                exitPending = false; // allow a retry on the next tick
            if (orderState != OrderState.Rejected) return;
            Log(string.Format("MarketRock: order {0} rejected ({1}: {2})", order.Name, error, comment), LogLevel.Alert);
            if (order.Name == LongName || order.Name == ShortName)
                return; // nothing opened; next signal may retry
            // A protective or exit order was rejected: the position may be naked.
            if (State == State.Realtime) Account.Flatten(new[] { Instrument });
            else FlattenStrategy("reject");
            guard.LockDay();
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            int n = SystemPerformance.AllTrades.Count;
            for (int i = lastTradeCount; i < n; i++)
            {
                Trade tr = SystemPerformance.AllTrades[i];
                double pnl = tr.ProfitCurrency;
                guard.OnRealized(pnl);
                if (tradeLog != null)
                {
                    long epoch = (long)(TimeZoneInfo.ConvertTimeToUtc(tr.Exit.Time, NinjaTrader.Core.Globals.GeneralOptions.TimeZoneInfo)
                        - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
                    int dir = tr.Entry.MarketPosition == MarketPosition.Long ? 1 : -1;
                    tradeLog.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6}",
                        epoch, currentSessionId, openTradeRegime, dir, tr.Quantity, pnl, prmHash));
                    tradeLog.Flush();
                }
            }
            lastTradeCount = n;
            if (Position.MarketPosition == MarketPosition.Flat) exitPending = false;
            if (guard.Persist) guard.Save(Path.Combine(DataDir, "riskstate.txt"));
        }

        // ------------------------------------------------------------------
        // Parameters
        // ------------------------------------------------------------------
        private void TryLoadParams(bool initial)
        {
            string jsonPath = Path.Combine(DataDir, "params.json"), shaPath = Path.Combine(DataDir, "params.sha256");
            try
            {
                if (!File.Exists(jsonPath) || !File.Exists(shaPath))
                {
                    if (initial) Log("MarketRock: params.json/params.sha256 missing - strategy will not trade", LogLevel.Error);
                    if (initial) prmValid = false;
                    return;
                }
                byte[] bytes = File.ReadAllBytes(jsonPath);
                string want = File.ReadAllText(shaPath).Trim().ToLowerInvariant();
                string got;
                using (SHA256 sha = SHA256.Create())
                    got = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                if (got == prmHash && prmValid) return; // unchanged
                if (got != want)
                {
                    // Possibly mid-write: keep the last verified set, retry next bar.
                    Log("MarketRock: params hash mismatch - keeping previous parameters", LogLevel.Warning);
                    if (initial) prmValid = false;
                    return;
                }
                var parsed = new Dictionary<string, double>();
                foreach (Match m in Regex.Matches(Encoding.ASCII.GetString(bytes), "\"([a-z_]+)\"\\s*:\\s*([-+0-9.eE]+)"))
                    parsed[m.Groups[1].Value] = double.Parse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture);
                if (parsed.Count != ParamBounds.Length)
                    throw new InvalidDataException("unexpected key count " + parsed.Count);
                foreach (object[] b in ParamBounds)
                {
                    string k = (string)b[0];
                    double v;
                    if (!parsed.TryGetValue(k, out v)) throw new InvalidDataException("missing " + k);
                    if (double.IsNaN(v) || v < (double)b[1] - 1e-12 || v > (double)b[2] + 1e-12)
                        throw new InvalidDataException(k + " out of bounds: " + v);
                }
                prm = parsed;
                prmHash = got;
                prmValid = true;
                Log("MarketRock: loaded params " + got.Substring(0, 12), LogLevel.Information);
            }
            catch (Exception ex)
            {
                // Fail closed: a bad file never replaces a good one, and with no
                // good one there are no entries.
                Log("MarketRock: params rejected: " + ex.Message, LogLevel.Error);
                if (initial) prmValid = false;
            }
        }

        private void Throttled(string msg)
        {
            if (DateTime.UtcNow - lastPrintUtc < TimeSpan.FromSeconds(5)) return;
            lastPrintUtc = DateTime.UtcNow;
            Print(msg);
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------
        private sealed class Ring
        {
            private readonly double[] buf;
            private int head, count;
            public Ring(int n) { buf = new double[n]; }
            public bool Full { get { return count == buf.Length; } }
            public void Push(double x) { buf[head] = x; head = (head + 1) % buf.Length; if (count < buf.Length) count++; }
            // 0 = oldest retained, count-1 = newest
            public double Get(int i) { return buf[(head - count + i + buf.Length) % buf.Length]; }
        }

        public enum RiskAction { Ok, FlattenDay, FlattenPermanent }

        public sealed class RiskGuard
        {
            public double StartEquity, TrailDd, TrailCap, DailyLoss, Buffer, ConsistencyPct, ConsistencyBase;
            public bool UseTrailCap, Persist;
            public double Hwm, CumRealized, DayStartRealized, PriorProfit, LastEquity;
            public bool PermLock, Fresh;
            public int CurrentDay, LockedDay;
            public double KnownFloor; // broker-reported liquidation threshold, 0 = unknown

            public void NewDay(int day)
            {
                if (day == CurrentDay) return;
                CurrentDay = day;
                DayStartRealized = CumRealized;
                PriorProfit = CumRealized;
            }

            public void OnRealized(double pnl) { CumRealized += pnl; }
            public void LockDay() { LockedDay = CurrentDay; }

            private double Floor()
            {
                double f = Hwm - TrailDd;
                if (UseTrailCap) f = Math.Min(f, StartEquity + TrailCap);
                return Math.Max(f, KnownFloor); // never looser than what the broker enforces
            }

            public RiskAction Check(double unrealized, double accountEquity)
            {
                if (Fresh && !double.IsNaN(accountEquity))
                {
                    // First live run without saved state: adopt the account's actual P&L.
                    CumRealized = accountEquity - unrealized - StartEquity;
                    DayStartRealized = CumRealized;
                    PriorProfit = CumRealized;
                    Fresh = false;
                }
                double own = StartEquity + CumRealized + unrealized;
                double eq = double.IsNaN(accountEquity) ? own : Math.Min(own, accountEquity);   // worse number for equity
                double peak = double.IsNaN(accountEquity) ? own : Math.Max(own, accountEquity); // higher number for the peak
                LastEquity = eq;
                if (peak > Hwm) Hwm = peak;
                if (eq <= Floor() + Buffer) { PermLock = true; return RiskAction.FlattenPermanent; }
                if ((CumRealized - DayStartRealized) + unrealized <= -DailyLoss + Buffer) { LockedDay = CurrentDay; return RiskAction.FlattenDay; }
                return RiskAction.Ok;
            }

            public bool CanEnter()
            {
                if (PermLock || LockedDay == CurrentDay) return false;
                double today = CumRealized - DayStartRealized;
                double cap = ConsistencyPct / (1 - ConsistencyPct) * Math.Max(PriorProfit, ConsistencyBase);
                return today < cap;
            }

            public int AllowedQty(int qty, double lossPerContract)
            {
                double eq = double.IsNaN(LastEquity) || LastEquity == 0 ? StartEquity + CumRealized : LastEquity;
                double roomTrail = eq - (Floor() + Buffer);
                double roomDay = DailyLoss - Buffer + (CumRealized - DayStartRealized);
                double room = Math.Min(roomTrail, roomDay);
                if (room <= 0 || lossPerContract <= 0) return 0;
                // Clamp in double space: casting a huge room straight to int overflows
                // to a negative number and silently sizes every trade to zero.
                return (int)Math.Min((double)qty, Math.Floor(room / lossPerContract));
            }

            public void Save(string path)
            {
                string tmp = path + ".tmp";
                File.WriteAllText(tmp, string.Format(CultureInfo.InvariantCulture,
                    "hwm={0:R}\ncum={1:R}\ndaystart={2:R}\nprior={3:R}\nperm={4}\nday={5}\nlocked={6}\n",
                    Hwm, CumRealized, DayStartRealized, PriorProfit, PermLock ? 1 : 0, CurrentDay, LockedDay));
                if (File.Exists(path)) File.Replace(tmp, path, null); else File.Move(tmp, path);
            }

            public void Load(string path)
            {
                if (!File.Exists(path)) { Fresh = true; return; } // first live run
                foreach (string line in File.ReadAllLines(path))
                {
                    int i = line.IndexOf('=');
                    if (i <= 0) continue;
                    string k = line.Substring(0, i), v = line.Substring(i + 1);
                    switch (k)
                    {
                        case "hwm": Hwm = double.Parse(v, CultureInfo.InvariantCulture); break;
                        case "cum": CumRealized = double.Parse(v, CultureInfo.InvariantCulture); break;
                        case "daystart": DayStartRealized = double.Parse(v, CultureInfo.InvariantCulture); break;
                        case "prior": PriorProfit = double.Parse(v, CultureInfo.InvariantCulture); break;
                        case "perm": PermLock = v == "1"; break;
                        case "day": CurrentDay = int.Parse(v, CultureInfo.InvariantCulture); break;
                        case "locked": LockedDay = int.Parse(v, CultureInfo.InvariantCulture); break;
                    }
                }
            }
        }

        #region Properties
        [NinjaScriptProperty, Range(1000, double.MaxValue)]
        [Display(Name = "Account start balance", GroupName = "1. Hard limits", Order = 1)]
        public double AccountStartBalance { get; set; }

        [NinjaScriptProperty, Range(100, double.MaxValue)]
        [Display(Name = "Trailing drawdown (USD)", GroupName = "1. Hard limits", Order = 2)]
        public double TrailingDrawdownUsd { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trailing floor stops at start + cap", GroupName = "1. Hard limits", Order = 3)]
        public bool UseTrailCap { get; set; }

        [NinjaScriptProperty, Range(0, double.MaxValue)]
        [Display(Name = "Trail cap (USD above start)", GroupName = "1. Hard limits", Order = 4)]
        public double TrailCapUsd { get; set; }

        [NinjaScriptProperty, Range(50, double.MaxValue)]
        [Display(Name = "Daily loss limit (USD)", GroupName = "1. Hard limits", Order = 5)]
        public double DailyLossLimitUsd { get; set; }

        [NinjaScriptProperty, Range(0, double.MaxValue)]
        [Display(Name = "Safety buffer before limits (USD)", GroupName = "1. Hard limits", Order = 6)]
        public double RiskBufferUsd { get; set; }

        [NinjaScriptProperty, Range(0.05, 0.95)]
        [Display(Name = "Consistency: max share of one day", GroupName = "1. Hard limits", Order = 7)]
        public double ConsistencyMaxPct { get; set; }

        [NinjaScriptProperty, Range(0, double.MaxValue)]
        [Display(Name = "Consistency base profit (USD)", GroupName = "1. Hard limits", Order = 8)]
        public double ConsistencyBaseUsd { get; set; }

        [NinjaScriptProperty, Range(0, double.MaxValue)]
        [Display(Name = "Broker liquidation threshold (USD, 0 = unknown)", GroupName = "1. Hard limits", Order = 9)]
        public double KnownFloorUsd { get; set; }

        [NinjaScriptProperty, Range(0, 100)]
        [Display(Name = "Commission per contract round trip", GroupName = "2. Execution", Order = 1)]
        public double CommissionPerContractRt { get; set; }

        [NinjaScriptProperty, Range(0, 600)]
        [Display(Name = "Reconnect cool-down (s)", GroupName = "2. Execution", Order = 2)]
        public int ReconnectCooldownSec { get; set; }

        [NinjaScriptProperty, Range(1, 600)]
        [Display(Name = "Stale data threshold (s)", GroupName = "2. Execution", Order = 3)]
        public int StaleDataSec { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Data directory", GroupName = "3. Files", Order = 1)]
        public string DataDir { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Export historical bars", GroupName = "3. Files", Order = 2)]
        public bool ExportHistory { get; set; }
        #endregion
    }
}
