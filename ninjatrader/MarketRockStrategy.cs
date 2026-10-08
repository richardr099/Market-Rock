// Market-Rock NT8 execution layer.
//
// Mirrors marketrock/features.py, genome.py and strategy.py. Any change to a
// feature, condition or rounding rule must be made in BOTH places
// (tests/test_parity.py replays ticks through this file and compares).
//
// Design summary (see docs/ARCHITECTURE.md):
//  * Instrument: MES (Micro E-mini S&P 500). Primary series: 1-minute bars
//    (apply the strategy to a 1-minute MES chart). Sizing is in dollars and
//    uses the instrument's own PointValue, so contract counts are always right
//    for the chart it runs on.
//    Secondary series: 1-tick, used to classify volume with the TICK RULE.
//    The tick rule is used both historically and live, so the bar log the
//    Python learner trains on is produced by the same classifier that trades.
//  * A completed bar is FINALIZED on the first tick of the next bar, not in
//    the primary OnBarUpdate: NT8 processes the primary series before the
//    secondary at equal timestamps, so ticks stamped exactly at the bar end
//    would otherwise be lost. Entry orders therefore go out at the next bar's
//    first tick, which is exactly the Python fill model (next bar's open).
//  * Strategies arrive as DATA: portfolio.txt holds up to 8 genomes (rule
//    text + $ risk). The evolution engine can invent new strategies without
//    any change to this file; this file only interprets the fixed vocabulary.
//  * Hard account limits, and MaxRiskPerTradeUsd, are user-set properties here
//    and nowhere else. The learning loop can only write portfolio.txt.
//  * portfolio.txt is accepted only if its SHA-256 matches portfolio.sha256,
//    every genome id equals sha256(rule)[:12], and every value is in bounds.
//    Otherwise the previous portfolio stays (or, with none, no entries).

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
        private const string LongName = "MR_L", ShortName = "MR_S";

        // ---- genome vocabulary and bounds: must equal marketrock/genome.py ----
        private const int OpCrossUp = 0, OpCrossDown = 1, OpAbove = 2, OpBelow = 3, OpGe = 4, OpLe = 5, OpTimeIn = 6;
        private static readonly string[] OpNames = { "CROSS_UP", "CROSS_DOWN", "ABOVE", "BELOW", "GE", "LE", "TIME_IN" };
        private static readonly string[] LevelNames = { "VAH", "VAL", "POC" };
        private static readonly string[] FeatNames = { "delta_z", "er", "vol_pct" };
        private static readonly double[] FeatMin = { -3.0, 0.0, 0.0 };
        private static readonly double[] FeatMax = { 3.0, 1.0, 1.0 };
        private const int MaxGenomes = 8;
        private const int MaxConds = 4;
        private const int TimeMax = 1440;
        private const double StopMin = 0.5, StopMax = 3.0, RrMin = 0.8, RrMax = 3.0;
        private const int HoldMin = 5, HoldMax = 120;
        private const double MaxRiskFile = 5000.0;
        private const double VaPct = 0.7;

        private sealed class Gen
        {
            public string Id;
            public int Dir, Hold, N;
            public double Stop, Rr, Risk;
            public readonly int[] Op = new int[MaxConds], Arg = new int[MaxConds];
            public readonly double[] X = new double[MaxConds], Y = new double[MaxConds];
        }

        // ---- live portfolio (from portfolio.txt) ----
        private Gen[] gens = new Gen[0];
        private string pfHash = "";
        private bool pfValid;

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

        private int barInSession = -1;

        // ---- trade state ----
        private int barsHeld;
        private string entryGenId = "", openGenId = "";
        private int entryHold, openHold;
        private double entryStopValue, openStopValue; // $ per contract to the stop
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
                Description = "Self-developing order-flow strategy portfolio with hard prop-firm risk guard.";
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
                MaxRiskPerTradeUsd = 250;
                CommissionPerContractRt = 1.5; // MES round trip incl. fees; set your broker's rate
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
                TryLoadPortfolio(true);
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
                tradeLog = OpenCsv("trades.csv", "exit_time,session,genome,direction,qty,pnl_usd,risk_usd,portfolio_sha256", true);
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
                TryLoadPortfolio(false); // portfolio changes take effect at session boundaries only
            }
            else if (!pfValid)
                TryLoadPortfolio(false);
            barInSession = pendingFirstOfSession || barInSession < 0 ? 0 : barInSession + 1;

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

            // --- time exit (per the genome that opened the position) ---
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                barsHeld++;
                if (openHold > 0 && barsHeld >= openHold)
                    FlattenStrategy("time");
            }

            // --- genome signals: first genome (portfolio order) whose conditions all hold ---
            if (!pfValid || double.IsNaN(prevClose) || double.IsNaN(dz) || double.IsNaN(volPct) || double.IsNaN(er)
                || double.IsNaN(atr) || double.IsNaN(sessVah) || double.IsNaN(sessVal) || double.IsNaN(sessPoc))
                return;
            Gen hit = null;
            for (int gi = 0; gi < gens.Length && hit == null; gi++)
            {
                Gen g = gens[gi];
                bool all = true;
                for (int k = 0; k < g.N && all; k++)
                    all = CondTrue(g, k, prevClose, pClose, dz, er, volPct);
                if (all) hit = g;
            }
            bool stale = staleSinceLastBar;
            staleSinceLastBar = false;
            if (hit == null || nextTickIsNewSession || stale || entryPending) return;
            if (Position.MarketPosition != MarketPosition.Flat) return;
            if (!guard.CanEnter() || !HealthyForEntry()) return;

            int stopTicks = Math.Max(1, (int)Math.Floor(hit.Stop * atr / tick + 0.5));
            int tgtTicks = Math.Max(1, (int)Math.Floor(stopTicks * hit.Rr + 0.5));
            double tickValue = Instrument.MasterInstrument.PointValue * tick;
            double risk = Math.Min(hit.Risk, MaxRiskPerTradeUsd); // the user's ceiling always wins
            int qty = (int)Math.Floor(risk / (stopTicks * tickValue));
            // Pre-trade: the worst case of this trade must fit inside BOTH limits.
            double lossPerContract = stopTicks * tickValue + CommissionPerContractRt + 2 * tickValue;
            qty = guard.AllowedQty(qty, lossPerContract);
            if (qty < 1) return;

            entryGenId = hit.Id;
            entryHold = hit.Hold;
            entryStopValue = stopTicks * tickValue;
            string name = hit.Dir > 0 ? LongName : ShortName;
            SetStopLoss(name, CalculationMode.Ticks, stopTicks, false);
            SetProfitTarget(name, CalculationMode.Ticks, tgtTicks);
            entryPending = true;
            if (hit.Dir > 0) EnterLong(0, qty, name); else EnterShort(0, qty, name);
        }

        private bool CondTrue(Gen g, int k, double prevClose, double c, double dz, double er, double volPct)
        {
            int a = g.Arg[k];
            switch (g.Op[k])
            {
                case OpCrossUp: { double L = LevelOf(a); return prevClose <= L && c > L; }
                case OpCrossDown: { double L = LevelOf(a); return prevClose >= L && c < L; }
                case OpAbove: return c > LevelOf(a);
                case OpBelow: return c < LevelOf(a);
                case OpGe: return (a == 0 ? dz : a == 1 ? er : volPct) >= g.X[k];
                case OpLe: return (a == 0 ? dz : a == 1 ? er : volPct) <= g.X[k];
                case OpTimeIn: return barInSession >= g.X[k] && barInSession < g.Y[k];
            }
            return false;
        }

        private double LevelOf(int a)
        {
            return a == 0 ? sessVah : a == 1 ? sessVal : sessPoc;
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
                int lo = poc, hi = poc; double acc = h[poc], target = VaPct * total;
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
                if (orderState == OrderState.Filled)
                {
                    barsHeld = 0;
                    openGenId = entryGenId; openHold = entryHold; openStopValue = entryStopValue;
                }
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
                    tradeLog.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7}",
                        epoch, currentSessionId, openGenId, dir, tr.Quantity, pnl, openStopValue * tr.Quantity, pfHash));
                    tradeLog.Flush();
                }
            }
            lastTradeCount = n;
            if (Position.MarketPosition == MarketPosition.Flat) exitPending = false;
            if (guard.Persist) guard.Save(Path.Combine(DataDir, "riskstate.txt"));
        }

        // ------------------------------------------------------------------
        // Portfolio loading (fail closed)
        // ------------------------------------------------------------------
        private static string Sha256Hex(byte[] bytes)
        {
            using (SHA256 sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }

        private static double Num(string v)
        {
            double x = double.Parse(v, NumberStyles.Float, CultureInfo.InvariantCulture);
            if (double.IsNaN(x) || double.IsInfinity(x)) throw new InvalidDataException("non-finite number");
            return x;
        }

        private static int IndexOf(string[] names, string v)
        {
            int i = Array.IndexOf(names, v);
            if (i < 0) throw new InvalidDataException("unknown token " + v);
            return i;
        }

        private static void Check(bool ok, string what)
        {
            if (!ok) throw new InvalidDataException(what);
        }

        private static Gen ParseRule(string id, string rule, double risk)
        {
            Check(Sha256Hex(Encoding.ASCII.GetBytes(rule)).Substring(0, 12) == id, "genome id does not match rule");
            var kv = new Dictionary<string, string>();
            foreach (string part in rule.Split(';'))
            {
                int e = part.IndexOf('=');
                Check(e > 0, "bad rule part");
                kv[part.Substring(0, e)] = part.Substring(e + 1);
            }
            var g = new Gen { Id = id, Risk = risk };
            g.Dir = int.Parse(kv["dir"], CultureInfo.InvariantCulture);
            g.Stop = Num(kv["stop"]);
            g.Rr = Num(kv["rr"]);
            g.Hold = int.Parse(kv["hold"], CultureInfo.InvariantCulture);
            Check(g.Dir == 1 || g.Dir == -1, "dir");
            Check(g.Stop >= StopMin && g.Stop <= StopMax && g.Rr >= RrMin && g.Rr <= RrMax && g.Hold >= HoldMin && g.Hold <= HoldMax, "exit bounds");
            string[] conds = kv["c"].Split('|');
            Check(conds.Length >= 1 && conds.Length <= MaxConds, "condition count");
            g.N = conds.Length;
            for (int k = 0; k < conds.Length; k++)
            {
                string[] t = conds[k].Split(' ');
                int op = IndexOf(OpNames, t[0]);
                g.Op[k] = op;
                if (op <= OpBelow) { Check(t.Length == 2, "level arity"); g.Arg[k] = IndexOf(LevelNames, t[1]); }
                else if (op == OpGe || op == OpLe)
                {
                    Check(t.Length == 3, "feature arity");
                    int f = IndexOf(FeatNames, t[1]);
                    g.Arg[k] = f;
                    g.X[k] = Num(t[2]);
                    Check(g.X[k] >= FeatMin[f] && g.X[k] <= FeatMax[f], "threshold bounds");
                }
                else
                {
                    Check(t.Length == 3, "time arity");
                    g.X[k] = int.Parse(t[1], CultureInfo.InvariantCulture);
                    g.Y[k] = int.Parse(t[2], CultureInfo.InvariantCulture);
                    Check(g.X[k] >= 0 && g.X[k] < g.Y[k] && g.Y[k] <= TimeMax, "time bounds");
                }
            }
            return g;
        }

        private void TryLoadPortfolio(bool initial)
        {
            string txtPath = Path.Combine(DataDir, "portfolio.txt"), shaPath = Path.Combine(DataDir, "portfolio.sha256");
            try
            {
                if (!File.Exists(txtPath) || !File.Exists(shaPath))
                {
                    if (initial) { Log("MarketRock: portfolio.txt/portfolio.sha256 missing - strategy will not trade", LogLevel.Error); pfValid = false; }
                    return;
                }
                byte[] bytes = File.ReadAllBytes(txtPath);
                string want = File.ReadAllText(shaPath).Trim().ToLowerInvariant();
                string got = Sha256Hex(bytes);
                if (got == pfHash && pfValid) return; // unchanged
                if (got != want)
                {
                    // Possibly mid-write: keep the last verified portfolio, retry next bar.
                    Log("MarketRock: portfolio hash mismatch - keeping previous portfolio", LogLevel.Warning);
                    if (initial) pfValid = false;
                    return;
                }
                var kv = new Dictionary<string, string>();
                foreach (string line in Encoding.ASCII.GetString(bytes).Split('\n'))
                {
                    if (line.Length == 0) continue;
                    int e = line.IndexOf('=');
                    Check(e > 0, "bad line");
                    kv[line.Substring(0, e)] = line.Substring(e + 1);
                }
                Check(kv["version"] == "2", "version");
                Check(Num(kv["va_pct"]) == VaPct, "va_pct");
                int n = int.Parse(kv["n"], CultureInfo.InvariantCulture);
                Check(n >= 0 && n <= MaxGenomes, "genome count");
                var loaded = new Gen[n];
                for (int i = 0; i < n; i++)
                {
                    string p = "g" + i.ToString(CultureInfo.InvariantCulture) + ".";
                    double risk = Num(kv[p + "risk_usd"]);
                    Check(risk >= 0 && risk <= MaxRiskFile, "risk bounds");
                    loaded[i] = ParseRule(kv[p + "id"], kv[p + "rule"], risk);
                }
                gens = loaded;
                pfHash = got;
                pfValid = true;
                Log("MarketRock: loaded portfolio " + got.Substring(0, 12) + " (" + n.ToString(CultureInfo.InvariantCulture) + " strategies)", LogLevel.Information);
            }
            catch (Exception ex)
            {
                // Fail closed: a bad file never replaces a good one; with no good one there are no entries.
                Log("MarketRock: portfolio rejected: " + ex.Message, LogLevel.Error);
                if (initial) pfValid = false;
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

        [NinjaScriptProperty, Range(0, 5000)]
        [Display(Name = "Max risk per trade (USD) - ceiling on automatic sizing", GroupName = "1. Hard limits", Order = 10)]
        public double MaxRiskPerTradeUsd { get; set; }

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
