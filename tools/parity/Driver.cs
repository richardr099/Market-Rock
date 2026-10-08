// Replays a tick CSV through MarketRockStrategy in NT8's event order
// (at equal timestamps the primary series is processed before the tick
// series) and writes every entry decision. Used by tests/test_parity.py.
// usage: parity <ticks.csv> <datadir> <entries_out.csv>
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;

class Series : ISeriesD { public double V; public double this[int i] { get { return V; } } }
class TSeries : ISeriesT { public DateTime V; public DateTime this[int i] { get { return V; } } }

static class Driver
{
    static int Main(string[] args)
    {
        var inv = CultureInfo.InvariantCulture;
        var t = new List<long>(); var px = new List<double>(); var vol = new List<double>(); var ses = new List<int>();
        foreach (var line in File.ReadLines(args[0]))
        {
            if (line.StartsWith("time")) continue;
            var f = line.Split(',');
            t.Add(long.Parse(f[0], inv)); px.Add(double.Parse(f[1], inv)); vol.Add(double.Parse(f[2], inv)); ses.Add(int.Parse(f[3], inv));
        }
        var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Unspecified);
        var sessionOfDay = new Dictionary<DateTime, int>();
        // bars: end = ceil(t/60)*60 ; tick at exactly T belongs to bar T
        var barEnd = new List<long>(); var bo = new List<double>(); var bh = new List<double>(); var bl = new List<double>();
        var bc = new List<double>(); var bv = new List<double>(); var bs = new List<int>();
        for (int i = 0; i < t.Count; i++)
        {
            long e = (t[i] + 59) / 60 * 60;
            if (barEnd.Count == 0 || barEnd[barEnd.Count - 1] != e)
            { barEnd.Add(e); bo.Add(px[i]); bh.Add(px[i]); bl.Add(px[i]); bc.Add(px[i]); bv.Add(0); bs.Add(ses[i]); }
            int k = barEnd.Count - 1;
            bh[k] = Math.Max(bh[k], px[i]); bl[k] = Math.Min(bl[k], px[i]); bc[k] = px[i]; bv[k] += vol[i];
        }
        for (int k = 0; k < barEnd.Count; k++) sessionOfDay[epoch.AddSeconds(barEnd[k])] = bs[k];
        SessionIterator.TradingDayOf = d => { int s = sessionOfDay[d]; return new DateTime(s / 10000, s / 100 % 100, s % 100); };

        var st = new MarketRockStrategy();
        var flags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        Action<string, object[]> call = (m, a) => typeof(MarketRockStrategy).GetMethod(m, flags).Invoke(st, a);
        st.State = State.SetDefaults; call("OnStateChange", new object[0]);
        st.DataDir = args[1]; st.ExportHistory = true;
        st.MaxRiskPerTradeUsd = 5000; st.DailyLossLimitUsd = 1e12; st.TrailingDrawdownUsd = 1e12; st.AccountStartBalance = 1e12; st.ConsistencyBaseUsd = 1e12;
        st.TickSize = 0.25; st.Instrument = new Instrument(); st.Instrument.MasterInstrument.PointValue = 5; // MES
        st.Position = new Position(); st.PositionAccount = new Position(); st.Account = new Account();
        st.SystemPerformance = new SystemPerformanceT();
        var b0 = new Bars(); var b1 = new Bars(); st.BarsArray = new[] { b0, b1 };
        var T0 = new TSeries(); var T1 = new TSeries(); st.Times = new ISeriesT[] { T0, T1 };
        var O0 = new Series(); var H0 = new Series(); var L0 = new Series(); var C0 = new Series(); var V0 = new Series();
        var C1 = new Series(); var V1 = new Series();
        st.Opens = new ISeriesD[] { O0, null }; st.Highs = new ISeriesD[] { H0, null }; st.Lows = new ISeriesD[] { L0, null };
        st.Closes = new ISeriesD[] { C0, C1 }; st.Volumes = new ISeriesD[] { V0, V1 };
        st.State = State.Configure; call("OnStateChange", new object[0]);
        st.State = State.DataLoaded; call("OnStateChange", new object[0]);
        st.State = State.Historical;

        var outw = new StreamWriter(args[2]); outw.WriteLine("bar_time,genome,direction,qty,stop_ticks,target_ticks");
        long lastBarClosed = 0; double stop = 0, tgt = 0;
        Strategy.OnSetStop = (n, v) => stop = v;
        Strategy.OnSetTarget = (n, v) => tgt = v;
        Strategy.OnEnter = (n, q) =>
        {
            string gid = (string)typeof(MarketRockStrategy).GetField("entryGenId", flags).GetValue(st);
            outw.WriteLine(string.Format(inv, "{0},{1},{2},{3},{4},{5}", lastBarClosed, gid, n == "MR_L" ? 1 : -1, q, stop, tgt));
            // no fills in this harness: report the entry as cancelled so the strategy stays flat
            call("OnOrderUpdate", new object[] { new Order { Name = n }, 0.0, 0.0, q, 0, 0.0, OrderState.Cancelled, DateTime.MinValue, ErrorCode.NoError, "" });
        };

        var featw = new StreamWriter(args[2] + ".features.csv"); featw.WriteLine("atr,delta_z,vol_pct,er,poc,vah,val,bar_in_session");
        Func<string, double> fld = nm => Convert.ToDouble(typeof(MarketRockStrategy).GetField(nm, flags).GetValue(st), inv);
        Func<double, string> fmt = x => x.ToString("R", inv);
        int lastCount = 0;
        int bi = 0;
        for (int i = 0; i < t.Count; i++)
        {
            // primary bars whose end <= this tick's time are processed first (equal stamps: primary first)
            while (bi < barEnd.Count && barEnd[bi] <= t[i])
            {
                st.BarsInProgress = 0; T0.V = epoch.AddSeconds(barEnd[bi]);
                O0.V = bo[bi]; H0.V = bh[bi]; L0.V = bl[bi]; C0.V = bc[bi]; V0.V = bv[bi];
                b0.IsFirstBarOfSession = bi == 0 || bs[bi] != bs[bi - 1];
                call("OnBarUpdate", new object[0]);
                lastBarClosed = barEnd[bi];
                bi++;
            }
            st.BarsInProgress = 1; T1.V = epoch.AddSeconds(t[i]); C1.V = px[i]; V1.V = vol[i];
            b1.IsFirstBarOfSession = i == 0 || ses[i] != ses[i - 1];
            call("OnBarUpdate", new object[0]);
            int done = (int)typeof(MarketRockStrategy).GetField("barCount", flags).GetValue(st);
            if (done != lastCount)
            {
                lastCount = done;
                featw.WriteLine(string.Join(",", new[] { fmt(fld("atr")), fmt(fld("fDz")), fmt(fld("fVolPct")), fmt(fld("fEr")),
                    fmt(fld("sessPoc")), fmt(fld("sessVah")), fmt(fld("sessVal")), fmt(fld("barInSession")) }));
            }
        }
        featw.Flush(); featw.Dispose();
        st.State = State.Terminated; call("OnStateChange", new object[0]);
        outw.Flush(); outw.Dispose();
        return 0;
    }
}
