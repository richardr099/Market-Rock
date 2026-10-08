// COMPILE-CHECK STUBS ONLY. Minimal signatures of the NinjaTrader 8 API used by
// MarketRockStrategy.cs, so CI can catch syntax/type errors without NT8.
// Passing this check does NOT prove the real NT8 API matches - only that the
// strategy is internally consistent with these signatures. Compile in NT8
// (NinjaScript Editor, F5) before any use.
using System;
using System.Collections.Generic;

namespace NinjaTrader.Gui { }
namespace NinjaTrader.Gui.Tools { }
namespace NinjaTrader.Core.FloatingPoint { }
namespace NinjaTrader.Core
{
    public class GeneralOptionsT { public TimeZoneInfo TimeZoneInfo = TimeZoneInfo.Utc; }
    public static class Globals { public static string UserDataDir = "."; public static GeneralOptionsT GeneralOptions = new GeneralOptionsT(); }
}
namespace NinjaTrader.Cbi
{
    public enum MarketPosition { Flat, Long, Short }
    public enum OrderState { Accepted, Working, Filled, Rejected, Cancelled }
    public enum ErrorCode { NoError }
    public enum AccountItem { NetLiquidation, CashValue }
    public enum Currency { UsDollar }
    public enum PerformanceUnit { Currency, Points }
    public enum ConnectionStatus { Connected, Disconnected, ConnectionLost }
    public enum LogLevel { Information, Warning, Error, Alert }
    public class MasterInstrument { public double PointValue; }
    public class Instrument { public MasterInstrument MasterInstrument = new MasterInstrument(); }
    public class Account
    {
        public double Get(AccountItem item, Currency c) { return 0; }
        public void Flatten(ICollection<Instrument> instruments) { }
    }
    public class Order { public string Name = ""; }
    public class Execution { }
    public class Position
    {
        public MarketPosition MarketPosition; public int Quantity;
        public double GetUnrealizedProfitLoss(PerformanceUnit u, double price) { return 0; }
    }
    public class TradeExecution { public DateTime Time; public MarketPosition MarketPosition; }
    public class Trade { public double ProfitCurrency; public int Quantity; public TradeExecution Entry = new TradeExecution(), Exit = new TradeExecution(); }
    public class TradeCollection { public int Count { get { return 0; } } public Trade this[int i] { get { return null; } } }
    public class SystemPerformanceT { public TradeCollection AllTrades = new TradeCollection(); }
    public class ConnectionStatusEventArgs : EventArgs { public ConnectionStatus Status, PriceStatus; }
}
namespace NinjaTrader.Data
{
    public enum BarsPeriodType { Tick, Minute }
    public class Bars { public bool IsFirstBarOfSession; }
    public class SessionIterator
    {
        public static Func<DateTime, DateTime> TradingDayOf = t => t.Date; // test hook
        public SessionIterator(Bars b) { }
        public bool GetNextSession(DateTime t, bool includesEndTimeStamp) { ActualTradingDayExchange = TradingDayOf(t); return true; }
        public DateTime ActualTradingDayExchange;
    }
}
namespace NinjaTrader.NinjaScript
{
    public enum State { SetDefaults, Configure, DataLoaded, Historical, Transition, Realtime, Terminated }
    public enum Calculate { OnBarClose, OnEachTick, OnPriceChange }
    public enum EntryHandling { AllEntries, UniqueEntries }
    public enum StartBehavior { WaitUntilFlat, ImmediatelySubmit }
    public enum RealtimeErrorHandling { StopCancelClose, IgnoreAllErrors, StopCancelCloseIgnoreRejects }
    public enum ConnectionLossHandling { Recalculate, KeepRunning, StopStrategy }
    public enum CalculationMode { Ticks, Price, Currency }
    [AttributeUsage(AttributeTargets.Property)] public class NinjaScriptPropertyAttribute : Attribute { }
    public interface ISeriesD { double this[int barsAgo] { get; } }
    public interface ISeriesT { DateTime this[int barsAgo] { get; } }
}
namespace NinjaTrader.NinjaScript.Strategies
{
    using NinjaTrader.Cbi;
    using NinjaTrader.Data;
    public abstract class Strategy
    {
        public string Name, Description;
        public Calculate Calculate; public int EntriesPerDirection; public EntryHandling EntryHandling;
        public bool IsExitOnSessionCloseStrategy; public int ExitOnSessionCloseSeconds;
        public StartBehavior StartBehavior; public RealtimeErrorHandling RealtimeErrorHandling;
        public ConnectionLossHandling ConnectionLossHandling; public int DisconnectDelaySeconds;
        public int BarsRequiredToTrade; public bool IsUnmanaged, TraceOrders;
        public State State; public int BarsInProgress; public double TickSize;
        public Bars[] BarsArray; public ISeriesT[] Times; public ISeriesD[] Opens, Highs, Lows, Closes, Volumes;
        public Instrument Instrument; public Account Account; public Position Position, PositionAccount;
        public SystemPerformanceT SystemPerformance;
        protected void AddDataSeries(BarsPeriodType t, int v) { }
        // test hooks (parity driver)
        public static Action<string, double> OnSetStop = (n, v) => { }, OnSetTarget = (n, v) => { };
        public static Action<string, int> OnEnter = (n, q) => { };
        protected void SetStopLoss(string fromEntrySignal, CalculationMode m, double v, bool isSimulatedStop) { OnSetStop(fromEntrySignal, v); }
        protected void SetProfitTarget(string fromEntrySignal, CalculationMode m, double v) { OnSetTarget(fromEntrySignal, v); }
        protected Order EnterLong(int barsInProgressIndex, int quantity, string signalName) { OnEnter(signalName, quantity); return new Order { Name = signalName }; }
        protected Order EnterShort(int barsInProgressIndex, int quantity, string signalName) { OnEnter(signalName, quantity); return new Order { Name = signalName }; }
        protected Order ExitLong(int barsInProgressIndex, int quantity, string signalName, string fromEntrySignal) { return null; }
        protected Order ExitShort(int barsInProgressIndex, int quantity, string signalName, string fromEntrySignal) { return null; }
        protected void Print(string s) { Console.Error.WriteLine(s); }
        protected void Log(string s, LogLevel l) { Console.Error.WriteLine(l + ": " + s); }
        protected virtual void OnStateChange() { }
        protected virtual void OnBarUpdate() { }
        protected virtual void OnConnectionStatusUpdate(ConnectionStatusEventArgs e) { }
        protected virtual void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled,
            double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment) { }
        protected virtual void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time) { }
    }
}
