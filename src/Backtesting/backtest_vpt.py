#!/usr/bin/env python3
import os
# Force matplotlib to use a non-interactive backend for rendering in Streamlit
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
# Use Agg backend explicitly
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
# Ensure pyplot uses Agg
plt.switch_backend("agg")

import streamlit as st  # Streamlit for interactive UI
from datetime import datetime  # For date handling
import logging  # Standard logging library
import backtrader as bt  # Backtesting framework
import pandas as pd  # Data manipulation
import sys  # For manipulating Python path

# Add project source directory to path so custom modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Custom modules
from src.Data_Retrieval.data_fetcher import DataFetcher  # Historical data fetcher
from src.Agents.VPT.vpt_agent import VPTAnalysisAgent  # AI-based decision agent
import crewai  # CrewAI orchestration library
from crewai import Task, Crew, Process  # Task and Crew abstractions
from langchain_openai import ChatOpenAI  # LLM interface

# ----------------------------------------
# VPT calculation function
# ----------------------------------------
def calculate_vpt(df: pd.DataFrame,
                  calc_period: int = 1,
                  weighting_factor: float = 1.0,
                  apply_smoothing: bool = False,
                  smoothing_window: int = 1) -> pd.DataFrame:
    """
    Compute the Volume-Price Trend (VPT) indicator on a DataFrame.

    - pct_change on 'close'
    - multiply by volume and weighting_factor
    - cumulative sum -> VPT
    - optional rolling smoothing
    """
    df = df.copy()
    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Compute percentage change of closing price over 'calc_period' bars
    df['price_change'] = df['close'].pct_change(periods=calc_period)

    # Compute raw VPT and accumulate
    df['vpt'] = (df['volume'] * df['price_change'] * weighting_factor).cumsum()

    # Apply optional rolling smoothing
    if apply_smoothing and smoothing_window > 1:
        df['vpt'] = df['vpt'].rolling(window=int(smoothing_window), min_periods=1).mean()

    return df

# ----------------------------------------
# VPT Indicator Wrapper for Backtrader
# ----------------------------------------
class VPTIndicatorBT(bt.Indicator):
    """
    Backtrader indicator that wraps the calculate_vpt logic.
    Writes the VPT values into self.lines.vpt for each bar.
    """
    lines = ('vpt',)
    params = (
        ('calc_period', 1),
        ('weighting_factor', 1.0),
        ('apply_smoothing', False),
        ('smoothing_window', 1),
    )

    def __init__(self):
        # Determine minimum required period before indicator fires
        minp = self.p.calc_period + (self.p.smoothing_window - 1 if self.p.apply_smoothing else 0)
        self.addminperiod(minp)

    def once(self, start, end):
        # Called once after all historical data is loaded
        size = self.data.buflen()
        # Build a small DataFrame from the internal data buffer
        df = pd.DataFrame({
            'high':   [self.data.high[i]   for i in range(size)],
            'low':    [self.data.low[i]    for i in range(size)],
            'close':  [self.data.close[i]  for i in range(size)],
            'volume': [self.data.volume[i] for i in range(size)],
        })
        # Add a dummy date index (not used by VPT computation itself)
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')

        # Calculate VPT using our pandas function
        res = calculate_vpt(
            df,
            calc_period=self.p.calc_period,
            weighting_factor=self.p.weighting_factor,
            apply_smoothing=self.p.apply_smoothing,
            smoothing_window=self.p.smoothing_window
        )

        # Write results back into the indicator line buffer
        for i in range(size):
            self.lines.vpt[i] = res['vpt'].iat[i]

# ----------------------------------------
# VPT Strategy for Backtrader
# ----------------------------------------
class VPTStrategy(bt.Strategy):
    """
    Simple strategy: buy when VPT increases from previous bar, sell when it decreases.
    Logs each trade in self.trade_log.
    """
    params = (
        ('calc_period', 1),
        ('weighting_factor', 1.0),
        ('apply_smoothing', False),
        ('smoothing_window', 1),
        ('allocation', 1.0),  # fraction of cash to allocate per trade
    )

    def __init__(self):
        self.trade_log = []
        # Attach the VPT indicator to the data feed
        self.vpt_ind = VPTIndicatorBT(
            self.data,
            calc_period=self.p.calc_period,
            weighting_factor=self.p.weighting_factor,
            apply_smoothing=self.p.apply_smoothing,
            smoothing_window=self.p.smoothing_window
        )

    def next(self):
        # Called on each new bar
        dt       = self.datas[0].datetime.date(0)
        close    = self.data.close[0]
        vpt_now  = self.vpt_ind.vpt[0]
        vpt_prev = self.vpt_ind.vpt[-1]

        # Buy when momentum (VPT) turns positive
        if not self.position and vpt_now > vpt_prev:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            self.buy(size=size)
            msg = f"{dt}: BUY  {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)
        # Sell when momentum turns negative
        elif self.position and vpt_now < vpt_prev:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    """
    Set up Backtrader Cerebro engine, attach strategy and data, run, and return results.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    # Attach analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logging.info(f"Running {strategy_class.__name__}…")
    strat = cerebro.run()[0]

    # Extract performance metrics
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    summary = {
        "Sharpe Ratio":         strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "Total Return (%)":     r.get('rtot', 0) * 100,
        "Avg Daily Return (%)": r.get('ravg', 0) * 100,
        "Max Drawdown (%)":     d.get('drawdown', 0) * 100,
    }

    # Generate equity curve figure
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Shared LLM for CrewAI (not explicitly used here)
# ----------------------------------------
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    # Page title
    st.title("VPT Backtest + CrewAI Signals")

    # Sidebar inputs for parameters
    st.sidebar.header("Backtest Parameters")
    ticker           = st.sidebar.text_input("Ticker", "AAPL")
    sd               = st.sidebar.date_input("Start", datetime(2020, 1, 1).date())
    ed               = st.sidebar.date_input("End",   datetime.today().date())
    cash             = st.sidebar.number_input("Cash", 10000)
    comm             = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    calc_period      = st.sidebar.number_input("VPT Calc Period", 1, step=1)
    weighting_factor = st.sidebar.number_input("Weighting Factor", 1.0, step=0.1)
    apply_smoothing  = st.sidebar.checkbox("Apply Smoothing", value=False)
    smoothing_window = st.sidebar.number_input("Smoothing Window", 1, value=5, step=1) if apply_smoothing else 1

    # Run button
    if st.sidebar.button("Run Backtest"):
        # Fetch historical OHLCV data
        df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)

        # Calculate VPT and rename for the agent
        df_vpt = calculate_vpt(
            df,
            calc_period=calc_period,
            weighting_factor=weighting_factor,
            apply_smoothing=apply_smoothing,
            smoothing_window=smoothing_window
        )
        df_vpt.rename(columns={'vpt': 'VPT'}, inplace=True)

        # Generate AI-based signal (logic retained but not displayed)
        globals()['data'] = df_vpt.assign(date=df_vpt.index)
        vpt_agent     = VPTAnalysisAgent()
        advisor_agent = vpt_agent.vpt_trading_advisor()
        current_price = df_vpt['close'].iloc[-1]
        task          = vpt_agent.vpt_analysis(advisor_agent, globals()['data'], current_price)
        crew          = Crew(agents=[advisor_agent], tasks=[task], verbose=True, process=Process.sequential)
        _             = crew.kickoff()  # run AI signal generation silently

        # Prepare Backtrader data feed from raw OHLCV
        feed = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)

        # Run the backtest and collect performance
        perf, trades, fig = run_backtest(VPTStrategy, feed, cash=cash, commission=comm)

        # Display performance summary
        st.subheader("Performance Summary")
        st.write(perf)

        # Display the executed trade log
        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        # Render the equity curve chart
        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == '__main__':
    # Enable logging when script runs
    logging.basicConfig(level=logging.INFO)
    main()
