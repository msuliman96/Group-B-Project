import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# Define tickers and date ranges
tickers = ["AAPL", "BTC-USD", "^FTSE", "GBPUSD=X", "GC=F"]
date_ranges = [("2010-01-01", "2024-12-31")]

# Create output directory
output_dir = "ROC_Strategy_Results"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate ROC with a given lookback period (default is 12 days)
def compute_roc(df, window=12):
    # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    roc = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    return roc

# Loop through tickers and date ranges
for ticker in tickers:
    for start_date, end_date in date_ranges:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculate ROC indicator
        df['ROC'] = compute_roc(df, window=12)
        
        # Generate raw trading signals based on ROC:
        # Signal = 1 if ROC > 0 (buy), otherwise -1 (sell)
        raw_signals = np.where(df['ROC'].to_numpy() > 0, 1, -1).flatten()
        
        # Enforce alternating buy-sell:
        # Only record a new signal if it differs from the last nonzero trade.
        signals = []
        last_trade = 0  # 0 indicates no trade yet; 1 = last trade was buy, -1 = last trade was sell.
        for s in raw_signals:
            if s != last_trade:
                signals.append(s)
                last_trade = s
            else:
                signals.append(0)  # No new trade if the same signal repeats
        
        # Create the final signal series and shift by one day to avoid lookahead bias
        df['Signal'] = pd.Series(signals, index=df.index).shift(1)
        
        # Calculate returns
        df['Market Return'] = df['Close'].pct_change()
        df['Strategy Return'] = df['Signal'] * df['Market Return']
        df.dropna(inplace=True)
        
        # Calculate cumulative returns
        df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
        df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
        
        # Create subdirectory for the ticker
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Plot price with ROC-based trading signals
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Close Price", color="blue")
        # Mark buy signals (Signal == 1)
        plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'],
                    marker="^", color="green", label="Buy Signal", alpha=1)
        # Mark sell signals (Signal == -1)
        plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'],
                    marker="v", color="red", label="Sell Signal", alpha=1)
        plt.legend()
        plt.title(f"ROC Trading Signals for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_signals.png"))
        plt.close()
        
        # Plot cumulative performance of strategy vs market
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color="blue")
        plt.plot(df.index, df['Cumulative Strategy Return'], label="ROC Strategy Return", color="orange")
        plt.legend()
        plt.title(f"ROC Strategy vs Market Performance for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_performance.png"))
        plt.close()
        
        # Calculate Sharpe Ratio (using a risk-free rate of 0.0113 as before)
        sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
        
        # Perform a one-sample T-Test on the strategy returns
        t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
        
        # Save statistical results to a text file
        with open(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_stats.txt"), "w") as f:
            f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("✅ ROC strategy provides statistically significant excess returns!\n")
            else:
                f.write("❌ ROC strategy does not significantly outperform random trading.\n")
