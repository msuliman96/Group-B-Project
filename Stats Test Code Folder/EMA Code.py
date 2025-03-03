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
output_dir = "EMA_Strategy_Results"
os.makedirs(output_dir, exist_ok=True)

# Compute 5-day and 100-day Exponential Moving Averages (EMA)
def compute_ema(data, span_short=5, span_long=100):
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data['EMA_5'] = data[price_col].ewm(span=span_short, adjust=False).mean()
    data['EMA_100'] = data[price_col].ewm(span=span_long, adjust=False).mean()
    return data

# Loop through tickers and date ranges
for ticker in tickers:
    for start_date, end_date in date_ranges:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = compute_ema(df)
        
        # Generate trading signals based on EMA crossovers:
        # Buy when the 5-day EMA crosses above the 100-day EMA.
        # Sell when the 5-day EMA crosses below the 100-day EMA.
        # Enforce that a buy must be followed by a sell and vice versa.
        df['Signal'] = 0
        last_trade = 0  # 0 indicates no trade, 1 indicates last trade was buy, -1 indicates last trade was sell
        
        for i in range(1, len(df)):
            # Buy condition: 5-day EMA crosses above 100-day EMA
            if (df['EMA_5'].iloc[i] > df['EMA_100'].iloc[i] and 
                df['EMA_5'].iloc[i-1] <= df['EMA_100'].iloc[i-1]):
                if last_trade != 1:  # Only issue a buy if the last trade was not a buy
                    df.loc[df.index[i], 'Signal'] = 1
                    last_trade = 1
            # Sell condition: 5-day EMA crosses below 100-day EMA
            elif (df['EMA_5'].iloc[i] < df['EMA_100'].iloc[i] and 
                  df['EMA_5'].iloc[i-1] >= df['EMA_100'].iloc[i-1]):
                if last_trade != -1:  # Only issue a sell if the last trade was not a sell
                    df.loc[df.index[i], 'Signal'] = -1
                    last_trade = -1
        
        # Shift signals to avoid lookahead bias
        df['Signal'] = df['Signal'].shift(1)
        
        # Calculate returns
        df['Market Return'] = df['Close'].pct_change()
        df['Strategy Return'] = df['Signal'] * df['Market Return']
        df.dropna(inplace=True)
        
        # Calculate cumulative returns
        df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
        df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
        
        # Create subdirectory for ticker
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Plot price with EMA crossovers and signals
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Close Price", color="blue")
        plt.plot(df.index, df['EMA_5'], label="5-Day EMA", color="green")
        plt.plot(df.index, df['EMA_100'], label="100-Day EMA", color="red")
        plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], 
                    marker="^", color="magenta", label="Buy Signal", alpha=1)
        plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], 
                    marker="v", color="black", label="Sell Signal", alpha=1)
        plt.legend()
        plt.title(f"EMA Crossover Trading Signals for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_signals.png"))
        plt.close()
        
        # Plot strategy vs market performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color="blue")
        plt.plot(df.index, df['Cumulative Strategy Return'], label="EMA Strategy Return", color="orange")
        plt.legend()
        plt.title(f"EMA Trading Strategy vs Market Performance for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_performance.png"))
        plt.close()
        
        # Calculate Sharpe Ratio (using a risk-free rate of 0.0113)
        sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
        
        # Perform a one-sample T-Test on the strategy returns
        t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
        
        # Save statistical results
        with open(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_stats.txt"), "w") as f:
            f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("✅ EMA strategy provides statistically significant excess returns!\n")
            else:
                f.write("❌ EMA strategy does not significantly outperform random trading.\n")
