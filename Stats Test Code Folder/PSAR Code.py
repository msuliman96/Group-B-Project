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
output_dir = "PSAR_Strategy_Results"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate PSAR
def calculate_psar(df, step=0.02, max_step=0.2):
    high = df['High']
    low = df['Low']
    length = len(df)
    psar = np.zeros(length)
    bull = True  # starting trend
    af = step    # acceleration factor
    psar[0] = float(low.iloc[0])
    ep = float(high.iloc[0]) if bull else float(low.iloc[0])
   
    for i in range(1, length):
        prior_psar = psar[i - 1]
        psar[i] = prior_psar + af * (ep - prior_psar)

        # In a bull trend, PSAR cannot be above the previous two lows.
        if bull:
            if i >= 2:
                psar[i] = min(psar[i], float(low.iloc[i - 1]), float(low.iloc[i - 2]))
            else:
                psar[i] = min(psar[i], float(low.iloc[i - 1]))
        # In a bear trend, PSAR cannot be below the previous two highs.
        else:
            if i >= 2:
                psar[i] = max(psar[i], float(high.iloc[i - 1]), float(high.iloc[i - 2]))
            else:
                psar[i] = max(psar[i], float(high.iloc[i - 1]))

        # Check for reversal conditions:
        if bull:
            if float(low.iloc[i]) < psar[i]:
                bull = False
                psar[i] = ep
                af = step
                ep = float(low.iloc[i])
        else:
            if float(high.iloc[i]) > psar[i]:
                bull = True
                psar[i] = ep
                af = step
                ep = float(high.iloc[i])

        # Update extreme point and acceleration factor
        if bull:
            if float(high.iloc[i]) > ep:
                ep = float(high.iloc[i])
                af = min(af + step, max_step)
        else:
            if float(low.iloc[i]) < ep:
                ep = float(low.iloc[i])
                af = min(af + step, max_step)
               
    return pd.Series(psar, index=df.index)

# Loop through tickers and date ranges
for ticker in tickers:
    for start_date, end_date in date_ranges:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculate PSAR
        df['PSAR'] = calculate_psar(df)
        
        # Generate raw trading signals:
        # 1 if Close > PSAR (buy), -1 if Close < PSAR (sell)
        close_array = df['Close'].to_numpy().reshape(-1)
        psar_array  = df['PSAR'].to_numpy().reshape(-1)
        raw_signals = np.where(close_array > psar_array, 1, -1)
        
        # Enforce alternating buy-sell:
        # Only record a new nonzero signal if it differs from the last trade.
        signals = []
        last_trade = 0  # 0 means no trade yet; 1 indicates last trade was buy, -1 indicates sell.
        for s in raw_signals:
            if s != last_trade:
                signals.append(s)
                last_trade = s
            else:
                signals.append(0)  # No new trade if the same signal repeats
        
        df['Signal'] = pd.Series(signals, index=df.index)
        # Shift signals by one day to avoid lookahead bias
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
        
        # Plot price with PSAR and trading signals
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Close Price", color="blue")
        plt.plot(df.index, df['PSAR'], label="PSAR", color="orange", linestyle="--")
        plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], marker="^", color="green", label="Buy Signal", alpha=1)
        plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], marker="v", color="red", label="Sell Signal", alpha=1)
        plt.legend()
        plt.title(f"PSAR Trading Signals for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_signals.png"))
        plt.close()
        
        # Plot strategy vs. market performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color="blue")
        plt.plot(df.index, df['Cumulative Strategy Return'], label="PSAR Strategy Return", color="orange")
        plt.legend()
        plt.title(f"PSAR Trading Strategy vs Market Performance for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_performance.png"))
        plt.close()
        
        # Calculate Sharpe Ratio (using a risk-free rate of 0.0113)
        sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
        
        # Perform a one-sample T-Test on the strategy returns
        t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
        
        # Save statistical results to a text file
        with open(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_stats.txt"), "w") as f:
            f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("✅ PSAR strategy provides statistically significant excess returns!\n")
            else:
                f.write("❌ PSAR strategy does not significantly outperform random trading.\n")
        