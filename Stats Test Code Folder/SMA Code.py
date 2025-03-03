#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:50:49 2025

@author: student
"""
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
output_dir = "SMA_Strategy_Results"
os.makedirs(output_dir, exist_ok=True)

# Compute 5-day and 100-day Simple Moving Averages (SMA)
def compute_sma(data, window_short=5, window_long=100):
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data['SMA_5'] = data[price_col].rolling(window=window_short).mean()
    data['SMA_100'] = data[price_col].rolling(window=window_long).mean()
    return data

# Loop through tickers and date ranges
for ticker in tickers:
    for start_date, end_date in date_ranges:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = compute_sma(df)
        
        # Initialize signal column and state variable for enforcing alternation
        df['Signal'] = 0
        last_trade = 0  # 0 indicates no previous trade, 1 indicates last trade was buy, -1 indicates last trade was sell
        
        # Loop through each day to generate signals based on SMA crossovers
        for i in range(1, len(df)):
            # Buy condition: 5-day SMA crosses above 100-day SMA
            if df['SMA_5'].iloc[i] > df['SMA_100'].iloc[i] and df['SMA_5'].iloc[i-1] <= df['SMA_100'].iloc[i-1]:
                # Enforce alternating trades: only buy if the last trade wasn't a buy
                if last_trade != 1:
                    df.loc[df.index[i], 'Signal'] = 1
                    last_trade = 1
            # Sell condition: 5-day SMA crosses below 100-day SMA
            elif df['SMA_5'].iloc[i] < df['SMA_100'].iloc[i] and df['SMA_5'].iloc[i-1] >= df['SMA_100'].iloc[i-1]:
                # Enforce alternating trades: only sell if the last trade wasn't a sell
                if last_trade != -1:
                    df.loc[df.index[i], 'Signal'] = -1
                    last_trade = -1
        
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
        
        # Plot SMA with buy and sell signals
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Close Price", color="blue")
        plt.plot(df.index, df['SMA_5'], label="5-Day SMA", color="green")
        plt.plot(df.index, df['SMA_100'], label="100-Day SMA", color="red")
        plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'],
                    marker="^", color="magenta", label="Buy Signal", alpha=1)
        plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'],
                    marker="v", color="black", label="Sell Signal", alpha=1)
        plt.legend()
        plt.title(f"SMA Crossover Trading Signals for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_signals.png"))
        plt.close()
        
        # Plot strategy versus market performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color="blue")
        plt.plot(df.index, df['Cumulative Strategy Return'], label="SMA Strategy Return", color="orange")
        plt.legend()
        plt.title(f"SMA Trading Strategy vs Market Performance for {ticker} ({start_date} to {end_date})")
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
                f.write("✅ SMA strategy provides statistically significant excess returns!\n")
            else:
                f.write("❌ SMA strategy does not significantly outperform random trading.\n")
