#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:04:03 2025

@author: student
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

def fetch_and_analyze(ticker, start, end):
    # Create folder structure
    folder_path = f"Bollinger_Bands_Results/{ticker}/{start}_to_{end}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Download stock data
    df = yf.download(ticker, start=start, end=end)
    
    # Flatten MultiIndex Column Names if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Compute Bollinger Bands
    window = 20
    df['SMA'] = df['Close'].rolling(window=window, min_periods=1).mean()
    df['Std Dev'] = df['Close'].rolling(window=window, min_periods=1).std()
    df['Upper Band'] = df['SMA'] + (2 * df['Std Dev'])
    df['Lower Band'] = df['SMA'] - (2 * df['Std Dev'])
    
    # Generate trading signals ensuring alternate buy and sell orders
    df['Signal'] = 0
    position = 0  # 1 for long, -1 for short, 0 for no position
    
    for i in range(1, len(df)):
        if position == 0:
            if df['Close'].iloc[i] < df['Lower Band'].iloc[i]:
                df.at[df.index[i], 'Signal'] = 1
                position = 1
            elif df['Close'].iloc[i] > df['Upper Band'].iloc[i]:
                df.at[df.index[i], 'Signal'] = -1
                position = -1
        elif position == 1 and df['Close'].iloc[i] > df['Upper Band'].iloc[i]:
            df.at[df.index[i], 'Signal'] = -1
            position = 0
        elif position == -1 and df['Close'].iloc[i] < df['Lower Band'].iloc[i]:
            df.at[df.index[i], 'Signal'] = 1
            position = 0
    
    # Calculate returns
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Signal'].shift(1) * df['Market Return']
    df['Strategy Return'].fillna(0, inplace=True)
    
    # Compute performance metrics
    sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
    t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
    
    # Save results
    stats_path = os.path.join(folder_path, "performance_metrics.txt")
    with open(stats_path, "w") as f:
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
        f.write("✅ Statistically Significant!" if p_value < 0.05 else "❌ Not Significant.")
    
    # Plot Strategy Performance
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color='blue')
    plt.plot(df.index, df['Cumulative Strategy Return'], label="Bollinger Bands Strategy", color='orange')
    plt.legend()
    plt.title(f"Bollinger Bands Strategy vs Market for {ticker}")
    plt.savefig(os.path.join(folder_path, "performance_plot.png"))
    plt.close()
    
    # Plot Bollinger Bands with Signals
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label="Close Price", color="blue", alpha=0.6)
    plt.plot(df.index, df['Upper Band'], label="Upper Bollinger Band", color="red", linestyle="--")
    plt.plot(df.index, df['SMA'], label="20-Day SMA", color="black", linestyle="--")
    plt.plot(df.index, df['Lower Band'], label="Lower Bollinger Band", color="green", linestyle="--")
    plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], color="green", label="Buy Signal", marker="^")
    plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], color="red", label="Sell Signal", marker="v")
    plt.legend()
    plt.title(f"Bollinger Bands Trading Signals for {ticker}")
    plt.savefig(os.path.join(folder_path, "bollinger_bands_plot.png"))
    plt.close()
    
    # Save DataFrame
    df.to_csv(os.path.join(folder_path, "Bollinger_Bands_Data.csv"))

# Tickers and Date Ranges
tickers = ["AAPL", "BTC-USD", "^FTSE", "GBPUSD=X", "GC=F"]
date_ranges = [("2010-01-01", "2024-12-31")]

for ticker in tickers:
    for start, end in date_ranges:
        fetch_and_analyze(ticker, start, end)