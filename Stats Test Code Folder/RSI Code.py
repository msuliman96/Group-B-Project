#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:54:59 2025

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
output_dir = "RSI_Strategy_Results"
os.makedirs(output_dir, exist_ok=True)

# Compute RSI
def compute_rsi(data, window=14):
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    delta = data[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi
    return data

# Loop through tickers and date ranges
for ticker in tickers:
    for start_date, end_date in date_ranges:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = compute_rsi(df)
        
        # Generate trading signals with buy/sell alternation rule
        df['Signal'] = 0
        trade_state = 0  # 0 = No trade, 1 = Last trade was buy, -1 = Last trade was sell
        
        for i in range(1, len(df)):
            if df['RSI'].iloc[i] < 30 and trade_state != 1:  # Buy only if last trade was a sell or no trade
                df.loc[df.index[i], 'Signal'] = 1
                trade_state = 1
            elif df['RSI'].iloc[i] > 70 and trade_state != -1:  # Sell only if last trade was a buy
                df.loc[df.index[i], 'Signal'] = -1
                trade_state = -1
        
        df['Signal'] = df['Signal'].shift(1)  # Shift signals to avoid lookahead bias
        
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
        
        # Plot RSI with Buy/Sell markers
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="Close Price", color="blue")
        plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], color="green", label="Buy (RSI < 30)", marker="^", alpha=1)
        plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], color="red", label="Sell (RSI > 70)", marker="v", alpha=1)
        plt.legend()
        plt.title(f"RSI Trading Signals for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_signals.png"))
        plt.close()
        
        # Plot strategy vs market performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color='blue')
        plt.plot(df.index, df['Cumulative Strategy Return'], label="RSI Strategy Return", color='orange')
        plt.legend()
        plt.title(f"RSI Trading Strategy vs Market Performance for {ticker} ({start_date} to {end_date})")
        plt.savefig(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_performance.png"))
        plt.close()
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
        
        # Perform a one-sample T-Test
        t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
        
        # Save statistical results
        with open(os.path.join(ticker_dir, f"{ticker}_{start_date}_{end_date}_stats.txt"), "w") as f:
            f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("✅ RSI strategy provides statistically significant excess returns!\n")
            else:
                f.write("❌ RSI strategy does not significantly outperform random trading.\n")
