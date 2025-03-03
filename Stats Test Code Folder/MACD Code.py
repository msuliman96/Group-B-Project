#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:01:47 2025

@author: student
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

def download_and_analyze_macd(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Signal'] = 0
    in_trade = False  # Track whether we are currently in a trade
    last_signal = 0    # Track the last signal (1 for buy, -1 for sell)
    
    for i in range(1, len(df)):
        if not in_trade:
            if df['MACD'].iloc[i-1] <= df['Signal Line'].iloc[i-1] and df['MACD'].iloc[i] > df['Signal Line'].iloc[i]:
                df.at[df.index[i], 'Signal'] = 1  # Buy Signal
                in_trade = True
                last_signal = 1
            elif df['MACD'].iloc[i-1] >= df['Signal Line'].iloc[i-1] and df['MACD'].iloc[i] < df['Signal Line'].iloc[i]:
                df.at[df.index[i], 'Signal'] = -1  # Sell Signal
                in_trade = True
                last_signal = -1
        else:
            if last_signal == 1 and df['MACD'].iloc[i-1] >= df['Signal Line'].iloc[i-1] and df['MACD'].iloc[i] < df['Signal Line'].iloc[i]:
                df.at[df.index[i], 'Signal'] = -1  # Sell Signal
                in_trade = False
            elif last_signal == -1 and df['MACD'].iloc[i-1] <= df['Signal Line'].iloc[i-1] and df['MACD'].iloc[i] > df['Signal Line'].iloc[i]:
                df.at[df.index[i], 'Signal'] = 1  # Buy Signal
                in_trade = False
    
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Signal'].shift(1) * df['Market Return']
    df['Strategy Return'].fillna(0, inplace=True)
    
    sharpe_ratio = (df['Strategy Return'].mean() - 0.00003) / df['Strategy Return'].std()
    t_stat, p_value = ttest_1samp(df['Strategy Return'].dropna(), 0)
    
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    
    folder_path = f"MACD_Strategy_Results/{ticker}/{start}_to_{end}"
    os.makedirs(folder_path, exist_ok=True)
    
    df.to_csv(f"{folder_path}/MACD_Trades.csv")
    
    with open(f"{folder_path}/MACD_Stats.txt", "w") as f:
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}\n")
        f.write("Significant Returns: Yes" if p_value < 0.05 else "Significant Returns: No")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", color='blue')
    plt.plot(df.index, df['Cumulative Strategy Return'], label="MACD Strategy", color='orange')
    plt.legend()
    plt.title(f"MACD Strategy vs Market Performance for {ticker}")
    plt.savefig(f"{folder_path}/Performance_Plot.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['MACD'], label="MACD", color="blue")
    plt.plot(df.index, df['Signal Line'], label="Signal Line", color="red", linestyle="--")
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['MACD'], color="green", label="Buy Signal", marker="^", alpha=1)
    plt.scatter(sell_signals.index, sell_signals['MACD'], color="red", label="Sell Signal", marker="v", alpha=1)
    plt.legend()
    plt.title(f"MACD Crossover Signals for {ticker}")
    plt.savefig(f"{folder_path}/MACD_Signals_Plot.png")
    plt.close()
    
    print(f"✅ Results saved for {ticker} ({start} to {end})")

tickers = ["AAPL", "BTC-USD", "^FTSE", "GBPUSD=X", "GC=F"]
date_ranges = [("2010-01-01", "2024-12-31")]

for ticker in tickers:
    for start, end in date_ranges:
        download_and_analyze_macd(ticker, start, end)
