import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------
# TECHNICAL INDICATOR FUNCTIONS
# ---------------------------
def compute_SMA(series, window):
    return series.rolling(window=window).mean()

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_MACD(series, fast=12, slow=26, signal=9):
    EMA_fast = series.ewm(span=fast, adjust=False).mean()
    EMA_slow = series.ewm(span=slow, adjust=False).mean()
    MACD_line = EMA_fast - EMA_slow
    signal_line = MACD_line.ewm(span=signal, adjust=False).mean()
    return MACD_line, signal_line

def compute_Bollinger_Bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

# ---------------------------
# DATA FUNCTIONS
# ---------------------------
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def compute_indicators(df, indicators):
    # Compute indicators using the 'Close' price
    if indicators.get("SMA50"):
        df["SMA50"] = compute_SMA(df["Close"], 50)
    if indicators.get("SMA200"):
        df["SMA200"] = compute_SMA(df["Close"], 200)
    if indicators.get("RSI"):
        df["RSI"] = compute_RSI(df["Close"], 14)
    if indicators.get("MACD"):
        df["MACD"], df["MACD_signal"] = compute_MACD(df["Close"])
    if indicators.get("Bollinger"):
        df["BB_upper"], df["BB_lower"] = compute_Bollinger_Bands(df["Close"])
    df.dropna(inplace=True)
    return df

def prepare_ml_data(df, lookahead):
    """
    Create a target column where:
      - 1 (BUY): if the price increases in the next `lookahead` days
      - 0 (SELL): if the price decreases in the next `lookahead` days
    """
    df = df.copy()
    # Ensure 'Close' is a 1D Series
    df["Close"] = df["Close"].squeeze()
    df["Future_Close"] = df["Close"].shift(-lookahead)
    # Flatten the arrays to ensure they are 1D
    close_arr = df["Close"].to_numpy().flatten()
    future_arr = df["Future_Close"].to_numpy().flatten()
    df["Target"] = np.where(future_arr > close_arr, 1, 0)
    df.dropna(inplace=True)
    return df

def select_features(df, indicators):
    """
    Select technical indicator columns (and Close) as features.
    """
    feature_cols = []
    if indicators.get("SMA50"):
        feature_cols.append("SMA50")
    if indicators.get("SMA200"):
        feature_cols.append("SMA200")
    if indicators.get("RSI"):
        feature_cols.append("RSI")
    if indicators.get("MACD"):
        feature_cols.append("MACD")
        # Optionally include the signal line:
        # feature_cols.append("MACD_signal")
    if indicators.get("Bollinger"):
        feature_cols.append("BB_upper")
        feature_cols.append("BB_lower")
    # It's often useful to include the closing price as well.
    feature_cols.append("Close")
    
    X = df[feature_cols]
    y = df["Target"]
    return X, y, feature_cols

# ---------------------------
# MACHINE LEARNING FUNCTIONS
# ---------------------------
def train_xgb_model(X_train, y_train):
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return acc, class_report, cm, y_pred

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature Importance")
    return fig

# ---------------------------
# BACKTESTING & BUY & HOLD COMPARISON
# ---------------------------
def backtest_strategy(df, model, feature_cols):
    """
    Simulate trading with your strategy and also compute a Buy & Hold portfolio.
    
    Trading Rules:
      - Start with $10,000.
      - On a BUY signal (model predicts 1) with no open position, buy as many shares as possible.
      - On a SELL signal (model predicts 0) with an open position, sell all shares.
      - All trades are executed at the market close price.
      - At the end, if a position is open, sell at the last available price.
      
    Buy & Hold:
      - Buy as many shares as possible on the first day and hold until the end.
    
    Returns:
      trade_log_df: DataFrame with trade log details.
      portfolio_df: Daily portfolio values for your strategy.
      buy_hold_df: Daily portfolio values for a Buy & Hold strategy.
    """
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trade_log = []
    portfolio_values = []
    
    # Loop over each day for strategy simulation
    for idx, row in df.iterrows():
        # Get current price and date as scalars
        current_price = float(row["Close"]) if not isinstance(row["Close"], pd.Series) else float(row["Close"].iloc[0])
        date = row["Date"] if not isinstance(row["Date"], pd.Series) else row["Date"].iloc[0]
        
        # Prepare feature vector and get model prediction
        X_day = row[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X_day)[0]
        action = None
        
        # BUY: if prediction is 1 and no open position
        if prediction == 1 and position == 0:
            shares_to_buy = int(balance // current_price)
            if shares_to_buy > 0:
                position = shares_to_buy
                balance -= shares_to_buy * current_price
                action = "BUY"
                trade_log.append({
                    "Date": date,
                    "Action": action,
                    "Price": current_price,
                    "Shares": shares_to_buy,
                    "Portfolio Value": balance + position * current_price
                })
        # SELL: if prediction is 0 and a position is held
        elif prediction == 0 and position > 0:
            balance += position * current_price
            action = "SELL"
            trade_log.append({
                "Date": date,
                "Action": action,
                "Price": current_price,
                "Shares": position,
                "Portfolio Value": balance
            })
            position = 0
        
        # Record the portfolio value for the day (cash + value of held shares)
        portfolio_value = balance + position * current_price
        portfolio_values.append({
            "Date": date,
            "Portfolio Value": portfolio_value,
            "Close": current_price,
            "Signal": prediction,
            "Action": action if action is not None else ""
        })
    
    # Close any open position on the final day
    if position > 0:
        final_row = df.iloc[-1]
        final_price = float(final_row["Close"]) if not isinstance(final_row["Close"], pd.Series) else float(final_row["Close"].iloc[0])
        final_date = final_row["Date"] if not isinstance(final_row["Date"], pd.Series) else final_row["Date"].iloc[0]
        balance += position * final_price
        trade_log.append({
            "Date": final_date,
            "Action": "SELL (Final)",
            "Price": final_price,
            "Shares": position,
            "Portfolio Value": balance
        })
        position = 0
        portfolio_values[-1]["Portfolio Value"] = balance
        portfolio_values[-1]["Action"] = "SELL (Final)"
    
    trade_log_df = pd.DataFrame(trade_log)
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df["Date"] = portfolio_df["Date"].apply(lambda d: pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d)
    
    # --- Calculate Buy & Hold Portfolio ---
    first_row = df.iloc[0]
    initial_price = float(first_row["Close"]) if not isinstance(first_row["Close"], pd.Series) else float(first_row["Close"].iloc[0])
    shares_bought = int(initial_balance // initial_price)
    cash_left = initial_balance - shares_bought * initial_price
    
    buy_hold_values = []
    for idx, row in df.iterrows():
        current_price = float(row["Close"]) if not isinstance(row["Close"], pd.Series) else float(row["Close"].iloc[0])
        date = row["Date"] if not isinstance(row["Date"], pd.Series) else row["Date"].iloc[0]
        portfolio_value = shares_bought * current_price + cash_left
        buy_hold_values.append({
            "Date": date,
            "BuyHold": portfolio_value
        })
    buy_hold_df = pd.DataFrame(buy_hold_values)
    buy_hold_df["Date"] = buy_hold_df["Date"].apply(lambda d: pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d)
    
    return trade_log_df, portfolio_df, buy_hold_df

# ---------------------------
# STREAMLIT USER INTERFACE
# ---------------------------
def main():
    st.title("XGBoost Trading Strategy vs. Buy & Hold")
    st.markdown("""
    This app implements an XGBoost-based trading strategy and compares it to a simple Buy & Hold approach.
    Explore the trade log, portfolio equity curves, and combined comparisons below.
    """)

    st.sidebar.header("User Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
    lookahead = st.sidebar.selectbox("Lookahead Period (Days)", options=[3, 5, 10], index=1)
    
    st.sidebar.subheader("Technical Indicators")
    indicators = {
        "SMA50": st.sidebar.checkbox("Simple Moving Average (50)", value=True),
        "SMA200": st.sidebar.checkbox("Simple Moving Average (200)", value=True),
        "RSI": st.sidebar.checkbox("Relative Strength Index (RSI)", value=True),
        "MACD": st.sidebar.checkbox("MACD", value=True),
        "Bollinger": st.sidebar.checkbox("Bollinger Bands", value=True)
    }
    
    if ticker and start_date and end_date:
        # Fetch historical data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the ticker and date range provided.")
            return
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Compute technical indicators
        df = compute_indicators(df, indicators)
        
        # Prepare data for ML
        df_ml = prepare_ml_data(df, lookahead)
        X, y, feature_cols = select_features(df_ml, indicators)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        st.subheader("Training XGBoost Model")
        model = train_xgb_model(X_train, y_train)
        acc, class_report, cm, y_pred = evaluate_model(model, X_test, y_test)
        st.write("**Model Accuracy:**", np.round(acc, 4))
        st.text(classification_report(y_test, y_pred))
        st.pyplot(plot_confusion_matrix(cm))
        st.pyplot(plot_feature_importance(model, feature_cols))
        
        st.subheader("Backtesting & Buy & Hold Comparison")
        trade_log_df, portfolio_df, buy_hold_df = backtest_strategy(df_ml, model, feature_cols)
        
        st.write("### Trade Log")
        st.dataframe(trade_log_df)
        
        # Plot Strategy Equity Curve
        fig_strategy, ax_strategy = plt.subplots(figsize=(10, 5))
        ax_strategy.plot(portfolio_df["Date"], portfolio_df["Portfolio Value"], label="Strategy Portfolio", color="blue")
        ax_strategy.set_title("Strategy Portfolio Equity Curve")
        ax_strategy.set_xlabel("Date")
        ax_strategy.set_ylabel("Portfolio Value ($)")
        ax_strategy.legend()
        st.pyplot(fig_strategy)
        
        # Plot Buy & Hold Equity Curve
        fig_bh, ax_bh = plt.subplots(figsize=(10, 5))
        ax_bh.plot(buy_hold_df["Date"], buy_hold_df["BuyHold"], label="Buy & Hold Portfolio", color="green")
        ax_bh.set_title("Buy & Hold Equity Curve")
        ax_bh.set_xlabel("Date")
        ax_bh.set_ylabel("Portfolio Value ($)")
        ax_bh.legend()
        st.pyplot(fig_bh)
        
        # Plot Combined Equity Curves for Comparison
        fig_combined, ax_combined = plt.subplots(figsize=(10, 5))
        ax_combined.plot(portfolio_df["Date"], portfolio_df["Portfolio Value"], label="Strategy", color="blue")
        ax_combined.plot(buy_hold_df["Date"], buy_hold_df["BuyHold"], label="Buy & Hold", color="green", linestyle="--")
        ax_combined.set_title("Strategy vs. Buy & Hold Comparison")
        ax_combined.set_xlabel("Date")
        ax_combined.set_ylabel("Portfolio Value ($)")
        ax_combined.legend()
        st.pyplot(fig_combined)
        
        # Optionally, overlay equity curves on the price chart
        fig_price, ax_price = plt.subplots(figsize=(10, 5))
        ax_price.plot(df_ml["Date"], df_ml["Close"], label="Close Price", color="black", alpha=0.5)
        ax_price2 = ax_price.twinx()
        ax_price2.plot(portfolio_df["Date"], portfolio_df["Portfolio Value"], label="Strategy Portfolio", color="blue")
        ax_price2.plot(buy_hold_df["Date"], buy_hold_df["BuyHold"], label="Buy & Hold Portfolio", color="green", linestyle="--")
        ax_price.set_title("Price Chart with Equity Curves")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)")
        ax_price2.set_ylabel("Portfolio Value ($)")
        ax_price.legend(loc="upper left")
        ax_price2.legend(loc="upper right")
        st.pyplot(fig_price)
        
        # Real-Time Prediction on Latest Data
        st.subheader("Real-Time Prediction on Latest Data")
        if st.button("Run Prediction on Most Recent Data Point"):
            latest_row = df_ml.iloc[[-1]]
            X_latest = latest_row[feature_cols]
            pred = model.predict(X_latest)[0]
            signal = "BUY" if pred == 1 else "SELL"
            st.write(f"The model signals a **{signal}** at the latest close price of ${latest_row['Close'].values[0]:.2f}")
    else:
        st.info("Please provide a stock ticker and date range.")

if __name__ == "__main__":
    main()