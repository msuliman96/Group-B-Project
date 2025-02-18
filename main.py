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
# HELPER FUNCTIONS
# ---------------------------
def get_float(val):
    """Safely extract a float from a value, even if it's a one-element Series."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

def get_date(val):
    """Safely extract a datetime value from a value, even if it's a one-element Series."""
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val

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
    df = df.copy()
    df["Close"] = df["Close"].squeeze()
    df["Future_Close"] = df["Close"].shift(-lookahead)
    close_arr = df["Close"].to_numpy().flatten()
    future_arr = df["Future_Close"].to_numpy().flatten()
    df["Target"] = np.where(future_arr > close_arr, 1, 0)
    df.dropna(inplace=True)
    return df

def select_features(df, indicators):
    feature_cols = []
    if indicators.get("SMA50"):
        feature_cols.append("SMA50")
    if indicators.get("SMA200"):
        feature_cols.append("SMA200")
    if indicators.get("RSI"):
        feature_cols.append("RSI")
    if indicators.get("MACD"):
        feature_cols.append("MACD")
    if indicators.get("Bollinger"):
        feature_cols.append("BB_upper")
        feature_cols.append("BB_lower")
    feature_cols.append("Close")
    
    X = df[feature_cols]
    y = df["Target"]
    return X, y, feature_cols

# ---------------------------
# MACHINE LEARNING FUNCTIONS
# ---------------------------
def train_xgb_model(X_train, y_train):
    # Removed use_label_encoder to avoid warning; eval_metric is set explicitly.
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return acc, class_report, cm, y_pred

def plot_confusion_matrix(cm):
    # Check if confusion matrix is empty
    if cm.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax.set_title("Confusion Matrix")
        ax.axis("off")
        return fig
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
# WALK-FORWARD BACKTESTING
# ---------------------------
def walk_forward_backtest_strategy(df, indicators, feature_cols):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trade_log = []
    portfolio_values = []

    # Use the first 80% of the data as the initial training window.
    train_window = int(len(df) * 0.8)

    # Walk-forward simulation: retrain the model each day using only past data.
    for idx in range(train_window, len(df)):
        df_train = df.iloc[:idx]
        test_row = df.iloc[idx:idx+1]

        X_train, y_train, _ = select_features(df_train, indicators)
        X_test, _, _ = select_features(test_row, indicators)

        model_loop = train_xgb_model(X_train, y_train)

        row = test_row.iloc[0]
        current_price = get_float(row["Close"])
        date = get_date(row["Date"])
        prediction = model_loop.predict(X_test)[0]
        action = None

        # BUY signal when prediction is 1 and no current position.
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
        # SELL signal when prediction is 0 and a position is held.
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

        portfolio_value = balance + position * current_price
        portfolio_values.append({
            "Date": date,
            "Portfolio Value": portfolio_value,
            "Close": current_price,
            "Signal": prediction,
            "Action": action if action is not None else ""
        })

    # Close any open position on the final day.
    if position > 0:
        final_row = df.iloc[-1]
        final_price = get_float(final_row["Close"])
        final_date = get_date(final_row["Date"])
        balance += position * final_price
        trade_log.append({
            "Date": final_date,
            "Action": "SELL (Final)",
            "Price": final_price,
            "Shares": position,
            "Portfolio Value": balance
        })
        position = 0

    trade_log_df = pd.DataFrame(trade_log)
    portfolio_df = pd.DataFrame(portfolio_values)
    # Ensure the Date column is in datetime format
    portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])

    # --- Buy & Hold Simulation ---
    first_test_row = df.iloc[train_window]
    initial_price = get_float(first_test_row["Close"])
    shares_bought = int(initial_balance // initial_price)
    cash_left = initial_balance - shares_bought * initial_price

    buy_hold_values = []
    for idx in range(train_window, len(df)):
        row = df.iloc[idx]
        current_price = get_float(row["Close"])
        date = get_date(row["Date"])
        portfolio_value = shares_bought * current_price + cash_left
        buy_hold_values.append({
            "Date": date,
            "BuyHold": portfolio_value
        })
    buy_hold_df = pd.DataFrame(buy_hold_values)
    buy_hold_df["Date"] = pd.to_datetime(buy_hold_df["Date"])

    return trade_log_df, portfolio_df, buy_hold_df

# ---------------------------
# STREAMLIT USER INTERFACE
# ---------------------------
def main():
    st.title("XGBoost Trading Strategy vs. Buy & Hold")
    st.markdown("""
    This app implements a walk-forward XGBoost-based trading strategy and compares it to a simple Buy & Hold approach.
    The model is retrained using only past data to avoid lookahead bias.
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
        df = fetch_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No data found for the ticker and date range provided.")
            return
        # 'fetch_data' already resets the index; we convert the Date column to datetime.
        df["Date"] = pd.to_datetime(df["Date"])
        df = compute_indicators(df, indicators)
        df_ml = prepare_ml_data(df, lookahead)

        # For an initial snapshot evaluation, split data into train/test (80%/20% split).
        X, y, feature_cols = select_features(df_ml, indicators)
        train_size = int(0.8 * len(df_ml))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        model = train_xgb_model(X_train, y_train)
        
        # Only evaluate if test data exists.
        if X_test.empty or y_test.empty:
            st.write("Not enough test data for evaluation. Please adjust the date range.")
        else:
            acc, class_report, cm, y_pred = evaluate_model(model, X_test, y_test)
            
            # Display larger accuracy score
            st.markdown(f"<h2>Model Accuracy: {np.round(acc, 4)}</h2>", unsafe_allow_html=True)
            
            # Display classification report as a table
            report_df = pd.DataFrame(class_report).transpose().round(4)
            st.subheader("Classification Report")
            st.table(report_df)
            
            st.pyplot(plot_confusion_matrix(cm))
            st.pyplot(plot_feature_importance(model, feature_cols))

        st.subheader("Walk-Forward Backtesting & Buy & Hold Comparison")
        trade_log_df, portfolio_df, buy_hold_df = walk_forward_backtest_strategy(df_ml, indicators, feature_cols)
        
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
            X_latest, _, _ = select_features(latest_row, indicators)
            pred = model.predict(X_latest)[0]
            signal = "BUY" if pred == 1 else "SELL"
            st.write(f"The model signals a **{signal}** at the latest close price of ${latest_row['Close'].values[0]:.2f}")
    else:
        st.info("Please provide a stock ticker and date range.")

if __name__ == "__main__":
    main()