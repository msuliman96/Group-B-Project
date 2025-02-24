import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

def get_date(val):
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val

# ---------------------------
# TECHNICAL INDICATOR FUNCTIONS
# ---------------------------
def compute_SMA(series, window):
    return series.rolling(window=window).mean()

def compute_EMA(series, window):
    return series.ewm(span=window, adjust=False).mean()

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

def compute_PSAR(high, low, close, start_af=0.02, step=0.02, max_af=0.2):
    high = high.squeeze()
    low = low.squeeze()
    close = close.squeeze()

    psar = close.copy()
    psar.iat[0] = low.iat[0]
    
    uptrend = True if close.iat[1] > close.iat[0] else False
    if uptrend:
        ep = high.iat[1]
        psar.iat[1] = low.iat[0]
    else:
        ep = low.iat[1]
        psar.iat[1] = high.iat[0]
    af = start_af

    for i in range(2, len(close)):
        prev_psar = psar.iat[i-1]
        curr_psar = prev_psar + af * (ep - prev_psar)
        if uptrend:
            curr_psar = min(curr_psar, low.iat[i-1], low.iat[i-2])
        else:
            curr_psar = max(curr_psar, high.iat[i-1], high.iat[i-2])
            
        reversal = False
        if uptrend and low.iat[i] < curr_psar:
            reversal = True
            curr_psar = ep
            ep = low.iat[i]
            af = start_af
            uptrend = False
        elif not uptrend and high.iat[i] > curr_psar:
            reversal = True
            curr_psar = ep
            ep = high.iat[i]
            af = start_af
            uptrend = True
        
        if not reversal:
            if uptrend:
                if high.iat[i] > ep:
                    ep = high.iat[i]
                    af = min(af + step, max_af)
            else:
                if low.iat[i] < ep:
                    ep = low.iat[i]
                    af = min(af + step, max_af)
                    
        psar.iat[i] = curr_psar
    return psar

def compute_ROC(series, period=12):
    return series.pct_change(periods=period) * 100

# ---------------------------
# DATA FUNCTIONS
# ---------------------------
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df.reset_index(inplace=True)
    return df

def compute_indicators(df, indicators):
    if indicators.get("Bollinger"):
        df["BB_upper"], df["BB_lower"] = compute_Bollinger_Bands(df["Close"])
    if indicators.get("EMA5"):
        df["EMA5"] = compute_EMA(df["Close"], 5)
    if indicators.get("EMA100"):
        df["EMA100"] = compute_EMA(df["Close"], 100)
    if indicators.get("SMA5"):
        df["SMA5"] = compute_SMA(df["Close"], 5)
    if indicators.get("SMA100"):
        df["SMA100"] = compute_SMA(df["Close"], 100)
    if indicators.get("RSI"):
        df["RSI"] = compute_RSI(df["Close"], 14)
    if indicators.get("MACD"):
        df["MACD"], df["MACD_signal"] = compute_MACD(df["Close"])
    if indicators.get("PSAR"):
        df["PSAR"] = compute_PSAR(df["High"], df["Low"], df["Close"])
    if indicators.get("ROC"):
        df["ROC"] = compute_ROC(df["Close"], period=12)
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
    if indicators.get("Bollinger"):
        feature_cols.extend(["BB_upper", "BB_lower"])
    if indicators.get("EMA5"):
        feature_cols.append("EMA5")
    if indicators.get("EMA100"):
        feature_cols.append("EMA100")
    if indicators.get("SMA5"):
        feature_cols.append("SMA5")
    if indicators.get("SMA100"):
        feature_cols.append("SMA100")
    if indicators.get("RSI"):
        feature_cols.append("RSI")
    if indicators.get("MACD"):
        feature_cols.extend(["MACD", "MACD_signal"])
    if indicators.get("PSAR"):
        feature_cols.append("PSAR")
    if indicators.get("ROC"):
        feature_cols.append("ROC")
    feature_cols.append("Close")
    
    X = df[feature_cols]
    y = df["Target"]
    return X, y, feature_cols

# ---------------------------
# ANTI-OVERFITTING MEASURES
# ---------------------------
def select_optimal_features(X_train, y_train, feature_names):
    """
    Perform feature selection based on importance to reduce dimensionality
    and minimize overfitting.
    """
    # Train a basic model
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Keep only the top 5 features to reduce overfitting
    top_n = min(5, len(feature_names))
    selected_indices = indices[:top_n]
    selected_features = [feature_names[i] for i in selected_indices]
    
    return selected_features, model

def standardize_features(X_train, X_test=None):
    """
    Standardize features to have zero mean and unit variance.
    This helps with model convergence and reduces sensitivity to outliers.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

def train_robust_model(X_train, y_train):
    """
    Train an XGBoost model with cross-validation and regularization
    to prevent overfitting.
    """
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define parameter grid with regularization
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [50, 100],
        'reg_alpha': [0.1, 0.5],  # L1 regularization
        'reg_lambda': [1.0, 5.0],  # L2 regularization
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Use grid search with time series cross-validation
    model = GridSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model.best_estimator_

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Indicator Importance")
    return fig

# ---------------------------
# WALK-FORWARD BACKTESTING
# ---------------------------
def walk_forward_backtest_strategy(df, indicators, feature_cols, initial_balance=10000, retrain_freq=20):
    """
    Perform walk-forward backtesting with reduced overfitting risk.
    We retrain the model less frequently and use anti-overfitting measures.
    """
    balance = initial_balance
    position = 0
    trade_log = []
    portfolio_values = []

    train_window = int(len(df) * 0.6)  # Use 60% for initial training, 40% for testing
    
    # Model retraining and prediction variables
    last_trained = 0
    model = None
    scaler = None
    selected_features = None
    
    for idx in range(train_window, len(df)):
        current_step = idx - train_window
        
        # Retrain model periodically instead of every day to reduce overfitting
        if current_step % retrain_freq == 0 or model is None:
            df_train = df.iloc[:idx]
            X_train, y_train, _ = select_features(df_train, indicators)
            
            # Apply feature selection to reduce dimensionality
            if selected_features is None:
                selected_features, _ = select_optimal_features(X_train, y_train, feature_cols)
            
            # Use only selected features
            X_train = X_train[selected_features]
            
            # Standardize features
            X_train_scaled, scaler = standardize_features(X_train)
            
            # Train robust model with regularization
            model = train_robust_model(X_train_scaled, y_train)
            last_trained = current_step
        
        # Make prediction for current step
        test_row = df.iloc[idx:idx+1]
        X_test, _, _ = select_features(test_row, indicators)
        
        # Apply same feature selection
        X_test = X_test[selected_features]
        
        # Apply same scaling
        X_test_scaled = scaler.transform(X_test)
        
        # Make prediction
        prediction = model.predict(X_test_scaled)[0]
        
        row = test_row.iloc[0]
        current_price = get_float(row["Close"])
        date = get_date(row["Date"])
        action = None

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
    
    # --- Random Trading Simulation ---
    # Generate random signals to match the testing window
    np.random.seed(42)  # For reproducibility
    test_size = len(df) - train_window
    random_signals = np.random.randint(0, 2, size=test_size)
    
    random_balance = initial_balance
    random_position = 0
    random_values = []
    
    for idx, signal in enumerate(random_signals):
        date = get_date(df.iloc[train_window + idx]["Date"])
        price = get_float(df.iloc[train_window + idx]["Close"])
        
        if signal == 1 and random_position == 0:  # Buy signal
            shares_to_buy = int(random_balance // price)
            if shares_to_buy > 0:
                random_position = shares_to_buy
                random_balance -= shares_to_buy * price
        elif signal == 0 and random_position > 0:  # Sell signal
            random_balance += random_position * price
            random_position = 0
        
        portfolio_value = random_balance + random_position * price
        random_values.append({
            "Date": date,
            "RandomTrading": portfolio_value
        })
    
    # Liquidate any remaining position
    if random_position > 0:
        final_price = get_float(df.iloc[-1]["Close"])
        random_balance += random_position * final_price
        random_values[-1]["RandomTrading"] = random_balance
    
    random_df = pd.DataFrame(random_values)
    random_df["Date"] = pd.to_datetime(random_df["Date"])

    return trade_log_df, portfolio_df, buy_hold_df, random_df, selected_features, model

# ---------------------------
# EMH ANALYSIS FUNCTIONS
# ---------------------------
def evaluate_emh_compliance(portfolio_df, buy_hold_df, random_df):
    # Calculate daily returns for all strategies
    portfolio_df['daily_return'] = portfolio_df['Portfolio Value'].pct_change()
    buy_hold_df['daily_return'] = buy_hold_df['BuyHold'].pct_change()
    random_df['daily_return'] = random_df['RandomTrading'].pct_change()
    
    # Drop NaN values
    strategy_returns = portfolio_df['daily_return'].dropna().values
    bh_returns = buy_hold_df['daily_return'].dropna().values
    random_returns = random_df['daily_return'].dropna().values
    
    # Calculate key metrics
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
    sortino_ratio = np.mean(strategy_returns) / np.std(strategy_returns[strategy_returns < 0]) * np.sqrt(252) if any(strategy_returns < 0) else float('inf')
    max_drawdown = calculate_max_drawdown(portfolio_df['Portfolio Value'])
    
    # Perform statistical tests
    # 1. Are returns statistically different from zero? (EMH prediction)
    t_stat, p_value_t = stats.ttest_1samp(strategy_returns, 0)
    
    # 2. Is strategy better than buy & hold?
    u_stat_bh, p_value_bh = stats.mannwhitneyu(strategy_returns, bh_returns, alternative='greater')
    
    # 3. Is strategy better than random trading?
    u_stat_random, p_value_random = stats.mannwhitneyu(strategy_returns, random_returns, alternative='greater')
    
    # 4. Test for randomness (runs test)
    runs, p_value_runs = runs_test(strategy_returns)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        't_stat': t_stat,
        'p_value_t': p_value_t,
        'p_value_bh': p_value_bh,
        'p_value_random': p_value_random,
        'runs_test_p': p_value_runs
    }

def runs_test(returns):
    # Convert returns to binary sequence (up or down)
    binary_seq = np.array([1 if r > 0 else 0 for r in returns])
    
    # Count runs
    n1 = sum(binary_seq)
    n0 = len(binary_seq) - n1
    
    # Count runs
    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1
    
    # Calculate expected runs under randomness
    expected_runs = ((2 * n1 * n0) / (n1 + n0)) + 1
    std_runs = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1)))
    
    z = (runs - expected_runs) / std_runs
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed test
    
    return runs, p_value

def calculate_max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())

# ---------------------------
# STREAMLIT UI
# ---------------------------
def main():
    st.title("Technical Analysis vs. Efficient Market Hypothesis")
    
    # Set fixed parameters as discussed
    initial_capital = 10000
    lookahead = 5
    
    # Sidebar configuration
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
    
    # Add info about anti-overfitting measures
    with st.sidebar.expander("Anti-Overfitting Measures"):
        st.markdown("""
        - Automatic feature selection (top 5 features)
        - Data standardization
        - Regularization (L1 & L2)
        - Less frequent model retraining
        - Cross-validation with time series splits
        """)
    
    # Technical indicators selection
    st.sidebar.subheader("Technical Indicators")
    indicators = {
        "Bollinger": st.sidebar.checkbox("Bollinger Bands", value=True),
        "EMA5": st.sidebar.checkbox("EMA (5-day)", value=True),
        "EMA100": st.sidebar.checkbox("EMA (100-day)", value=True),
        "SMA5": st.sidebar.checkbox("SMA (5-day)", value=True),
        "SMA100": st.sidebar.checkbox("SMA (100-day)", value=True),
        "RSI": st.sidebar.checkbox("RSI", value=True),
        "MACD": st.sidebar.checkbox("MACD", value=True),
        "PSAR": st.sidebar.checkbox("PSAR", value=True),
        "ROC": st.sidebar.checkbox("ROC", value=True)
    }
    
    if ticker and start_date and end_date:
        with st.spinner("Processing data..."):
            df = fetch_data(ticker, start_date, end_date)
            
            if df.empty:
                st.error("No data found for the ticker and date range provided.")
                return
                
            df["Date"] = pd.to_datetime(df["Date"])
            df = compute_indicators(df, indicators)
            df_ml = prepare_ml_data(df, lookahead)
            
            # Feature selection and model training
            X, y, feature_cols = select_features(df_ml, indicators)
            
            # Run backtesting with anti-overfitting measures
            with st.spinner("Running strategy simulation with anti-overfitting measures..."):
                trade_log_df, portfolio_df, buy_hold_df, random_df, selected_features, model = walk_forward_backtest_strategy(
                    df_ml, indicators, feature_cols, initial_capital, retrain_freq=20
                )
                
                # Calculate returns for all strategies
                initial_value = portfolio_df["Portfolio Value"].iloc[0]
                final_value = portfolio_df["Portfolio Value"].iloc[-1]
                strategy_return = ((final_value / initial_value) - 1) * 100
                
                initial_bh = buy_hold_df["BuyHold"].iloc[0]
                final_bh = buy_hold_df["BuyHold"].iloc[-1]
                bh_return = ((final_bh / initial_bh) - 1) * 100
                
                initial_random = random_df["RandomTrading"].iloc[0]
                final_random = random_df["RandomTrading"].iloc[-1]
                random_return = ((final_random / initial_random) - 1) * 100
                
                # Display returns comparison
                st.subheader("Strategy Comparison")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("TA Strategy Return", f"{strategy_return:.2f}%")
                
                with col2:
                    st.metric("Buy & Hold Return", f"{bh_return:.2f}%", 
                             delta=f"{strategy_return - bh_return:.2f}%")
                
                with col3:
                    st.metric("Random Trading Return", f"{random_return:.2f}%",
                             delta=f"{strategy_return - random_return:.2f}%")
                
                # Display selected features
                st.subheader("Optimally Selected Features")
                st.write("To reduce overfitting, only these top features were used:")
                st.write(selected_features)
                
                # Display feature importance
                if model is not None and selected_features is not None:
                    st.pyplot(plot_feature_importance(model, selected_features))
                
                # Performance chart
                st.subheader("Performance Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(portfolio_df["Date"], portfolio_df["Portfolio Value"], label="TA Strategy", linewidth=2)
                ax.plot(buy_hold_df["Date"], buy_hold_df["BuyHold"], label="Buy & Hold", linewidth=2, linestyle="--")
                ax.plot(random_df["Date"], random_df["RandomTrading"], label="Random Trading", linewidth=1, linestyle=":", color="gray")
                
                # Add buy/sell markers
                buy_signals = portfolio_df[portfolio_df["Action"] == "BUY"]
                if not buy_signals.empty:
                    ax.scatter(buy_signals["Date"], buy_signals["Portfolio Value"], 
                              marker="^", color="green", s=100, label="Buy Signal")
                
                sell_signals = portfolio_df[portfolio_df["Action"] == "SELL"]
                if not sell_signals.empty:
                    ax.scatter(sell_signals["Date"], sell_signals["Portfolio Value"], 
                              marker="v", color="red", s=100, label="Sell Signal")
                
                ax.set_title(f"Strategy Comparison: {ticker}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio Value ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Trade log
                with st.expander("Trade Log"):
                    st.dataframe(trade_log_df)
                
                # EMH Analysis
                st.subheader("EMH Analysis")
                emh_stats = evaluate_emh_compliance(portfolio_df, buy_hold_df, random_df)
                
                # Display key EMH stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sharpe Ratio", f"{emh_stats['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{emh_stats['max_drawdown']*100:.2f}%")
                
                with col2:
                    alpha = 0.05
                    t_test_result = "Reject EMH" if emh_stats['p_value_t'] < alpha else "Cannot reject EMH"
                    st.metric("Returns ≠ 0 (p-value)", f"{emh_stats['p_value_t']:.4f}", delta=t_test_result)
                    
                # Key EMH comparisons
                st.subheader("Statistical Tests")
                col1, col2 = st.columns(2)
                
                with col1:
                    bh_test_result = "Strategy > Buy & Hold" if emh_stats['p_value_bh'] < alpha else "Not better than Buy & Hold"
                    st.metric("vs Buy & Hold (p-value)", f"{emh_stats['p_value_bh']:.4f}", delta=bh_test_result)
                
                with col2:
                    random_test_result = "Strategy > Random" if emh_stats['p_value_random'] < alpha else "Not better than Random"
                    st.metric("vs Random Trading (p-value)", f"{emh_stats['p_value_random']:.4f}", delta=random_test_result)
                
                # Overall EMH conclusion
                if emh_stats['p_value_t'] < alpha and emh_stats['p_value_random'] < alpha:
                    st.success("🚫 Your strategy shows evidence against the weak form of EMH!")
                elif emh_stats['p_value_t'] < 0.1 and emh_stats['p_value_random'] < 0.1:
                    st.info("⚠️ Your strategy shows some evidence against EMH, but it's not strongly conclusive.")
                else:
                    st.warning("✅ Your strategy does not provide evidence against EMH. Results are consistent with market efficiency.")
    else:
        st.info("Please provide a stock ticker and date range to analyze.")

if __name__ == "__main__":
    main()