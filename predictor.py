import contextlib
import io

import pandas as pd
import ta
import yfinance as yf

import os

from FinMind.data import DataLoader
from xgboost import XGBClassifier


# ===== 自動判斷股票代號 =====
def resolve_stock_code(stock_code):

    stock_code = stock_code.strip()

    if "." not in stock_code:

        original_code = stock_code

        tw_code = original_code + ".TW"
        two_code = original_code + ".TWO"

        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):

                tw_data = yf.download(
                    tw_code,
                    period="5d",
                    progress=False
                )

        if not tw_data.empty:
            return tw_code

        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):

                two_data = yf.download(
                    two_code,
                    period="5d",
                    progress=False
                )

        if not two_data.empty:
            return two_code

        raise ValueError(
            f"找不到股票代號：{original_code}"
        )

    return stock_code


# ===== 股票名稱 =====
def get_stock_name(stock_code):

    csv_path = os.path.join(
        "data",
        "stock_names.csv"
    )

    if os.path.exists(csv_path):

        stock_name_df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig"
        )

        matched = stock_name_df[
            stock_name_df["stock_code"] == stock_code
        ]

        if not matched.empty:
            return matched.iloc[0]["stock_name"]

    ticker = yf.Ticker(stock_code)

    try:
        info = ticker.info

        stock_name = (
            info.get("longName")
            or info.get("shortName")
            or stock_code
        )

    except:
        stock_name = stock_code

    return stock_name


# ===== 最新收盤價 =====
def get_latest_close(stock_code, df):

    ticker = yf.Ticker(stock_code)

    try:
        latest_close = ticker.fast_info["last_price"]

        if latest_close is None or latest_close == 0:
            raise ValueError("fast_info 無效")

    except:
        latest_close = float(
            df["Close"]
            .dropna()
            .iloc[-1]
        )

    return latest_close


# ===== 建立資料與特徵 =====
def build_feature_data(stock_code):

    api = DataLoader()

    finmind_stock_id = stock_code.split(".", 1)[0]

    # ===== 個股日 K =====
    finmind_df = api.taiwan_stock_daily(
        stock_id=finmind_stock_id,
        start_date="2020-01-01"
    )

    if finmind_df.empty:
        raise ValueError("FinMind 找不到日 K 資料")

    # ===== 三大法人 =====
    institution_df = api.taiwan_stock_institutional_investors(
        stock_id=finmind_stock_id,
        start_date="2024-01-01"
    )

    if not institution_df.empty:

        institution_df["net_buy"] = (
            institution_df["buy"]
            - institution_df["sell"]
        )

        institution_pivot = institution_df.pivot_table(
            index="date",
            columns="name",
            values="net_buy",
            aggfunc="sum"
        ).reset_index()

        institution_pivot = institution_pivot.fillna(0)

    else:

        institution_pivot = pd.DataFrame({
            "date": finmind_df["date"],
            "Foreign_Investor": 0,
            "Investment_Trust": 0,
            "Dealer_self": 0,
            "Dealer_Hedging": 0,
        })

    # ===== 欄位改名 =====
    finmind_df = finmind_df.rename(columns={
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume"
    })

    # ===== 合併法人 =====
    finmind_df = pd.merge(
        finmind_df,
        institution_pivot,
        on="date",
        how="left"
    )

    finmind_df = finmind_df.fillna(0)

    # ===== 確保法人欄位存在 =====
    institution_columns = [
        "Foreign_Investor",
        "Investment_Trust",
        "Dealer_self",
        "Dealer_Hedging"
    ]

    for col in institution_columns:
        if col not in finmind_df.columns:
            finmind_df[col] = 0

    # ===== 台股加權指數 =====
    twii = yf.download(
        "^TWII",
        start="2024-01-01",
        auto_adjust=False,
        progress=False
    )

    if isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.get_level_values(0)

    twii = twii.reset_index()

    twii = twii.rename(columns={
        "Date": "date"
    })

    twii["date"] = (
        pd.to_datetime(twii["date"])
        .dt.strftime("%Y-%m-%d")
    )

    twii["Market_Return"] = twii["Close"].pct_change()

    twii["Market_RSI"] = ta.momentum.RSIIndicator(
        twii["Close"],
        14
    ).rsi()

    twii["Market_Volatility"] = (
        twii["Market_Return"]
        .rolling(10)
        .std()
    )

    market_df = twii[
        [
            "date",
            "Market_Return",
            "Market_RSI",
            "Market_Volatility"
        ]
    ]

    finmind_df = pd.merge(
        finmind_df,
        market_df,
        on="date",
        how="left"
    )

    finmind_df = finmind_df.fillna(0)

    # ===== 美股領先指標 =====
    us_symbols = {
        "NVDA": "NVDA",
        "QQQ": "QQQ",
        "SOX": "^SOX"
    }

    for name, symbol in us_symbols.items():

        us_df = yf.download(
            symbol,
            start="2024-01-01",
            auto_adjust=False,
            progress=False
        )

        if isinstance(us_df.columns, pd.MultiIndex):
            us_df.columns = us_df.columns.get_level_values(0)

        us_df = us_df.reset_index()

        us_df = us_df.rename(columns={
            "Date": "date"
        })

        us_df["date"] = (
            pd.to_datetime(us_df["date"])
            .dt.strftime("%Y-%m-%d")
        )

        us_df[f"{name}_Return"] = (
            us_df["Close"]
            .pct_change()
            .shift(1)
        )

        us_feature_df = us_df[
            [
                "date",
                f"{name}_Return"
            ]
        ]

        finmind_df = pd.merge(
            finmind_df,
            us_feature_df,
            on="date",
            how="left"
        )

    finmind_df = finmind_df.fillna(0)

    df = finmind_df.copy()

    df = df[
        (df["Close"] > 0)
        & (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
    ].copy()

    # ===== 技術指標 =====
    df["Return"] = df["Close"].pct_change()
    df["Return_1"] = df["Return"].shift(1)
    df["Return_2"] = df["Return"].shift(2)
    df["Return_3"] = df["Return"].shift(3)

    df["Volatility"] = (
        df["Return"]
        .rolling(10)
        .std()
    )

    stoch = ta.momentum.StochasticOscillator(
        df["High"],
        df["Low"],
        df["Close"],
        14,
        3
    )

    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    df["Volume_MA5"] = (
        df["Volume"]
        .rolling(window=5)
        .mean()
    )

    df["Tomorrow_Close"] = (
        df["Close"]
        .shift(-1)
    )

    df["Target"] = (
        df["Tomorrow_Close"] > df["Close"]
    ).astype(int)

    return df


# ===== 真正 XGBoost 預測 =====
def run_xgboost_prediction(df):

    feature_columns = [
        "Volume_MA5",
        "Return",
        "Return_1",
        "Return_2",
        "Return_3",
        "K",
        "D",

        "Foreign_Investor",
        "Investment_Trust",
        "Dealer_self",
        "Dealer_Hedging",

        "Market_Return",
        "Market_RSI",
        "Market_Volatility",

        "NVDA_Return",
        "SOX_Return",
        "QQQ_Return",
    ]

    feature_data = df[feature_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    feature_data = feature_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid_prediction_rows = feature_data.dropna()

    if valid_prediction_rows.empty:
        raise ValueError(
            "有效特徵資料不足，無法預測"
        )

    latest_data = valid_prediction_rows.iloc[[-1]]

    model_df = df.loc[
        valid_prediction_rows.index
    ].dropna(
        subset=["Tomorrow_Close"]
    ).copy()

    if len(model_df) < 60:
        raise ValueError(
            "訓練資料不足，無法建立模型"
        )

    model_df["Target"] = (
        model_df["Tomorrow_Close"]
        > model_df["Close"]
    ).astype(int)

    X = model_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = model_df["Target"]

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    model.fit(X, y)

    prediction = model.predict(
        latest_data
    )[0]

    probability = model.predict_proba(
        latest_data
    )[0]

    up_probability = probability[1] * 100
    down_probability = probability[0] * 100

    if prediction == 1:
        direction = "上漲"
        confidence = up_probability
    else:
        direction = "下跌"
        confidence = down_probability

    return {
        "direction": direction,
        "confidence": confidence,
        "up_probability": up_probability,
        "down_probability": down_probability,
    }


# ===== AI 預測主函式 =====
def predict_stock(stock_id):

    stock_code = resolve_stock_code(stock_id)

    stock_name = get_stock_name(stock_code)

    df = build_feature_data(stock_code)

    latest_close = get_latest_close(
        stock_code,
        df
    )

    model_result = run_xgboost_prediction(df)

    recent_volatility = (
        df["Volatility"]
        .dropna()
        .iloc[-1]
    )

    price_range_value = (
        latest_close
        * recent_volatility
    )

    if model_result["direction"] == "上漲":

        lower_price = latest_close
        upper_price = latest_close + price_range_value

    else:

        lower_price = latest_close - price_range_value
        upper_price = latest_close

    price_range = (
        f"{round(lower_price)}"
        f" ~ "
        f"{round(upper_price)}"
    )

    result = {
        "stock_id": stock_code,
        "stock_name": stock_name,

        "direction": model_result["direction"],

        "confidence": round(
            model_result["confidence"],
            2
        ),

        "up_probability": round(
            model_result["up_probability"],
            2
        ),

        "down_probability": round(
            model_result["down_probability"],
            2
        ),

        "latest_close": round(
            latest_close,
            2
        ),

        "price_range": price_range
    }

    return result