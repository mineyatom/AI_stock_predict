



from multiprocessing.reduction import duplicate

from typing import final
from xxlimited import Str

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.base import clone

from datetime import datetime, timedelta

from FinMind.data import DataLoader



import ta.trend
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta
import os
import contextlib
import io

from predictor import predict_stock



# ===== matplotlib 中文字體 =====
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei"
]

plt.rcParams["axes.unicode_minus"] = False

# ===== 股票代號 =====
stock_code = input(
    "請輸入股票代號："
).strip()

# 如果輸入數字，自動判斷台股
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
        stock_code = tw_code

    else:
        with contextlib.redirect_stdout(io.StringIO()):
         with contextlib.redirect_stderr(io.StringIO()):
            two_data = yf.download(
                two_code,
                period="5d",
            progress=False
        )

        if not two_data.empty:
            stock_code = two_code

        else:
            raise ValueError(
                f"找不到股票代號：{original_code}"
            )

print(f"使用股票代號：{stock_code}")


# ===== 抓股票名稱 =====
def get_stock_name_from_csv(stock_code):

    stock_names = pd.read_csv(
        "data/stock_names.csv",
        encoding="utf-8-sig"
    )

    matched = stock_names[
        stock_names["stock_code"] == stock_code
    ]

    if not matched.empty:
        return matched.iloc[0]["stock_name"]

    return stock_code


stock_name = get_stock_name_from_csv(stock_code)


# ===== FinMind 測試 =====
api = DataLoader()

finmind_stock_id = stock_code.split(".", 1)[0]

finmind_df = api.taiwan_stock_daily(
    stock_id=finmind_stock_id,
    start_date="2020-01-01"
)




# ===== 三大法人資料 =====
institution_df = api.taiwan_stock_institutional_investors(
    stock_id=finmind_stock_id,
    start_date="2024-01-01"
)

institution_df["net_buy"] = (
    institution_df["buy"] - institution_df["sell"]
)

institution_pivot = institution_df.pivot_table(
    index="date",
    columns="name",
    values="net_buy",
    aggfunc="sum"
).reset_index()

institution_pivot = institution_pivot.fillna(0)




# ===== FinMind 欄位改名 =====
finmind_df = finmind_df.rename(columns={
    "open": "Open",
    "max": "High",
    "min": "Low",
    "close": "Close",
    "Trading_Volume": "Volume"
})


# ===== 合併法人資料 =====
finmind_df = pd.merge(
    finmind_df,
    institution_pivot,
    on="date",
    how="left"
)

finmind_df = finmind_df.fillna(0)


# ===== 台股加權指數 =====
twii = yf.download(
    "^TWII",
    start="2024-01-01",
    auto_adjust=False
)

if isinstance(twii.columns, pd.MultiIndex):
    twii.columns = twii.columns.get_level_values(0)

twii = twii.reset_index()

twii = twii.rename(columns={
    "Date": "date"
})

twii["date"] = pd.to_datetime(twii["date"]).dt.strftime("%Y-%m-%d")

twii["Market_Return"] = twii["Close"].pct_change()

twii["Market_RSI"] = ta.momentum.RSIIndicator(
    twii["Close"],
    14
).rsi()

twii["Market_Volatility"] = (
    twii["Market_Return"].rolling(10).std()
)

market_df = twii[
    [
        "date",
        "Market_Return",
        "Market_RSI",
        "Market_Volatility"
    ]
]


# ===== 合併大盤資料 =====
finmind_df = pd.merge(
    finmind_df,
    market_df,
    on="date",
    how="left"
)

finmind_df = finmind_df.fillna(0)


# ====== 美股領先指標 ======
us_symbols = {
    "NVDA": "NVDA",
    "QQQ" : "QQQ",
    "SPY" : "SPY",
    "SOX": "^SOX"
}

for name,symbol in us_symbols.items():

    us_df = yf.download(
        symbol,
        start="2024-01-01",
        auto_adjust=False
        )
    
    if isinstance(us_df.columns, pd.MultiIndex):
        us_df.columns = us_df.columns.get_level_values(0)

    us_df = us_df.reset_index()

    us_df = us_df.rename(
        columns={
            "Date": "date"
        }
    ) 

    us_df["date"] = pd.to_datetime(us_df["date"]).dt.strftime("%Y-%m-%d") 

    us_df[f"{name}_Return"] = us_df["Close"].pct_change()

    #!!!shift(1)，用前一天的美股資料，避免偷看未來
    us_df[f"{name}_Return"]= us_df[f"{name}_Return"].shift(1)

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

# ===== 最後才建立 df =====
df = finmind_df.copy()

# ===== 移除價格異常資料 =====
# 如果收盤價、開盤價、最高價、最低價小於等於 0，
# 代表資料異常，不能拿來訓練或回測
df = df[
    (df["Close"] > 0) &
    (df["Open"] > 0) &
    (df["High"] > 0) &
    (df["Low"] > 0)
].copy()

# ===== 儲存今日預測紀錄 =====
def save_prediction_log(
        predict_date,
        stock_code,
        stock_name,
        prediction_text,
        confidence,
        up_probability,
        down_probability,
        predict_close,
        lower_price,
        upper_price
):
    
    # ===== 日期格式統一 =====
    date = pd.to_datetime(
        predict_date
    )

    predict_date = (
        f"{date.year}/"
        f"{date.month}/"
        f"{date.day}"
    )

    log_file = "prediction_log.csv"

    new_record = pd.DataFrame([
        {
            "預測日期": predict_date,
            "股票代號": stock_code,
            "股票名稱": stock_name,

            "預測結果": prediction_text,

            "信心值": round(
                confidence,
                2
            ),

            "上漲機率": round(
                up_probability,
                2
            ),

            "下跌機率": round(
                down_probability,
                2
            ),

            "隔日預測參考價": round(
                predict_close
            ),

            "預測區間下緣": round(
                lower_price
            ),

            "預測區間上緣": round(
                upper_price
            ),

            "實際收盤價": None,
            "實際漲跌": None,
            "是否預測正確": None
        }
    ])

    # ===== 檢查是否已有 CSV =====
    if os.path.exists(log_file):

        old_log = pd.read_csv(
            log_file
        )

        duplicated = (
            (old_log["預測日期"] == predict_date)
            &
            (old_log["股票代號"] == stock_code)
        )

        # ===== 已存在 → 更新 =====
        if duplicated.any():

            old_log.loc[
                duplicated,
                new_record.columns
            ] = (
                new_record
                .iloc[0]
                .values
            )

            final_log = old_log

        # ===== 不存在 → 新增 =====
        else:

            final_log = pd.concat(
                [
                    old_log,
                    new_record
                ],
                ignore_index=True
            )

    else:

        final_log = new_record

    # ===== 統一整理數字格式 =====
    final_log["信心值"] = (
        final_log["信心值"]
        .round(2)
    )

    final_log["上漲機率"] = (
        final_log["上漲機率"]
        .round(2)
    )

    final_log["下跌機率"] = (
        final_log["下跌機率"]
        .round(2)
    )

    final_log["隔日預測參考價"] = (
        final_log["隔日預測參考價"]
        .round(0)
        .astype(int)
    )

    final_log["預測區間下緣"] = (
        final_log["預測區間下緣"]
        .round(0)
        .astype(int)
    )

    final_log["預測區間上緣"] = (
        final_log["預測區間上緣"]
        .round(0)
        .astype(int)
    )

    columns_order = [
        "預測日期",
        "股票代號",
        "股票名稱",
        "預測結果",
        "信心值",
        "上漲機率",
        "下跌機率",
        "隔日預測參考價",
        "預測區間下緣",
        "預測區間上緣",
        "實際收盤價",
        "實際漲跌",
        "是否預測正確"
    ]

    for col in columns_order:
        if col not in final_log.columns:
            final_log[col] = ""

    final_log = final_log[
        columns_order
    ]

    # ===== 存檔 =====
    final_log.to_csv(
        log_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n✅ 預測紀錄已存入 "
        "prediction_log.csv"
    )


# ===== 更新預測結果 =====
def update_prediction_result():

    print("🔥 update_prediction_result 啟動")

    log_file = "prediction_log.csv"

    if not os.path.exists(log_file):
        print("❌ 找不到 prediction_log.csv")
        return

    print("✅ 找到 prediction_log.csv")

    df_log = pd.read_csv(
        log_file,
        encoding="utf-8-sig",
        dtype={
        "實際漲跌": "object",
        "是否預測正確": "object"
        }
    )
    df_log = df_log.dropna(
    subset=["預測日期", "股票代號"]
    )

    

    print(f"📄 CSV 筆數：{len(df_log)}")
    today = pd.Timestamp.today().normalize()

    for index, row in df_log.iterrows():

        #if pd.notna(row["實際收盤價"]):
         #   continue

        predict_date = pd.to_datetime(
            row["預測日期"],
            errors="coerce")
        
        if pd.isna(predict_date):
            print(
                f"⚠️ 跳過日期格式錯誤的資料列：第 {index + 2} 列"
        )
            continue

        predict_date = predict_date.normalize()

        # 預測日還沒過，不更新
        if predict_date > today:
            print(
                f"⏳ 尚未到可更新日期："
                f"{row['股票代號']} "
                f"{predict_date.date()}"
            )
            continue
        stock_code = row["股票代號"]

        try:
            ticker = yf.Ticker(stock_code)

            history = ticker.history(
                start=predict_date.strftime("%Y-%m-%d"),
                end=(
                    predict_date
                    + pd.Timedelta(days=7)
                ).strftime("%Y-%m-%d")
            )

            if history.empty:
                continue

            actual_close = float(
                history["Close"].iloc[0]
            )

            reference_close = float(
                row["隔日預測參考價"]
            )

            prediction = row["預測結果"]

            if actual_close > reference_close:
                actual_direction = "上漲"
            elif actual_close < reference_close:
                actual_direction = "下跌"
            else:
                actual_direction = "持平"

            is_correct = (
                prediction == actual_direction
            )
            if is_correct:
                result_text = "正確"
            else:
                result_text = "錯誤"
                    

            

            df_log.loc[index, "實際收盤價"] = round(actual_close , 2)
            df_log.loc[index, "實際漲跌"] = str(actual_direction)
            df_log.loc[index, "是否預測正確"] = result_text

            print(f"✅ 已更新：{stock_code} {predict_date.date()}")

        except Exception as e:
            print(f"⚠️ 更新失敗：{stock_code}，原因：{e}")

    # ===== 存檔前整理格式 =====
    df_log["實際收盤價"] =(
        pd.to_numeric(
            df_log["實際收盤價"],
            errors="coerce"
        ).round(2)
    )

    # ===== 去除.0 =====
    df_log["實際收盤價"] = (
    df_log["實際收盤價"]
    .apply(
        lambda x:
        int(x)
        if pd.notna(x)
        and x.is_integer()
        else x
        )
    )

    df_log["實際漲跌"] = (
    df_log["實際漲跌"]
    .astype(str)
    )

    df_log["是否預測正確"] = (
    df_log["是否預測正確"]
    .astype(str)
    )
    df_log.to_csv(
        log_file,
        index=False,
        encoding="utf-8-sig"
    )

def fill_stock_names():

    df = pd.read_csv(
        "prediction_log.csv",
        encoding="utf-8-sig"
    )

    stock_names = pd.read_csv(
        "data/stock_names.csv",
        encoding="utf-8-sig"
    )

    for index, row in df.iterrows():

        stock_code = row["股票代號"]

        matched = stock_names[
            stock_names["stock_code"] == stock_code
        ]

        if not matched.empty:
            stock_name = matched.iloc[0]["stock_name"]
        else:
            stock_name = row["股票名稱"]

        df.loc[index, "股票名稱"] = stock_name

    df.to_csv(
        "prediction_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 股票名稱補完")


# ===== 補股票名稱（只跑一次）=====
#fill_stock_names()


# ===== 更新昨日預測 =====
update_prediction_result()    


# ===== 最新收盤價（Yahoo）=====
ticker = yf.Ticker(stock_code)

try:
    latest_close = ticker.fast_info[
        "last_price"
    ]
except:
    latest_close = float(
        df["Close"]
        .dropna()
        .iloc[-1]
    )


# 報酬率
df["Return"] = df["Close"].pct_change()

df["Return_1"] = df["Return"].shift(1)

df["Return_2"] = df["Return"].shift(2)

df["Return_3"] = df["Return"].shift(3)

df["Price_Change"] = (df["Close"] - df["Open"])
# 波動率
df["Volatility"] = (df["Return"].rolling(10).std() )

# RSI
df["RSI"] = ta.momentum.RSIIndicator(df["Close"],14).rsi()


# ====== MACD ======
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Diff"] = macd.macd_diff()


# ====== KD 指標 ======
stoch = ta.momentum.StochasticOscillator(
    df["High"],
    df["Low"],
    df["Close"],
    14,
    3
)
df["K"] = stoch.stoch()
df["D"] = stoch.stoch_signal()

# ===== 移動平均線 =====
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA60"] = df["Close"].rolling(window=60).mean()



# ===== 成交量平均 =====
df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()

# ===== 建立明天收盤價 =====
df["Tomorrow_Close"] = df["Close"].shift(-1)


# ===== 建立 Target =====
df["Target"] = (
    df["Tomorrow_Close"] > df["Close"]
).astype(int)






# ====== Feature(特徵) =====
X = df[
    [
        
        "Volume_MA5",
        "Return",
        "Return_1",
        "Return_2",
        "Return_3",
        "K",
        "D",

        # 法人籌碼
         "Foreign_Investor",
         "Investment_Trust",
         "Dealer_self",
         "Dealer_Hedging",

        #大盤特徵
         "Market_Return",
         "Market_RSI",
         "Market_Volatility",

         #美股領先指標
         "NVDA_Return",
         "SOX_Return",
         "QQQ_Return",
        

    ]

]

# 保留最新完整資料（用來預測明天）
# 最新一天還不知道明天收盤價，
# 避免被誤判成「下跌」標籤
feature_columns = list(X.columns)
feature_data = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

# ===== 確保所有特徵都是數字型態 =====
feature_data = feature_data.apply(
    pd.to_numeric,
    errors="coerce"
)


valid_prediction_rows = feature_data.dropna()

if valid_prediction_rows.empty:
    raise ValueError("Not enough valid feature data to make a prediction.")

latest_data = valid_prediction_rows.iloc[[-1]]

model_df = df.loc[valid_prediction_rows.index].dropna(
    subset=["Tomorrow_Close"]
).copy()

if len(model_df) < 2:
    raise ValueError("Not enough labeled data to train and test the models.")

model_df["Target"] = (
    model_df["Tomorrow_Close"] > model_df["Close"]
).astype(int)

# ===== 保持特徵、標籤與回測資料對齊 =====
df = model_df

# ===== 確保特徵全部都是數字格式 =====
X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

y = df["Target"]
# ===== 時間序列切分 =====
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ===== Baseline（基準模型）=====
# 永遠猜「上漲」

baseline_pred = [1] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

print("\n===== Baseline =====")
print(
    f"永遠猜上漲 Accuracy："
    f"{baseline_accuracy:.4f}"
)




# ===== 儲存最佳模型 =====
best_model_name = ""
best_model = None
best_accuracy = 0

# ===== 建立模型 =====
models = {
   

    "XGBoost":
      XGBClassifier(
          n_estimators=150,
          learning_rate=0.05,
          max_depth=5,
          random_state=42,
      )
}

for model_name, model in models.items():

    print(f"\n===== {model_name} =====")

    # ===== 訓練模型 =====
    model.fit(X_train, y_train)

    # ===== 預測 =====
    y_pred = model.predict(X_test)

    # ===== 準確率 =====
    accuracy = accuracy_score(y_test, y_pred)


    # 如果目前模型比較好，就記錄起來
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model


    print("\n=== Accuracy ===")
    print(accuracy)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))


   

    if hasattr(model, "feature_importances_"):

        print("\n=== Feature Importance ===")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })



        importance = importance.sort_values(
            by= "Importance",
            ascending=False 
        )
        print(importance)





# ===== 取得現在時間 =====
now = datetime.now()

# ===== 台股時間判斷 =====
# 凌晨到下午 13:30 前 → 預測今天
# 下午 13:30 後 → 預測下一交易日

if now.hour < 13 or (
    now.hour == 13 and now.minute < 30
):
    predict_date = now
else:
    predict_date = now + timedelta(days=1)

# ===== 跳過六日 =====
if predict_date.weekday() == 5:
    predict_date += timedelta(days=2)

elif predict_date.weekday() == 6:
    predict_date += timedelta(days=1)

# ===== 使用測試集表現最佳的分類模型進行最終預測 =====
if best_model is None:
    raise RuntimeError("No classification model was selected.")

# 用全部有標籤的資料重新訓練，讓最終預測更準確
final_model = clone(best_model)
final_model.fit(X, y)



# 最新一天資料
# 預測明天漲跌
prediction = final_model.predict(latest_data)[0]

# 預測機率
probability = final_model.predict_proba(latest_data)[0]


# 上漲與下跌機率
up_probability = probability[1] * 100
down_probability = probability[0] * 100


# ===== 多時間回測 =====

backtest_days_list = [
    30,
    60,
    90,
    120
]
# 每筆交易的來回手續費估算
fee_rate = 0.003


# ===== 停損停利設定 =====
stop_loss = -0.02
take_profit = 0.06


#===== Rolling Window 設定=====
rolling_window = 180
min_train_size = 60

# ===== 使用者輸入初始資金 =====
initial_capital = float(
    input("請輸入初始資金：")
)

# ===== 策略模式 =====

print("\n請選擇策略模式")
print("1. 積極模式（50%）")
print("2. 保守模式（65%）")

strategy_mode = input(
    "請輸入模式（1 或 2）："
)

# 根據模式設定門檻
if strategy_mode == "1":

    confidence_threshold = 50
    strategy_name = "積極模式"

elif strategy_mode == "2":

    confidence_threshold = 65
    strategy_name = "保守模式"

else:

    # 預設使用積極模式
    confidence_threshold = 50
    strategy_name = "積極模式（預設）"

print("\n📊" + "=" * 50)
print("          多時間回測結果")
print("=" * 52)

for backtest_days in backtest_days_list:

    capital = initial_capital

    correct_predictions = 0
    wrong_predictions = 0
    trade_count = 0

    capital_history = [capital]

    start_index = len(X) - backtest_days

    if start_index < 0:
        print(f"\n⚠️ 資料不足 {backtest_days} 天，跳過此回測。")
        continue

    for i in range(start_index, len(X) - 1):

        # 只學最近 180 天資料
        train_start = max(
            0,
            i - rolling_window
        )
        X_train_walk = X.iloc[train_start:i]
        y_train_walk = y.iloc[train_start:i]

        # 如果訓練資料沒有同時包含上漲和下跌，就跳過
        if y_train_walk.nunique() < 2:
            continue

        # 訓練資料太少時跳過
        if len(X_train_walk) < min_train_size:
            continue

        # 用今天資料預測明天
        X_today = X.iloc[[i]]
        y_actual = y.iloc[i]

        # 每一天重新建立模型，避免吃到未來資料
        walk_model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        walk_model.fit(
            X_train_walk,
            y_train_walk
        )

        # 預測明天漲跌
        y_pred_walk = walk_model.predict(
            X_today
        )[0]

        # 預測機率
        probability = walk_model.predict_proba(
            X_today
        )[0]

        # 上漲機率
        backtest_up_probability = (probability[1] * 100)

        today_close = df.iloc[i]["Close"]
        tomorrow_close = df.iloc[i + 1]["Close"]

        real_return = (
            tomorrow_close - today_close
        ) / today_close

        # ===== 停損停利 =====
        # 如果跌超過停損
        if real_return < stop_loss:
            real_return = stop_loss

        # 如果漲超過停利
        elif real_return > take_profit:
            real_return = take_profit

        # 預測正確率
        if y_pred_walk == y_actual:
            correct_predictions += 1
        else:
            wrong_predictions += 1

        # 根據策略模式決定是否交易
        if (
            y_pred_walk == 1
            and backtest_up_probability
            >= confidence_threshold
            ):

            trade_count += 1

            capital = capital * (
                1 + real_return - fee_rate
            )

        capital_history.append(capital)

    total_predictions = (
        correct_predictions
        + wrong_predictions
    )

    if total_predictions > 0:
        backtest_accuracy = (
            correct_predictions
            / total_predictions
        ) * 100
    else:
        backtest_accuracy = 0

    profit_percent = (
        (capital - initial_capital)
        / initial_capital
    ) * 100

    # ===== 買進持有比較 =====

    buy_hold_capital = (
        initial_capital *
        (
            df.iloc[-1]["Close"]
            / df.iloc[start_index]["Close"]
        )
    )

    buy_hold_return = (
        (buy_hold_capital - initial_capital)
        / initial_capital
    ) * 100

    # ====== 買進持有最大回撤 ======

    buy_hold_prices = df.iloc[
        start_index:
    ]["Close"]

    buy_hold_peak = buy_hold_prices.iloc[0]

    buy_hold_max_drawdown = 0

    for price in buy_hold_prices:

        # 更新最高價格
        if price > buy_hold_peak:
            buy_hold_peak = price

        # 計算回撤
        drawdown = (
            (buy_hold_peak - price)
            /buy_hold_peak
        ) * 100

        #更新最大回撤
        if drawdown > buy_hold_max_drawdown:
            buy_hold_max_drawdown = drawdown    


    # ===== 最大回撤 =====

    peak = capital_history[0]
    max_drawdown = 0

    for value in capital_history:

        if value > peak:
            peak = value

        drawdown = (
            (peak - value)
            / peak
        ) * 100

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    print("-" * 52)
    print(f"回測天數：{backtest_days} 天")
    print(f"策略模式：{strategy_name}")
    print(f"信心門檻：{confidence_threshold}%")
    print(f"每筆交易成本：{fee_rate * 100:.2f}%")
    print(f"停損設定：{stop_loss * 100:.1f}%")
    print(f"停利設定：{take_profit * 100:.1f}%")
    print(f"預測正確：{correct_predictions}")
    print(f"預測錯誤：{wrong_predictions}")
    print(f"勝率：{backtest_accuracy:.2f}%")
    print(f"交易次數：{trade_count}")
    print(f"策略報酬率：{profit_percent:.2f}%")
    print(f"買進持有報酬率：{buy_hold_return:.2f}%")
    print(
    f"買進持有最大回撤："
    f"{buy_hold_max_drawdown:.2f}%"
    )
    print(f"最大回撤：{max_drawdown:.2f}%")

print("=" * 52 + "📊")



 # ===== 信心門檻回測 =====

# thresholds = [50, 55, 60, 65]

# print("\n📈" + "=" * 50)
# print("          信心門檻回測")
# print("=" * 52)

# for threshold in thresholds:

#     threshold_capital = initial_capital
#     threshold_trade_count = 0
#     threshold_correct = 0
#     threshold_wrong = 0

#     for i in range(start_index, len(X) -1):

#         # 只有當天以前資料訓練
#         X_train_walk = X.iloc[:i]
#         y_train_walk = y.iloc[:i]

#         # 如果訓練資料只有一種結果，就跳過這一天
#         if y_train_walk.nunique() < 2:
#             continue

#         # 用當天資料預測明天
#         X_today = X.iloc[[i]]
#         y_actual = y.iloc[i]

#         walk_model = XGBClassifier(
#             n_estimators=150,
#             learning_rate =0.05,
#             max_depth=5,
#             random_state=42
#         )

#         walk_model.fit(X_train_walk, y_train_walk)

#         #預測上漲 / 下跌
#         y_pred_walk = walk_model.predict(X_today)[0]

#         #預測機率
#         probability = walk_model.predict_proba(X_today)[0]

#         #上漲機率
#         up_probability = probability[1] * 100

#         today_close = df.iloc[i]["Close"]
#         tommorow_close = df.iloc[i + 1]["Close"]

#         real_return = (
#             tommorow_close - today_close
#         ) / today_close

#         # 只有「預測上漲」且「上漲機率超過門檻」才買
#         if y_pred_walk == 1 and up_probability >= threshold:

#             threshold_trade_count += 1

#             threshold_capital = (
#                 threshold_capital * 
#                 (1 + real_return)
#             )

#             if y_pred_walk == y_actual:
#                 threshold_correct += 1
#             else:
#                 threshold_wrong += 1

#     if threshold_trade_count > 0:
#         threshold_win_rate = (
#             threshold_correct /
#             threshold_trade_count 
#         ) * 100
#     else:
#         threshold_win_rate = 0

#     threshold_return = (
#         (threshold_capital - initial_capital)
#         / initial_capital
#     )   * 100 
                        
                

#     print("-" * 52)
#     print(f"信心門檻：{threshold}%")
#     print(f"交易次數：{threshold_trade_count}")
#     print(f"交易勝率：{threshold_win_rate:.2f}%")
#     print(f"最終資金：{threshold_capital:.0f}")
#     print(f"策略報酬率：{threshold_return:.2f}%")

# print("=" * 52 + "📈")

# ===== AI 股票最終結果 =====

print("\n🤖" + "=" * 50)
print("          AI 股票預測系統 V5")
print("=" * 52)

print(
    f"股票：{stock_name}"
    f" ({stock_code})"
)
print(
    f"預測時間："
    f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
)

print(
    f"預測日期："
    f"{predict_date.strftime('%Y-%m-%d')}"
)
print("-" * 52)

print("【模型名稱】")
print(f"模型名稱：{best_model_name}")
print(f"模型準確率：{best_accuracy * 100:.2f}%")

print("-" * 52)

print("【明天漲跌預測】")

if prediction == 1:
    prediction_text = "上漲"
    confidence = up_probability
    print("預測結果：上漲 📈")
else:
    prediction_text = "下跌"
    confidence = down_probability
    print("預測結果：下跌 📉")

print(f"信心值：{confidence:.2f}%")
print(f"上漲機率：{up_probability:.2f}%")
print(f"下跌機率：{down_probability:.2f}%")

print("-" * 52)

print("【明天價格區間預測】")

# 最新收盤價
last_close = latest_close

# ===== 保守版價格區間 =====
recent_volatility = (
    df["Volatility"]
    .iloc[-1]
)

# 縮小波動幅度（避免 range 過大）
price_range = (
    last_close
    * recent_volatility
    
)


if prediction ==1:
    # 預測上漲
    lower_price = last_close
    upper_price = (
        last_close + price_range
    )

else:
    # 預測下跌    
    lower_price = (
        last_close - price_range
    )
    upper_price = last_close

print(f"最新收盤價：{last_close:.2f}")

if prediction == 1:
    print("預測方向：上漲 📈")
else:
    print("預測方向：下跌 📉")

print(
     f"預測區間："
    f"{lower_price:.2f}"
    f" ~ "
    f"{upper_price:.2f}"
)

save_prediction_log(
    predict_date.strftime("%Y-%m-%d"),
    stock_code,
    stock_name,
    prediction_text,
    confidence,
    up_probability,
    down_probability,
    last_close,
    lower_price,
    upper_price
)
print("-" * 52)
print("提醒：此結果僅供學習與研究，不構成投資建議。")
print("=" * 52 + "🤖")


# ===== 畫圖 =====
plt.figure(figsize=(14, 7))

# 日期轉 datetime
plot_date = pd.to_datetime(
    df["date"]
)

# 股價與均線
plt.plot(
    plot_date,
    df["Close"],
    label="收盤價"
)

plt.plot(
    plot_date,
    df["MA5"],
    label="5日均線"
)

plt.plot(
    plot_date,
    df["MA20"],
    label="20日均線"
)

plt.plot(
    plot_date,
    df["MA60"],
    label="60日均線"
)

# 最新收盤價水平線
plt.axhline(
    y=last_close,
    linestyle="--",
    label=f"最新收盤價：{last_close:.0f}"
)

# 預測區間
plt.axhspan(
    lower_price,
    upper_price,
    alpha=0.25,
    label=(
        f"預測區間："
        f"{lower_price:.0f}"
        f" ~ "
        f"{upper_price:.0f}"
    )
)

# 標題
plt.title(
    f"{stock_name} "
    f"({stock_code}) 股價預測"
)

plt.xlabel("日期")
plt.ylabel("價格")

# 日期旋轉
plt.xticks(rotation=30)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
