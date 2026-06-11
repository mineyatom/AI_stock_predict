

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import LogisticRegression

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error
)

from datetime import datetime, timedelta



import ta.trend
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta


# ===== 股票代號 =====
stock_code = input("請輸入股票代號：")


## ===== 抓股票名稱 =====
ticker = yf.Ticker(stock_code)

try:
    info = ticker.info

    stock_name = (
        info.get("longName") or
        info.get("shortName") or
        stock_code
    )

except:
    stock_name = stock_code


# ===== 抓股票資料 =====
df = yf.download(stock_code, start="2015-01-01",auto_adjust=False)

# ===== 保留原始股價資料 =====
raw_df = df.copy()


# ===== 嘗試抓 Yahoo 最新價格 =====
ticker = yf.Ticker(stock_code)

try:
    latest_close = ticker.fast_info[
        "last_price"
    ]
except:
    latest_close = float(
        raw_df["Close"]
        .dropna()
        .squeeze()
        .iloc[-1]
    )

#  =====修正 MultiIndex 問題 =====
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ====== 技術指標 ======

# 報酬率
df["Return"] = df["Close"].pct_change()

df["Return_1"] = df["Return"].shift(1)

df["Return_2"] = df["Return"].shift(2)

df["Return_3"] = df["Return"].shift(3)

df["Price_Change"] = (df["Close"] - df["Open"])
# 波動率
df["Volatility"] = (df["Return"].rolling(10).std() )

# RSI
df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"],n=14).rsi()


# ====== MACD ======
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Diff"] = macd.macd_diff()


# ====== KD 指標 ======
stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    n=14,
    d_n=3
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

# ===== 清理空值 =====
df.dropna(inplace=True)

# ===== 價格預測 Target =====
y_price = df["Tomorrow_Close"]



# ====== Feature(特徵) =====
X = df[
    [
        
        "Volume_MA5",
        "Return",
        "Return_1",
        "Return_2",
        "Return_3",
        "Volatility",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Diff",
        "K",
        "D",
    ]

]
print(X.columns)
# ===== Label =====
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


# ===== 價格預測資料切分 =====
y_price_train = y_price.iloc[:split_index]
y_price_test = y_price.iloc[split_index:]


# ===== 儲存最佳模型 =====
best_model_name = ""
best_model = None
best_accuracy = 0

# ===== 建立模型 =====
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost":
      XGBClassifier(
          n_estimators=300,
          learning_rate=0.05,
          max_depth=5,
          random_state=42,
      )
}

# ===== 價格預測模型 =====
price_models = {
    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost Regressor":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
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

# ===== 用最佳分類模型重新訓練 =====
best_model.fit(X, y)



# 最新一天資料
latest_data = X.iloc[[-1]]

# 預測明天漲跌
prediction = best_model.predict(latest_data)[0]

# 預測機率
probability = best_model.predict_proba(latest_data)[0]

# 上漲與下跌機率
up_probability = probability[1] * 100
down_probability = probability[0] * 100


# ===== 價格預測 =====

best_price_model = None
best_mae = 999999
best_price_model_name = ""

for model_name, model in price_models.items():

    # 訓練價格模型
    model.fit(X_train, y_price_train)

    # 預測價格
    y_price_pred = model.predict(X_test)

    # 平均誤差（越低越好）
    mae = mean_absolute_error(
        y_price_test,
        y_price_pred
    )

    # 記錄最佳價格模型
    if mae < best_mae:
        best_mae = mae
        best_price_model = model
        best_price_model_name = model_name


# ===== 預測明天收盤價 =====
predicted_price = best_price_model.predict(
    latest_data
)[0]


# ===== AI 股票最終結果 =====

print("\n🤖" + "=" * 50)
print("          AI 股票預測系統 V1")
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

print("【最佳分類模型】")
print(f"模型名稱：{best_model_name}")
print(f"模型準確率：{best_accuracy * 100:.2f}%")

print("-" * 52)

print("【明天漲跌預測】")

if prediction == 1:
    confidence = up_probability
    print("預測結果：上漲 📈")
else:
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
    * 0.20
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

print("-" * 52)
print("提醒：此結果僅供學習與研究，不構成投資建議。")
print("=" * 52 + "🤖")


# ===== 畫圖 =====
plt.figure(figsize=(14, 7))

plt.plot(df.index, df["Close"], label="Close Price")
plt.plot(df.index, df["MA5"], label="MA5")
plt.plot(df.index, df["MA20"], label="MA20")
plt.plot(df.index, df["MA60"], label="MA60")

# 最新收盤價水平線
plt.axhline(
    y=last_close,
    linestyle="--",
    label=f"Latest Close: {last_close:.2f}"
)

# 預測區間
plt.axhspan(
    lower_price,
    upper_price,
    alpha=0.15,
    label=f"Predicted Range: {lower_price:.2f} ~ {upper_price:.2f}"
)

plt.title(f"{stock_name} ({stock_code}) Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()