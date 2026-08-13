import requests
import pandas as pd

URL = "https://api.finmindtrade.com/api/v4/data"

symbols = [
    "NVDA",
    "QQQ",
    "^SOX",
    "SOX"
]

for symbol in symbols:

    print("\n==========================")
    print(f"測試：{symbol}")
    print("==========================")

    try:
        params = {
            "dataset": "USStockPrice",
            "data_id": symbol,
            "start_date": "2026-08-01"
        }

        response = requests.get(
            URL,
            params=params,
            timeout=30
        )

        result = response.json()

        data = pd.DataFrame(
            result.get("data", [])
        )

        if data.empty:
            print(f"❌ FinMind 找不到 {symbol}")
        else:
            print(f"✅ FinMind 支援 {symbol}")
            print(data.tail())

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")