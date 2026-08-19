from datetime import datetime, timedelta

from FinMind.data import DataLoader
from pytz import timezone


# ==========================
# 台灣時區
# ==========================

TAIPEI_TZ = timezone("Asia/Taipei")


api = DataLoader()


def get_stock_price(stock_id):
    """
    使用 FinMind 取得台股最新收盤價
    避免 yfinance 台股資料延遲或價格尺度不一致
    """

    try:
        finmind_stock_id = (
            stock_id
            .replace(".TW", "")
            .replace(".TWO", "")
        )

        # 使用台灣時間計算資料查詢起始日期
        start_date = (
            datetime.now(TAIPEI_TZ)
            - timedelta(days=14)
        ).strftime("%Y-%m-%d")

        df = api.taiwan_stock_daily(
            stock_id=finmind_stock_id,
            start_date=start_date
        )

        if df.empty:
            return None

        df = df.dropna(
            subset=["close"]
        )

        if len(df) < 2:
            return None

        df = df.sort_values("date")

        current_row = df.iloc[-1]
        previous_row = df.iloc[-2]

        current_price = round(
            float(current_row["close"]),
            2
        )

        previous_close = float(
            previous_row["close"]
        )

        if previous_close == 0:
            return None

        change_percent = round(
            (
                (current_price - previous_close)
                / previous_close
            ) * 100,
            2
        )

        direction = (
            "rise"
            if change_percent >= 0
            else "fall"
        )

        latest_market_date = (
            current_row["date"]
            .replace("-", "/")
        )

        return {
            "price": current_price,
            "change": abs(change_percent),
            "direction": direction,

            # FinMind 最新交易日收盤時間
            "market_datetime": (
                f"{latest_market_date} 13:30"
            ),

            # 網頁顯示的資料更新時間固定使用台灣時間
            "updated_at": datetime.now(
                TAIPEI_TZ
            ).strftime(
                "%Y/%m/%d %H:%M"
            )
        }

    except Exception as e:
        print(
            f"FinMind 股價取得失敗：{e}"
        )
        return None