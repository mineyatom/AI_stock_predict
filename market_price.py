import yfinance as yf
from datetime import datetime


def get_stock_price(stock_id):

    try:
        ticker = yf.Ticker(stock_id)

        history = ticker.history(period="5d")

        if history.empty or len(history) < 2:
            return None

        current_price = round(
            float(history["Close"].iloc[-1]),
            2
        )

        previous_close = float(
            history["Close"].iloc[-2]
        )

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
            history.index[-1]
            .strftime("%Y/%m/%d")
        )

        return {
            "price": current_price,
            "change": abs(change_percent),
            "direction": direction,

            # 股價資料本身的時間
            "market_datetime": (
                f"{latest_market_date} 13:30"
            ),

            # 你的系統抓資料的時間
            "updated_at": datetime.now().strftime(
                "%Y/%m/%d %H:%M"
            )
        }

    except Exception as e:
        print(f"股價取得失敗：{e}")
        return None