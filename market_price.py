import yfinance as yf
from datetime import datetime


def get_stock_price(stock_id):

    try:
        ticker = yf.Ticker(stock_id)

        history = ticker.history(period="10d")

        if history.empty or "Close" not in history.columns:
            return None

        close_data = history["Close"].dropna()

        if len(close_data) < 2:
            return None

        current_price = round(
            float(close_data.iloc[-1]),
            2
        )

        previous_close = float(
            close_data.iloc[-2]
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
            close_data.index[-1]
            .strftime("%Y/%m/%d")
        )

        return {
            "price": current_price,
            "change": abs(change_percent),
            "direction": direction,
            "market_datetime": f"{latest_market_date} 13:30",
            "updated_at": datetime.now().strftime(
                "%Y/%m/%d %H:%M"
            )
        }

    except Exception as e:
        print(f"股價取得失敗：{e}")
        return None