import yfinance as yf


def get_stock_price(stock_id):

    try:
        ticker = yf.Ticker(stock_id)

        history = ticker.history(period="2d")

        if history.empty or len(history) < 2:
            return None

        current_price = round(float(history["Close"].iloc[-1]), 2)
        previous_close = float(history["Close"].iloc[-2])

        change_percent = round(
            ((current_price - previous_close) / previous_close) * 100,
            2
        )

        direction = "rise" if change_percent >= 0 else "fall"

        return {
            "price": current_price,
            "change": abs(change_percent),
            "direction": direction
        }

    except Exception as e:
        print(f"股價取得失敗：{e}")
        return None