import yfinance as yf
import pandas as pd


def get_market_data():

    symbols = {
        "台股加權": "^TWII",
        "SOX 指數": "^SOX",
        "NVDA": "NVDA",
        "QQQ": "QQQ"
    }

    market_data = []

    for name, symbol in symbols.items():

        try:
            df = yf.download(
                symbol,
                period="5d",
                auto_adjust=False,
                progress=False
            )

            if df.empty or len(df) < 2:
                continue

            # ===== 處理 MultiIndex =====
            if isinstance(
                df.columns,
                pd.MultiIndex
            ):
                df.columns = (
                    df.columns
                    .get_level_values(0)
                )

            close_data = (
                df["Close"]
                .dropna()
            )

            if len(close_data) < 2:
                continue

            latest_close = float(
                close_data.iloc[-1]
            )

            previous_close = float(
                close_data.iloc[-2]
            )

            change_percent = round(
                (
                    (
                        latest_close
                        - previous_close
                    )
                    / previous_close
                ) * 100,
                2
            )

            direction = (
                "rise"
                if change_percent >= 0
                else "fall"
            )

            market_data.append(
                {
                    "name": name,
                    "price":format(round(
                        latest_close,2),
                        ","
                    ),
                    "change": change_percent,
                    "direction": direction
                }
            )

        except Exception as e:
            print(
                f"{symbol} 錯誤：{e}"
            )

    return market_data