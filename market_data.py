import yfinance as yf
import pandas as pd
from FinMind.data import DataLoader

from datetime import datetime, timedelta


def get_market_data():

    market_data = []

   # ==========================
    # 台股加權（FinMind）
    # ==========================
    try:

        api = DataLoader()

        start_date = (
            datetime.today()
            - timedelta(days=10)
         ).strftime("%Y-%m-%d")

        twii = api.taiwan_stock_daily(
            stock_id="TAIEX",
            start_date=start_date
        )

        latest_close = float(
            twii.iloc[-1]["close"]
        )

        previous_close = float(
            twii.iloc[-2]["close"]
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
                "name": "台股加權",
                "price": format(
                    round(
                        latest_close,
                        2
                    ),
                    ","
                ),
                "change": change_percent,
                "direction": direction
            }
        )

    except Exception as e:

        print(
            f"台股加權錯誤：{e}"
        )

    # ==========================
    # 美股市場（Yahoo）
    # ==========================
    symbols = {
        "SOX 指數": "^SOX",
        "NVDA": "NVDA",
        "QQQ": "QQQ"
    }

    for name, symbol in symbols.items():

        try:

            df = yf.download(
                symbol,
                period="5d",
                auto_adjust=False,
                progress=False
            )

            if (
                df.empty
                or len(df) < 2
            ):
                continue

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
                    "price": format(
                        round(
                            latest_close,
                            2
                        ),
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