import os
import pandas as pd
import yfinance as yf

LOG_FILE = "prediction_log.csv"


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
    date = pd.to_datetime(predict_date)

    predict_date = (
        f"{date.year}/"
        f"{date.month}/"
        f"{date.day}"
    )

    new_record = pd.DataFrame([
        {
            "預測日期": predict_date,
            "股票代號": stock_code,
            "股票名稱": stock_name,
            "預測結果": prediction_text,
            "信心值": round(float(confidence), 2),
            "上漲機率": round(float(up_probability), 2),
            "下跌機率": round(float(down_probability), 2),
            "隔日預測參考價": round(float(predict_close)),
            "預測區間下緣": round(float(lower_price)),
            "預測區間上緣": round(float(upper_price)),
            "實際收盤價": None,
            "實際漲跌": None,
            "是否預測正確": None
        }
    ])

    if os.path.exists(LOG_FILE):
        old_log = pd.read_csv(LOG_FILE, encoding="utf-8-sig")

        duplicated = (
            (old_log["預測日期"] == predict_date)
            &
            (old_log["股票代號"] == stock_code)
        )

        if duplicated.any():
            old_log.loc[
                duplicated,
                new_record.columns
            ] = new_record.iloc[0].values

            final_log = old_log

        else:
            final_log = pd.concat(
                [old_log, new_record],
                ignore_index=True
            )

    else:
        final_log = new_record

    final_log.to_csv(
        LOG_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 預測紀錄已存入 prediction_log.csv")




def get_latest_prediction():

    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        return None

    latest = df.iloc[-1]

    return {
         "predict_date":
        latest["預測日期"],

    "stock_id":
        latest["股票代號"],

    "stock_name":
        latest["股票名稱"],

    "prediction":
        latest["預測結果"],

    "confidence":
        latest["信心值"],

    "lower_bound":
        latest["預測區間下緣"],

    "upper_bound":
        latest["預測區間上緣"],

    "model":
        "XGBoost"
    }



def get_prediction_history():

    if not os.path.exists(LOG_FILE):
        return {
            "history": [],
            "accuracy": 0,
            "validated_count": 0,
            "total_count": 0
        }

    df = pd.read_csv(
        LOG_FILE,
        encoding="utf-8-sig"
    )

    if df.empty:
        return {
            "history": [],
            "accuracy": 0,
            "validated_count": 0,
            "total_count": 0
        }

    total_count = len(df)

    validated_df = df[
        df["是否預測正確"].notna()
        & (
            df["是否預測正確"] != ""
        )
    ]

    validated_count = len(
        validated_df
    )

    if validated_count > 0:

        correct_count = len(
            validated_df[
                validated_df[
                    "是否預測正確"
                ] == "正確"
            ]
        )

        accuracy = round(
            (
                correct_count
                / validated_count
            ) * 100,
            2
        )

    else:
        accuracy = 0

    history = df.to_dict(
        orient="records"
    )

    return {
        "history": history,
        "accuracy": accuracy,
        "validated_count": validated_count,
        "total_count": total_count
    }


def get_accuracy_chart_data():

    if not os.path.exists(LOG_FILE):
        return {
            "labels": [],
            "values": []
        }

    df = pd.read_csv(
        LOG_FILE,
        encoding="utf-8-sig"
    )

    if df.empty:
        return {
            "labels": [],
            "values": []
        }

    df = df[
        df["是否預測正確"].notna()
        & (df["是否預測正確"] != "")
    ]

    if df.empty:
        return {
            "labels": [],
            "values": []
        }

    # 日期轉換
    df["預測日期"] = pd.to_datetime(
        df["預測日期"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["預測日期"]
    )

    # 每日統計
    daily_result = []

    grouped = df.groupby(
        df["預測日期"].dt.strftime("%Y/%m/%d")
    )

    for date, group in grouped:

        total_count = len(group)

        correct_count = len(
            group[
                group["是否預測正確"] == "正確"
            ]
        )

        accuracy = round(
            correct_count / total_count * 100,
            2
        )

        daily_result.append({
            "date": date,
            "accuracy": accuracy
        })

    daily_result = daily_result[-30:]

    labels = [
        item["date"]
        for item in daily_result
    ]

    values = [
        item["accuracy"]
        for item in daily_result
    ]

    return {
        "labels": labels,
        "values": values
    }


def update_prediction_result():

    print(
        "🔥 update_prediction_result 啟動"
    )

    if not os.path.exists(
        LOG_FILE
    ):
        print(
            "❌ 找不到 prediction_log.csv"
        )
        return

    df_log = pd.read_csv(
        LOG_FILE,
        encoding="utf-8-sig",
        dtype={
            "實際漲跌": "object",
            "是否預測正確": "object"
        }
    )

    if df_log.empty:
        print(
            "❌ prediction_log.csv 沒有資料"
        )
        return

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    for index, row in (
        df_log.iterrows()
    ):

        # 已驗證過跳過
        if (
            pd.notna(
                row["是否預測正確"]
            )
            and row[
                "是否預測正確"
            ] != ""
        ):
            continue

        predict_date = (
            pd.to_datetime(
                row["預測日期"],
                errors="coerce"
            )
        )

        if pd.isna(
            predict_date
        ):
            print(
                f"⚠️ 日期格式錯誤：第 {index + 2} 列"
            )
            continue

        predict_date = (
            predict_date
            .normalize()
        )

        # 今天以前才驗證
        if predict_date >= today:
            continue

        stock_code = str(
            row["股票代號"]
        )

        # 自動補 .TW
        if (
            ".TW"
            not in stock_code
        ):
            stock_code = (
                stock_code
                + ".TW"
            )

        try:

            ticker = yf.Ticker(
                stock_code
            )

            history = (
                ticker.history(
                    start=(
                        predict_date
                        + pd.Timedelta(
                            days=1
                        )
                    ).strftime(
                        "%Y-%m-%d"
                    ),

                    end=(
                        predict_date
                        + pd.Timedelta(
                            days=7
                        )
                    ).strftime(
                        "%Y-%m-%d"
                    )
                )
            )

            if history.empty:
                print(
                    f"⚠️ 無交易資料：{stock_code}"
                )
                continue

            actual_close = float(
                history[
                    "Close"
                ]
                .dropna()
                .iloc[0]
            )

            reference_close = float(
                row[
                    "隔日預測參考價"
                ]
            )

            prediction = row[
                "預測結果"
            ]

            if (
                actual_close
                > reference_close
            ):
                actual_direction = "上漲"

            elif (
                actual_close
                < reference_close
            ):
                actual_direction = "下跌"

            else:
                actual_direction = "持平"

            result_text = (
                "正確"
                if prediction
                == actual_direction
                else "錯誤"
            )

            df_log.loc[
                index,
                "實際收盤價"
            ] = round(
                actual_close,
                2
            )

            df_log.loc[
                index,
                "實際漲跌"
            ] = (
                actual_direction
            )

            df_log.loc[
                index,
                "是否預測正確"
            ] = result_text

            print(
                f"✅ 已更新："
                f"{stock_code} "
                f"{predict_date.date()}"
            )

        except Exception as e:

            print(
                f"⚠️ 更新失敗："
                f"{stock_code}"
                f" 原因：{e}"
            )

    df_log.to_csv(
        LOG_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "✅ 預測驗證更新完成"
    )
