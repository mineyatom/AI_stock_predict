import os
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

from market_calendar import (
    get_market_data_status,
    get_next_trade_day,
    has_market_data,
)

from prediction_repository import (
    create_prediction,
    get_prediction_history_from_db,
    get_stock_accuracy_stats_from_db,
    get_validated_predictions_from_db,
    update_prediction_date,
    update_prediction_validation,
)
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
        old_log = pd.read_csv(
        LOG_FILE,
        encoding="utf-8-sig",
        dtype={
            "股票代號": str
            }
        )

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

    # ==========================
    # 同步寫入 SQLite
    # ==========================
    create_prediction(
        predict_date=str(predict_date),
        stock_code=str(stock_code),
        stock_name=str(stock_name),
        prediction_text=str(prediction_text),
        confidence=float(confidence),
        up_probability=float(up_probability),
        down_probability=float(down_probability),
        predict_close=float(predict_close),
        lower_price=float(lower_price),
        upper_price=float(upper_price)
    )




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
    """
    從 SQLite 取得歷史預測資料。
    """

    return get_prediction_history_from_db()

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

def get_stock_accuracy_stats():
    """
    從 SQLite 取得各股票模型準確率。
    """

    return get_stock_accuracy_stats_from_db()

# ==========================
# 休市預測日期自動順延
# ==========================
def shift_untraded_prediction_dates():
    """
    將因颱風假或臨時休市而沒有交易的預測日期，
    自動順延到下一個預期交易日。
    """

    print("🔄 開始檢查休市預測日期")

    if not os.path.exists(LOG_FILE):
        print("❌ 找不到 prediction_log.csv")
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
        print("❌ prediction_log.csv 沒有資料")
        return

    today = pd.Timestamp.today().normalize()
    updated = False

    for index, row in df_log.iterrows():

        # 已驗證完成，不處理
        if (
            pd.notna(row["是否預測正確"])
            and str(row["是否預測正確"]).strip()
            not in ("", "nan")
        ):
            continue

        predict_date = pd.to_datetime(
            row["預測日期"],
            errors="coerce"
        )

        if pd.isna(predict_date):
            continue

        predict_date = predict_date.normalize()

        # 未來日期暫時不處理
        if predict_date > today:
            continue

        market_status = get_market_data_status(
            predict_date.strftime("%Y-%m-%d")
        )

        # FinMind 查詢失敗，不修改日期
        if market_status is None:
            print(
                f"⚠️ 無法確認市場狀態，保留原日期："
                f"{predict_date.date()}"
            )
            continue

        # 有交易資料，不需順延
        if market_status is True:
            continue

        # 沒有交易資料，尋找下一個預期交易日
        next_date = get_next_trade_day(
            predict_date.to_pydatetime()
            + timedelta(days=1)
        )

        next_date_str = next_date.strftime("%Y-%m-%d")
        old_date_str = predict_date.strftime("%Y-%m-%d")


        # ==========================
        # 同步更新 SQLite
        # ==========================
        sqlite_updated = update_prediction_date(
            old_predict_date=old_date_str,
            new_predict_date=next_date_str,
            stock_code=str(row["股票代號"]),
        )


        # ==========================
        #更新 CSV
        # ==========================
        df_log.loc[
            index,
            "預測日期"
        ] = next_date_str

        updated = True

        print(
            f"📅 預測日期已順延："
            f"{row['股票代號']} "
            f"{predict_date.date()} → {next_date_str}"
        )

    if updated:
        df_log.to_csv(
            LOG_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print("✅ 休市預測日期順延完成")

    else:
        print("✅ 沒有需要順延的預測日期")



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

    now = pd.Timestamp.now()

    market_verify_time = (
        today
        + pd.Timedelta(
            hours=15,
            minutes=0
        )
    )

    for index, row in (
        df_log.iterrows()
    ):

        # 已驗證過跳過
        if (
            pd.notna(
                row["是否預測正確"]
            )
            and str(
                row["是否預測正確"]
            ).strip()
            != ""
            and str(
                row["是否預測正確"]
            ).strip()
            != "nan"
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

        stock_code = str(
            row["股票代號"]
        )

        # ==========================
        # 驗證時間控制
        # ==========================

        # 未來日期不驗證
        if predict_date > today:
            print(
                f"⏳ 尚未到預測日期："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            continue

        # 今天的預測，15:00 後才驗證
        if (
            predict_date == today
            and now < market_verify_time
        ):
            print(
                f"⏳ 尚未到驗證時間："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            continue

        
        # ==========================
        # 確認是否真的有市場資料
        # ==========================
        if not has_market_data(
            predict_date.strftime("%Y-%m-%d")
        ):
            print(
                    f"🌀 無實際交易資料，略過驗證："
                    f"{stock_code} "
                    f"{predict_date.date()}"
            )
            continue

        # 自動補 .TW
        if (
            ".TW"
            not in stock_code
            and ".TWO"
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

            history = ticker.history(
                start=predict_date.strftime(
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

            prediction = str(prediction).strip()
            actual_direction = str(actual_direction).strip()
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
            ] = actual_direction

            df_log.loc[
                index,
                "是否預測正確"
            ] = result_text


            # ==========================
            # 同步更新 SQLite
            # ==========================
            update_prediction_validation(
                predict_date=str(row["預測日期"]),
                stock_code=str(row["股票代號"]),
                actual_close=round(actual_close, 2),
                actual_change=actual_direction,
                is_correct=result_text,
            )

            print(
                f"✅ 已更新："
                f"{stock_code} "
                f"{predict_date.date()}"
            )

        except Exception as e:

            print(
                f"⚠️ 更新失敗："
                f"{stock_code} "
                f"原因：{e}"
            )

    df_log.to_csv(
        LOG_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "✅ 預測驗證更新完成"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

def get_confidence_stats():
    """
    從 SQLite 計算各信心區間實際勝率。
    """

    try:
        data = get_validated_predictions_from_db()

        if not data:
            return []

        df = pd.DataFrame(data)

        df["confidence"] = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["confidence"]
        )

        bins = [
            0,
            60,
            70,
            80,
            100,
        ]

        labels = [
            "50~60%",
            "60~70%",
            "70~80%",
            "80%以上",
        ]

        df["confidence_range"] = pd.cut(
            df["confidence"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        result = []

        for label in labels:
            group = df[
                df["confidence_range"] == label
            ]

            if group.empty:
                continue

            accuracy = (
                (
                    group["is_correct"]
                    == "正確"
                )
                .mean()
                * 100
            )

            result.append({
                "range": label,
                "count": len(group),
                "accuracy": round(
                    accuracy,
                    2
                ),
            })

        return result

    except Exception as e:
        print(
            f"信心區間分析失敗: {e}"
        )

        return []
    

def get_recent_accuracy_stats():
    """
    從 SQLite 計算最近 10 / 20 / 30 筆
    已驗證預測的勝率。
    """

    try:
        data = get_validated_predictions_from_db()

        if not data:
            return []

        df = pd.DataFrame(data)

        recent_settings = [
            10,
            20,
            30,
        ]

        result = []

        for n in recent_settings:
            recent_df = df.tail(n)

            if recent_df.empty:
                continue

            accuracy = (
                (
                    recent_df["is_correct"]
                    == "正確"
                )
                .mean()
                * 100
            )

            result.append({
                "label": f"最近{n}筆",
                "accuracy": round(
                    accuracy,
                    2
                ),
                "count": len(recent_df),
            })

        return result

    except Exception as e:
        print(
            f"近期勝率分析失敗: {e}"
        )

        return []
    

def get_high_confidence_accuracy(
    threshold=70
):
    """
    從 SQLite 計算高信心預測的實際勝率。
    預設信心值 >= 70%。
    """

    try:
        data = get_validated_predictions_from_db()

        if not data:
            return {
                "accuracy": 0,
                "count": 0,
                "label": f"信心≥{threshold}%",
            }

        df = pd.DataFrame(data)

        df["confidence"] = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["confidence"]
        )

        high_confidence_df = df[
            df["confidence"] >= threshold
        ]

        if high_confidence_df.empty:
            return {
                "accuracy": 0,
                "count": 0,
                "label": f"信心≥{threshold}%",
            }

        accuracy = (
            (
                high_confidence_df[
                    "is_correct"
                ]
                == "正確"
            )
            .mean()
            * 100
        )

        return {
            "accuracy": round(
                accuracy,
                2
            ),
            "count": len(
                high_confidence_df
            ),
            "label": f"信心≥{threshold}%",
        }

    except Exception as e:
        print(
            f"高信心勝率分析失敗: {e}"
        )

        return {
            "accuracy": 0,
            "count": 0,
            "label": f"信心≥{threshold}%",
        }  
    
      
def prediction_exists_for_date(predict_date):
    """
    檢查指定預測日期是否已經有預測紀錄
    """

    try:
        df = pd.read_csv("prediction_log.csv")

        if df.empty:
            return False

        df["預測日期"] = pd.to_datetime(
            df["預測日期"]
        ).dt.strftime("%Y-%m-%d")

        predict_date = pd.to_datetime(
            predict_date
        ).strftime("%Y-%m-%d")

        exists = (
            df["預測日期"] == predict_date
        ).any()

        return exists

    except Exception as e:
        print(f"檢查預測紀錄失敗：{e}")
        return False    
