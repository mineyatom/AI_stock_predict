import os
import pandas as pd
import yfinance as yf

from datetime import  timedelta

from market_calendar import (
    get_market_data_status,
    get_next_trade_day,
    can_verify_market_data,
)

from prediction_repository import (
    create_prediction,
    get_prediction_history_from_db,
    get_stock_accuracy_stats_from_db,
    get_validated_predictions_from_db,
    update_prediction_date,
    update_prediction_validation,
    get_unvalidated_predictions_from_db,
    get_latest_prediction_from_db,
    prediction_exists_for_date_from_db,
    get_accuracy_chart_data_from_db,
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
    upper_price,
):
    """
    將預測結果寫入 SQLite。

    SQLite 是唯一正式資料來源，
    不再同步寫入 prediction_log.csv。
    """

    normalized_predict_date = (
        pd.to_datetime(
            predict_date,
            errors="coerce",
        )
    )

    if pd.isna(normalized_predict_date):
        print(
            f"❌ 無效的預測日期：{predict_date}"
        )
        return False

    normalized_predict_date = (
        normalized_predict_date
        .strftime("%Y-%m-%d")
    )

    try:
        success = create_prediction(
            predict_date=normalized_predict_date,
            stock_code=str(stock_code).strip(),
            stock_name=str(stock_name).strip(),
            prediction_text=str(
                prediction_text
            ).strip(),
            confidence=round(
                float(confidence),
                2,
            ),
            up_probability=round(
                float(up_probability),
                2,
            ),
            down_probability=round(
                float(down_probability),
                2,
            ),
            predict_close=round(
                float(predict_close),
                2,
            ),
            lower_price=round(
                float(lower_price),
                2,
            ),
            upper_price=round(
                float(upper_price),
                2,
            ),
        )

        if success:
            print(
                "✅ 預測紀錄已存入 SQLite："
                f"{stock_code} "
                f"{normalized_predict_date}"
            )
            return True

        print(
            "ℹ️ 預測紀錄已存在或未新增："
            f"{stock_code} "
            f"{normalized_predict_date}"
        )
        return False

    except Exception as e:
        print(
            "❌ SQLite 預測紀錄寫入失敗："
            f"{stock_code} "
            f"{normalized_predict_date}，"
            f"原因：{e}"
        )
        return False




def get_latest_prediction():
    """
    從 SQLite 取得最新預測。
    """

    return get_latest_prediction_from_db()



def get_prediction_history():
    """
    從 SQLite 取得歷史預測資料。
    """

    return get_prediction_history_from_db()

def get_accuracy_chart_data():
    """
    從 SQLite 取得每日勝率圖表。
    """

    return get_accuracy_chart_data_from_db()


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

        # ==========================
        # 尚未到市場確認時間，不可判定休市
        # ==========================
        if not can_verify_market_data(
            predict_date.to_pydatetime()
        ):
            print(
                f"⏳ 尚未到市場確認時間，保留原日期："
                f"{predict_date.date()}"
            )
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


def shift_untraded_prediction_dates():
    """
    檢查 SQLite 中尚未驗證的預測。

    若預測日期遇到颱風假、臨時休市或無成交資料，
    將預測日期順延至下一個預定交易日。
    """

    print("🔄 開始檢查 SQLite 休市預測日期")

    predictions = (
        get_unvalidated_predictions_from_db()
    )

    if not predictions:
        print("✅ SQLite 沒有待檢查的預測")
        return

    today = pd.Timestamp.today().normalize()
    now = pd.Timestamp.now()

    market_verify_time = (
        today
        + pd.Timedelta(
            hours=15,
            minutes=0,
        )
    )

    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for prediction_data in predictions:
        predict_date = pd.to_datetime(
            prediction_data["predict_date"],
            errors="coerce",
        )

        if pd.isna(predict_date):
            print(
                "⚠️ SQLite 日期格式錯誤："
                f"{prediction_data['predict_date']}"
            )
            failed_count += 1
            continue

        predict_date = predict_date.normalize()

        stock_code = str(
            prediction_data["stock_code"]
        ).strip()

        # 未來日期不能判斷是否真正休市
        if predict_date > today:
            skipped_count += 1
            continue

        # 當天 15:00 前不判斷休市
        if (
            predict_date == today
            and now < market_verify_time
        ):
            print(
                "⏳ 尚未到休市確認時間："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            skipped_count += 1
            continue

        try:
            market_status = (
                get_market_data_status(
                    predict_date.strftime(
                        "%Y-%m-%d"
                    )
                )
            )

            if market_status is None:
                print(
                    "⚠️ 無法確認市場狀態："
                    f"{stock_code} "
                    f"{predict_date.date()}"
                )
                failed_count += 1
                continue

            # 有交易，不需順延
            if market_status is True:
                skipped_count += 1
                continue

            next_trade_date = (
                get_next_trade_day(
                    (
                        predict_date
                        + pd.Timedelta(days=1)
                    ).to_pydatetime()
                )
            )

            next_trade_date_str = (
                pd.to_datetime(
                    next_trade_date
                ).strftime("%Y-%m-%d")
            )

            success = update_prediction_date(
                old_predict_date=(
                    predict_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                new_predict_date=(
                    next_trade_date_str
                ),
                stock_code=stock_code,
            )

            if not success:
                print(
                    "⚠️ SQLite 日期順延失敗："
                    f"{stock_code} "
                    f"{predict_date.date()}"
                )
                failed_count += 1
                continue

            updated_count += 1

            print(
                "📅 SQLite 預測日期已順延："
                f"{stock_code} "
                f"{predict_date.date()} "
                f"→ {next_trade_date_str}"
            )

        except Exception as e:
            failed_count += 1

            print(
                "⚠️ 休市日期檢查失敗："
                f"{stock_code}，"
                f"原因：{e}"
            )

    print()
    print("=" * 50)
    print("SQLite 休市日期檢查完成")
    print("=" * 50)
    print(f"更新：{updated_count}")
    print(f"略過：{skipped_count}")
    print(f"失敗：{failed_count}")        


def update_prediction_result():
    """
    從 SQLite 讀取尚未驗證的預測，
    取得實際收盤價後直接更新 SQLite。

    CSV 不再作為驗證流程的主資料來源。
    """

    print("🔥 update_prediction_result 啟動")

    predictions = get_unvalidated_predictions_from_db()

    if not predictions:
        print("✅ SQLite 沒有待驗證的預測")
        return

    today = pd.Timestamp.today().normalize()
    now = pd.Timestamp.now()

    market_verify_time = (
        today
        + pd.Timedelta(
            hours=15,
            minutes=0
        )
    )

    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for prediction_data in predictions:

        predict_date = pd.to_datetime(
            prediction_data["predict_date"],
            errors="coerce"
        )

        if pd.isna(predict_date):
            print(
                f"⚠️ SQLite 日期格式錯誤："
                f"{prediction_data['predict_date']}"
            )
            failed_count += 1
            continue

        predict_date = predict_date.normalize()

        stock_code = str(
            prediction_data["stock_code"]
        ).strip()

        # ==========================
        # 驗證時間控制
        # ==========================

        if predict_date > today:
            print(
                f"⏳ 尚未到預測日期："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            skipped_count += 1
            continue

        if (
            predict_date == today
            and now < market_verify_time
        ):
            print(
                f"⏳ 尚未到驗證時間："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            skipped_count += 1
            continue

        # ==========================
        # FinMind 確認是否有交易
        # ==========================

        market_status = get_market_data_status(
            predict_date.strftime("%Y-%m-%d")
        )

        if market_status is None:
            print(
                f"⚠️ 無法確認市場狀態："
                f"{predict_date.date()}"
            )
            failed_count += 1
            continue

        if market_status is False:
            print(
                f"🌀 無實際交易資料，略過驗證："
                f"{stock_code} "
                f"{predict_date.date()}"
            )
            skipped_count += 1
            continue

        # ==========================
        # Yahoo Finance 股票代號
        # ==========================

        yahoo_stock_code = stock_code

        if (
            ".TW" not in yahoo_stock_code
            and ".TWO" not in yahoo_stock_code
        ):
            yahoo_stock_code = (
                yahoo_stock_code
                + ".TW"
            )

        try:
            ticker = yf.Ticker(
                yahoo_stock_code
            )

            history = ticker.history(
                start=predict_date.strftime(
                    "%Y-%m-%d"
                ),
                end=(
                    predict_date
                    + pd.Timedelta(days=7)
                ).strftime(
                    "%Y-%m-%d"
                )
            )

            if history.empty:
                print(
                    f"⚠️ Yahoo 無交易資料："
                    f"{yahoo_stock_code}"
                )
                failed_count += 1
                continue

            close_series = (
                history["Close"]
                .dropna()
            )

            if close_series.empty:
                print(
                    f"⚠️ Yahoo 收盤價為空："
                    f"{yahoo_stock_code}"
                )
                failed_count += 1
                continue

            actual_close = float(
                close_series.iloc[0]
            )

            reference_close = float(
                prediction_data["predict_close"]
            )

            prediction_text = str(
                prediction_data["prediction_text"]
            ).strip()

            if actual_close > reference_close:
                actual_direction = "上漲"

            elif actual_close < reference_close:
                actual_direction = "下跌"

            else:
                actual_direction = "持平"

            result_text = (
                "正確"
                if prediction_text == actual_direction
                else "錯誤"
            )

            sqlite_updated = update_prediction_validation(
                predict_date=predict_date.strftime(
                    "%Y-%m-%d"
                ),
                stock_code=stock_code,
                actual_close=round(
                    actual_close,
                    2
                ),
                actual_change=actual_direction,
                is_correct=result_text,
            )

            if not sqlite_updated:
                print(
                    f"⚠️ SQLite 驗證更新失敗："
                    f"{stock_code} "
                    f"{predict_date.date()}"
                )
                failed_count += 1
                continue

            updated_count += 1

            print(
                f"✅ SQLite 已驗證："
                f"{stock_code} "
                f"{predict_date.date()} "
                f"{result_text}"
            )

        except Exception as e:
            failed_count += 1

            print(
                f"⚠️ 驗證失敗："
                f"{stock_code} "
                f"原因：{e}"
            )

    print()
    print("=" * 50)
    print("SQLite 預測驗證完成")
    print("=" * 50)
    print(f"成功：{updated_count}")
    print(f"略過：{skipped_count}")
    print(f"失敗：{failed_count}")

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
    
      
def prediction_exists_for_date(
    predict_date
):
    """
    檢查指定日期是否已有預測。
    """

    return prediction_exists_for_date_from_db(
        predict_date
    )
       
