from apscheduler.schedulers.background import (
    BackgroundScheduler
)



from predictor import predict_stock
from log_manager import (
    save_prediction_log,
    update_prediction_result,
    prediction_exists_for_date
)

from datetime import datetime
    


from taiwan_holidays.taiwan_calendar import TaiwanCalendar



import pandas as pd

scheduler = BackgroundScheduler()


# ==========================
# 熱門股票
# ==========================
HOT_STOCKS = [
    "2330",
    "2454",
    "2317",
    "3661",
    "0050",
    "2233",
    "1303",
    "2337",
    "2308",
    "2382",
    "3231",
    "3017",
    "6669"
]


# ==========================
# 取得下一交易日
# ==========================
def get_next_trade_date():

    calendar = TaiwanCalendar()

    today = pd.Timestamp.today().normalize()
    now = pd.Timestamp.now()

    if now.hour >= 13:
        predict_date = today + pd.Timedelta(days=1)
    else:
        predict_date = today

    for _ in range(30):

        date_text = predict_date.strftime(
            "%Y-%m-%d"
        )

        # 平日 + 台灣非休假日
        if (
            predict_date.weekday() < 5
            and not calendar.is_holiday(date_text)
        ):
            
            print(
                f"📅 下一交易日：{date_text}"
)
            return date_text

        predict_date = (
            predict_date
            + pd.Timedelta(days=1)
        )

    print("⚠️ 找不到交易日，改用週末判斷")

    predict_date = today + pd.Timedelta(days=1)

    while predict_date.weekday() >= 5:
        predict_date = (
            predict_date
            + pd.Timedelta(days=1)
        )

    return predict_date.strftime(
        "%Y-%m-%d"
    )


# ==========================
# 每日自動預測
# ==========================
def run_daily_prediction():

    print(
        f"開始自動預測："
        f"{datetime.now()}"
    )

    for stock_id in HOT_STOCKS:

        try:

            result = predict_stock(
                stock_id
            )

            lower_price, upper_price = (
                result["price_range"]
                .split(" ~ ")
            )

            save_prediction_log(

                predict_date=(
                    get_next_trade_date()
                ),

                stock_code=result[
                    "stock_id"
                ],

                stock_name=result[
                    "stock_name"
                ],

                prediction_text=result[
                    "direction"
                ],

                confidence=result[
                    "confidence"
                ],

                up_probability=result[
                    "up_probability"
                ],

                down_probability=result[
                    "down_probability"
                ],

                predict_close=result[
                    "latest_close"
                ],

                lower_price=lower_price,

                upper_price=upper_price,
            )

            print(
                f"{stock_id} "
                f"預測完成，"
                f"已寫入 CSV"
            )

        except Exception as e:

            print(
                f"{stock_id} "
                f"預測失敗：{e}"
            )

    print("本輪自動預測完成")

# ==========================
# 補跑遺漏預測
# ==========================
def recover_missing_prediction():

    target_date = get_next_trade_date()

    print(
        f"🔍 檢查是否需要補預測：{target_date}"
    )

    if prediction_exists_for_date(
        target_date
    ):
        print(
            f"✅ {target_date} 已有預測紀錄，不需補跑"
        )
        return

    print(
        f"⚠️ {target_date} 尚無預測紀錄，開始補跑"
    )

    run_daily_prediction() 



# ==========================
# 啟動 Scheduler
# ==========================
def start_scheduler():

    print(
        f"🕒 啟動補驗證："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    update_prediction_result()

    print(
        f"🧩 啟動補預測檢查："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    recover_missing_prediction()

    scheduler.add_job(
        update_prediction_result,
        trigger="cron",
        hour=15,
        minute=0,
        id="daily_validation_job",
        replace_existing=True,
    )

    scheduler.add_job(
        run_daily_prediction,
        trigger="cron",
        hour=21,
        minute=0,
        id="daily_prediction_job",
        replace_existing=True,
    )

    scheduler.start()

    print(
        f"⏰ 排程已建立："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"✅ Scheduler 已啟動："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    
