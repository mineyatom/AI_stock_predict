from apscheduler.schedulers.background import (
    BackgroundScheduler
)

from predictor import predict_stock
from log_manager import (
    save_prediction_log
)

from datetime import (
    datetime,
    timedelta
)


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
]


# ==========================
# 取得下一交易日
# ==========================
def get_next_trade_date():

    predict_date = datetime.now()

    # 下午 13:00 後
    # 預測下一個交易日
    if predict_date.hour >= 13:

        predict_date = (
            predict_date
            + timedelta(days=1)
        )

    # 跳過六日
    while predict_date.weekday() >= 5:

        predict_date = (
            predict_date
            + timedelta(days=1)
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
# 啟動 Scheduler
# ==========================
def start_scheduler():

    scheduler.add_job(
        run_daily_prediction,

        trigger="cron",

        hour=21,
        minute=0,

        id="daily_prediction_job",

        replace_existing=True,
    )

    scheduler.start()
    print("⏰ 每日預測排程已建立")
    print("Scheduler 已啟動")