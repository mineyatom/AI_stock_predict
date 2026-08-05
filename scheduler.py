from apscheduler.schedulers.background import (
    BackgroundScheduler
)


from predictor import predict_stock


from ollama_analyzer import (
    analyze_prediction_with_ollama
)


from log_manager import (
    save_prediction_log,
    update_prediction_result,
    prediction_exists_for_date,
    shift_untraded_prediction_dates,
)
from pytz import timezone

from datetime import datetime, timedelta


from market_calendar import get_next_trade_day



scheduler = BackgroundScheduler(
    timezone=timezone("Asia/Taipei")
)



# ==========================
# 熱門股票
# ==========================

HOT_STOCKS = [
    "2330",
    "2454",
    "2317",
    "3661",
    "6669",
    "2382",
    "3231",
    "2356",
    "3017",
    "3443",
    "2308",
    "3711",
    "3037",
    "2379",
    "2408",
    "2337",
    "2357",
    "1303",
    "0050",
    "2345",
    "2303",
    
]



# ==========================
# 取得下一交易日
# ==========================

def get_next_trade_date(
    now: datetime | None = None
) -> str:

    """
    取得下一個真實交易日。
    """

    if now is None:
        now = datetime.now()


    market_close_time = now.replace(
        hour=13,
        minute=30,
        second=0,
        microsecond=0
    )


    if now >= market_close_time:

        start_date = (
            now + timedelta(days=1)
        )

    else:

        start_date = now


    next_trade_day = get_next_trade_day(
        start_date
    )


    return next_trade_day.strftime(
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


            # ==========================
            # 模型預測
            # ==========================

            result = predict_stock(
                stock_id
            )


            # ==========================
            # Gemini AI 分析
            # ==========================

            ai_analysis = (
                analyze_prediction_with_ollama(
                    result
                )
            )


            lower_price, upper_price = (
                result["price_range"]
                .split(" ~ ")
            )



            # ==========================
            # 寫入 SQLite
            # ==========================

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

                ai_summary=ai_analysis,

            )


            print(
                f"{stock_id} "
                f"預測完成，"
                f"AI摘要已保存"
            )


        except Exception as e:

            print(
                f"{stock_id} "
                f"預測失敗：{e}"
            )


    print(
        "本輪自動預測完成"
    )



# ==========================
# 補跑遺漏預測
# ==========================

def recover_missing_prediction():

    now = datetime.now()

    # 尚未到每日預測時間，不補跑
    if now.hour < 21:

        print(
            "[CHECK] 尚未到預測時間，不補跑"
        )

        return


    target_date = get_next_trade_date()


    print(
        f"[CHECK] 檢查是否需要補預測：{target_date}"
    )


    if prediction_exists_for_date(target_date):

        print(
            f"[OK] {target_date} 已有預測紀錄，不需補跑"
        )

        return


    print(
        f"[WARN] {target_date} 尚無預測紀錄，開始補跑"
    )


    try:

        run_daily_prediction()


    except ValueError as e:

        print(
            f"[WARN] 補預測略過：{e}"
        )


    except Exception as e:

        print(
            f"[ERROR] 補預測失敗：{e}"
        )



# ==========================
# 每日驗證流程
# ==========================

def run_daily_validation():

    print(
        f"[INFO] 開始每日休市檢查："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    shift_untraded_prediction_dates()


    print(
        f"[RUN] 開始每日預測驗證："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    update_prediction_result()


    print(
        "[OK] 每日驗證流程完成"
    )



# ==========================
# 啟動 Scheduler
# ==========================

def start_scheduler():


    run_daily_validation()



    print(
        f"[CHECK] 啟動補預測檢查："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    recover_missing_prediction()



    scheduler.add_job(

        run_daily_validation,

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
        f"[INFO] 排程已建立："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    print(
        f"[OK] Scheduler 已啟動："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )