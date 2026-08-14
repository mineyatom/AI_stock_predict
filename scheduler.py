from apscheduler.schedulers.background import BackgroundScheduler

from predictor import predict_stock

from ollama_analyzer import analyze_prediction_with_ollama

from log_manager import (
    save_prediction_log,
    update_prediction_result,
    prediction_exists_for_date,
    shift_untraded_prediction_dates,
)

from pytz import timezone
from datetime import datetime, timedelta
import traceback

from market_calendar import get_next_trade_day


# ==========================
# Scheduler
# ==========================

TAIPEI_TZ = timezone("Asia/Taipei")


def taipei_now() -> datetime:
    """取得台北時間，並轉為 naive datetime 以相容既有日期比較邏輯。"""
    return datetime.now(TAIPEI_TZ).replace(tzinfo=None)


scheduler = BackgroundScheduler(
    timezone=TAIPEI_TZ
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
        now = taipei_now()

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

def run_daily_prediction(
    target_date=None
):

    """
    執行每日股票預測。

    target_date:
        None
            → 正常排程模式，自動取得下一交易日

        "2026-08-14"
            → 手動補跑模式，強制將預測日期寫成指定日期
    """

    # ==========================
    # 決定本輪預測日期
    # ==========================

    if target_date is None:

        target_date = get_next_trade_date()

    print(
        f"開始自動預測："
        f"{taipei_now()}"
    )

    print(
        f"[INFO] 本輪預測日期："
        f"{target_date}"
    )

    # ==========================
    # 逐檔預測
    # ==========================

    for stock_id in HOT_STOCKS:

        try:

            # ==========================
            # 模型預測
            # ==========================

            result = predict_stock(
                stock_id
            )

            # ==========================
            # AI 分析
            # ==========================

            ai_analysis = (
                analyze_prediction_with_ollama(
                    result
                )
            )

            # ==========================
            # 預測價格區間
            # ==========================

            lower_price, upper_price = (
                result["price_range"]
                .split(" ~ ")
            )

            # ==========================
            # 寫入 SQLite
            # ==========================

            save_prediction_log(

                # ★ 使用本輪已決定的日期
                predict_date=target_date,

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
                f"預測日期：{target_date}，"
                f"AI摘要已保存"
            )

        except Exception as e:

            print(
                f"[ERROR] {stock_id} "
                f"預測失敗：{e}"
            )
            traceback.print_exc()

    print(
        f"本輪自動預測完成，"
        f"預測日期：{target_date}"
    )


# ==========================
# 補跑遺漏預測
# ==========================

def recover_missing_prediction():

    now = taipei_now()

    # ==========================
    # 尚未到每日預測時間
    # ==========================
    # ==========================
    # 取得應預測的下一交易日
    # ==========================

    target_date = get_next_trade_date()

    print(
        f"[CHECK] 檢查是否需要補預測："
        f"{target_date}"
    )

    # ==========================
    # 已存在預測
    # ==========================

    if prediction_exists_for_date(
        target_date
    ):

        print(
            f"[OK] {target_date} "
            f"已有預測紀錄，不需補跑"
        )

        return

    # ==========================
    # 找不到 → 補跑
    # ==========================

    print(
        f"[WARN] {target_date} "
        f"尚無預測紀錄，開始補跑"
    )

    try:

        # ★ 把已經決定好的日期傳進去
        run_daily_prediction(
            target_date=target_date
        )

    except ValueError as e:

        print(
            f"[WARN] 補預測略過：{e}"
        )

    except Exception as e:

        print(
            f"[ERROR] 補預測失敗：{e}"
        )
        traceback.print_exc()


# ==========================
# 每日驗證流程
# ==========================

def run_daily_validation():

    print(
        f"[INFO] 開始每日休市檢查："
        f"{taipei_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ==========================
    # 處理休市日期
    # ==========================

    shift_untraded_prediction_dates()

    print(
        f"[RUN] 開始每日預測驗證："
        f"{taipei_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ==========================
    # 更新實際結果
    # ==========================

    update_prediction_result()

    print(
        "[OK] 每日驗證流程完成"
    )


# ==========================
# 啟動 Scheduler
# ==========================

def start_scheduler():

    # ==========================
    # 啟動時先執行驗證
    # ==========================

    run_daily_validation()

    # ==========================
    # 啟動時檢查是否漏預測
    # ==========================

    print(
        f"[CHECK] 啟動補預測檢查："
        f"{taipei_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    recover_missing_prediction()

    # ==========================
    # 每日 15:00 驗證
    # ==========================

    scheduler.add_job(

        run_daily_validation,

        trigger="cron",

        hour=15,

        minute=0,

        id="daily_validation_job",

        replace_existing=True,

    )

    # ==========================
    # 每日 21:00 預測下一交易日
    # ==========================

    scheduler.add_job(

        run_daily_prediction,

        trigger="cron",

        hour=21,

        minute=0,

        id="daily_prediction_job",

        replace_existing=True,

    )

    # ==========================
    # 啟動 Scheduler
    # ==========================


    # 每日 02:30：SQLite 缺漏檢查；缺少下一交易日預測才補跑
    scheduler.add_job(
        recover_missing_prediction,
        "cron",
        hour=2,
        minute=30,
        id="prediction_recovery_0230",
        replace_existing=True,
    )

    scheduler.start()

    print(
        f"[INFO] 排程已建立："
        f"{taipei_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"[OK] Scheduler 已啟動："
        f"{taipei_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )