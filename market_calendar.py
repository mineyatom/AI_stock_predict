# market_calendar.py

from datetime import datetime, timedelta
from FinMind.data import DataLoader
from taiwan_holidays.taiwan_calendar import TaiwanCalendar


calendar = TaiwanCalendar()


def has_market_data(
    date_value: datetime | str,
    stock_id: str = "2330"
) -> bool:
    """
    使用 FinMind 確認指定日期是否真的有台股成交資料。
    適合用在驗證，不適合拿來判斷未來交易日。
    """

    if isinstance(date_value, datetime):
        date_str = date_value.strftime("%Y-%m-%d")
    else:
        date_str = date_value

    api = DataLoader()

    try:
        df = api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=date_str,
            end_date=date_str
        )

        return not df.empty

    except Exception as e:
        print(f"[market_calendar] FinMind 查詢失敗：{date_str}, error={e}")
        return False


def is_scheduled_trade_day(date_value: datetime | str) -> bool:
    """
    判斷是否為預期交易日。
    用於未來日期，例如：明天、下一個預測日。
    """

    if isinstance(date_value, str):
        date_obj = datetime.strptime(date_value, "%Y-%m-%d")
    else:
        date_obj = date_value

    if date_obj.weekday() >= 5:
        return False

    date_str = date_obj.strftime("%Y-%m-%d")

    if calendar.is_holiday(date_str):
        return False

    return True


def get_next_trade_day(start_date: datetime | str) -> datetime:
    """
    取得下一個預期交易日。

    用於 Scheduler 的預測日期。

    規則：
    1. 未來日期：用 TaiwanCalendar 判斷
    2. 今天或過去日期：若沒有 FinMind 交易資料，視為非交易日
    3. 最多往後找 30 天
    """

    if isinstance(start_date, str):
        current_date = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )
    else:
        current_date = start_date

    today = datetime.now().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    for _ in range(30):

        current_day = current_date.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        if not is_scheduled_trade_day(
            current_day
        ):
            current_date += timedelta(days=1)
            continue

        # 今天或過去日期，要確認真的有市場資料
        if current_day <= today:
            if not has_market_data(
                current_day
            ):
                current_date += timedelta(days=1)
                continue

        return current_day

    raise ValueError(
        "30 天內找不到下一個可用交易日"
    )


def is_real_trade_day(date_value: datetime | str) -> bool:
    """
    判斷指定日期是否真的有交易。
    用於驗證，不用於未來預測。
    """
    return has_market_data(date_value)