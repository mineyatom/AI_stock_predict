from datetime import datetime

from market_calendar import (
    is_scheduled_trade_day,
    has_market_data,
    get_next_trade_day,
)

print("=" * 50)
print("今天")
print("=" * 50)

today = datetime.now()

print(today.strftime("%Y-%m-%d"))
print("預期交易日：", is_scheduled_trade_day(today))
print("已有成交資料：", has_market_data(today))

print()

print("=" * 50)
print("星期六")
print("=" * 50)

sat = datetime(2026, 7, 11)

print(sat.strftime("%Y-%m-%d"))
print("預期交易日：", is_scheduled_trade_day(sat))
print("已有成交資料：", has_market_data(sat))

print()

print("=" * 50)
print("下一交易日")
print("=" * 50)

next_day = get_next_trade_day(today)

print(next_day.strftime("%Y-%m-%d"))